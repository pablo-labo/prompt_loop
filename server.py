import argparse
import base64
import io
import json
import os
import random
import threading
import time
import urllib.parse
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from comfy_client import ComfyClient, MockComfyClient
from rl_engine import RLEngine
from storage import LoopStorage
from tag_classifier import CURATED_EXPANSION_PACKS, SLOT_META, SLOT_ORDER
from tagger import LocalWDTagger

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
CONFIG_EXAMPLE_PATH = BASE_DIR / "config.example.json"
TAG_DICT_PATH = BASE_DIR / "tag_dict.json"
WEB_INDEX_PATH = BASE_DIR / "web" / "index.html"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "loop.db"
IMAGES_DIR = DATA_DIR / "images"

DEFAULT_PORT = 7866
MAX_BODY_BYTES = 30 * 1024 * 1024


def _decode_image_to_png_bytes(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if "," in payload and payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    raw_bytes = base64.b64decode(payload, validate=False)
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


class PromptLoopApp:
    """Coordinator holding config, storage, tagger, RL engine, and batch workers."""

    def __init__(self) -> None:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self._cfg_lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._batch_timing: dict[int, dict[str, Any]] = {}
        self._manual_timing: dict[int, dict[str, Any]] = {}
        self._tag_dict_lock = threading.Lock()
        self._tag_dict_cache: tuple[float, bytes] | None = None
        self.storage = LoopStorage(DB_PATH)
        self._cleanup_orphaned_batches()
        self.tagger = LocalWDTagger()
        self.rl_engine = RLEngine(self.storage, self.tagger)

    def _cleanup_orphaned_batches(self) -> None:
        active_id = self.storage.has_active_batch()
        if active_id is not None:
            self.storage.update_batch(
                active_id,
                status="error",
                progress_text="上次生成被中断",
                error_message="服务在上次生成中途关闭，请重新生成新批次。",
            )
        active_manual_id = self.storage.has_active_manual_generation()
        if active_manual_id is not None:
            self.storage.update_manual_generation(
                active_manual_id,
                status="error",
                progress_text="上次生成被中断",
                error_message="服务在上次生成中途关闭，请重新提交生成。",
            )

    def _ensure_config_exists(self) -> None:
        if CONFIG_PATH.is_file():
            return
        if CONFIG_EXAMPLE_PATH.is_file():
            CONFIG_PATH.write_text(
                CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return
        fallback = {
            "comfy_url": "http://127.0.0.1:8188",
            "mock_mode": False,
            "enable_wd14": True,
            "batch_size": 2,
            "workflow_path": "workflows/anima_base_skp.json",
            "workflow_nodes": {
                "unet": "1",
                "positive": "4",
                "negative": "5",
                "latent": "6",
                "sampler": "7",
                "save": "9",
                "lora": "10",
            },
            "generation": {"width": 748, "height": 512, "steps": 30, "cfg": 4},
            "loras": [],
            "fixed_prefix": "masterpiece, best quality, score_7, score_8, very aesthetic, absurdres, highres, 1girl",
            "negative_base": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, extra limbs, extra fingers, artist name, watermark, text",
            "rl": {
                "prior_m": 3.0,
                "cluster_prior_alpha": 0.25,
                "ucb_beta": 0.45,
                "weight_decay": 0.985,
                "cooldown_rounds": 3,
                "cooldown_penalty": 0.45,
                "temp_start": 1.2,
                "temp_min": 0.55,
                "temp_decay_rounds": 40,
                "max_prob_cap": 0.45,
                "dropout_prob": 0.25,
                "auto_promote": True,
                "promote_min_obs": 2,
                "promote_min_gain": 0.25,
                "max_auto_negatives": 10,
                "synergy_weight": 0.55,
                "target_tag_min": 5,
                "target_tag_max": 8,
                "explore_slots_per_img": 1,
                "explore_wip_limit": 25,
                "contrast_prob": 0.5,
                "contrast_same_seed": True,
                "pair_explore_beta": 0.25,
                "pair_decay": 0.997,
            },
        }
        CONFIG_PATH.write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_config(self) -> dict[str, Any]:
        with self._cfg_lock:
            self._ensure_config_exists()
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def load_tag_dict_bytes(self) -> bytes:
        with self._tag_dict_lock:
            if not TAG_DICT_PATH.is_file():
                return b'{"entries":{}}'
            mtime = TAG_DICT_PATH.stat().st_mtime
            if self._tag_dict_cache is not None and self._tag_dict_cache[0] == mtime:
                return self._tag_dict_cache[1]
            raw = TAG_DICT_PATH.read_bytes()
            self._tag_dict_cache = (mtime, raw)
            return raw

    @staticmethod
    def _normalize_loras(raw_loras: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_loras, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in raw_loras:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            cleaned.append(
                {
                    "name": name,
                    "strength": round(float(item.get("strength", 1.0)), 4),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        return cleaned

    def save_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._cfg_lock:
            self._ensure_config_exists()
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            updated = _deep_merge(current, patch)
            updated["batch_size"] = max(1, min(4, int(updated.get("batch_size", 2))))
            if isinstance(updated.get("generation"), dict):
                updated["generation"].pop("lora_strength", None)
            if "loras" in updated:
                updated["loras"] = self._normalize_loras(updated["loras"])
            CONFIG_PATH.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return updated

    def make_client(self, cfg: dict[str, Any]) -> ComfyClient | MockComfyClient:
        if bool(cfg.get("mock_mode", False)):
            return MockComfyClient()
        wf_path = Path(cfg.get("workflow_path", "workflows/anima_base_skp.json"))
        if not wf_path.is_absolute():
            wf_path = BASE_DIR / wf_path
        return ComfyClient(
            base_url=str(cfg.get("comfy_url", "http://127.0.0.1:8188")),
            workflow_path=wf_path,
            node_map=dict(cfg.get("workflow_nodes", {})),
        )

    def get_status(self) -> dict[str, Any]:
        cfg = self.load_config()
        client = self.make_client(cfg)
        comfy_health = client.check_health()

        tags = self.storage.get_all_tags()
        stage_counts = {"core": 0, "explore": 0, "staging": 0, "negative": 0, "ignored": 0}
        for t in tags:
            st = t["stage"]
            if st in stage_counts:
                stage_counts[st] += 1

        active_loras = [
            item
            for item in self._normalize_loras(cfg.get("loras", []))
            if item["enabled"] and abs(item["strength"]) > 1e-6
        ]

        return {
            "comfy": comfy_health,
            "mock_mode": bool(cfg.get("mock_mode", False)),
            "active_loras": active_loras,
            "wd14": {
                "available": self.tagger.is_available(),
                "enabled": bool(cfg.get("enable_wd14", True)),
                "model_dir": str(self.tagger.model_dir),
            },
            "meta": {
                "round_num": int(self.storage.get_meta("round_num", "0")),
                "reward_ema": round(float(self.storage.get_meta("reward_ema", "0.0")), 4),
            },
            "active_batch_id": self.storage.has_active_batch(),
            "stage_counts": stage_counts,
        }

    def enrich_batch(self, batch: dict[str, Any] | None) -> dict[str, Any] | None:
        if batch is None:
            return None
        if batch.get("status") == "generating":
            bid = int(batch["id"])
            with self._timing_lock:
                t = self._batch_timing.get(bid)
                t_copy = dict(t) if t is not None else None
            if t_copy is not None:
                now_mono = time.monotonic()
                batch["elapsed_sec"] = round(max(0.0, now_mono - t_copy["batch_started_at"]), 2)
                batch["step_elapsed_sec"] = round(max(0.0, now_mono - t_copy["step_started_at"]), 2)
                batch["expected_total"] = int(t_copy["expected_total"])
                batch["current_index"] = int(t_copy["current_index"])
                batch["phase"] = str(t_copy["phase"])
            else:
                try:
                    created = datetime.strptime(
                        batch["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)
                    elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
                except Exception:
                    elapsed = 0.0
                cfg = self.load_config()
                batch["elapsed_sec"] = round(elapsed, 2)
                batch["step_elapsed_sec"] = round(elapsed, 2)
                batch["expected_total"] = max(
                    int(cfg.get("batch_size", 2)),
                    len(batch.get("generations", [])),
                )
                batch["current_index"] = len(batch.get("generations", []))
                batch["phase"] = "generating"
        return batch

    def start_batch_generation(
        self, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.storage.has_active_batch() is not None:
            raise RuntimeError("当前已有正在生成的批次，请等待完成")

        data = payload or {}
        prompt_mode = str(data.get("prompt_mode", "auto")).strip().lower()
        if prompt_mode not in ("auto", "manual", "hybrid", "contrast"):
            raise ValueError(f"无效的生成模式: {prompt_mode}")

        raw_bs = data.get("batch_size")
        batch_size_override: int | None = None
        if raw_bs is not None and str(raw_bs).strip() != "":
            batch_size_override = max(1, min(4, int(raw_bs)))

        cfg = self.load_config()
        run_cfg = dict(cfg)
        run_gen = dict(cfg.get("generation", {}))

        raw_w = data.get("width")
        raw_h = data.get("height")
        has_w = raw_w is not None and str(raw_w).strip() != ""
        has_h = raw_h is not None and str(raw_h).strip() != ""

        if has_w:
            w = int(raw_w)
            if not (64 <= w <= 4096):
                raise ValueError("宽度需在 64..4096 之间")
            run_gen["width"] = w
        if has_h:
            h = int(raw_h)
            if not (64 <= h <= 4096):
                raise ValueError("高度需在 64..4096 之间")
            run_gen["height"] = h
        if not has_w and not has_h and bool(data.get("swap_aspect", False)):
            run_gen["width"], run_gen["height"] = (
                int(run_gen.get("height", 512)),
                int(run_gen.get("width", 512)),
            )
        run_cfg["generation"] = run_gen

        mode = "mock" if bool(run_cfg.get("mock_mode", False)) else "comfyui"

        if prompt_mode == "manual":
            plans = self.rl_engine.build_manual_batch_prompts(
                run_cfg,
                positive_prompt=str(data.get("positive_prompt", "")),
                negative_prompt=str(data.get("negative_prompt", "")),
                include_prefix=bool(data.get("include_prefix", True)),
                batch_size=batch_size_override,
            )
        elif prompt_mode == "hybrid":
            raw_slots = data.get("roll_slots")
            roll_slots = raw_slots if isinstance(raw_slots, list) else None
            plans = self.rl_engine.build_hybrid_batch_prompts(
                run_cfg,
                positive_prompt=str(data.get("positive_prompt", "")),
                negative_prompt=str(data.get("negative_prompt", "")),
                include_prefix=bool(data.get("include_prefix", True)),
                batch_size=batch_size_override,
                roll_slots=roll_slots,
            )
        elif prompt_mode == "contrast":
            raw_slots = data.get("roll_slots")
            roll_slots = raw_slots if isinstance(raw_slots, list) else None
            same_seed = bool(data.get("same_seed", True))
            raw_vc = data.get("variant_count")
            variant_count = (
                int(raw_vc)
                if (raw_vc is not None and str(raw_vc).strip() != "")
                else None
            )
            plans = self.rl_engine.build_contrast_batch_prompts(
                run_cfg,
                positive_prompt=str(data.get("positive_prompt", "")),
                negative_prompt=str(data.get("negative_prompt", "")),
                include_prefix=bool(data.get("include_prefix", True)),
                batch_size=batch_size_override,
                roll_slots=roll_slots,
                same_seed=same_seed,
                variant_count=variant_count,
            )
        else:
            plans = self.rl_engine.sample_batch_prompts(
                run_cfg,
                batch_size_override=batch_size_override,
            )

        batch_id = self.storage.create_batch(
            mode=mode,
            progress_text="正在提交生成队列...",
            prompt_mode=prompt_mode,
        )

        now_mono = time.monotonic()
        with self._timing_lock:
            self._batch_timing[batch_id] = {
                "batch_started_at": now_mono,
                "step_started_at": now_mono,
                "expected_total": len(plans),
                "current_index": 0,
                "phase": "queued",
            }

        worker = threading.Thread(
            target=self._run_batch_worker,
            args=(batch_id, plans, run_cfg),
            daemon=True,
        )
        worker.start()
        return {
            "batch_id": batch_id,
            "status": "generating",
            "prompt_mode": prompt_mode,
            "width": int(run_gen.get("width", 512)),
            "height": int(run_gen.get("height", 512)),
        }

    def _run_batch_worker(
        self,
        batch_id: int,
        plans: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> None:
        client = self.make_client(cfg)
        gen_cfg = dict(cfg.get("generation", {}))
        enable_wd14 = bool(cfg.get("enable_wd14", True)) and self.tagger.is_available()
        fixed_prefix = str(cfg.get("fixed_prefix", ""))
        total = len(plans)

        try:
            for idx, plan in enumerate(plans):
                mode_label = "Mock 本地画图" if isinstance(client, MockComfyClient) else "远端 ComfyUI 生图"
                with self._timing_lock:
                    if batch_id in self._batch_timing:
                        self._batch_timing[batch_id].update(
                            {
                                "step_started_at": time.monotonic(),
                                "current_index": idx,
                                "phase": "generating",
                            }
                        )
                self.storage.update_batch(
                    batch_id,
                    progress_text=f"{mode_label}中（第 {idx + 1}/{total} 张）...",
                )
                gen_out = client.generate_one(
                    positive=plan["positive_prompt"],
                    negative=plan["negative_prompt"],
                    seed=plan["seed"],
                    gen_cfg=gen_cfg,
                    loras=self._normalize_loras(cfg.get("loras", [])),
                )

                img_name = f"batch_{batch_id}_slot_{plan['slot_index']}_{plan['seed']}.png"
                img_path = IMAGES_DIR / img_name
                img_path.write_bytes(gen_out["image_bytes"])

                detected_tags: list[dict[str, Any]] = []
                emergent_tags: list[dict[str, Any]] = []
                ratings: list[dict[str, Any]] = []
                tag_elapsed_ms = 0.0

                if enable_wd14:
                    with self._timing_lock:
                        if batch_id in self._batch_timing:
                            self._batch_timing[batch_id].update(
                                {
                                    "step_started_at": time.monotonic(),
                                    "current_index": idx,
                                    "phase": "tagging",
                                }
                            )
                    self.storage.update_batch(
                        batch_id,
                        progress_text=f"正在运行本地 WD14 识图（第 {idx + 1}/{total} 张）...",
                    )
                    tag_out = self.tagger.tag_image_bytes(gen_out["image_bytes"])
                    detected_tags = tag_out["tags"]
                    ratings = tag_out["ratings"]
                    tag_elapsed_ms = float(tag_out["elapsed_ms"])
                    emergent_tags = self.rl_engine.compute_emergent_tags(
                        positive_prompt=plan["positive_prompt"],
                        sampled_tags=plan["sampled_tags"],
                        detected_tags=detected_tags,
                        fixed_prefix=fixed_prefix,
                    )

                self.storage.add_generation(
                    batch_id=batch_id,
                    slot_index=plan["slot_index"],
                    seed=plan["seed"],
                    is_dropout_round=plan["is_dropout_round"],
                    positive_prompt=plan["positive_prompt"],
                    negative_prompt=plan["negative_prompt"],
                    sampled_tags=plan["sampled_tags"],
                    detected_tags=detected_tags,
                    emergent_tags=emergent_tags,
                    ratings=ratings,
                    image_rel_path=img_name,
                    comfy_prompt_id=str(gen_out.get("comfy_prompt_id", "")),
                    gen_elapsed_ms=float(gen_out.get("elapsed_ms", 0.0)),
                    tag_elapsed_ms=tag_elapsed_ms,
                )

            self.storage.update_batch(
                batch_id,
                status="ready",
                progress_text="生成与识图完成，等待评分",
            )
        except Exception as exc:
            self.storage.update_batch(
                batch_id,
                status="error",
                progress_text="生成失败",
                error_message=str(exc),
            )
        finally:
            with self._timing_lock:
                self._batch_timing.pop(batch_id, None)

    def enrich_manual_gen(self, item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None:
            return None
        if item.get("status") == "generating":
            gid = int(item["id"])
            with self._timing_lock:
                t = self._manual_timing.get(gid)
                t_copy = dict(t) if t is not None else None
            if t_copy is not None:
                item["elapsed_sec"] = round(
                    max(0.0, time.monotonic() - t_copy["started_at"]), 2
                )
                item["phase"] = str(t_copy["phase"])
            else:
                try:
                    created = datetime.strptime(
                        item["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)
                    elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
                except Exception:
                    elapsed = 0.0
                item["elapsed_sec"] = round(elapsed, 2)
                item["phase"] = "generating"
        return item

    def start_manual_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.storage.has_active_manual_generation() is not None:
            raise RuntimeError("当前已有正在进行的自由生图任务，请等待完成")

        positive = str(payload.get("positive_prompt", "")).strip()
        if not positive:
            raise ValueError("正向 Prompt 不能为空")

        cfg = self.load_config()
        gen_defaults = dict(cfg.get("generation", {}))
        negative = str(
            payload.get("negative_prompt")
            if payload.get("negative_prompt") is not None
            else cfg.get("negative_base", "")
        ).strip()

        raw_seed = payload.get("seed")
        if raw_seed is None or str(raw_seed).strip() == "" or int(raw_seed) <= 0:
            seed = random.randint(1, 2**31 - 1)
        else:
            seed = int(raw_seed)

        width = int(payload.get("width") or gen_defaults.get("width", 512))
        height = int(payload.get("height") or gen_defaults.get("height", 512))
        steps = int(payload.get("steps") or gen_defaults.get("steps", 30))
        cfg_val = float(payload.get("cfg") if payload.get("cfg") is not None else gen_defaults.get("cfg", 4.0))
        run_wd14 = bool(payload.get("run_wd14", False))

        loras = self._normalize_loras(cfg.get("loras", []))
        mode = "mock" if bool(cfg.get("mock_mode", False)) else "comfyui"

        gen_id = self.storage.create_manual_generation(
            mode=mode,
            positive_prompt=positive,
            negative_prompt=negative,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg_val,
            loras=loras,
            progress_text="正在提交生图任务...",
        )

        with self._timing_lock:
            self._manual_timing[gen_id] = {
                "started_at": time.monotonic(),
                "phase": "queued",
            }

        worker = threading.Thread(
            target=self._run_manual_worker,
            args=(gen_id, positive, negative, seed, width, height, steps, cfg_val, loras, run_wd14, cfg),
            daemon=True,
        )
        worker.start()
        return {"id": gen_id, "seed": seed, "status": "generating"}

    def _run_manual_worker(
        self,
        gen_id: int,
        positive: str,
        negative: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_val: float,
        loras: list[dict[str, Any]],
        run_wd14: bool,
        cfg: dict[str, Any],
    ) -> None:
        client = self.make_client(cfg)
        mode_label = "Mock 本地画图" if isinstance(client, MockComfyClient) else "远端 ComfyUI 生图"
        try:
            with self._timing_lock:
                if gen_id in self._manual_timing:
                    self._manual_timing[gen_id]["phase"] = "generating"
            self.storage.update_manual_generation(
                gen_id,
                progress_text=f"{mode_label}中...",
            )

            gen_out = client.generate_one(
                positive=positive,
                negative=negative,
                seed=seed,
                gen_cfg={
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg_val,
                },
                loras=loras,
            )

            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            img_name = f"manual_{gen_id}_{seed}.png"
            (IMAGES_DIR / img_name).write_bytes(gen_out["image_bytes"])

            detected_tags: list[dict[str, Any]] = []
            tag_elapsed_ms = 0.0
            if run_wd14 and self.tagger.is_available():
                with self._timing_lock:
                    if gen_id in self._manual_timing:
                        self._manual_timing[gen_id]["phase"] = "tagging"
                self.storage.update_manual_generation(
                    gen_id,
                    progress_text="正在运行本地 WD14 识图...",
                )
                tag_out = self.tagger.tag_image_bytes(gen_out["image_bytes"])
                detected_tags = tag_out["tags"]
                tag_elapsed_ms = float(tag_out["elapsed_ms"])

            self.storage.update_manual_generation(
                gen_id,
                status="done",
                progress_text="生成完成",
                image_rel_path=img_name,
                comfy_prompt_id=str(gen_out.get("comfy_prompt_id", "")),
                gen_elapsed_ms=float(gen_out.get("elapsed_ms", 0.0)),
                tag_elapsed_ms=tag_elapsed_ms,
                detected_tags=detected_tags,
            )
        except Exception as exc:
            self.storage.update_manual_generation(
                gen_id,
                status="error",
                progress_text="生成失败",
                error_message=str(exc),
            )
        finally:
            with self._timing_lock:
                self._manual_timing.pop(gen_id, None)


def make_handler(app: PromptLoopApp) -> type[BaseHTTPRequestHandler]:
    class LoopHTTPHandler(BaseHTTPRequestHandler):
        server_version = "PromptLoop/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            raw_len = self.headers.get("Content-Length")
            if not raw_len:
                return {}
            length = int(raw_len)
            if length > MAX_BODY_BYTES:
                raise OverflowError("请求体过大")
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/":
                if not WEB_INDEX_PATH.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "未找到 web/index.html"})
                    return
                html_bytes = WEB_INDEX_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            if path.startswith("/images/"):
                rel_name = urllib.parse.unquote(path[len("/images/") :])
                if not rel_name or Path(rel_name).name != rel_name:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "非法文件名"})
                    return
                img_path = (IMAGES_DIR / rel_name).resolve()
                if img_path.parent != IMAGES_DIR.resolve() or not img_path.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "图片不存在"})
                    return
                img_bytes = img_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(img_bytes)
                return

            if path == "/api/status":
                self._send_json(HTTPStatus.OK, app.get_status())
                return

            if path == "/api/config":
                self._send_json(HTTPStatus.OK, app.load_config())
                return

            if path == "/api/tag_dict":
                raw_bytes = app.load_tag_dict_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw_bytes)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw_bytes)
                return

            if path == "/api/loras":
                cfg = app.load_config()
                client = app.make_client(cfg)
                remote = client.list_loras()
                remote["configured_loras"] = app._normalize_loras(cfg.get("loras", []))
                self._send_json(HTTPStatus.OK, remote)
                return

            if path == "/api/batches":
                self._send_json(HTTPStatus.OK, {"batches": app.storage.list_batches(limit=30)})
                return

            if path == "/api/batches/latest":
                self._send_json(
                    HTTPStatus.OK,
                    {"batch": app.enrich_batch(app.storage.get_latest_batch())},
                )
                return

            if path.startswith("/api/batches/"):
                id_str = path[len("/api/batches/") :]
                if id_str.isdigit():
                    batch = app.storage.get_batch(int(id_str))
                    if batch is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "批次不存在"})
                    else:
                        self._send_json(HTTPStatus.OK, {"batch": app.enrich_batch(batch)})
                    return

            if path == "/api/tags":
                qs = urllib.parse.parse_qs(parsed.query)
                stage = qs.get("stage", [None])[0]
                slot = qs.get("slot", [None])[0]
                sort_mode = qs.get("sort", ["default"])[0]
                cfg = app.load_config()
                prior_m = float(cfg["rl"].get("prior_m", 3.0))
                tags = app.storage.get_all_tags(stage=stage, slot=slot)
                for t in tags:
                    obs_c = int(t["observed_count"])
                    obs_eff = float(t.get("observed_eff_count", 0.0))
                    denom_cnt = obs_eff if obs_eff > 1e-6 else float(obs_c)
                    t["observed_gain"] = (
                        round(float(t["observed_gain_sum"]) / (denom_cnt + prior_m), 4)
                        if obs_c > 0
                        else 0.0
                    )
                    trials = int(t["exec_trials"])
                    t["exec_rate"] = (
                        round(int(t["exec_hits"]) / trials, 4) if trials > 0 else None
                    )
                if sort_mode == "gain_asc":
                    tags.sort(
                        key=lambda t: (
                            float(t["observed_gain"]),
                            -int(t["observed_count"]),
                            str(t["tag"]),
                        )
                    )
                elif stage == "staging":
                    tags.sort(
                        key=lambda t: (
                            -float(t["observed_gain"]),
                            -int(t["observed_count"]),
                            str(t["tag"]),
                        )
                    )
                elif stage == "negative":
                    tags.sort(
                        key=lambda t: (
                            float(t["weight"]),
                            float(t["observed_gain"]),
                            str(t["tag"]),
                        )
                    )
                all_db_tags = app.storage.get_all_tags() if (stage or slot) else tags
                by_lower_all = {
                    str(x["tag"]).strip().lower(): str(x["stage"]) for x in all_db_tags
                }
                expansion_packs = []
                for key, pack in CURATED_EXPANSION_PACKS.items():
                    pack_tags = pack.get("tags", [])
                    new_cnt = sum(
                        1
                        for item in pack_tags
                        if item["tag"].strip().lower() not in by_lower_all
                    )
                    stg_cnt = sum(
                        1
                        for item in pack_tags
                        if by_lower_all.get(item["tag"].strip().lower()) == "staging"
                    )
                    expansion_packs.append(
                        {
                            "key": key,
                            "label": pack["label"],
                            "description": pack["description"],
                            "total_count": len(pack_tags),
                            "new_count": new_cnt,
                            "staging_count": stg_cnt,
                            "preview_tags": [item["tag"] for item in pack_tags],
                        }
                    )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "tags": tags,
                        "clusters": app.storage.get_all_clusters(),
                        "slot_order": SLOT_ORDER,
                        "slot_meta": SLOT_META,
                        "slot_summary": app.storage.get_stage_slot_summary(),
                        "expansion_packs": expansion_packs,
                        "meta": {
                            "round_num": int(app.storage.get_meta("round_num", "0")),
                            "reward_ema": round(
                                float(app.storage.get_meta("reward_ema", "0.0")), 4
                            ),
                        },
                    },
                )
                return

            if path == "/api/external/history":
                self._send_json(
                    HTTPStatus.OK,
                    {"imports": app.storage.list_external_imports(limit=30)},
                )
                return

            if path == "/api/manual/history":
                self._send_json(
                    HTTPStatus.OK,
                    {"items": app.storage.list_manual_generations(limit=30)},
                )
                return

            if path.startswith("/api/manual/"):
                id_str = path[len("/api/manual/") :]
                if id_str.isdigit():
                    item = app.storage.get_manual_generation(int(id_str))
                    if item is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "记录不存在"})
                    else:
                        self._send_json(HTTPStatus.OK, {"item": app.enrich_manual_gen(item)})
                    return

            if path == "/api/synergy/graph":
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    min_co = max(1, int(qs.get("min_co", ["2"])[0]))
                    inc_raw = str(qs.get("include_staging", ["0"])[0]).strip().lower()
                    include_staging = inc_raw in ("1", "true", "yes")
                    max_edges = max(20, min(600, int(qs.get("max_edges", ["260"])[0])))
                    metric = str(qs.get("metric", ["affinity"])[0]).strip().lower()
                    graph = app.rl_engine.get_synergy_graph(
                        app.load_config(),
                        min_co_count=min_co,
                        include_staging=include_staging,
                        max_edges=max_edges,
                        metric=metric,
                    )
                    self._send_json(HTTPStatus.OK, graph)
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            try:
                data = self._read_json()
            except OverflowError as exc:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"无效 JSON: {exc}"})
                return

            if path == "/api/config":
                try:
                    updated = app.save_config(data)
                    self._send_json(HTTPStatus.OK, updated)
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path == "/api/batches/generate":
                try:
                    res = app.start_batch_generation(data)
                    self._send_json(HTTPStatus.ACCEPTED, res)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except RuntimeError as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            if path.startswith("/api/batches/") and path.endswith("/rate"):
                id_part = path[len("/api/batches/") : -len("/rate")]
                if not id_part.isdigit():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "无效的 batch_id"})
                    return
                batch_id = int(id_part)
                ratings_input = data.get("ratings")
                if not isinstance(ratings_input, list):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 ratings 数组"})
                    return
                try:
                    cfg = app.load_config()
                    summary = app.rl_engine.apply_batch_ratings(batch_id, ratings_input, cfg)
                    batch = app.storage.get_batch(batch_id)
                    self._send_json(HTTPStatus.OK, {"summary": summary, "batch": batch})
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            if path == "/api/tags/action":
                action = str(data.get("action", "")).strip()
                try:
                    cfg = app.load_config()
                    res = app.rl_engine.perform_tag_action(action, data, cfg)
                    self._send_json(HTTPStatus.OK, res)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            if path == "/api/external/preview":
                image_base64 = data.get("image_base64")
                if not image_base64 or not isinstance(image_base64, str):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 image_base64 字段"})
                    return
                if not app.tagger.is_available():
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": f"本地 WD14 模型不可用: {app.tagger.model_dir}"},
                    )
                    return
                threshold = float(data.get("threshold", 0.35))
                try:
                    png_bytes = _decode_image_to_png_bytes(image_base64)
                    tag_out = app.tagger.tag_image_bytes(png_bytes)
                    preview_tags = app.rl_engine.preview_external_tags(
                        tag_out["tags"], threshold=threshold
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "elapsed_ms": tag_out["elapsed_ms"],
                            "ratings": tag_out["ratings"],
                            "detected_tags": tag_out["tags"],
                            "preview_tags": preview_tags,
                        },
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"图片解析或识图失败: {exc}"})
                return

            if path == "/api/external/apply":
                image_base64 = data.get("image_base64")
                if not image_base64 or not isinstance(image_base64, str):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 image_base64 字段"})
                    return
                selected_tags = data.get("selected_tags")
                if not isinstance(selected_tags, list):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 selected_tags 数组"})
                    return
                filename = str(data.get("filename", "")).strip()
                rating = data.get("rating", 2)
                direct_explore = bool(data.get("direct_explore", False))
                detected_tags = data.get("detected_tags") or []
                ratings = data.get("ratings") or []
                tag_elapsed_ms = float(data.get("tag_elapsed_ms", 0.0))
                try:
                    cfg = app.load_config()
                    png_bytes = _decode_image_to_png_bytes(image_base64)
                    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    img_name = f"ext_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.png"
                    (IMAGES_DIR / img_name).write_bytes(png_bytes)

                    summary = app.rl_engine.apply_external_image(
                        selected_tags=selected_tags,
                        rating=int(rating),
                        direct_explore=direct_explore,
                        config=cfg,
                    )
                    import_id = app.storage.add_external_import(
                        filename=filename,
                        image_rel_path=img_name,
                        rating=int(rating),
                        direct_explore=direct_explore,
                        tag_elapsed_ms=tag_elapsed_ms,
                        detected_tags=detected_tags if isinstance(detected_tags, list) else [],
                        applied_tags=selected_tags,
                        ratings=ratings if isinstance(ratings, list) else [],
                        summary=summary,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "import_id": import_id,
                            "image_rel_path": img_name,
                            "summary": summary,
                        },
                    )
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            if path == "/api/manual/roll":
                try:
                    cfg = app.load_config()
                    mode = str(data.get("mode", "weighted"))
                    slots = data.get("slots")
                    if slots is not None and not isinstance(slots, list):
                        slots = None
                    include_prefix = bool(data.get("include_prefix", True))
                    res = app.rl_engine.roll_manual_prompt(
                        cfg,
                        mode=mode,
                        slots=slots,
                        include_prefix=include_prefix,
                    )
                    self._send_json(HTTPStatus.OK, res)
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path == "/api/manual/generate":
                try:
                    res = app.start_manual_generation(data)
                    self._send_json(HTTPStatus.ACCEPTED, res)
                except RuntimeError as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            if path == "/api/synergy/rebuild":
                try:
                    cfg = app.load_config()
                    pair_sum = app.rl_engine.rebuild_pair_stats(cfg)
                    obs_sum = app.rl_engine.rebuild_observed_eff_counts(cfg)
                    summary = {**pair_sum, **obs_sum}
                    self._send_json(HTTPStatus.OK, {"ok": True, "summary": summary})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})

    return LoopHTTPHandler


def bind_server(start_port: int, app: PromptLoopApp) -> tuple[ThreadingHTTPServer, int]:
    handler_cls = make_handler(app)
    last_err: OSError | None = None
    for port in range(start_port, start_port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
            return server, port
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f"端口 {start_port}~{start_port + 9} 均已被占用: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anima Prompt 闭环演化工作台")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    args = parser.parse_args()

    app = PromptLoopApp()
    server, port = bind_server(args.port, app)
    url = f"http://127.0.0.1:{port}/"
    print(f"Anima Prompt 闭环工作台已启动: {url}")
    print("按 Ctrl+C 可停止服务")

    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()
        app.storage.close()


if __name__ == "__main__":
    main()
