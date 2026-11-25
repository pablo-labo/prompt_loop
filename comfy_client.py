import copy
import hashlib
import io
import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _should_bypass_proxy(base_url: str) -> bool:
    host = (urllib.parse.urlparse(base_url).hostname or "").strip().lower()
    if not host or host in ("localhost", "0.0.0.0", "::1") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


class ComfyClient:
    """HTTP API client for remote ComfyUI accessed over SSH port forwarding."""

    def __init__(
        self,
        base_url: str,
        workflow_path: Path,
        node_map: dict[str, str],
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.workflow_path = workflow_path
        self.node_map = node_map
        self.client_id = str(uuid.uuid4())
        self._opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if _should_bypass_proxy(self.base_url)
            else urllib.request.build_opener()
        )

    def check_health(self) -> dict[str, Any]:
        url = f"{self.base_url}/system_stats"
        try:
            with self._opener.open(url, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "ok": True,
                    "url": self.base_url,
                    "system": data.get("system", {}),
                }
        except urllib.error.URLError as exc:
            return {
                "ok": False,
                "url": self.base_url,
                "error": f"无法连接 ComfyUI ({exc.reason})，请确认 SSH 端口转发与远端服务已启动。",
            }
        except Exception as exc:
            return {
                "ok": False,
                "url": self.base_url,
                "error": f"ComfyUI 健康检查失败: {exc}",
            }

    def list_loras(self) -> dict[str, Any]:
        # 1. 优先请求 /models/loras
        try:
            with self._opener.open(f"{self.base_url}/models/loras", timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    names = sorted({str(x).strip() for x in data if str(x).strip()})
                    return {"ok": True, "loras": names, "source": "/models/loras"}
        except Exception:
            pass

        # 2. 回退 /object_info/LoraLoaderModelOnly
        try:
            with self._opener.open(
                f"{self.base_url}/object_info/LoraLoaderModelOnly", timeout=5.0
            ) as resp:
                info = json.loads(resp.read().decode("utf-8"))
                spec = (
                    info.get("LoraLoaderModelOnly", {})
                    .get("input", {})
                    .get("required", {})
                    .get("lora_name")
                )
                raw_list: list[Any] = []
                if isinstance(spec, list) and spec:
                    if isinstance(spec[0], list):
                        raw_list = spec[0]
                    elif (
                        spec[0] == "COMBO"
                        and len(spec) > 1
                        and isinstance(spec[1], dict)
                        and isinstance(spec[1].get("options"), list)
                    ):
                        raw_list = spec[1]["options"]
                names = sorted({str(x).strip() for x in raw_list if str(x).strip()})
                return {
                    "ok": True,
                    "loras": names,
                    "source": "/object_info/LoraLoaderModelOnly",
                }
        except urllib.error.URLError as exc:
            return {
                "ok": False,
                "loras": [],
                "error": f"无法拉取远端 LoRA 列表 ({exc.reason})，请确认 SSH 端口转发已连通。",
            }
        except Exception as exc:
            return {
                "ok": False,
                "loras": [],
                "error": f"拉取远端 LoRA 列表失败: {exc}",
            }

    def _load_and_patch_workflow(
        self,
        positive: str,
        negative: str,
        seed: int,
        gen_cfg: dict[str, Any],
        loras: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.workflow_path.is_file():
            raise FileNotFoundError(f"未找到工作流文件: {self.workflow_path}")

        workflow = copy.deepcopy(json.loads(self.workflow_path.read_text(encoding="utf-8")))

        unet_id = str(self.node_map.get("unet", "1"))
        pos_id = str(self.node_map.get("positive", "4"))
        neg_id = str(self.node_map.get("negative", "5"))
        lat_id = str(self.node_map.get("latent", "6"))
        smp_id = str(self.node_map.get("sampler", "7"))
        sav_id = str(self.node_map.get("save", "9"))
        lor_id = str(self.node_map.get("lora", "10"))

        if pos_id not in workflow or neg_id not in workflow:
            raise KeyError(f"工作流中缺少正/负向节点 ({pos_id}, {neg_id})")

        workflow[pos_id]["inputs"]["text"] = positive
        workflow[neg_id]["inputs"]["text"] = negative

        if lat_id in workflow:
            workflow[lat_id]["inputs"]["width"] = int(gen_cfg.get("width", 512))
            workflow[lat_id]["inputs"]["height"] = int(gen_cfg.get("height", 512))
            workflow[lat_id]["inputs"]["batch_size"] = 1

        if smp_id in workflow:
            workflow[smp_id]["inputs"]["seed"] = int(seed)
            workflow[smp_id]["inputs"]["steps"] = int(gen_cfg.get("steps", 30))
            workflow[smp_id]["inputs"]["cfg"] = float(gen_cfg.get("cfg", 4.0))

        if sav_id in workflow:
            workflow[sav_id]["inputs"]["filename_prefix"] = "anima_loop"

        if loras is None:
            if lor_id in workflow and "lora_strength" in gen_cfg:
                workflow[lor_id]["inputs"]["strength_model"] = float(gen_cfg["lora_strength"])
            return workflow

        active_loras = [
            {
                "name": str(item.get("name", "")).strip(),
                "strength": float(item.get("strength", 1.0)),
            }
            for item in loras
            if bool(item.get("enabled", True))
            and str(item.get("name", "")).strip()
            and abs(float(item.get("strength", 1.0))) > 1e-6
        ]

        upstream_model: list[Any] = [unet_id, 0]
        if lor_id in workflow:
            lor_inputs = workflow[lor_id].get("inputs", {})
            if isinstance(lor_inputs, dict) and isinstance(lor_inputs.get("model"), list):
                upstream_model = copy.deepcopy(lor_inputs["model"])
            workflow.pop(lor_id, None)

        if not active_loras:
            if smp_id in workflow:
                workflow[smp_id]["inputs"]["model"] = upstream_model
            return workflow

        prev_ref = upstream_model
        next_extra_id = 1001
        for idx, item in enumerate(active_loras):
            if idx == 0 and lor_id not in workflow:
                node_id = lor_id
            else:
                while str(next_extra_id) in workflow:
                    next_extra_id += 1
                node_id = str(next_extra_id)
                next_extra_id += 1

            workflow[node_id] = {
                "inputs": {
                    "lora_name": item["name"],
                    "strength_model": item["strength"],
                    "model": prev_ref,
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": {
                    "title": f"LoRA #{idx + 1}",
                },
            }
            prev_ref = [node_id, 0]

        if smp_id in workflow:
            workflow[smp_id]["inputs"]["model"] = prev_ref

        return workflow

    def generate_one(
        self,
        positive: str,
        negative: str,
        seed: int,
        gen_cfg: dict[str, Any],
        loras: list[dict[str, Any]] | None = None,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        workflow = self._load_and_patch_workflow(
            positive, negative, seed, gen_cfg, loras=loras
        )

        req_payload = json.dumps(
            {"prompt": workflow, "client_id": self.client_id},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=req_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=15.0) as resp:
                queued = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI 拒绝提交任务 (HTTP {exc.code}): {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"提交 ComfyUI 任务失败: {exc}") from exc

        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI 返回缺少 prompt_id: {queued}")

        save_id = str(self.node_map.get("save", "9"))
        deadline = time.perf_counter() + timeout_s
        image_meta: dict[str, Any] | None = None

        while time.perf_counter() < deadline:
            time.sleep(0.8)
            with self._opener.open(
                f"{self.base_url}/history/{prompt_id}", timeout=10.0
            ) as resp:
                history = json.loads(resp.read().decode("utf-8"))

            if prompt_id not in history:
                continue

            entry = history[prompt_id]
            status_obj = entry.get("status", {})
            if status_obj.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI 执行报错: {json.dumps(status_obj.get('messages', []), ensure_ascii=False)}"
                )

            outputs = entry.get("outputs", {})
            if save_id in outputs and outputs[save_id].get("images"):
                image_meta = outputs[save_id]["images"][0]
                break
            for node_out in outputs.values():
                if isinstance(node_out, dict) and node_out.get("images"):
                    image_meta = node_out["images"][0]
                    break
            if image_meta is not None:
                break

        if image_meta is None:
            raise TimeoutError(f"等待 ComfyUI 出图超时 (>{timeout_s:.0f}s, prompt_id={prompt_id})")

        query = urllib.parse.urlencode(
            {
                "filename": image_meta["filename"],
                "subfolder": image_meta.get("subfolder", ""),
                "type": image_meta.get("type", "output"),
            }
        )
        with self._opener.open(f"{self.base_url}/view?{query}", timeout=30.0) as resp:
            image_bytes = resp.read()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "image_bytes": image_bytes,
            "comfy_prompt_id": prompt_id,
            "elapsed_ms": elapsed_ms,
        }


class MockComfyClient:
    """Offline visual mock generator for verifying the loop and real WD14 tagging."""

    def check_health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "url": "mock://local",
            "mode": "mock",
        }

    def list_loras(self) -> dict[str, Any]:
        return {
            "ok": True,
            "loras": [
                "style_lora_v1.safetensors",
                "style/anime_lineart_v2.safetensors",
                "lighting/dramatic_rim.safetensors",
            ],
            "source": "mock",
        }

    def generate_one(
        self,
        positive: str,
        negative: str,
        seed: int,
        gen_cfg: dict[str, Any],
        loras: list[dict[str, Any]] | None = None,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        width = int(gen_cfg.get("width", 512))
        height = int(gen_cfg.get("height", 512))

        digest = hashlib.sha256(f"{seed}:{positive}".encode("utf-8")).digest()
        bg_color = (
            180 + (digest[0] % 70),
            180 + (digest[1] % 70),
            185 + (digest[2] % 65),
        )
        accent = (
            40 + (digest[3] % 150),
            50 + (digest[4] % 150),
            90 + (digest[5] % 150),
        )
        accent2 = (
            200 + (digest[6] % 55),
            90 + (digest[7] % 120),
            120 + (digest[8] % 120),
        )

        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw distinct stylized shapes so WD14 predicts varied non-empty visual tags
        band_y = height // 2 + (digest[9] % 80) - 40
        draw.rectangle([0, band_y, width, height], fill=accent)

        cx = width // 2 + (digest[10] % 90) - 45
        cy = height // 2 - 40 + (digest[11] % 60) - 30
        radius = 65 + (digest[12] % 45)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=accent2,
            outline=(255, 255, 255),
            width=4,
        )

        # Secondary motif for seed diversity
        if digest[13] % 2 == 0:
            draw.polygon(
                [(cx - 90, cy + 120), (cx + 90, cy + 120), (cx, cy - 20)],
                outline=(30, 35, 50),
                width=4,
            )
        else:
            for i in range(4):
                sx = 60 + i * 110 + (digest[14 + i] % 25)
                sy = 70 + (digest[18 + i] % 70)
                draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill=(255, 250, 210))

        draw.rectangle([16, 16, width - 16, 84], fill=(18, 22, 32))
        snippet = positive[:54] + ("..." if len(positive) > 54 else "")
        draw.text((26, 26), f"MOCK ANIMA | seed={seed}", fill=(190, 210, 255))
        draw.text((26, 48), snippet, fill=(235, 240, 248))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        time.sleep(0.05)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "image_bytes": buf.getvalue(),
            "comfy_prompt_id": f"mock-{seed}",
            "elapsed_ms": elapsed_ms,
        }
