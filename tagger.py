import csv
import io
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional when running without local WD14
    ort = None  # type: ignore[assignment]

KAOMOJI = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "O_o",
    "o_O",
    "u_u",
    "x_x",
    "X_X",
    "t_t",
    "T_T",
    "-_-",
    ";_;",
    "v_v",
    ">_>",
    "<_<",
    "^_~",
    "~_^",
    "|_|",
    "||_||",
}

RATING_ORDER = ["general", "sensitive", "questionable", "explicit"]
MIN_RETURN_SCORE = 0.10


def format_display_tag(raw_tag: str) -> str:
    if raw_tag in KAOMOJI:
        return raw_tag
    return raw_tag.replace("_", " ")


class LocalWDTagger:
    """Thread-safe wrapper around the shared local WD14/WDv3 ONNX model."""

    def __init__(self, model_dir: Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        env_dir = os.environ.get("WD14_MODEL_DIR", "").strip()
        local_dir = base_dir / "models" / "wd-swinv2-tagger-v3"
        parent_dir = base_dir.parent / "models" / "wd-swinv2-tagger-v3"
        if model_dir is not None:
            self.model_dir = model_dir
        elif env_dir:
            self.model_dir = Path(env_dir).expanduser()
        elif (local_dir / "model.onnx").is_file():
            self.model_dir = local_dir
        else:
            self.model_dir = parent_dir
        self.onnx_path = self.model_dir / "model.onnx"
        self.csv_path = self.model_dir / "selected_tags.csv"

        self._lock = threading.Lock()
        self._session: Any = None
        self._tags: list[dict[str, str]] | None = None
        self._known_display_tags: set[str] | None = None
        self._target_size: int = 448

    def is_available(self) -> bool:
        return (
            ort is not None
            and self.onnx_path.is_file()
            and self.onnx_path.stat().st_size > 0
            and self.csv_path.is_file()
            and self.csv_path.stat().st_size > 0
        )

    def _ensure_tags_loaded(self) -> list[dict[str, str]]:
        if self._tags is not None:
            return self._tags
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"未找到 WD14 标签词表: {self.csv_path}")

        tags: list[dict[str, str]] = []
        known: set[str] = set()
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row["name"].strip()
                display_name = format_display_tag(raw_name)
                cat = str(row["category"]).strip()
                tags.append(
                    {
                        "raw_tag": raw_name,
                        "display_tag": display_name,
                        "category": cat,
                    }
                )
                if cat in ("0", "4"):
                    known.add(display_name.lower())
                    known.add(raw_name.lower())

        self._tags = tags
        self._known_display_tags = known
        return tags

    def is_known_tag(self, display_tag: str) -> bool:
        with self._lock:
            if self._known_display_tags is None:
                if not self.csv_path.is_file():
                    return False
                self._ensure_tags_loaded()
            assert self._known_display_tags is not None
            return display_tag.strip().lower() in self._known_display_tags

    def _ensure_session(self) -> tuple[Any, list[dict[str, str]], int]:
        tags = self._ensure_tags_loaded()
        if self._session is not None:
            return self._session, tags, self._target_size
        if ort is None:
            raise RuntimeError("未安装 onnxruntime，请执行 pip install onnxruntime 后再使用本地 WD14")
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"未找到 WD14 模型文件: {self.onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        input_shape = session.get_inputs()[0].shape
        target_size = 448
        if len(input_shape) >= 3 and isinstance(input_shape[1], int) and input_shape[1] > 0:
            target_size = input_shape[1]

        self._session = session
        self._target_size = target_size
        return session, tags, target_size

    @staticmethod
    def _preprocess_bytes(image_bytes: bytes, target_size: int) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)

        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )
        if has_alpha:
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, rgba).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        max_dim = max(width, height)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))

        if max_dim != target_size:
            padded = padded.resize((target_size, target_size), Image.Resampling.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(arr, axis=0)

    def tag_image_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        with self._lock:
            session, tags, target_size = self._ensure_session()
            tensor = self._preprocess_bytes(image_bytes, target_size)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            t0 = time.perf_counter()
            probs = session.run([output_name], {input_name: tensor})[0][0]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        rating_map: dict[str, float] = {}
        detected_tags: list[dict[str, Any]] = []

        for idx, info in enumerate(tags):
            score = float(probs[idx])
            cat = info["category"]
            if cat == "9":
                rating_map[info["raw_tag"]] = score
            elif cat in ("0", "4") and score >= MIN_RETURN_SCORE:
                detected_tags.append(
                    {
                        "tag": info["display_tag"],
                        "raw_tag": info["raw_tag"],
                        "score": round(score, 4),
                        "category": "general" if cat == "0" else "character",
                    }
                )

        detected_tags.sort(key=lambda x: x["score"], reverse=True)
        ratings = [
            {"tag": name, "score": round(rating_map.get(name, 0.0), 4)}
            for name in RATING_ORDER
        ]
        return {
            "elapsed_ms": elapsed_ms,
            "ratings": ratings,
            "tags": detected_tags,
        }
