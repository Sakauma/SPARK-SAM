from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _safe_segment(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {arr.shape}.")
    min_v = float(arr.min())
    max_v = float(arr.max())
    denom = max(1e-6, max_v - min_v)
    return (arr - min_v) / denom


def _image_to_rgb(image: Any, height: int, width: int) -> np.ndarray:
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil:
            arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB-compatible image, got shape {arr.shape}.")
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[:2] == (height, width):
        return arr
    pil = Image.fromarray(arr, mode="RGB").resize((width, height), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _heatmap_color(norm: np.ndarray) -> np.ndarray:
    red = np.clip(norm * 255.0, 0.0, 255.0)
    green = np.clip((1.0 - np.abs(norm - 0.5) * 2.0) * 180.0, 0.0, 180.0)
    blue = np.clip((1.0 - norm) * 80.0, 0.0, 80.0)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def write_heatmap_artifact(
    *,
    root: Path,
    experiment_id: str,
    dataset: str,
    sample_id: str,
    stage: str,
    heatmap: np.ndarray,
    image: Any | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    norm = _normalize_heatmap(heatmap)
    height, width = norm.shape
    out_dir = root / _safe_segment(experiment_id) / _safe_segment(dataset) / _safe_segment(sample_id) / _safe_segment(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw.npz"
    heatmap_path = out_dir / "heatmap.png"
    overlay_path = out_dir / "overlay.png"
    meta_path = out_dir / "meta.json"

    np.savez_compressed(raw_path, heatmap=np.asarray(heatmap, dtype=np.float32), heatmap_normalized=norm.astype(np.float32))
    heat_color = _heatmap_color(norm)
    Image.fromarray(heat_color, mode="RGB").save(heatmap_path)
    base = _image_to_rgb(image, height, width).astype(np.float32)
    overlay = np.clip(base * 0.55 + heat_color.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    metadata = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "sample_id": sample_id,
        "stage": stage,
        "shape": [int(height), int(width)],
        "heatmap_min": float(np.asarray(heatmap, dtype=np.float32).min()),
        "heatmap_max": float(np.asarray(heatmap, dtype=np.float32).max()),
        **(meta or {}),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "raw": str(raw_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
        "meta": str(meta_path),
    }
