from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

from .prompt_synthesis import clamp_box_xyxy


HEURISTIC_IR_AUTO_PROMPT_PROTOCOL = "heuristic_ir_auto_prompt_v1"


@dataclass(frozen=True)
class AutoPrompt:
    point: list[float]
    box: list[float]
    points: list[list[float]]
    point_labels: list[int]
    metadata: dict[str, Any]


def load_gray_float(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    denom = max(1e-6, max_v - min_v)
    return (arr - min_v) / denom


def _normalize_map(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    denom = max(1e-6, max_v - min_v)
    return (arr - min_v) / denom


def _box_mean(gray: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    padded = np.pad(gray.astype(np.float32), ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return total / float(size * size)


def _white_tophat(gray: np.ndarray, size: int) -> np.ndarray:
    kernel = max(3, int(size))
    if kernel % 2 == 0:
        kernel += 1
    image = Image.fromarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8))
    opened = image.filter(ImageFilter.MinFilter(kernel)).filter(ImageFilter.MaxFilter(kernel))
    opened_arr = np.asarray(opened, dtype=np.float32) / 255.0
    return np.maximum(gray - opened_arr, 0.0)


def _local_maxima(score: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    padded = np.pad(score, ((radius, radius), (radius, radius)), mode="edge")
    local_max = np.zeros_like(score, dtype=np.float32)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            local_max = np.maximum(local_max, padded[dy : dy + score.shape[0], dx : dx + score.shape[1]])
    return (score >= local_max).astype(np.float32)


def _greedy_nms(score: np.ndarray, *, top_k: int, radius: int, threshold: float) -> list[tuple[int, int, float]]:
    work = np.asarray(score, dtype=np.float32).copy()
    h, w = work.shape
    candidates: list[tuple[int, int, float]] = []
    for _ in range(max(1, int(top_k))):
        flat_idx = int(np.argmax(work))
        value = float(work.flat[flat_idx])
        if value < threshold:
            break
        y, x = divmod(flat_idx, w)
        candidates.append((x, y, value))
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        work[y1:y2, x1:x2] = -1.0
    return candidates


def _component_box(score: np.ndarray, x: int, y: int, *, threshold: float, fallback_half_side: int) -> list[float]:
    binary = score >= threshold
    h, w = score.shape
    if not (0 <= x < w and 0 <= y < h) or not binary[y, x]:
        return [float(x - fallback_half_side), float(y - fallback_half_side), float(x + fallback_half_side + 1), float(y + fallback_half_side + 1)]
    stack = [(y, x)]
    visited = np.zeros_like(binary, dtype=bool)
    visited[y, x] = True
    xs: list[int] = []
    ys: list[int] = []
    while stack:
        cy, cx = stack.pop()
        xs.append(cx)
        ys.append(cy)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = cy + dy
            nx = cx + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= w:
                continue
            if visited[ny, nx] or not binary[ny, nx]:
                continue
            visited[ny, nx] = True
            stack.append((ny, nx))
    return [float(min(xs)), float(min(ys)), float(max(xs) + 1), float(max(ys) + 1)]


def _expand_candidate_box(box: Iterable[float], width: int, height: int, pad: float) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return clamp_box_xyxy([x1 - pad, y1 - pad, x2 + pad, y2 + pad], width=width, height=height)


def _negative_ring_points(box: list[float], width: int, height: int, offset: float) -> list[list[float]]:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    points = [
        [cx, y1 - offset],
        [cx, y2 + offset],
        [x1 - offset, cy],
        [x2 + offset, cy],
    ]
    clipped: list[list[float]] = []
    for x, y in points:
        clipped.append([float(min(max(0.0, x), max(0.0, width - 1.0))), float(min(max(0.0, y), max(0.0, height - 1.0)))])
    return clipped


def generate_heuristic_ir_auto_prompt(
    gray: np.ndarray,
    *,
    top_k: int = 1,
    nms_radius: int = 8,
    response_threshold: float = 0.1,
    box_threshold: float = 0.55,
    fallback_half_side: int = 4,
    negative_ring: bool = False,
) -> AutoPrompt:
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {arr.shape}.")
    h, w = arr.shape
    arr = _normalize_map(arr)
    small_mean = _box_mean(arr, radius=2)
    large_mean = _box_mean(arr, radius=8)
    local_contrast = np.maximum(small_mean - large_mean, 0.0)
    local_std = np.sqrt(np.maximum(_box_mean((arr - large_mean) ** 2, radius=8), 1e-6))
    snr_like = np.maximum(arr - large_mean, 0.0) / (local_std + 1e-3)
    top_hat = np.maximum.reduce([_white_tophat(arr, size=3), _white_tophat(arr, size=5), _white_tophat(arr, size=9)])
    bright_peak = arr * _local_maxima(arr, radius=2)
    score = _normalize_map(0.35 * _normalize_map(top_hat) + 0.3 * _normalize_map(local_contrast) + 0.2 * _normalize_map(snr_like) + 0.15 * _normalize_map(bright_peak))
    candidates = _greedy_nms(score, top_k=top_k, radius=nms_radius, threshold=response_threshold)
    fallback = False
    if not candidates:
        flat_idx = int(np.argmax(score))
        y, x = divmod(flat_idx, w)
        candidates = [(x, y, float(score[y, x]))]
        fallback = True
    x, y, candidate_score = candidates[0]
    component = _component_box(score, x, y, threshold=max(box_threshold * candidate_score, response_threshold), fallback_half_side=fallback_half_side)
    box = _expand_candidate_box(component, width=w, height=h, pad=max(2.0, float(fallback_half_side) * 0.5))
    point = [float(x), float(y)]
    points = [point]
    labels = [1]
    negative_points: list[list[float]] = []
    if negative_ring:
        negative_points = _negative_ring_points(box, width=w, height=h, offset=max(2.0, float(fallback_half_side)))
        points.extend(negative_points)
        labels.extend([0] * len(negative_points))
    metadata = {
        "source": "synthesized",
        "protocol": HEURISTIC_IR_AUTO_PROMPT_PROTOCOL,
        "point": point,
        "box": box,
        "points": points,
        "point_labels": labels,
        "candidate_score": float(candidate_score),
        "candidate_rank": 0,
        "candidate_count": len(candidates),
        "candidate_top_k": int(top_k),
        "candidate_nms_radius": int(nms_radius),
        "fallback": fallback,
        "negative_point_count": len(negative_points),
        "positive_point_count": 1,
        "top_k": int(top_k),
        "nms_radius": int(nms_radius),
        "response_threshold": float(response_threshold),
        "box_threshold": float(box_threshold),
        "fallback_half_side": int(fallback_half_side),
        "box_width": float(box[2] - box[0]),
        "box_height": float(box[3] - box[1]),
    }
    return AutoPrompt(point=point, box=box, points=points, point_labels=labels, metadata=metadata)


def generate_heuristic_ir_auto_prompt_from_path(path: Path, **kwargs: Any) -> AutoPrompt:
    return generate_heuristic_ir_auto_prompt(load_gray_float(path), **kwargs)
