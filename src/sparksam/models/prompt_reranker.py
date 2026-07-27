from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..data.prompt_synthesis import clamp_box_xyxy
from .candidate_reranker import score_candidates
from .prompt_estimator import LearnedAutoPrompt, ir_prior_stack, load_ir_gray


@dataclass(frozen=True)
class PromptRerankerConfig:
    window_sizes: tuple[int, ...] = (9, 17, 33)
    use_frequency: bool = True
    use_mask_feedback: bool = True
    max_feedback_candidates: int = 5
    box_scales: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    min_box_side: float = 2.0
    prior_weight_objectness: float = 0.4
    prior_weight_local_contrast: float = 0.2
    prior_weight_top_hat: float = 0.15
    prior_weight_peak_sharpness: float = 0.15
    prior_weight_frequency: float = 0.1
    final_weight_prior: float = 0.45
    final_weight_feedback: float = 0.55
    feedback_weight_sam_score: float = 0.35
    feedback_weight_contrast: float = 0.25
    feedback_weight_area: float = 0.2
    feedback_weight_compactness: float = 0.1
    feedback_weight_center: float = 0.1
    min_area_ratio: float = 1e-6
    max_area_ratio: float = 0.02
    center_sigma_fraction: float = 0.08
    box_enable_margin: float = 0.02
    box_enable_min_score: float = 0.0
    box_enable_min_point_feedback_score: float = 0.0
    learned_candidate_reranker_path: str = ""
    learned_candidate_reranker_mode: str = "blend"
    learned_candidate_reranker_weight: float = 0.7
    learned_candidate_reranker_device: str = "cpu"
    learned_candidate_reranker_required: bool = True


@dataclass(frozen=True)
class PromptCandidate:
    index: int
    point: list[float]
    objectness: float
    prior_score: float
    final_score: float
    features: dict[str, float]
    feedback_score: float | None = None
    feedback: dict[str, float] | None = None
    sam_result: dict[str, Any] | None = None
    learned_score: float | None = None


@dataclass(frozen=True)
class RerankResult:
    candidates: list[PromptCandidate]
    selected: PromptCandidate
    policy: str


@dataclass(frozen=True)
class BoxCalibrationCandidate:
    index: int
    box: list[float]
    scale: float
    score: float
    feedback: dict[str, float]
    sam_result: dict[str, Any]


@dataclass(frozen=True)
class BoxCalibrationResult:
    candidates: list[BoxCalibrationCandidate]
    selected: BoxCalibrationCandidate
    policy: str


def prompt_reranker_config_from_dict(raw: dict[str, Any] | None) -> PromptRerankerConfig:
    if not raw:
        return PromptRerankerConfig()
    payload = dict(raw)
    allowed = {item.name for item in fields(PromptRerankerConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    if "window_sizes" in payload:
        payload["window_sizes"] = tuple(int(value) for value in payload["window_sizes"])
    if "box_scales" in payload:
        payload["box_scales"] = tuple(float(value) for value in payload["box_scales"])
    return PromptRerankerConfig(**payload)


def rank_prompt_candidates(auto_prompt: LearnedAutoPrompt, image_path: Path, config: PromptRerankerConfig) -> RerankResult:
    gray = load_ir_gray(image_path)
    prior = ir_prior_stack(gray)
    candidate_points = _candidate_points(auto_prompt)
    candidates = [
        _score_prior_candidate(
            index=idx,
            candidate=item,
            gray=gray,
            local_contrast=prior[1],
            top_hat=prior[2],
            config=config,
        )
        for idx, item in enumerate(candidate_points)
    ]
    ranked, policy = _rank_candidates(candidates, gray=gray, config=config, base_policy="prior_multi_cue")
    return RerankResult(candidates=ranked, selected=ranked[0], policy=policy)


def apply_mask_feedback(
    candidates: Iterable[PromptCandidate],
    *,
    image_path: Path,
    sam_results: list[dict[str, Any]],
    config: PromptRerankerConfig,
) -> RerankResult:
    gray = load_ir_gray(image_path)
    updated: list[PromptCandidate] = []
    source_candidates = list(candidates)
    for candidate, result in zip(source_candidates, sam_results):
        mask, sam_score = select_best_mask(result)
        feedback = score_mask_feedback(mask, sam_score=sam_score, gray=gray, point=candidate.point, config=config)
        final_score = _weighted_sum(
            [
                (candidate.prior_score, config.final_weight_prior),
                (feedback["feedback_score"], config.final_weight_feedback),
            ]
        )
        updated.append(
            replace(
                candidate,
                feedback_score=feedback["feedback_score"],
                final_score=final_score,
                feedback=feedback,
                sam_result=result,
            )
        )
    untouched = [item for item in source_candidates[len(updated) :]]
    ranked, policy = _rank_candidates(
        updated + untouched,
        gray=gray,
        config=config,
        base_policy="prior_multi_cue_sam2_mask_feedback" if updated else "prior_multi_cue",
    )
    return RerankResult(candidates=ranked, selected=ranked[0], policy=policy)


def make_scaled_boxes(
    *,
    point: list[float],
    base_box: list[float],
    image_width: int,
    image_height: int,
    min_box_side: float,
    scales: Iterable[float],
) -> list[tuple[float, list[float]]]:
    cx, cy = float(point[0]), float(point[1])
    base_w = max(float(min_box_side), float(base_box[2]) - float(base_box[0]))
    base_h = max(float(min_box_side), float(base_box[3]) - float(base_box[1]))
    boxes: list[tuple[float, list[float]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw_scale in scales:
        scale = max(1e-3, float(raw_scale))
        width = max(float(min_box_side), base_w * scale)
        height = max(float(min_box_side), base_h * scale)
        box = clamp_box_xyxy(
            [cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width + 1.0, cy + 0.5 * height + 1.0],
            width=image_width,
            height=image_height,
        )
        key = tuple(int(round(value * 1000.0)) for value in box)
        if key in seen:
            continue
        seen.add(key)
        boxes.append((scale, box))
    return boxes


def calibrate_box_from_results(
    *,
    boxes: list[tuple[float, list[float]]],
    image_path: Path,
    point: list[float],
    sam_results: list[dict[str, Any]],
    config: PromptRerankerConfig,
) -> BoxCalibrationResult:
    gray = load_ir_gray(image_path)
    candidates: list[BoxCalibrationCandidate] = []
    for idx, ((scale, box), result) in enumerate(zip(boxes, sam_results)):
        mask, sam_score = select_best_mask(result)
        feedback = score_mask_feedback(mask, sam_score=sam_score, gray=gray, point=point, config=config)
        candidates.append(
            BoxCalibrationCandidate(
                index=idx,
                box=box,
                scale=float(scale),
                score=feedback["feedback_score"],
                feedback=feedback,
                sam_result=result,
            )
        )
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    return BoxCalibrationResult(candidates=ranked, selected=ranked[0], policy="sam2_mask_feedback_multi_scale_box")


def select_best_mask(result: dict[str, Any]) -> tuple[np.ndarray, float]:
    masks = np.asarray(result.get("masks"), dtype=np.float32)
    scores = np.asarray(result.get("scores"), dtype=np.float32).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError(f"Expected SAM2 masks with shape NxHxW, got {masks.shape!r}.")
    if scores.size == 0:
        scores = np.zeros((masks.shape[0],), dtype=np.float32)
    best_idx = int(np.argmax(scores[: masks.shape[0]]))
    return masks[best_idx].astype(np.float32), float(scores[best_idx])


def score_mask_feedback(
    mask: np.ndarray,
    *,
    sam_score: float,
    gray: np.ndarray,
    point: list[float],
    config: PromptRerankerConfig,
) -> dict[str, float]:
    mask_bin = np.asarray(mask, dtype=np.float32) > 0.5
    height, width = gray.shape
    if mask_bin.shape != gray.shape:
        mask_bin = _fit_mask(mask_bin, height=height, width=width)
    area = float(mask_bin.sum())
    area_ratio = area / float(max(1, height * width))
    if area <= 0.0:
        return {
            "feedback_score": 0.0,
            "sam_score": _clamp01(float(sam_score)),
            "area_ratio": 0.0,
            "area_score": 0.0,
            "compactness": 0.0,
            "contrast_score": 0.0,
            "center_score": 0.0,
        }
    ys, xs = np.where(mask_bin)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    box_area = float(max(1, (x2 - x1) * (y2 - y1)))
    compactness = _clamp01(area / box_area)
    area_score = _area_score(area_ratio, config)
    contrast_score = _mask_contrast_score(mask_bin, gray, x1=x1, y1=y1, x2=x2, y2=y2)
    center_score = _center_score(mask_bin, point, width=width, height=height, config=config)
    feedback_score = _weighted_sum(
        [
            (_clamp01(float(sam_score)), config.feedback_weight_sam_score),
            (contrast_score, config.feedback_weight_contrast),
            (area_score, config.feedback_weight_area),
            (compactness, config.feedback_weight_compactness),
            (center_score, config.feedback_weight_center),
        ]
    )
    return {
        "feedback_score": feedback_score,
        "sam_score": _clamp01(float(sam_score)),
        "area_ratio": float(area_ratio),
        "area_score": float(area_score),
        "compactness": float(compactness),
        "contrast_score": float(contrast_score),
        "center_score": float(center_score),
    }


def candidate_metadata(candidates: Iterable[PromptCandidate]) -> list[dict[str, float | int | list[float]]]:
    rows: list[dict[str, float | int | list[float]]] = []
    for item in candidates:
        row: dict[str, float | int | list[float]] = {
            "index": int(item.index),
            "point": [float(item.point[0]), float(item.point[1])],
            "objectness": float(item.objectness),
            "prior_score": float(item.prior_score),
            "final_score": float(item.final_score),
        }
        if item.feedback_score is not None:
            row["feedback_score"] = float(item.feedback_score)
        if item.learned_score is not None:
            row["learned_score"] = float(item.learned_score)
        rows.append(row)
    return rows


def _rank_candidates(
    candidates: list[PromptCandidate],
    *,
    gray: np.ndarray,
    config: PromptRerankerConfig,
    base_policy: str,
) -> tuple[list[PromptCandidate], str]:
    ranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)
    checkpoint_path = str(config.learned_candidate_reranker_path or "").strip()
    if not checkpoint_path:
        return ranked, base_policy
    try:
        height, width = int(gray.shape[0]), int(gray.shape[1])
        learned_scores = score_candidates(
            ranked,
            checkpoint_path=checkpoint_path,
            device=str(config.learned_candidate_reranker_device or "cpu"),
            image_width=width,
            image_height=height,
        )
    except Exception:
        if bool(config.learned_candidate_reranker_required):
            raise
        return ranked, f"{base_policy}_response_candidate_reranker_unavailable"
    weight = min(1.0, max(0.0, float(config.learned_candidate_reranker_weight)))
    mode = str(config.learned_candidate_reranker_mode or "blend").strip().lower()
    updated: list[PromptCandidate] = []
    for candidate, learned_score in zip(ranked, learned_scores):
        old_score = float(candidate.final_score)
        if mode == "replace":
            final_score = float(learned_score)
        elif mode == "add":
            final_score = old_score + weight * float(learned_score)
        else:
            final_score = (1.0 - weight) * old_score + weight * float(learned_score)
        updated.append(replace(candidate, final_score=float(final_score), learned_score=float(learned_score)))
    return sorted(updated, key=lambda item: item.final_score, reverse=True), f"{base_policy}_response_candidate_reranker"


def feedback_metadata(feedback: dict[str, float] | None) -> dict[str, float]:
    if not feedback:
        return {}
    return {key: float(value) for key, value in feedback.items()}


def box_calibration_metadata(candidates: Iterable[BoxCalibrationCandidate]) -> list[dict[str, float | int | list[float]]]:
    rows: list[dict[str, float | int | list[float]]] = []
    for item in candidates:
        rows.append(
            {
                "index": int(item.index),
                "scale": float(item.scale),
                "box": [float(value) for value in item.box],
                "score": float(item.score),
            }
        )
    return rows


def _candidate_points(auto_prompt: LearnedAutoPrompt) -> list[list[float]]:
    raw_points = auto_prompt.metadata.get("candidate_points")
    if isinstance(raw_points, list) and raw_points:
        candidates = []
        for item in raw_points:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                score = float(item[2]) if len(item) >= 3 else float(auto_prompt.metadata.get("candidate_score", 0.0))
                candidates.append([float(item[0]), float(item[1]), score])
        if candidates:
            return candidates
    return [[float(auto_prompt.point[0]), float(auto_prompt.point[1]), float(auto_prompt.metadata.get("candidate_score", 0.0))]]


def _score_prior_candidate(
    *,
    index: int,
    candidate: list[float],
    gray: np.ndarray,
    local_contrast: np.ndarray,
    top_hat: np.ndarray,
    config: PromptRerankerConfig,
) -> PromptCandidate:
    x = int(round(float(candidate[0])))
    y = int(round(float(candidate[1])))
    h, w = gray.shape
    x = min(max(0, x), max(0, w - 1))
    y = min(max(0, y), max(0, h - 1))
    objectness = _clamp01(float(candidate[2]))
    local_value = _clamp01(float(local_contrast[y, x]))
    top_hat_value = _clamp01(float(top_hat[y, x]))
    peak = _peak_sharpness(gray, x=x, y=y, window=max(config.window_sizes or (9,)))
    frequency = _frequency_score(gray, x=x, y=y, window=max(config.window_sizes or (17,))) if config.use_frequency else 0.0
    features = {
        "objectness": objectness,
        "local_contrast": local_value,
        "top_hat": top_hat_value,
        "peak_sharpness": peak,
        "frequency_score": frequency,
    }
    prior_score = _weighted_sum(
        [
            (objectness, config.prior_weight_objectness),
            (local_value, config.prior_weight_local_contrast),
            (top_hat_value, config.prior_weight_top_hat),
            (peak, config.prior_weight_peak_sharpness),
            (frequency, config.prior_weight_frequency if config.use_frequency else 0.0),
        ]
    )
    return PromptCandidate(
        index=index,
        point=[float(x), float(y)],
        objectness=objectness,
        prior_score=prior_score,
        final_score=prior_score,
        features=features,
    )


def _window(arr: np.ndarray, *, x: int, y: int, size: int) -> np.ndarray:
    size = max(3, int(size))
    radius = size // 2
    h, w = arr.shape
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    return arr[y1:y2, x1:x2]


def _peak_sharpness(gray: np.ndarray, *, x: int, y: int, window: int) -> float:
    crop = _window(gray, x=x, y=y, size=window)
    if crop.size <= 1:
        return 0.0
    center = float(gray[y, x])
    return _clamp01(center - float(crop.mean()))


def _frequency_score(gray: np.ndarray, *, x: int, y: int, window: int) -> float:
    crop = _window(gray, x=x, y=y, size=window).astype(np.float32)
    if min(crop.shape) < 4:
        return 0.0
    crop = crop - float(crop.mean())
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    total = float(spectrum.sum())
    if total <= 1e-12:
        return 0.0
    yy, xx = np.indices(spectrum.shape)
    cy = (spectrum.shape[0] - 1) * 0.5
    cx = (spectrum.shape[1] - 1) * 0.5
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = 0.25 * float(min(spectrum.shape))
    high_energy = float(spectrum[radius >= cutoff].sum())
    return _clamp01(high_energy / total)


def _mask_contrast_score(mask: np.ndarray, gray: np.ndarray, *, x1: int, y1: int, x2: int, y2: int) -> float:
    h, w = gray.shape
    pad = max(3, int(max(x2 - x1, y2 - y1)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = np.zeros_like(mask, dtype=bool)
    ring[ry1:ry2, rx1:rx2] = True
    ring &= ~mask
    inside = gray[mask]
    outside = gray[ring]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return _clamp01(float(inside.mean()) - float(outside.mean()))


def _center_score(mask: np.ndarray, point: list[float], *, width: int, height: int, config: PromptRerankerConfig) -> float:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0.0
    cx = float(xs.mean())
    cy = float(ys.mean())
    distance = float(((cx - float(point[0])) ** 2 + (cy - float(point[1])) ** 2) ** 0.5)
    sigma = max(1.0, ((width * width + height * height) ** 0.5) * float(config.center_sigma_fraction))
    return _clamp01(float(np.exp(-distance / sigma)))


def _area_score(area_ratio: float, config: PromptRerankerConfig) -> float:
    if area_ratio <= 0.0:
        return 0.0
    min_ratio = max(1e-12, float(config.min_area_ratio))
    max_ratio = max(min_ratio, float(config.max_area_ratio))
    if area_ratio < min_ratio:
        return _clamp01(area_ratio / min_ratio)
    if area_ratio > max_ratio:
        return _clamp01(max_ratio / area_ratio)
    return 1.0


def _fit_mask(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    work = np.asarray(mask, dtype=bool)
    fitted = np.zeros((height, width), dtype=bool)
    h = min(height, work.shape[0])
    w = min(width, work.shape[1])
    fitted[:h, :w] = work[:h, :w]
    return fitted


def _weighted_sum(values: Iterable[tuple[float, float]]) -> float:
    total_weight = 0.0
    total = 0.0
    for value, weight in values:
        weight = float(weight)
        if weight <= 0.0:
            continue
        total += _clamp01(float(value)) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return _clamp01(total / total_weight)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
