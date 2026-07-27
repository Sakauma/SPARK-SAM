from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BASE_FEATURE_NAMES: tuple[str, ...] = (
    "objectness",
    "prior_score",
    "final_score",
    "feedback_score",
    "local_contrast",
    "top_hat",
    "peak_sharpness",
    "frequency_score",
    "sam_score",
    "area_ratio",
    "area_score",
    "compactness",
    "contrast_score",
    "center_score",
    "point_x_norm",
    "point_y_norm",
    "candidate_index_norm",
)

RELATIVE_FEATURE_NAMES: tuple[str, ...] = (
    "candidate_count_norm",
    "objectness_rank_norm",
    "objectness_zscore",
    "objectness_margin_to_top",
    "final_score_rank_norm",
    "final_score_zscore",
    "final_score_margin_to_top",
    "feedback_score_rank_norm",
    "feedback_score_zscore",
    "feedback_score_margin_to_top",
    "local_contrast_rank_norm",
    "local_contrast_zscore",
    "area_ratio_rank_norm",
    "area_ratio_zscore",
    "point_center_distance_norm",
    "point_center_distance_rank_norm",
)

FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES
RELATIVE_TOPK_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + RELATIVE_FEATURE_NAMES
PROMPT_LOCAL_FEATURE_NAMES: tuple[str, ...] = (
    "objectness",
    "prior_score",
    "local_contrast",
    "top_hat",
    "peak_sharpness",
    "frequency_score",
    "point_x_norm",
    "point_y_norm",
    "candidate_index_norm",
)
SAM_FEEDBACK_FEATURE_NAMES: tuple[str, ...] = (
    "feedback_score",
    "sam_score",
    "area_ratio",
    "area_score",
    "compactness",
    "contrast_score",
    "center_score",
)
FEATURE_MODE_NAMES: tuple[str, ...] = (
    "base",
    "relative_topk",
    "full_relative_topk",
    "base_no_relative",
    "prompt_local_only",
    "sam_feedback_only",
)


@dataclass(frozen=True)
class CandidateRerankerCheckpoint:
    feature_names: tuple[str, ...]
    state_dict: dict[str, Any]
    hidden_channels: tuple[int, ...]
    dropout: float
    metadata: dict[str, Any]


def candidate_feature_dict(
    candidate: Any,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    candidate_count: int | None = None,
) -> dict[str, float]:
    features = _as_dict(getattr(candidate, "features", None))
    feedback = _as_dict(getattr(candidate, "feedback", None))
    point = getattr(candidate, "point", [0.0, 0.0])
    index = int(getattr(candidate, "index", 0) or 0)
    width = max(1.0, float(image_width or 1))
    height = max(1.0, float(image_height or 1))
    count = max(1, int(candidate_count or 1))
    payload: dict[str, float] = {
        "objectness": _finite(getattr(candidate, "objectness", features.get("objectness", 0.0))),
        "prior_score": _finite(getattr(candidate, "prior_score", 0.0)),
        "final_score": _finite(getattr(candidate, "final_score", 0.0)),
        "feedback_score": _finite(getattr(candidate, "feedback_score", 0.0)),
        "local_contrast": _finite(features.get("local_contrast", 0.0)),
        "top_hat": _finite(features.get("top_hat", 0.0)),
        "peak_sharpness": _finite(features.get("peak_sharpness", 0.0)),
        "frequency_score": _finite(features.get("frequency_score", 0.0)),
        "sam_score": _finite(feedback.get("sam_score", 0.0)),
        "area_ratio": _finite(feedback.get("area_ratio", 0.0)),
        "area_score": _finite(feedback.get("area_score", 0.0)),
        "compactness": _finite(feedback.get("compactness", 0.0)),
        "contrast_score": _finite(feedback.get("contrast_score", 0.0)),
        "center_score": _finite(feedback.get("center_score", 0.0)),
        "point_x_norm": _finite(float(point[0]) / width if len(point) >= 1 else 0.0),
        "point_y_norm": _finite(float(point[1]) / height if len(point) >= 2 else 0.0),
        "candidate_index_norm": _finite(float(index) / float(max(1, count - 1))),
    }
    return payload


def feature_vector_from_dict(payload: dict[str, Any], feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([_finite(payload.get(name, 0.0)) for name in feature_names], dtype=np.float32)


def feature_names_for_mode(feature_mode: str) -> tuple[str, ...]:
    mode = str(feature_mode or "base")
    if mode in {"relative_topk", "full_relative_topk"}:
        return RELATIVE_TOPK_FEATURE_NAMES
    if mode in {"base", "base_no_relative"}:
        return FEATURE_NAMES
    if mode == "prompt_local_only":
        return PROMPT_LOCAL_FEATURE_NAMES
    if mode == "sam_feedback_only":
        return SAM_FEEDBACK_FEATURE_NAMES
    return FEATURE_NAMES


def feature_mode_uses_relative_features(feature_mode: str) -> bool:
    return str(feature_mode or "") in {"relative_topk", "full_relative_topk"}


def candidate_feature_vector(
    candidate: Any,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    candidate_count: int | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    return feature_vector_from_dict(
        candidate_feature_dict(candidate, image_width=image_width, image_height=image_height, candidate_count=candidate_count),
        feature_names=feature_names,
    )


def build_candidate_reranker_model(
    input_dim: int,
    *,
    hidden_channels: Sequence[int] = (64, 32),
    dropout: float = 0.05,
) -> Any:
    from torch import nn

    layers: list[nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_channels:
        hidden_dim = max(1, int(hidden))
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        prev = hidden_dim
    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def enrich_relative_feature_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str] = ("experiment", "dataset", "method", "sample_id"),
) -> list[dict[str, Any]]:
    """Add within-image relative Top-K features used by Response Reranking reranking."""
    output = [dict(row) for row in rows]
    groups: dict[tuple[str, ...], list[int]] = {}
    if group_keys:
        for index, row in enumerate(output):
            key = tuple(str(row.get(name, "")) for name in group_keys)
            groups.setdefault(key, []).append(index)
    else:
        groups[("",)] = list(range(len(output)))
    for indices in groups.values():
        _enrich_group(output, indices)
    return output


@lru_cache(maxsize=8)
def load_candidate_reranker(path: str, device: str = "cpu") -> tuple[Any, tuple[str, ...]]:
    import torch

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device)
    feature_names = tuple(str(item) for item in payload.get("feature_names", FEATURE_NAMES))
    hidden_channels = tuple(int(item) for item in payload.get("hidden_channels", (64, 32)))
    dropout = float(payload.get("dropout", 0.0))
    model = build_candidate_reranker_model(len(feature_names), hidden_channels=hidden_channels, dropout=dropout)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, feature_names


def score_feature_rows(
    rows: Iterable[dict[str, Any]],
    *,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> list[float]:
    import torch

    model, feature_names = load_candidate_reranker(str(checkpoint_path), device)
    vectors = [feature_vector_from_dict(row, feature_names=feature_names) for row in rows]
    if not vectors:
        return []
    tensor = torch.from_numpy(np.stack(vectors, axis=0)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        scores = model(tensor).detach().cpu().numpy().reshape(-1)
    return [float(score) for score in scores]


def score_candidates(
    candidates: Sequence[Any],
    *,
    checkpoint_path: str | Path,
    device: str = "cpu",
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[float]:
    rows = [
        candidate_feature_dict(
            candidate,
            image_width=image_width,
            image_height=image_height,
            candidate_count=len(candidates),
        )
        for candidate in candidates
    ]
    rows = enrich_relative_feature_rows(rows, group_keys=())
    return score_feature_rows(rows, checkpoint_path=checkpoint_path, device=device)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _enrich_group(rows: list[dict[str, Any]], indices: list[int]) -> None:
    count = len(indices)
    if count <= 0:
        return
    for index in indices:
        row = rows[index]
        row["candidate_count_norm"] = min(1.0, float(count) / 10.0)
        x = _finite(row.get("point_x_norm", 0.5))
        y = _finite(row.get("point_y_norm", 0.5))
        row["point_center_distance_norm"] = float(min(1.0, np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) / np.sqrt(0.5)))
    for field in ("objectness", "final_score", "feedback_score", "local_contrast", "area_ratio"):
        _add_rank_features(rows, indices, field)
    _add_rank_features(rows, indices, "point_center_distance_norm", feature_prefix="point_center_distance", descending=False, zscore=False, margin=False)


def _add_rank_features(
    rows: list[dict[str, Any]],
    indices: list[int],
    field: str,
    *,
    feature_prefix: str | None = None,
    descending: bool = True,
    zscore: bool = True,
    margin: bool = True,
) -> None:
    prefix = feature_prefix or field
    values = np.asarray([_finite(rows[index].get(field, 0.0)) for index in indices], dtype=np.float32)
    if values.size == 0:
        return
    order = np.argsort(-values if descending else values)
    denom = float(max(1, values.size - 1))
    ranks = np.zeros_like(values, dtype=np.float32)
    for rank, ordered_position in enumerate(order.tolist()):
        ranks[ordered_position] = float(rank) / denom
    mean = float(values.mean())
    std = float(values.std())
    top = float(values[order[0]]) if order.size else 0.0
    for local_index, row_index in enumerate(indices):
        rows[row_index][f"{prefix}_rank_norm"] = float(ranks[local_index])
        if zscore:
            rows[row_index][f"{prefix}_zscore"] = float((float(values[local_index]) - mean) / max(1e-6, std))
        if margin:
            rows[row_index][f"{prefix}_margin_to_top"] = float(top - float(values[local_index]) if descending else float(values[local_index]) - top)
