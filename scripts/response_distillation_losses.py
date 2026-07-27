"""Losses shared by the SPARK-SAM training pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(0)
    weights = weights.to(dtype=values.dtype, device=values.device)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _teacher_mask_kd_loss(
    student_logits: torch.Tensor,
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    target: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Distill response masks while reducing the influence of teacher false positives."""

    loss_type = str(cfg.get("type", "soft_bce_plus_prob_mse") or "soft_bce_plus_prob_mse")
    if loss_type in {"soft_bce_plus_prob_mse", "soft_bce_mse"}:
        return F.binary_cross_entropy_with_logits(student_logits, teacher_prob) + F.mse_loss(student_prob, teacher_prob)
    if loss_type != "teacher_reliability_weighted_soft_bce_plus_prob_mse":
        raise RuntimeError(f"Unsupported teacher-mask distillation loss type: {loss_type!r}")

    teacher_fp_threshold = float(cfg.get("teacher_fp_threshold", 0.5) or 0.5)
    teacher_positive_threshold = float(cfg.get("teacher_positive_threshold", 0.5) or 0.5)
    aligned_weight = float(cfg.get("aligned_weight", 1.0) or 1.0)
    default_weight = float(cfg.get("default_weight", 0.75) or 0.75)
    suspicious_weight = float(cfg.get("suspicious_teacher_fp_weight", 0.25) or 0.25)
    target_foreground = target >= 0.5
    teacher_foreground = teacher_prob >= teacher_positive_threshold
    teacher_false_positive = (target < 0.5) & (teacher_prob >= teacher_fp_threshold)
    aligned = (target_foreground & teacher_foreground) | ((target < 0.5) & (teacher_prob < teacher_fp_threshold))
    reliability = torch.full_like(teacher_prob, default_weight)
    reliability = torch.where(aligned, torch.full_like(reliability, aligned_weight), reliability)
    reliability = torch.where(teacher_false_positive, torch.full_like(reliability, suspicious_weight), reliability)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(student_logits, teacher_prob, reduction="none")
    squared_error = torch.square(student_prob - teacher_prob)
    return _weighted_mean(binary_cross_entropy, reliability) + _weighted_mean(squared_error, reliability)


def _topk_background_probability(
    student_prob: torch.Tensor,
    background: torch.Tensor,
    *,
    topk_ratio: float,
    min_k: int,
) -> torch.Tensor:
    probabilities = student_prob[background > 0.5]
    if probabilities.numel() == 0:
        return torch.zeros((), dtype=student_prob.dtype, device=student_prob.device)
    count = max(int(min_k), int(float(probabilities.numel()) * float(topk_ratio)))
    count = max(1, min(count, int(probabilities.numel())))
    return torch.topk(probabilities, k=count, largest=True, sorted=False).values.mean()


def _largest_component_mask(binary: np.ndarray, *, max_components: int, min_area: int) -> np.ndarray:
    work = np.asarray(binary, dtype=bool)
    if not work.any():
        return np.zeros_like(work, dtype=bool)
    visited = np.zeros_like(work, dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    height, width = work.shape
    for start_y, start_x in np.argwhere(work):
        start = (int(start_y), int(start_x))
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for neighbor_y, neighbor_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and work[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        if len(pixels) >= min_area:
            components.append((len(pixels), pixels))
    components.sort(key=lambda item: item[0], reverse=True)
    output = np.zeros_like(work, dtype=bool)
    for _, pixels in components[: max(1, max_components)]:
        for y, x in pixels:
            output[y, x] = True
    return output


def _component_false_alarm_loss(student_prob: torch.Tensor, background: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    threshold = float(cfg.get("component_threshold", 0.5) or 0.5)
    max_components = max(1, int(cfg.get("max_components", 8) or 8))
    min_area = max(1, int(cfg.get("min_component_area", 1) or 1))
    masks: list[torch.Tensor] = []
    predicted = ((student_prob.detach() >= threshold) & (background > 0.5)).float().cpu().numpy()
    for batch_index in range(predicted.shape[0]):
        for channel_index in range(predicted.shape[1]):
            mask = _largest_component_mask(
                predicted[batch_index, channel_index] > 0.5,
                max_components=max_components,
                min_area=min_area,
            )
            masks.append(torch.from_numpy(mask.astype(np.float32)))
    if not masks:
        return torch.zeros((), dtype=student_prob.dtype, device=student_prob.device)
    component_mask = torch.stack(masks, dim=0).view_as(student_prob).to(device=student_prob.device, dtype=student_prob.dtype)
    if float(component_mask.sum().detach().cpu()) <= 0.0:
        if str(cfg.get("empty_fallback", "zero") or "zero") == "topk":
            return _topk_background_probability(
                student_prob,
                background,
                topk_ratio=float(cfg.get("topk_ratio", 0.01) or 0.01),
                min_k=int(cfg.get("min_k", 64) or 64),
            )
        return torch.zeros((), dtype=student_prob.dtype, device=student_prob.device)
    return (student_prob * component_mask).sum() / component_mask.sum().clamp_min(1.0)


def _false_alarm_single_loss(
    student_prob: torch.Tensor,
    target: torch.Tensor,
    teacher_prob: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    background = (target < 0.5).float()
    loss_type = str(cfg.get("type", "background_mean_probability") or "background_mean_probability")
    if loss_type == "background_mean_probability":
        base = (student_prob * background).sum() / background.sum().clamp_min(1.0)
    elif loss_type == "background_topk_probability":
        base = _topk_background_probability(
            student_prob,
            background,
            topk_ratio=float(cfg.get("topk_ratio", 0.01) or 0.01),
            min_k=int(cfg.get("min_k", 64) or 64),
        )
    elif loss_type == "background_focal_probability":
        probabilities = torch.clamp(student_prob[background > 0.5], 1e-6, 1.0 - 1e-6)
        base = (
            torch.zeros((), dtype=student_prob.dtype, device=student_prob.device)
            if probabilities.numel() == 0
            else (torch.pow(probabilities, float(cfg.get("gamma", 2.0) or 2.0)) * (-torch.log1p(-probabilities))).mean()
        )
    elif loss_type == "background_component_probability":
        base = _component_false_alarm_loss(student_prob, background, cfg)
    else:
        raise RuntimeError(f"Unsupported false-alarm penalty type: {loss_type!r}")

    multiplier = float(cfg.get("hard_background_multiplier", 1.0) or 1.0)
    if multiplier != 1.0:
        threshold = float(cfg.get("teacher_fp_threshold", 0.5) or 0.5)
        hard_background = ((target < 0.5) & (teacher_prob >= threshold)).float()
        if float(hard_background.sum().detach().cpu()) > 0.0:
            hard_term = (student_prob * hard_background).sum() / hard_background.sum().clamp_min(1.0)
            base = base + (multiplier - 1.0) * hard_term
    return base


def _false_alarm_loss(
    student_prob: torch.Tensor,
    target: torch.Tensor,
    teacher_prob: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one or more weighted false-alarm penalties."""

    terms = cfg.get("terms")
    if not isinstance(terms, list) or not terms:
        return _false_alarm_single_loss(student_prob, target, teacher_prob, cfg), {}
    total = torch.zeros((), dtype=student_prob.dtype, device=student_prob.device)
    details: dict[str, float] = {}
    seen: dict[str, int] = {}
    for index, raw_term in enumerate(terms):
        if not isinstance(raw_term, dict):
            raise RuntimeError(f"False-alarm penalty term #{index} must be a mapping.")
        term_cfg = dict(raw_term)
        term_type = str(term_cfg.get("type", "background_mean_probability") or "background_mean_probability")
        term_weight = float(term_cfg.get("weight", 1.0) or 0.0)
        term_loss = _false_alarm_single_loss(student_prob, target, teacher_prob, term_cfg)
        total = total + term_weight * term_loss
        seen[term_type] = seen.get(term_type, 0) + 1
        suffix = term_type if seen[term_type] == 1 else f"{term_type}_{seen[term_type]}"
        details[suffix] = float(term_loss.detach().cpu())
        details[f"{suffix}_weight"] = term_weight
        details[f"{suffix}_weighted"] = float((term_weight * term_loss).detach().cpu())
    return total, details
