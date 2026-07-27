from __future__ import annotations

import numpy as np


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    # mIoU/Dice 都在二值 mask 上计算，阈值固定为 0.5。
    pred_b = pred > 0.5
    target_b = target > 0.5
    union = float(np.logical_or(pred_b, target_b).sum())
    if union <= 0.0:
        return 1.0
    return float(np.logical_and(pred_b, target_b).sum()) / union


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred > 0.5
    target_b = target > 0.5
    denom = float(pred_b.sum() + target_b.sum())
    if denom <= 0.0:
        return 1.0
    return 2.0 * float(np.logical_and(pred_b, target_b).sum()) / denom


def _boundary(mask: np.ndarray) -> np.ndarray:
    # 简单 4 邻域边界提取，用于无额外依赖的 boundary F1。
    mask_b = mask > 0.5
    up = np.pad(mask_b[1:, :], ((0, 1), (0, 0)))
    down = np.pad(mask_b[:-1, :], ((1, 0), (0, 0)))
    left = np.pad(mask_b[:, 1:], ((0, 0), (0, 1)))
    right = np.pad(mask_b[:, :-1], ((0, 0), (1, 0)))
    same = up & down & left & right & mask_b
    return (mask_b ^ same).astype(np.float32)


def boundary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = _boundary(pred) > 0.5
    target_b = _boundary(target) > 0.5
    return boundary_f1_from_masks(pred_b, target_b)


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    output = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            src_y1 = max(0, -dy)
            src_y2 = min(h, h - dy)
            src_x1 = max(0, -dx)
            src_x2 = min(w, w - dx)
            dst_y1 = max(0, dy)
            dst_y2 = min(h, h + dy)
            dst_x1 = max(0, dx)
            dst_x2 = min(w, w + dx)
            output[dst_y1:dst_y2, dst_x1:dst_x2] |= binary[src_y1:src_y2, src_x1:src_x2]
    return output


def boundary_f1_from_masks(pred_b: np.ndarray, target_b: np.ndarray) -> float:
    tp = float(np.logical_and(pred_b, target_b).sum())
    pred_sum = float(pred_b.sum())
    target_sum = float(target_b.sum())
    if pred_sum <= 0.0 and target_sum <= 0.0:
        return 1.0
    if pred_sum <= 0.0 or target_sum <= 0.0:
        return 0.0
    precision = tp / pred_sum
    recall = tp / target_sum
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def boundary_f1_tolerance(pred: np.ndarray, target: np.ndarray, radius: int = 1) -> float:
    # 小目标边界常有 1 像素级偏移，tolerance 版本允许半径内匹配。
    pred_b = _boundary(pred) > 0.5
    target_b = _boundary(target) > 0.5
    pred_sum = float(pred_b.sum())
    target_sum = float(target_b.sum())
    if pred_sum <= 0.0 and target_sum <= 0.0:
        return 1.0
    if pred_sum <= 0.0 or target_sum <= 0.0:
        return 0.0
    target_dilated = _dilate(target_b, radius)
    pred_dilated = _dilate(pred_b, radius)
    precision = float(np.logical_and(pred_b, target_dilated).sum()) / pred_sum
    recall = float(np.logical_and(target_b, pred_dilated).sum()) / target_sum
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    # bbox IoU 使用预测 mask 的外接框与 GT prompt box 比较，bbox-only 数据集也可输出。
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return intersection / max(1.0, area_a + area_b - intersection)
