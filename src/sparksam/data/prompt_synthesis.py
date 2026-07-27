from __future__ import annotations

from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


DEFAULT_BOX_PAD_RATIO = 0.15
DEFAULT_BOX_MIN_PAD = 2.0
DEFAULT_BOX_MIN_SIDE = 0.0
DEFAULT_BOX_MAX_SIDE_MULTIPLIER = 2.0
# prompt protocol 字符串会写入每个 eval row；修改规则时必须同步更新协议名。
MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL = "mask_derived_adaptive_loose_box_centroid_point_v2"
MASK_DERIVED_LEGACY_LOOSE_BOX_CENTROID_POINT_PROTOCOL = "mask_derived_loose_box_centroid_point_v1"
MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL = "mask_derived_tight_box_centroid_point_v1"
MASK_DERIVED_CENTROID_POINT_PROTOCOL = "mask_derived_centroid_point_v1"
MASK_DERIVED_PROMPT_PROTOCOL = MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL


def clamp_box_xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    # SAM2 接受原图像素坐标。这里保证 x2/y2 至少比 x1/y1 大 1 个像素并裁剪到图像内。
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(0.0, x1), max(0.0, width - 1.0))
    y1 = min(max(0.0, y1), max(0.0, height - 1.0))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return [x1, y1, x2, y2]


def mask_to_tight_box(mask: np.ndarray) -> list[float]:
    # tight box 是 GT 前景像素的最小轴对齐外接矩形。
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def expand_box_xyxy(
    box: Iterable[float],
    width: int,
    height: int,
    pad_ratio: float = DEFAULT_BOX_PAD_RATIO,
    min_pad: float = DEFAULT_BOX_MIN_PAD,
    min_side: float = DEFAULT_BOX_MIN_SIDE,
    max_side_multiplier: float | None = DEFAULT_BOX_MAX_SIDE_MULTIPLIER,
) -> list[float]:
    # loose box 模拟实际人工框提示，同时对极小目标限制扩张比例，避免框面积被 padding 主导。
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(min_pad, bw * pad_ratio)
    pad_y = max(min_pad, bh * pad_ratio)
    if max_side_multiplier is not None and max_side_multiplier > 0:
        max_pad_x = max(0.0, (bw * float(max_side_multiplier) - bw) * 0.5)
        max_pad_y = max(0.0, (bh * float(max_side_multiplier) - bh) * 0.5)
        pad_x = min(pad_x, max_pad_x)
        pad_y = min(pad_y, max_pad_y)
    loose = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    if loose[2] - loose[0] < min_side:
        extra = 0.5 * (min_side - (loose[2] - loose[0]))
        loose[0] -= extra
        loose[2] += extra
    if loose[3] - loose[1] < min_side:
        extra = 0.5 * (min_side - (loose[3] - loose[1]))
        loose[1] -= extra
        loose[3] += extra
    return clamp_box_xyxy(loose, width, height)


def mask_to_point_prompt(mask: np.ndarray) -> list[float]:
    # point prompt 使用前景像素质心；对不规则目标比 bbox 中心更贴近真实前景。
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def mask_derived_prompt_metadata() -> dict[str, object]:
    # 这些字段会进入结果行，便于论文中明确声明 prompt 来源和生成规则。
    return {
        "source": "gt_mask",
        "protocol": "mask_derived_gt_prompt_rules_v1",
        "loose_box_point_protocol": MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL,
        "tight_box_point_protocol": MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL,
        "point_protocol": MASK_DERIVED_CENTROID_POINT_PROTOCOL,
        "tight_box_rule": "axis_aligned_foreground_bounding_rectangle",
        "loose_box_rule": "expand_tight_box_with_adaptive_small_target_cap_then_clip_to_image",
        "loose_box_pad_ratio": DEFAULT_BOX_PAD_RATIO,
        "loose_box_min_pad": DEFAULT_BOX_MIN_PAD,
        "loose_box_min_side": DEFAULT_BOX_MIN_SIDE,
        "loose_box_max_side_multiplier": DEFAULT_BOX_MAX_SIDE_MULTIPLIER,
        "point_rule": "foreground_pixel_centroid",
    }


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    # 小目标召回按连通域评估；使用 4 邻域可以避免斜角接触目标被合并。
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[np.ndarray] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            coords: list[Tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx] or binary[ny, nx] == 0:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            component = np.zeros_like(mask, dtype=np.float32)
            for cy, cx in coords:
                component[cy, cx] = 1.0
            components.append(component)
    return components
