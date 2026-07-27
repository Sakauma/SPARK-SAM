from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .sample import Sample

MASK_SOURCE_KEY = "mask_source"


def polygon_to_mask(points: Sequence[float], height: int, width: int) -> np.ndarray:
    """把 MultiModal/COCO polygon rasterize 成二值 float32 mask。"""
    canvas = Image.new("L", (width, height), 0)
    xy = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
    ImageDraw.Draw(canvas).polygon(xy, outline=1, fill=1)
    return np.array(canvas, dtype=np.float32)


def _is_polygon_points(value: object) -> bool:
    return isinstance(value, list) and len(value) >= 6 and all(isinstance(item, (int, float)) for item in value)


def coco_segmentation_to_polygons(segmentation: object) -> list[list[float]]:
    """Return all polygon lists from a COCO polygon segmentation object."""
    if _is_polygon_points(segmentation):
        return [[float(item) for item in segmentation]]  # type: ignore[arg-type]
    if not isinstance(segmentation, list):
        return []
    polygons: list[list[float]] = []
    for item in segmentation:
        polygons.extend(coco_segmentation_to_polygons(item))
    return polygons


def coco_polygon_to_mask(segmentation: object, height: int, width: int) -> np.ndarray | None:
    polygons = coco_segmentation_to_polygons(segmentation)
    if not polygons:
        return None
    canvas = np.zeros((height, width), dtype=np.float32)
    for polygon in polygons:
        canvas = np.maximum(canvas, polygon_to_mask(polygon, height=height, width=width))
    return canvas if canvas.any() else None


def is_coco_rle_segmentation(segmentation: object) -> bool:
    if not isinstance(segmentation, dict):
        return False
    counts = segmentation.get("counts")
    return isinstance(counts, (list, str))


def coco_rle_is_decodable(segmentation: object) -> bool:
    if not is_coco_rle_segmentation(segmentation):
        return False
    assert isinstance(segmentation, dict)
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        return True
    try:
        from pycocotools import mask as mask_utils  # noqa: F401
    except Exception:
        return False
    return True


def _rle_size(segmentation: dict[str, Any], height: int, width: int) -> tuple[int, int]:
    size = segmentation.get("size")
    if isinstance(size, list) and len(size) == 2:
        return max(1, int(size[0])), max(1, int(size[1]))
    return height, width


def _decode_uncompressed_rle(counts: Sequence[object], height: int, width: int) -> np.ndarray | None:
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        next_offset = min(offset + max(0, count), flat.size)
        if value == 1 and next_offset > offset:
            flat[offset:next_offset] = 1
        offset = next_offset
        value = 1 - value
        if offset >= flat.size:
            break
    mask = flat.reshape((height, width), order="F").astype(np.float32)
    return mask if mask.any() else None


def _decode_compressed_rle(segmentation: dict[str, Any], height: int, width: int) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None
    counts = segmentation.get("counts")
    encoded_counts = counts.encode("utf-8") if isinstance(counts, str) else counts
    decoded = mask_utils.decode({"size": [height, width], "counts": encoded_counts})
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    mask = np.asarray(decoded, dtype=np.float32)
    return mask if mask.any() else None


def coco_rle_to_mask(segmentation: object, height: int, width: int) -> np.ndarray | None:
    if not is_coco_rle_segmentation(segmentation):
        return None
    assert isinstance(segmentation, dict)
    rle_height, rle_width = _rle_size(segmentation, height, width)
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        return _decode_uncompressed_rle(counts, height=rle_height, width=rle_width)
    if isinstance(counts, str):
        return _decode_compressed_rle(segmentation, height=rle_height, width=rle_width)
    return None


def coco_segmentation_to_mask(segmentation: object, height: int, width: int) -> np.ndarray | None:
    polygon_mask = coco_polygon_to_mask(segmentation, height=height, width=width)
    if polygon_mask is not None:
        return polygon_mask
    return coco_rle_to_mask(segmentation, height=height, width=width)


def _mask_path_to_array(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.float32)


def sample_mask_array(sample: Sample) -> np.ndarray | None:
    """按统一优先级读取样本 GT mask。

    优先级为 eager mask -> mask_path -> lazy metadata source。
    这样评估代码不需要关心某个数据集是直接保存 mask、从文件读 mask，还是按需从 polygon 解码。
    """
    if sample.mask_array is not None:
        return np.asarray(sample.mask_array, dtype=np.float32)
    if sample.mask_path is not None:
        return _mask_path_to_array(sample.mask_path)

    source = sample.metadata.get(MASK_SOURCE_KEY)
    if not isinstance(source, dict):
        return None
    height = int(source.get("height", sample.height))
    width = int(source.get("width", sample.width))
    source_type = source.get("type")
    if source_type == "polygon":
        points = source.get("points")
        if not isinstance(points, list) or len(points) < 6:
            return None
        return polygon_to_mask(points, height=height, width=width)
    if source_type == "coco_polygon":
        segmentation: Any = source.get("segmentation", source.get("polygons"))
        return coco_polygon_to_mask(segmentation, height=height, width=width)
    if source_type in {"coco_rle", "coco_segmentation"}:
        segmentation = source.get("segmentation")
        return coco_segmentation_to_mask(segmentation, height=height, width=width)
    return None


def sample_mask_or_zeros(sample: Sample) -> np.ndarray:
    """返回可直接参与指标计算的 GT mask；无 GT mask 时返回同尺寸全零 mask。"""
    mask = sample_mask_array(sample)
    if mask is not None:
        return mask
    return np.zeros((sample.height, sample.width), dtype=np.float32)
