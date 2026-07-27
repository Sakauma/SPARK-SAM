from __future__ import annotations

import json
import os
import random
import sys
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml
from PIL import Image

from ..config import load_app_config
from ..data import build_dataset_adapter
from ..data.prompt_synthesis import clamp_box_xyxy
from ..data.sample import Sample
from ..evaluation.heatmaps import write_heatmap_artifact
from ..models import AutoPromptModelConfig, build_infrared_prompt_estimator, count_auto_prompt_parameters, ir_prior_stack, save_auto_prompt_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _require_torch():
    try:
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("auto prompt training requires PyTorch.") from exc
    return torch, F, DataLoader, Dataset


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return payload


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [PROJECT_ROOT / path, Path.cwd() / path, base / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if path.parts and path.parts[0] in {"artifacts", "configs", "splits"}:
        return (PROJECT_ROOT / path).resolve()
    return (base / path).resolve()


def _load_training_samples(config_path: Path, *, max_samples: int = 0) -> tuple[list[Sample], dict[str, int]]:
    raw = _read_yaml(config_path)
    dataset_configs = raw.get("dataset_configs", [])
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError("auto prompt training config requires a non-empty dataset_configs list.")

    samples: list[Sample] = []
    counts: dict[str, int] = {}
    for item in dataset_configs:
        dataset_config = _resolve_path(config_path.parent, str(item))
        app_config = load_app_config(dataset_config)
        loaded = build_dataset_adapter(app_config).load(app_config)
        usable = [sample for sample in loaded.samples if sample.bbox_tight is not None or sample.bbox_loose is not None]
        for sample in usable:
            sample.metadata.setdefault("dataset_id", app_config.dataset.dataset_id)
            counts[f"dataset:{app_config.dataset.dataset_id}"] = counts.get(f"dataset:{app_config.dataset.dataset_id}", 0) + 1
            counts[f"supervision:{sample.supervision_type}"] = counts.get(f"supervision:{sample.supervision_type}", 0) + 1
        samples.extend(usable)
        if max_samples > 0 and len(samples) >= max_samples:
            clipped = samples[:max_samples]
            clipped_counts: dict[str, int] = {}
            for sample in clipped:
                clipped_counts[f"supervision:{sample.supervision_type}"] = clipped_counts.get(f"supervision:{sample.supervision_type}", 0) + 1
            return clipped, {**counts, **{f"used_{key}": value for key, value in clipped_counts.items()}}
    return samples, counts


def _bool_setting(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _training_dataset_paths(config_path: Path, *, key: str = "dataset_configs") -> list[str]:
    raw = _read_yaml(config_path)
    dataset_configs = raw.get(key, [])
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError(f"auto prompt training config requires a non-empty {key} list.")
    return [str(item) for item in dataset_configs]


def _iter_training_samples(
    config_path: Path,
    *,
    dataset_configs: list[str] | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> Iterator[Sample]:
    if dataset_configs is None:
        dataset_configs = _training_dataset_paths(config_path)
    shard_text = f" shard={shard_id}/{num_shards}" if num_shards > 1 else ""
    for item in dataset_configs:
        dataset_config = _resolve_path(config_path.parent, str(item))
        app_config = load_app_config(dataset_config)
        adapter = build_dataset_adapter(app_config)
        started_at = time.perf_counter()
        print(
            f"[train-load] stream_start dataset={app_config.dataset.dataset_id} "
            f"adapter={adapter.adapter_name} max_samples={app_config.runtime.max_samples} "
            f"max_images={app_config.runtime.max_images}{shard_text}",
            file=sys.stderr,
            flush=True,
        )
        loaded_count = 0
        usable_count = 0
        first_usable_reported = False
        for sample in adapter.iter_samples(app_config, shard_id=shard_id, num_shards=num_shards):
            loaded_count += 1
            if sample.bbox_tight is None and sample.bbox_loose is None:
                continue
            usable_count += 1
            sample.metadata.setdefault("dataset_id", app_config.dataset.dataset_id)
            if not first_usable_reported:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[train-load] first_sample dataset={app_config.dataset.dataset_id} "
                    f"elapsed={elapsed:.1f}s sample_id={sample.sample_id}{shard_text}",
                    file=sys.stderr,
                    flush=True,
                )
                first_usable_reported = True
            yield sample
        elapsed = time.perf_counter() - started_at
        print(
            f"[train-load] stream_done dataset={app_config.dataset.dataset_id} "
            f"loaded={loaded_count} usable={usable_count} elapsed={elapsed:.1f}s{shard_text}",
            file=sys.stderr,
            flush=True,
        )


def _shuffle_buffer(samples: Iterable[Sample], *, buffer_size: int, rng: random.Random) -> Iterator[Sample]:
    if buffer_size <= 1:
        yield from samples
        return
    buffer: list[Sample] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample
    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _limit_samples(samples: Iterable[Sample], *, limit: int) -> Iterator[Sample]:
    if limit <= 0:
        yield from samples
        return
    iterator = iter(samples)
    for _ in range(limit):
        try:
            yield next(iterator)
        except StopIteration:
            return


def _load_resized_gray_and_box(sample: Sample, max_long_side: int) -> tuple[np.ndarray, list[float]]:
    with Image.open(sample.image_path) as image:
        gray_image = image.convert("L")
        src_w, src_h = gray_image.size
        scale = 1.0
        if max_long_side > 0 and max(src_h, src_w) > max_long_side:
            scale = float(max_long_side) / float(max(src_h, src_w))
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            gray_image = gray_image.resize((dst_w, dst_h), Image.BILINEAR)
        arr = np.asarray(gray_image, dtype=np.float32)

    raw_box = sample.bbox_tight or sample.bbox_loose
    if raw_box is None:
        raise ValueError(f"Sample {sample.sample_id!r} has no training box.")
    box = [float(value) * scale for value in raw_box[:4]]
    return arr, clamp_box_xyxy(box, width=arr.shape[1], height=arr.shape[0])


def _target_from_box(
    *,
    box: list[float],
    height: int,
    width: int,
    gaussian_sigma: float,
    positive_radius: int,
    min_box_side: float,
    hard_negative_weight: float,
    hard_negative_percentile: float,
    prior_score: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = clamp_box_xyxy(box, width=width, height=height)
    cx = float(np.clip(0.5 * (x1 + x2), 0.0, max(0.0, width - 1.0)))
    cy = float(np.clip(0.5 * (y1 + y2), 0.0, max(0.0, height - 1.0)))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if gaussian_sigma > 0.0:
        objectness = np.exp(-dist2 / (2.0 * gaussian_sigma * gaussian_sigma)).astype(np.float32)
    else:
        objectness = np.zeros((height, width), dtype=np.float32)
        objectness[int(round(cy)), int(round(cx))] = 1.0
    positive = (dist2 <= float(max(0, positive_radius)) ** 2).astype(np.float32)
    if float(positive.sum()) <= 0.0:
        positive[int(round(cy)), int(round(cx))] = 1.0
    size = np.zeros((2, height, width), dtype=np.float32)
    size[0, :, :] = max(float(min_box_side), float(x2 - x1))
    size[1, :, :] = max(float(min_box_side), float(y2 - y1))
    objectness_weight = np.ones((height, width), dtype=np.float32)
    if hard_negative_weight > 1.0 and prior_score is not None:
        x1i, y1i, x2i, y2i = [int(round(value)) for value in (x1, y1, x2, y2)]
        inside = np.zeros((height, width), dtype=bool)
        inside[max(0, y1i) : min(height, y2i), max(0, x1i) : min(width, x2i)] = True
        prior = np.asarray(prior_score, dtype=np.float32)
        threshold = float(np.percentile(prior, min(max(float(hard_negative_percentile), 0.0), 100.0)))
        hard_negative = np.logical_and(~inside, prior >= threshold)
        objectness_weight[hard_negative] = float(hard_negative_weight)
    return objectness[None], size, positive[None], objectness_weight[None]


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    def pad(arr: np.ndarray, channels: int) -> np.ndarray:
        out = np.zeros((channels, max_h, max_w), dtype=np.float32)
        out[:, : arr.shape[1], : arr.shape[2]] = arr
        return out

    return {
        "image": torch.from_numpy(np.stack([pad(item["image"], 3) for item in batch])),
        "objectness": torch.from_numpy(np.stack([pad(item["objectness"], 1) for item in batch])),
        "objectness_weight": torch.from_numpy(np.stack([pad(item["objectness_weight"], 1) for item in batch])),
        "box_size": torch.from_numpy(np.stack([pad(item["box_size"], 2) for item in batch])),
        "box_weight": torch.from_numpy(np.stack([pad(item["box_weight"], 1) for item in batch])),
        "sample_weight": torch.tensor([float(item.get("sample_weight", 1.0)) for item in batch], dtype=torch.float32),
        "sample_ids": [item["sample_id"] for item in batch],
        "dataset_ids": [item["dataset_id"] for item in batch],
        "area_buckets": [item["area_bucket"] for item in batch],
        "supervision_types": [item["supervision_type"] for item in batch],
        "samples": [item["sample"] for item in batch],
    }


def _box_area(sample: Sample) -> float:
    box = sample.bbox_tight or sample.bbox_loose
    if box is None:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _sample_area_pixels(sample: Sample) -> float:
    value = sample.metadata.get("gt_area_pixels")
    if isinstance(value, bool):
        return _box_area(sample)
    if isinstance(value, (int, float)):
        return float(value)
    return _box_area(sample)


def _area_bucket_from_pixels(area: float) -> str:
    if area < 16:
        return "tiny_0_15"
    if area < 64:
        return "tiny_16_63"
    if area < 256:
        return "small_64_255"
    if area < 1024:
        return "small_256_1023"
    return "large_ge_1024"


def _sample_area_bucket(sample: Sample) -> str:
    value = sample.metadata.get("area_bucket")
    if value:
        return str(value)
    return _area_bucket_from_pixels(_sample_area_pixels(sample))


def _metadata_object_count(sample: Sample) -> int:
    value = sample.metadata.get("objects_per_image", 1)
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 1


def _sample_density_bucket(sample: Sample) -> str:
    value = sample.metadata.get("density_bucket")
    if value:
        return str(value)
    count = _metadata_object_count(sample)
    if count <= 1:
        return "single_1"
    if count <= 3:
        return "low_2_3"
    if count <= 10:
        return "mid_4_10"
    if count <= 45:
        return "high_11_45"
    return "dense_gt_45"


def _density_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = train_cfg.get("density_sampling", {})
    return cfg if isinstance(cfg, dict) else {}


def _density_weight_from_sample(sample: Sample, *, train_cfg: dict[str, Any], purpose: str) -> float:
    cfg = _density_config(train_cfg)
    if not cfg or not _bool_setting(cfg.get("enabled"), default=False):
        return 1.0
    flag = "loss_weighting" if purpose == "loss" else "sampling_weighting"
    if not _bool_setting(cfg.get(flag), default=True):
        return 1.0
    bucket_weights = cfg.get("bucket_weights", {})
    bucket = _sample_density_bucket(sample)
    if isinstance(bucket_weights, dict) and bucket in bucket_weights:
        weight = float(bucket_weights[bucket])
    else:
        mode = str(cfg.get("mode", "inverse_sqrt_objects")).strip().lower()
        count = _metadata_object_count(sample)
        if mode in {"inverse_objects", "inverse_count"}:
            weight = 1.0 / float(count)
        elif mode in {"none", "off", "uniform"}:
            weight = 1.0
        else:
            weight = 1.0 / (float(count) ** 0.5)
    min_weight = float(cfg.get("min_weight", 0.1))
    max_weight = float(cfg.get("max_weight", 1.0))
    if max_weight < min_weight:
        min_weight, max_weight = max_weight, min_weight
    return float(min(max(weight, min_weight), max_weight))


def _configured_sample_weight(sample: Sample, *, train_cfg: dict[str, Any]) -> float:
    weight = 1.0
    domain_weights = train_cfg.get("domain_loss_weights", {})
    if isinstance(domain_weights, dict):
        dataset_id = str(sample.metadata.get("dataset_id", "unknown"))
        weight *= float(domain_weights.get(dataset_id, domain_weights.get(dataset_id.lower(), 1.0)))
    area_weights = train_cfg.get("area_loss_weights", {})
    if isinstance(area_weights, dict):
        bucket = _sample_area_bucket(sample)
        weight *= float(area_weights.get(bucket, 1.0))
    weight *= _density_weight_from_sample(sample, train_cfg=train_cfg, purpose="loss")
    return float(weight)


def _sample_to_training_item(
    sample: Sample,
    *,
    max_long_side: int,
    target_config: dict[str, Any],
    model_config: AutoPromptModelConfig,
    train_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gray, box = _load_resized_gray_and_box(sample, max_long_side)
    image = ir_prior_stack(
        gray,
        use_local_contrast=model_config.use_local_contrast,
        use_top_hat=model_config.use_top_hat,
        use_dog_filter=model_config.use_dog_filter,
        use_fft_highpass=model_config.use_fft_highpass,
    )
    prior_score = np.maximum.reduce(image)
    objectness, box_size, box_weight, objectness_weight = _target_from_box(
        box=box,
        height=int(image.shape[1]),
        width=int(image.shape[2]),
        gaussian_sigma=float(target_config.get("gaussian_sigma", 2.0)),
        positive_radius=int(target_config.get("positive_radius", 1)),
        min_box_side=float(model_config.min_box_side),
        hard_negative_weight=float(target_config.get("hard_negative_weight", 1.0)),
        hard_negative_percentile=float(target_config.get("hard_negative_percentile", 95.0)),
        prior_score=prior_score,
    )
    return {
        "image": image,
        "objectness": objectness,
        "objectness_weight": objectness_weight,
        "box_size": box_size,
        "box_weight": box_weight,
        "sample_id": sample.sample_id,
        "dataset_id": str(sample.metadata.get("dataset_id", "unknown")),
        "area_bucket": _sample_area_bucket(sample),
        "sample_weight": _configured_sample_weight(sample, train_cfg=train_config or {}),
        "supervision_type": sample.supervision_type,
        "sample": sample,
    }


def _load_resized_gray_uint8_and_box(sample: Sample, max_long_side: int) -> tuple[np.ndarray, list[float]]:
    gray, box = _load_resized_gray_and_box(sample, max_long_side)
    return np.clip(gray, 0.0, 255.0).astype(np.uint8), box


def _torch_normalize_maps(torch: Any, x: Any) -> Any:
    flat = x.flatten(2)
    min_v = flat.min(dim=2).values.view(x.shape[0], x.shape[1], 1, 1)
    max_v = flat.max(dim=2).values.view(x.shape[0], x.shape[1], 1, 1)
    return (x - min_v) / (max_v - min_v).clamp_min(1e-6)


def _pad_for_pool(torch: Any, x: Any, radius: int) -> Any:
    if radius <= 0:
        return x
    mode = "reflect" if x.shape[-2] > radius and x.shape[-1] > radius else "replicate"
    return torch.nn.functional.pad(x, (radius, radius, radius, radius), mode=mode)


def _avg_pool_reflect(torch: Any, F: Any, x: Any, radius: int) -> Any:
    kernel = 2 * int(radius) + 1
    return F.avg_pool2d(_pad_for_pool(torch, x, int(radius)), kernel_size=kernel, stride=1)


def _min_pool_reflect(torch: Any, F: Any, x: Any, kernel: int) -> Any:
    radius = int(kernel) // 2
    return -F.max_pool2d(-_pad_for_pool(torch, x, radius), kernel_size=int(kernel), stride=1)


def _max_pool_reflect(torch: Any, F: Any, x: Any, kernel: int) -> Any:
    radius = int(kernel) // 2
    return F.max_pool2d(_pad_for_pool(torch, x, radius), kernel_size=int(kernel), stride=1)


def _fft_highpass_residual_batch(torch: Any, gray: Any, cutoff_fraction: float = 0.12) -> Any:
    if min(int(gray.shape[-2]), int(gray.shape[-1])) < 4:
        return torch.zeros_like(gray)
    centered = gray - gray.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered.float(), dim=(-2, -1)), dim=(-2, -1))
    h, w = int(gray.shape[-2]), int(gray.shape[-1])
    yy = torch.arange(h, device=gray.device, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=gray.device, dtype=torch.float32).view(1, 1, 1, w)
    cy = float(h - 1) * 0.5
    cx = float(w - 1) * 0.5
    radius = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
    cutoff = float(cutoff_fraction) * float(min(h, w))
    spectrum = spectrum * (radius >= cutoff)
    residual = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)).abs()
    return _torch_normalize_maps(torch, residual)


def _ir_prior_stack_batch(
    torch: Any,
    F: Any,
    gray: Any,
    *,
    use_local_contrast: bool = True,
    use_top_hat: bool = True,
    use_dog_filter: bool = False,
    use_fft_highpass: bool = False,
) -> Any:
    arr = _torch_normalize_maps(torch, gray)
    if use_local_contrast:
        small_mean = _avg_pool_reflect(torch, F, arr, radius=2)
        large_mean = _avg_pool_reflect(torch, F, arr, radius=8)
        local_contrast = _torch_normalize_maps(torch, (small_mean - large_mean).clamp_min(0.0))
    else:
        local_contrast = torch.zeros_like(arr)
    if use_top_hat:
        hats = []
        for kernel in (3, 5, 9):
            opened = _max_pool_reflect(torch, F, _min_pool_reflect(torch, F, arr, kernel), kernel)
            hats.append((arr - opened).clamp_min(0.0))
        top_hat = _torch_normalize_maps(torch, torch.stack(hats, dim=0).max(dim=0).values)
    else:
        top_hat = torch.zeros_like(arr)
    channels = [arr, local_contrast, top_hat]
    if use_dog_filter:
        dog = _torch_normalize_maps(torch, (_avg_pool_reflect(torch, F, arr, radius=1) - _avg_pool_reflect(torch, F, arr, radius=5)).clamp_min(0.0))
        channels.append(dog)
    if use_fft_highpass:
        channels.append(_fft_highpass_residual_batch(torch, arr))
    return torch.cat(channels, dim=1).float()


def _target_from_box_batch(
    torch: Any,
    *,
    boxes: Any,
    height: int,
    width: int,
    gaussian_sigma: float,
    positive_radius: int,
    min_box_side: float,
    hard_negative_weight: float,
    hard_negative_percentile: float,
    prior_score: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    device = boxes.device
    boxes = boxes.float()
    x1 = boxes[:, 0].clamp(0.0, max(0.0, float(width - 1)))
    y1 = boxes[:, 1].clamp(0.0, max(0.0, float(height - 1)))
    x2 = boxes[:, 2].clamp(0.0, float(width))
    y2 = boxes[:, 3].clamp(0.0, float(height))
    cx = (0.5 * (x1 + x2)).clamp(0.0, max(0.0, float(width - 1)))
    cy = (0.5 * (y1 + y2)).clamp(0.0, max(0.0, float(height - 1)))
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    dist2 = (xx - cx.view(-1, 1, 1)) ** 2 + (yy - cy.view(-1, 1, 1)) ** 2
    if gaussian_sigma > 0.0:
        objectness = torch.exp(-dist2 / (2.0 * float(gaussian_sigma) * float(gaussian_sigma)))
    else:
        objectness = torch.zeros_like(dist2)
        objectness[
            torch.arange(boxes.shape[0], device=device),
            cy.round().long().clamp(0, height - 1),
            cx.round().long().clamp(0, width - 1),
        ] = 1.0
    positive = (dist2 <= float(max(0, int(positive_radius))) ** 2).float()
    empty_positive = positive.flatten(1).sum(dim=1) <= 0
    if bool(empty_positive.any()):
        positive[
            torch.where(empty_positive)[0],
            cy[empty_positive].round().long().clamp(0, height - 1),
            cx[empty_positive].round().long().clamp(0, width - 1),
        ] = 1.0
    box_w = (x2 - x1).clamp_min(float(min_box_side)).view(-1, 1, 1)
    box_h = (y2 - y1).clamp_min(float(min_box_side)).view(-1, 1, 1)
    box_size = torch.stack([box_w.expand(-1, height, width), box_h.expand(-1, height, width)], dim=1)
    objectness_weight = torch.ones((boxes.shape[0], height, width), device=device, dtype=torch.float32)
    if hard_negative_weight > 1.0 and prior_score is not None:
        inside = (
            (xx >= x1.view(-1, 1, 1).round())
            & (xx < x2.view(-1, 1, 1).round())
            & (yy >= y1.view(-1, 1, 1).round())
            & (yy < y2.view(-1, 1, 1).round())
        )
        q = min(max(float(hard_negative_percentile) / 100.0, 0.0), 1.0)
        threshold = torch.quantile(prior_score.float().flatten(1), q=q, dim=1).view(-1, 1, 1)
        objectness_weight[(~inside) & (prior_score.float() >= threshold)] = float(hard_negative_weight)
    return objectness[:, None], box_size, positive[:, None], objectness_weight[:, None]


def _positive_balance_weight(torch: Any, positive: Any, *, max_positive_weight: float) -> Any:
    positive = positive.float()
    negative = 1.0 - positive
    total = positive.flatten(1).shape[1]
    pos_count = positive.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    neg_count = negative.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    pos_weight = (float(total) / (2.0 * pos_count)).clamp(max=float(max_positive_weight))
    neg_weight = float(total) / (2.0 * neg_count)
    return positive * pos_weight + negative * neg_weight


def _objectness_loss(
    *,
    torch: Any,
    F: Any,
    logits: Any,
    target: Any,
    positive: Any,
    objectness_weight: Any,
    train_cfg: dict[str, Any],
) -> Any:
    mode = str(train_cfg.get("objectness_loss", "weighted_bce")).strip().lower()
    base = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = objectness_weight.float()
    if mode in {"balanced_bce", "balanced_focal", "focal_balanced"}:
        weight = weight * _positive_balance_weight(
            torch,
            (positive > 0.0).float(),
            max_positive_weight=float(train_cfg.get("max_positive_weight", 256.0)),
        )
    if mode in {"focal", "balanced_focal", "focal_balanced"}:
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        alpha = float(train_cfg.get("focal_alpha", 0.75))
        pt = torch.exp(-base)
        alpha_factor = torch.where((positive > 0.0), torch.full_like(base, alpha), torch.full_like(base, 1.0 - alpha))
        base = alpha_factor * (1.0 - pt).clamp_min(0.0).pow(gamma) * base
    return (base * weight).mean()


def _ranking_loss(
    *,
    torch: Any,
    logits: Any,
    positive: Any,
    train_cfg: dict[str, Any],
) -> Any:
    weight = float(train_cfg.get("ranking_loss_weight", 0.0))
    if weight <= 0.0:
        return logits.new_tensor(0.0)
    margin = float(train_cfg.get("ranking_margin", 1.0))
    negative_top_k = max(1, int(train_cfg.get("ranking_negative_top_k", 32)))
    flat_logits = logits.flatten(1)
    flat_positive = (positive > 0.0).flatten(1)
    losses = []
    for sample_logits, sample_positive in zip(flat_logits, flat_positive):
        if not bool(sample_positive.any()):
            continue
        pos_score = sample_logits[sample_positive].max()
        neg_logits = sample_logits[~sample_positive]
        if int(neg_logits.numel()) <= 0:
            continue
        top_k = min(negative_top_k, int(neg_logits.numel()))
        neg_score = torch.topk(neg_logits, k=top_k).values.mean()
        losses.append((margin + neg_score - pos_score).clamp_min(0.0))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _heuristic_distill_loss(*, F: Any, logits: Any, image: Any, train_cfg: dict[str, Any]) -> Any:
    weight = float(train_cfg.get("heuristic_distill_weight", 0.0))
    if weight <= 0.0:
        return logits.new_tensor(0.0)
    teacher = image.float().max(dim=1, keepdim=True).values.clamp(0.0, 1.0)
    return F.mse_loss(logits.sigmoid(), teacher)


def _dinov3_distill_loss(*, F: Any, logits: Any, batch: dict[str, Any], train_cfg: dict[str, Any]) -> Any:
    weight = float(train_cfg.get("dinov3_distill_weight", 0.0))
    if weight <= 0.0:
        return logits.new_tensor(0.0)
    target = batch.get("dinov3_targetness")
    if target is None:
        missing_policy = str(train_cfg.get("dinov3_missing_policy", "zero_loss")).strip().lower()
        if missing_policy in {"error", "raise"}:
            raise RuntimeError("dinov3_distill_weight > 0 but batch does not contain dinov3_targetness.")
        return logits.new_tensor(0.0)
    target = target.to(device=logits.device, dtype=logits.dtype)
    if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
        target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    target = target.clamp(0.0, 1.0)
    loss_name = str(train_cfg.get("dinov3_distill_loss", "mse")).strip().lower()
    if loss_name in {"bce", "binary_cross_entropy"}:
        return F.binary_cross_entropy_with_logits(logits, target)
    return F.mse_loss(logits.sigmoid(), target)


def _compute_auto_prompt_losses(
    *,
    torch: Any,
    F: Any,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
    teacher_outputs: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    objectness = batch["objectness"]
    objectness_weight = batch["objectness_weight"]
    box_size = batch["box_size"]
    box_weight = batch["box_weight"]
    logits = outputs["objectness_logits"]
    box_loss_weight = float(train_cfg.get("box_loss_weight", 0.1))
    if str(train_cfg.get("box_supervision_mode", "")).strip().lower() in {"point_only", "objectness_only", "center_only"}:
        box_loss_weight = 0.0
    objectness_loss = _objectness_loss(
        torch=torch,
        F=F,
        logits=logits,
        target=objectness,
        positive=box_weight,
        objectness_weight=objectness_weight,
        train_cfg=train_cfg,
    )
    raw_box_loss = F.smooth_l1_loss(outputs["box_size"], box_size, reduction="none")
    weighted_box_loss = raw_box_loss * box_weight
    box_loss = weighted_box_loss.sum() / box_weight.sum().clamp_min(1.0)
    ranking_loss = _ranking_loss(torch=torch, logits=logits, positive=box_weight, train_cfg=train_cfg)
    distill_loss = _heuristic_distill_loss(F=F, logits=logits, image=batch["image"], train_cfg=train_cfg)
    dinov3_distill_loss = _dinov3_distill_loss(F=F, logits=logits, batch=batch, train_cfg=train_cfg)
    teacher_loss = logits.new_tensor(0.0)
    teacher_weight = float(train_cfg.get("teacher_loss_weight", 0.0))
    if teacher_outputs is not None and teacher_weight > 0.0:
        teacher_logits = teacher_outputs["objectness_logits"].detach()
        teacher_loss = F.mse_loss(logits.sigmoid(), teacher_logits.sigmoid())
    total_distill_loss = distill_loss + dinov3_distill_loss
    loss = (
        objectness_loss
        + box_loss_weight * box_loss
        + float(train_cfg.get("ranking_loss_weight", 0.0)) * ranking_loss
        + float(train_cfg.get("heuristic_distill_weight", 0.0)) * distill_loss
        + float(train_cfg.get("dinov3_distill_weight", 0.0)) * dinov3_distill_loss
        + teacher_weight * teacher_loss
    )
    return loss, {
        "objectness_loss": objectness_loss,
        "box_loss": box_loss,
        "ranking_loss": ranking_loss,
        "heuristic_distill_loss": distill_loss,
        "dinov3_distill_loss": dinov3_distill_loss,
        "distill_loss": total_distill_loss,
        "teacher_loss": teacher_loss,
    }


def _build_dataset_class():
    _, _, _, Dataset = _require_torch()

    class AutoPromptDataset(Dataset):
        def __init__(
            self,
            samples: list[Sample],
            *,
            max_long_side: int,
            target_config: dict[str, Any],
            model_config: AutoPromptModelConfig,
        ) -> None:
            self.samples = samples
            self.max_long_side = int(max_long_side)
            self.target_config = target_config
            self.model_config = model_config

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = self.samples[index]
            return _sample_to_training_item(
                sample,
                max_long_side=self.max_long_side,
                target_config=self.target_config,
                model_config=self.model_config,
            )

    return AutoPromptDataset


def _build_streaming_dataset_class():
    torch, _, _, _ = _require_torch()
    IterableDataset = torch.utils.data.IterableDataset

    class StreamingAutoPromptDataset(IterableDataset):
        def __init__(
            self,
            *,
            config_path: Path,
            max_long_side: int,
            target_config: dict[str, Any],
            model_config: AutoPromptModelConfig,
            shuffle_buffer_size: int,
            max_samples: int,
            seed: int,
        ) -> None:
            super().__init__()
            self.config_path = config_path
            self.max_long_side = int(max_long_side)
            self.target_config = target_config
            self.model_config = model_config
            self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
            self.max_samples = max(0, int(max_samples))
            self.seed = int(seed)
            self.iteration = 0

        def _sample_to_item(self, sample: Sample) -> dict[str, Any]:
            return _sample_to_training_item(
                sample,
                max_long_side=self.max_long_side,
                target_config=self.target_config,
                model_config=self.model_config,
            )

        def __iter__(self) -> Iterator[dict[str, Any]]:
            worker_info = torch.utils.data.get_worker_info()
            iteration = self.iteration
            self.iteration += 1
            shard_id = int(worker_info.id) if worker_info is not None else 0
            num_shards = int(worker_info.num_workers) if worker_info is not None else 1
            rng = random.Random(self.seed + iteration * max(1, num_shards) + shard_id)
            sample_limit = self.max_samples
            if sample_limit > 0 and num_shards > 1:
                sample_limit = (sample_limit + num_shards - 1) // num_shards
            sample_iterable = _shuffle_buffer(
                _limit_samples(
                    _iter_training_samples(self.config_path, shard_id=shard_id, num_shards=num_shards),
                    limit=sample_limit,
                ),
                buffer_size=self.shuffle_buffer_size,
                rng=rng,
            )
            for sample in sample_iterable:
                yield self._sample_to_item(sample)

    return StreamingAutoPromptDataset


class _LineProgress:
    def __init__(self, *, desc: str, total: int | None, unit: str, mininterval: float) -> None:
        self.desc = desc
        self.total = total
        self.unit = unit
        self.mininterval = max(0.0, float(mininterval))
        self.count = 0
        self.started_at = time.perf_counter()
        self.last_print_at = 0.0
        self.postfix: dict[str, str] = {}
        self._print(force=True)

    def set_postfix(self, **kwargs: Any) -> None:
        self.postfix = {key: str(value) for key, value in kwargs.items()}

    def update(self, n: int) -> None:
        self.count += int(n)
        self._print(force=self.total is not None and self.count >= self.total)

    def close(self) -> None:
        self._print(force=True)

    def _print(self, *, force: bool) -> None:
        now = time.perf_counter()
        if not force and now - self.last_print_at < self.mininterval:
            return
        elapsed = max(0.0, now - self.started_at)
        rate = self.count / elapsed if elapsed > 0 else 0.0
        postfix = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        postfix_text = f" {postfix}" if postfix else ""
        total_text = str(self.total) if self.total is not None else "?"
        print(
            f"[train-progress] {self.desc} {self.count}/{total_text} {self.unit} "
            f"elapsed={elapsed:.1f}s rate={rate:.2f}/{self.unit}{postfix_text}",
            file=sys.stderr,
            flush=True,
        )
        self.last_print_at = now


def _training_progress_bar(
    *,
    desc: str,
    total: int | None,
    enabled: bool,
    backend: str,
    mininterval: float,
    unit: str = "batch",
) -> Any | None:
    if not enabled:
        return None
    if total is not None and total <= 0:
        return None
    backend = str(backend).strip().lower()
    if backend in {"none", "off", "false", "0"}:
        return None
    if backend not in {"auto", "tqdm", "line"}:
        backend = "auto"
    if backend == "line" or (backend == "auto" and not sys.stderr.isatty()):
        return _LineProgress(desc=desc, total=total, unit=unit, mininterval=mininterval)
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(
        total=total,
        unit=unit,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        mininterval=max(0.0, float(mininterval)),
        file=sys.stderr,
    )


def _optional_progress_path(
    *,
    config_path: Path,
    output_dir: Path,
    train_cfg: dict[str, Any],
    key: str,
    default_name: str,
) -> Path | None:
    value = train_cfg.get(key)
    if value is None or str(value).strip() == "":
        return output_dir / default_name
    lowered = str(value).strip().lower()
    if lowered in {"none", "off", "false", "0"}:
        return None
    return _resolve_path(config_path.parent, str(value))


def _checkpoint_epoch_from_metadata(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("epoch")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _checkpoint_epoch_from_path(path: Path) -> int | None:
    stem = path.stem
    prefix = "checkpoint_epoch_"
    if not stem.startswith(prefix):
        return None
    try:
        return max(0, int(stem[len(prefix) :]))
    except ValueError:
        return None


def _load_checkpoint_payload(*, torch: Any, checkpoint_path: Path, map_location: str | None = None) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Auto prompt checkpoint has no state_dict: {checkpoint_path}")
    return checkpoint


def _validate_checkpoint_architecture(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    model_cfg: AutoPromptModelConfig,
    label: str,
) -> None:
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_architecture = str(checkpoint_config.get("architecture", "")).strip().lower()
    requested_architecture = str(model_cfg.architecture).strip().lower()
    if checkpoint_architecture and checkpoint_architecture != requested_architecture:
        raise ValueError(
            f"Auto prompt {label} architecture mismatch: "
            f"checkpoint={checkpoint_architecture!r} requested={requested_architecture!r} path={checkpoint_path}"
        )


def _load_auto_prompt_checkpoint_if_requested(
    *,
    torch: Any,
    model: Any,
    config_path: Path,
    train_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    key: str,
    label: str,
) -> tuple[str | None, dict[str, Any]]:
    value = train_cfg.get(key)
    if value is None or str(value).strip() == "":
        return None, {}
    if str(value).strip().lower() in {"none", "off", "false", "0"}:
        return None, {}
    checkpoint_path = _resolve_path(config_path.parent, str(value))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Auto prompt {label} does not exist: {checkpoint_path}")
    checkpoint = _load_checkpoint_payload(torch=torch, checkpoint_path=checkpoint_path, map_location=device)
    _validate_checkpoint_architecture(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        label=label,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metadata = checkpoint.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    print(
        f"[train-init] loaded {label}={checkpoint_path} architecture={str(model_cfg.architecture).strip().lower()}",
        file=sys.stderr,
        flush=True,
    )
    return str(checkpoint_path), metadata


def _load_init_checkpoint_if_requested(
    *,
    torch: Any,
    model: Any,
    config_path: Path,
    train_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
) -> str | None:
    checkpoint_path, _ = _load_auto_prompt_checkpoint_if_requested(
        torch=torch,
        model=model,
        config_path=config_path,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        device=device,
        key="init_checkpoint",
        label="init_checkpoint",
    )
    return checkpoint_path


def _load_resume_checkpoint_if_requested(
    *,
    torch: Any,
    model: Any,
    config_path: Path,
    train_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
) -> tuple[str | None, int, dict[str, Any]]:
    checkpoint_path, metadata = _load_auto_prompt_checkpoint_if_requested(
        torch=torch,
        model=model,
        config_path=config_path,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        device=device,
        key="resume_checkpoint",
        label="resume_checkpoint",
    )
    if checkpoint_path is None:
        return None, 0, {}
    resume_epoch = _checkpoint_epoch_from_metadata(metadata)
    if resume_epoch is None:
        resume_epoch = _checkpoint_epoch_from_path(Path(checkpoint_path)) or 0
    return checkpoint_path, resume_epoch, metadata


def _build_teacher_model_if_requested(
    *,
    torch: Any,
    config_path: Path,
    train_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
) -> tuple[Any | None, str | None]:
    teacher_weight = float(train_cfg.get("teacher_loss_weight", 0.0))
    value = train_cfg.get("teacher_checkpoint")
    if teacher_weight <= 0.0 or value is None or str(value).strip() == "":
        return None, None
    teacher_path = _resolve_path(config_path.parent, str(value))
    required = _bool_setting(train_cfg.get("teacher_checkpoint_required"), default=True)
    if not teacher_path.exists():
        if required:
            raise FileNotFoundError(f"Auto prompt teacher_checkpoint does not exist: {teacher_path}")
        print(f"[train-teacher] skipped missing teacher_checkpoint={teacher_path}", file=sys.stderr, flush=True)
        return None, None
    checkpoint = _load_checkpoint_payload(torch=torch, checkpoint_path=teacher_path, map_location=device)
    teacher_raw_config = checkpoint.get("config", {})
    teacher_cfg = AutoPromptModelConfig(**teacher_raw_config) if isinstance(teacher_raw_config, dict) else model_cfg
    teacher_model = build_infrared_prompt_estimator(teacher_cfg).to(device)
    teacher_model.load_state_dict(checkpoint["state_dict"], strict=True)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    print(
        "[train-teacher] loaded "
        f"teacher_checkpoint={teacher_path} "
        f"teacher_architecture={str(teacher_cfg.architecture).strip().lower()} "
        f"student_architecture={str(model_cfg.architecture).strip().lower()}",
        file=sys.stderr,
        flush=True,
    )
    return teacher_model, str(teacher_path)


def _checkpoint_metadata_from_path(*, torch: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = _load_checkpoint_payload(torch=torch, checkpoint_path=checkpoint_path, map_location="cpu")
    except Exception:
        return {}
    metadata = checkpoint.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _restore_checkpoint_history_from_disk(*, torch: Any, output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint_path in sorted(output_dir.glob("checkpoint_epoch_*.pt")):
        metadata = _checkpoint_metadata_from_path(torch=torch, checkpoint_path=checkpoint_path)
        epoch = _checkpoint_epoch_from_metadata(metadata) or _checkpoint_epoch_from_path(checkpoint_path)
        if epoch is None:
            continue
        metric_name = str(metadata.get("metric_name", "loss"))
        metric_value = _float_or_none(metadata.get("metric_value"))
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            "metric_name": metric_name,
            "metric_value": float(metric_value) if metric_value is not None else 0.0,
        }
        records.append(record)
    records.sort(key=lambda item: int(item.get("epoch", 0) or 0))
    return records


def _restore_best_checkpoint_state(
    *,
    torch: Any,
    best_checkpoint_path: Path,
    checkpoint_history: list[dict[str, Any]],
) -> tuple[Path | None, int | None, str | None, float | None]:
    if best_checkpoint_path.exists():
        metadata = _checkpoint_metadata_from_path(torch=torch, checkpoint_path=best_checkpoint_path)
        epoch = _checkpoint_epoch_from_metadata(metadata)
        metric_name = str(metadata.get("metric_name", "")).strip() or None
        metric_value = _float_or_none(metadata.get("metric_value"))
        return best_checkpoint_path, epoch, metric_name, metric_value
    if not checkpoint_history:
        return None, None, None, None
    candidates = [record for record in checkpoint_history if record.get("checkpoint_path") and Path(str(record["checkpoint_path"])).exists()]
    if not candidates:
        return None, None, None, None
    best_record = min(candidates, key=lambda item: float(item.get("metric_value", 0.0)))
    return (
        Path(str(best_record["checkpoint_path"])),
        int(best_record.get("epoch", 0) or 0),
        str(best_record.get("metric_name", "loss")),
        float(best_record.get("metric_value", 0.0)),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _progress_history_values(history_entry: dict[str, Any] | None) -> dict[str, float]:
    if not history_entry:
        return {}
    output: dict[str, float] = {}
    for key, value in history_entry.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            output[str(key)] = float(value)
    return output


def _write_training_progress_state(
    *,
    state_path: Path | None,
    events_path: Path | None,
    experiment_id: str,
    train_seed: int | None,
    epoch: int,
    epochs: int,
    status: str,
    phase: str,
    started_at: float,
    history_entry: dict[str, Any] | None = None,
    best_checkpoint_epoch: int | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
) -> None:
    if state_path is None:
        return
    metrics = _progress_history_values(history_entry)
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "train_seed": train_seed,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "status": status,
        "phase": phase,
        "elapsed_s": round(max(0.0, time.perf_counter() - started_at), 3),
        "updated_at_unix": round(time.time(), 3),
        "metrics": metrics,
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value) if best_metric_value is not None else None,
    }
    _atomic_write_json(state_path, payload)
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _cuda_grad_scaler(torch: Any) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler()


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


@dataclass
class DenseGpuCache:
    image: Any
    objectness: Any
    objectness_weight: Any
    box_size: Any
    box_weight: Any
    sample_weight: Any
    sample_ids: list[str]
    dataset_ids: list[str]
    area_buckets: list[str]
    supervision_types: list[str]
    samples: list[Sample]
    cached_gb: float

    def __len__(self) -> int:
        return len(self.sample_ids)

    def batch(self, indices: Any) -> dict[str, Any]:
        index_list = [int(value) for value in indices.detach().cpu().tolist()]
        return {
            "image": self.image[indices],
            "objectness": self.objectness[indices],
            "objectness_weight": self.objectness_weight[indices],
            "box_size": self.box_size[indices],
            "box_weight": self.box_weight[indices],
            "sample_weight": self.sample_weight[indices],
            "sample_ids": [self.sample_ids[index] for index in index_list],
            "dataset_ids": [self.dataset_ids[index] for index in index_list],
            "area_buckets": [self.area_buckets[index] for index in index_list],
            "supervision_types": [self.supervision_types[index] for index in index_list],
            "samples": [self.samples[index] for index in index_list],
            "source": "gpu_cache",
        }


@dataclass
class LightGrayCache:
    grays: list[np.ndarray]
    boxes: list[list[float]]
    sample_ids: list[str]
    dataset_ids: list[str]
    area_buckets: list[str]
    sample_weights: list[float]
    supervision_types: list[str]
    samples: list[Sample]
    cached_gb: float

    def __len__(self) -> int:
        return len(self.sample_ids)

    def grays_for_indices(self, indices: list[int]) -> list[np.ndarray]:
        return [self.grays[index] for index in indices]


@dataclass
class DiskLightGrayShard:
    images_path: Path
    boxes_path: Path
    shapes_path: Path
    meta_path: Path
    count: int
    image_array: Any | None = None
    boxes_array: Any | None = None
    shapes_array: Any | None = None

    def open(self) -> None:
        if self.image_array is None:
            self.image_array = np.load(self.images_path, mmap_mode="r")
        if self.shapes_array is None:
            self.shapes_array = np.load(self.shapes_path, mmap_mode="r")

    def close(self) -> None:
        for attr in ("image_array", "boxes_array", "shapes_array"):
            array = getattr(self, attr)
            setattr(self, attr, None)
            mmap_obj = getattr(array, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()


@dataclass
class DiskLightGrayCache:
    cache_dir: Path
    shards: list[DiskLightGrayShard]
    index_map: list[tuple[int, int]]
    boxes: list[list[float]]
    sample_ids: list[str]
    dataset_ids: list[str]
    area_buckets: list[str]
    sample_weights: list[float]
    supervision_types: list[str]
    samples: list[Sample]
    cached_gb: float
    max_open_shards: int = 64
    _open_shards: OrderedDict[int, None] = field(default_factory=OrderedDict, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _ensure_open(self, shard_index: int) -> DiskLightGrayShard:
        shard = self.shards[shard_index]
        shard.open()
        if shard_index in self._open_shards:
            self._open_shards.move_to_end(shard_index)
        else:
            self._open_shards[shard_index] = None
        self._trim_open_shards()
        return shard

    def _trim_open_shards(self) -> None:
        limit = max(1, int(self.max_open_shards))
        while len(self._open_shards) > limit:
            shard_index, _ = self._open_shards.popitem(last=False)
            self.shards[shard_index].close()

    def grays_for_indices(self, indices: list[int]) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for index in indices:
            shard_index, local_index = self.index_map[index]
            shard = self._ensure_open(shard_index)
            h, w = [int(value) for value in shard.shapes_array[local_index][:2]]
            output.append(np.array(shard.image_array[local_index, :h, :w], dtype=np.uint8, copy=True))
        return output

    def close(self) -> None:
        for shard_index in list(self._open_shards.keys()):
            self.shards[shard_index].close()
        self._open_shards.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


LightCache = LightGrayCache | DiskLightGrayCache


def _close_light_cache(cache: LightCache | None) -> None:
    close = getattr(cache, "close", None)
    if callable(close):
        close()


@lru_cache(maxsize=8)
def _load_dinov3_targetness_index(index_path: str) -> dict[str, str]:
    path = Path(index_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DINOv3 targetness index must be a JSON object: {path}")
    base = path.parent
    mapping: dict[str, str] = {}

    def add(key: object, value: object) -> None:
        text = str(key or "").strip()
        if not text:
            return
        map_path = Path(str(value))
        if not map_path.is_absolute():
            map_path = base / map_path
        mapping[text] = str(map_path)

    sample_map = payload.get("sample_id_to_map", {})
    if isinstance(sample_map, dict):
        for key, value in sample_map.items():
            add(key, value)
    image_map = payload.get("image_path_to_map", {})
    if isinstance(image_map, dict):
        for key, value in image_map.items():
            add(key, value)
    records = payload.get("records", [])
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            map_value = record.get("map_path")
            if not map_value:
                continue
            add(record.get("sample_id"), map_value)
            add(record.get("image_path"), map_value)
    return mapping


def _dinov3_targetness_batch(
    *,
    samples: list[Sample],
    height: int,
    width: int,
    train_cfg: dict[str, Any] | None,
    torch: Any,
    F: Any,
    device: str,
) -> Any | None:
    if not train_cfg:
        return None
    index_value = train_cfg.get("dinov3_targetness_index")
    if not index_value:
        return None
    weight = float(train_cfg.get("dinov3_distill_weight", 0.0))
    if weight <= 0.0:
        return None
    index_path = _resolve_path(Path.cwd(), str(index_value))
    mapping = _load_dinov3_targetness_index(str(index_path))
    missing_policy = str(train_cfg.get("dinov3_missing_policy", "zeros")).strip().lower()
    maps: list[np.ndarray] = []
    missing: list[str] = []
    for sample in samples:
        map_path_text = (
            mapping.get(sample.sample_id)
            or mapping.get(str(sample.image_path))
            or mapping.get(str(sample.image_path.resolve()))
        )
        if not map_path_text:
            missing.append(sample.sample_id)
            maps.append(np.zeros((1, 1), dtype=np.float32))
            continue
        map_path = Path(map_path_text)
        if not map_path.exists():
            missing.append(sample.sample_id)
            maps.append(np.zeros((1, 1), dtype=np.float32))
            continue
        arr = np.asarray(np.load(map_path), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"DINOv3 targetness map must be 2D: {map_path} shape={arr.shape}")
        maps.append(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0))
    if missing and missing_policy in {"error", "raise"}:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing DINOv3 targetness maps for {len(missing)} samples; examples: {preview}")
    max_h = max(int(arr.shape[0]) for arr in maps)
    max_w = max(int(arr.shape[1]) for arr in maps)
    stacked = np.zeros((len(maps), 1, max_h, max_w), dtype=np.float32)
    for index, arr in enumerate(maps):
        h, w = int(arr.shape[0]), int(arr.shape[1])
        stacked[index, 0, :h, :w] = arr
    tensor = torch.from_numpy(stacked).to(device=device, dtype=torch.float32, non_blocking=True)
    if (max_h, max_w) != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.clamp(0.0, 1.0)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sample_to_light_cache_record(sample: Sample) -> dict[str, Any]:
    return {
        "image_path": str(sample.image_path),
        "sample_id": sample.sample_id,
        "frame_id": sample.frame_id,
        "sequence_id": sample.sequence_id,
        "frame_index": int(sample.frame_index),
        "temporal_key": sample.temporal_key,
        "track_id": sample.track_id,
        "width": int(sample.width),
        "height": int(sample.height),
        "category": sample.category,
        "target_scale": sample.target_scale,
        "device_source": sample.device_source,
        "annotation_protocol_flag": sample.annotation_protocol_flag,
        "supervision_type": sample.supervision_type,
        "bbox_tight": sample.bbox_tight,
        "bbox_loose": sample.bbox_loose,
        "point_prompt": sample.point_prompt,
        "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
        "metadata": _json_safe(sample.metadata),
    }


def _sample_from_light_cache_record(record: dict[str, Any]) -> Sample:
    return Sample(
        image_path=Path(str(record["image_path"])),
        sample_id=str(record["sample_id"]),
        frame_id=str(record.get("frame_id", record["sample_id"])),
        sequence_id=str(record.get("sequence_id", "")),
        frame_index=int(record.get("frame_index", 0)),
        temporal_key=str(record.get("temporal_key", record.get("frame_id", record["sample_id"]))),
        track_id=record.get("track_id"),
        width=int(record.get("width", 0)),
        height=int(record.get("height", 0)),
        category=str(record.get("category", "unknown")),
        target_scale=str(record.get("target_scale", "unknown")),
        device_source=str(record.get("device_source", "unknown")),
        annotation_protocol_flag=str(record.get("annotation_protocol_flag", "cached_light_gray")),
        supervision_type=str(record.get("supervision_type", "bbox")),
        bbox_tight=record.get("bbox_tight"),
        bbox_loose=record.get("bbox_loose"),
        point_prompt=record.get("point_prompt"),
        mask_path=Path(str(record["mask_path"])) if record.get("mask_path") else None,
        metadata=dict(record.get("metadata", {})),
    )


def _torch_cache_dtype(torch: Any, name: str) -> Any:
    text = str(name).strip().lower()
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _tensor_gb(*tensors: Any) -> float:
    total = 0
    for tensor in tensors:
        total += int(tensor.nelement()) * int(tensor.element_size())
    return total / float(1024**3)


def _ensure_cuda_cache_device(torch: Any, device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("auto-prompt gpu/light cache requires a CUDA training device.")


def _build_dense_gpu_cache(
    *,
    torch: Any,
    config_path: Path,
    dataset_configs: list[str],
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    cache_dtype: Any,
) -> DenseGpuCache | None:
    if not dataset_configs:
        return None
    _ensure_cuda_cache_device(torch, device)
    max_samples = max(0, int(train_cfg.get("max_samples", 0)))
    max_long_side = int(train_cfg.get("max_long_side", 512))
    started_at = time.perf_counter()
    items: list[dict[str, Any]] = []
    for sample in _limit_samples(_iter_training_samples(config_path, dataset_configs=dataset_configs), limit=max_samples):
        items.append(
            _sample_to_training_item(
                sample,
                max_long_side=max_long_side,
                target_config=target_cfg,
                model_config=model_cfg,
                train_config=train_cfg,
            )
        )
    if not items:
        return None
    max_h = max(int(item["image"].shape[1]) for item in items)
    max_w = max(int(item["image"].shape[2]) for item in items)
    image = torch.zeros((len(items), 3, max_h, max_w), device=device, dtype=cache_dtype)
    objectness = torch.zeros((len(items), 1, max_h, max_w), device=device, dtype=cache_dtype)
    objectness_weight = torch.zeros((len(items), 1, max_h, max_w), device=device, dtype=cache_dtype)
    box_size = torch.zeros((len(items), 2, max_h, max_w), device=device, dtype=cache_dtype)
    box_weight = torch.zeros((len(items), 1, max_h, max_w), device=device, dtype=cache_dtype)
    sample_weight = torch.ones((len(items),), device=device, dtype=torch.float32)
    for index, item in enumerate(items):
        h, w = int(item["image"].shape[1]), int(item["image"].shape[2])
        image[index, :, :h, :w] = torch.from_numpy(item["image"]).to(device=device, dtype=cache_dtype)
        objectness[index, :, :h, :w] = torch.from_numpy(item["objectness"]).to(device=device, dtype=cache_dtype)
        objectness_weight[index, :, :h, :w] = torch.from_numpy(item["objectness_weight"]).to(device=device, dtype=cache_dtype)
        box_size[index, :, :h, :w] = torch.from_numpy(item["box_size"]).to(device=device, dtype=cache_dtype)
        box_weight[index, :, :h, :w] = torch.from_numpy(item["box_weight"]).to(device=device, dtype=cache_dtype)
        sample_weight[index] = float(item.get("sample_weight", 1.0))
    cached_gb = _tensor_gb(image, objectness, objectness_weight, box_size, box_weight, sample_weight)
    elapsed = time.perf_counter() - started_at
    dataset_counts: dict[str, int] = {}
    for item in items:
        key = str(item["dataset_id"])
        dataset_counts[key] = dataset_counts.get(key, 0) + 1
    print(
        f"[cache-build] mode=dense_gpu samples={len(items)} shape={max_h}x{max_w} "
        f"cached_gb={cached_gb:.2f} elapsed={elapsed:.1f}s datasets={dataset_counts}",
        file=sys.stderr,
        flush=True,
    )
    return DenseGpuCache(
        image=image,
        objectness=objectness,
        objectness_weight=objectness_weight,
        box_size=box_size,
        box_weight=box_weight,
        sample_weight=sample_weight,
        sample_ids=[str(item["sample_id"]) for item in items],
        dataset_ids=[str(item["dataset_id"]) for item in items],
        area_buckets=[str(item["area_bucket"]) for item in items],
        supervision_types=[str(item["supervision_type"]) for item in items],
        samples=[item["sample"] for item in items],
        cached_gb=cached_gb,
    )


def _light_cache_progress(
    *,
    train_cfg: dict[str, Any],
    desc: str,
    total: int | None,
    unit: str = "sample",
) -> Any | None:
    return _training_progress_bar(
        desc=desc,
        total=total,
        enabled=bool(train_cfg.get("show_progress", True)),
        backend=str(train_cfg.get("progress_backend", "auto")),
        mininterval=float(train_cfg.get("progress_update_interval_s", 1.0)),
        unit=unit,
    )


def _light_cache_source_fingerprint(config_path: Path, dataset_configs: list[str], train_cfg: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"sparksam-light-gray-cache-v2\n")
    digest.update(str(int(train_cfg.get("max_long_side", 512))).encode("utf-8"))
    digest.update(b"\n")
    digest.update(str(max(0, int(train_cfg.get("light_cache_max_samples", 0)))).encode("utf-8"))
    digest.update(b"\n")
    for key in ("density_sampling", "domain_loss_weights", "area_loss_weights"):
        digest.update(json.dumps(train_cfg.get(key, {}), sort_keys=True, ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    for item in dataset_configs:
        dataset_config = _resolve_path(config_path.parent, str(item))
        digest.update(str(dataset_config).encode("utf-8"))
        digest.update(b"\n")
        if dataset_config.exists():
            digest.update(dataset_config.read_bytes())
        try:
            app_config = load_app_config(dataset_config)
            root = Path(str(app_config.dataset.root))
            ann_dir = root / (app_config.dataset.annotations_dir or "annotations_coco")
            if app_config.dataset.annotations_file:
                ann_files = [ann_dir / app_config.dataset.annotations_file]
            else:
                ann_files = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []
            for ann_path in ann_files:
                if ann_path.exists():
                    stat = ann_path.stat()
                    digest.update(str(ann_path).encode("utf-8"))
                    digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        except Exception:
            continue
    return digest.hexdigest()[:24]


def _light_cache_disk_root(config_path: Path, train_cfg: dict[str, Any]) -> Path:
    configured = train_cfg.get("light_cache_disk_root")
    if configured:
        return _resolve_path(config_path.parent, str(configured))
    raw = _read_yaml(config_path)
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    return output_root.parent / "cache" / "light_gray"


def _disk_cache_manifest_ready(cache_dir: Path) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not bool(manifest.get("complete")):
        return False
    for shard in manifest.get("shards", []):
        for key in ("images", "boxes", "shapes", "meta"):
            if not (cache_dir / str(shard.get(key, ""))).exists():
                return False
    return True


def _load_disk_light_gray_cache(
    cache_dir: Path,
    *,
    announce: bool = True,
    max_open_shards: int = 64,
) -> DiskLightGrayCache | None:
    if not _disk_cache_manifest_ready(cache_dir):
        return None
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    shards: list[DiskLightGrayShard] = []
    index_map: list[tuple[int, int]] = []
    boxes: list[list[float]] = []
    sample_ids: list[str] = []
    dataset_ids: list[str] = []
    area_buckets: list[str] = []
    sample_weights: list[float] = []
    supervision_types: list[str] = []
    samples: list[Sample] = []
    for shard_index, shard_payload in enumerate(manifest.get("shards", [])):
        shard = DiskLightGrayShard(
            images_path=cache_dir / str(shard_payload["images"]),
            boxes_path=cache_dir / str(shard_payload["boxes"]),
            shapes_path=cache_dir / str(shard_payload["shapes"]),
            meta_path=cache_dir / str(shard_payload["meta"]),
            count=int(shard_payload["count"]),
        )
        meta = json.loads(shard.meta_path.read_text(encoding="utf-8"))
        records = list(meta.get("records", []))
        shards.append(shard)
        for local_index, record in enumerate(records):
            index_map.append((shard_index, local_index))
            boxes.append([float(value) for value in record["box"][:4]])
            sample_ids.append(str(record["sample_id"]))
            dataset_ids.append(str(record["dataset_id"]))
            area_buckets.append(str(record["area_bucket"]))
            sample_weights.append(float(record.get("sample_weight", 1.0)))
            supervision_types.append(str(record.get("supervision_type", "bbox")))
            samples.append(_sample_from_light_cache_record(record["sample"]))
    cache = DiskLightGrayCache(
        cache_dir=cache_dir,
        shards=shards,
        index_map=index_map,
        boxes=boxes,
        sample_ids=sample_ids,
        dataset_ids=dataset_ids,
        area_buckets=area_buckets,
        sample_weights=sample_weights,
        supervision_types=supervision_types,
        samples=samples,
        cached_gb=float(manifest.get("cached_gb", 0.0)),
        max_open_shards=max(1, int(max_open_shards)),
    )
    if announce:
        print(
            f"[cache-hit] mode=light_gray_disk samples={len(cache)} cached_gb={cache.cached_gb:.2f} "
            f"max_open_shards={cache.max_open_shards} dir={cache_dir}",
            file=sys.stderr,
            flush=True,
        )
    return cache


def _disk_cache_lock_pid(lock_path: Path) -> int | None:
    try:
        text = lock_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for token in text.replace("\n", " ").split():
        if not token.startswith("pid="):
            continue
        try:
            return int(token.split("=", 1)[1])
        except ValueError:
            return None
    return None


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _disk_cache_lock_age_s(lock_path: Path) -> float:
    try:
        return max(0.0, time.time() - lock_path.stat().st_mtime)
    except OSError:
        return 0.0


def _acquire_disk_cache_lock(cache_dir: Path, train_cfg: dict[str, Any], *, accept_ready: bool) -> int | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "build.lock"
    wait_timeout_s = max(1.0, float(train_cfg.get("light_cache_lock_timeout_s", 86400.0)))
    stale_timeout_s = max(0.0, float(train_cfg.get("light_cache_lock_stale_timeout_s", 21600.0)))
    started_at = time.perf_counter()
    progress = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} started_at={time.time()}\n".encode("utf-8"))
            if progress is not None:
                progress.close()
            return fd
        except FileExistsError:
            if progress is None:
                print(f"[cache-wait] mode=light_gray_disk lock={lock_path}", file=sys.stderr, flush=True)
                progress = _light_cache_progress(train_cfg=train_cfg, desc="light-cache wait", total=None, unit="s")
            lock_pid = _disk_cache_lock_pid(lock_path)
            lock_age_s = _disk_cache_lock_age_s(lock_path)
            if lock_pid is not None and not _process_exists(lock_pid):
                if progress is not None:
                    progress.close()
                    progress = None
                print(
                    f"[cache-lock] removing stale lock with dead pid={lock_pid} age={lock_age_s:.0f}s path={lock_path}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if stale_timeout_s > 0.0 and lock_age_s > stale_timeout_s:
                if progress is not None:
                    progress.close()
                owner = f"pid={lock_pid}" if lock_pid is not None else "pid=unknown"
                raise TimeoutError(
                    f"Light cache lock is older than stale timeout but may still belong to a live process "
                    f"({owner}, age={lock_age_s:.0f}s, timeout={stale_timeout_s:.0f}s): {lock_path}. "
                    "Inspect the process and remove the lock manually if it is safe."
                )
            elapsed = time.perf_counter() - started_at
            if elapsed > wait_timeout_s:
                if progress is not None:
                    progress.close()
                raise TimeoutError(f"Timed out waiting for light cache lock: {lock_path}")
            time.sleep(5.0)
            if progress is not None:
                progress.set_postfix(waited=f"{elapsed:.0f}s")
                progress.update(5)
            if accept_ready and _disk_cache_manifest_ready(cache_dir):
                if progress is not None:
                    progress.close()
                return None


def _release_disk_cache_lock(cache_dir: Path, fd: int | None) -> None:
    if fd is None:
        return
    os.close(fd)
    try:
        (cache_dir / "build.lock").unlink()
    except FileNotFoundError:
        pass


def _light_cache_sample_payload(sample: Sample, *, max_long_side: int, train_cfg: dict[str, Any]) -> dict[str, Any]:
    gray, box = _load_resized_gray_uint8_and_box(sample, max_long_side)
    return {
        "gray": gray,
        "box": box,
        "sample_id": sample.sample_id,
        "dataset_id": str(sample.metadata.get("dataset_id", "unknown")),
        "area_bucket": _sample_area_bucket(sample),
        "sample_weight": _configured_sample_weight(sample, train_cfg=train_cfg),
        "supervision_type": sample.supervision_type,
        "sample": _sample_to_light_cache_record(sample),
    }


def _write_light_cache_shard(cache_dir: Path, shard_index: int, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    max_h = max(int(item["gray"].shape[0]) for item in payloads)
    max_w = max(int(item["gray"].shape[1]) for item in payloads)
    images = np.zeros((len(payloads), max_h, max_w), dtype=np.uint8)
    boxes = np.zeros((len(payloads), 4), dtype=np.float32)
    shapes = np.zeros((len(payloads), 2), dtype=np.int32)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payloads):
        gray = item["gray"]
        h, w = int(gray.shape[0]), int(gray.shape[1])
        images[index, :h, :w] = gray
        boxes[index] = np.asarray(item["box"], dtype=np.float32)
        shapes[index] = [h, w]
        records.append(
            {
                "sample_id": item["sample_id"],
                "dataset_id": item["dataset_id"],
                "area_bucket": item["area_bucket"],
                "sample_weight": float(item["sample_weight"]),
                "supervision_type": item["supervision_type"],
                "box": [float(value) for value in item["box"][:4]],
                "shape": [h, w],
                "sample": item["sample"],
            }
        )
    prefix = f"shard_{shard_index:06d}"
    images_name = f"{prefix}_images.npy"
    boxes_name = f"{prefix}_boxes.npy"
    shapes_name = f"{prefix}_shapes.npy"
    meta_name = f"{prefix}_meta.json"
    np.save(cache_dir / images_name, images)
    np.save(cache_dir / boxes_name, boxes)
    np.save(cache_dir / shapes_name, shapes)
    (cache_dir / meta_name).write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")
    return {
        "images": images_name,
        "boxes": boxes_name,
        "shapes": shapes_name,
        "meta": meta_name,
        "count": len(payloads),
        "bytes": int(images.nbytes + boxes.nbytes + shapes.nbytes),
    }


def _collect_light_cache_samples(config_path: Path, dataset_configs: list[str], *, max_samples: int) -> list[Sample]:
    print(
        f"[cache-scan] mode=light_gray dataset_configs={len(dataset_configs)} max_samples={max_samples}",
        file=sys.stderr,
        flush=True,
    )
    return list(_limit_samples(_iter_training_samples(config_path, dataset_configs=dataset_configs), limit=max_samples))


def _build_disk_light_gray_cache(
    *,
    config_path: Path,
    dataset_configs: list[str],
    train_cfg: dict[str, Any],
) -> DiskLightGrayCache | None:
    max_long_side = int(train_cfg.get("max_long_side", 512))
    max_samples = max(0, int(train_cfg.get("light_cache_max_samples", 0)))
    max_open_shards = max(1, int(train_cfg.get("light_cache_max_open_shards", 64)))
    cache_root = _light_cache_disk_root(config_path, train_cfg)
    cache_key = _light_cache_source_fingerprint(config_path, dataset_configs, train_cfg)
    cache_dir = cache_root / cache_key
    rebuild = _bool_setting(train_cfg.get("light_cache_rebuild"), default=False)
    if not rebuild:
        hit = _load_disk_light_gray_cache(cache_dir, max_open_shards=max_open_shards)
        if hit is not None:
            return hit
    lock_fd = _acquire_disk_cache_lock(cache_dir, train_cfg, accept_ready=not rebuild)
    if lock_fd is None and not rebuild:
        return _load_disk_light_gray_cache(cache_dir, max_open_shards=max_open_shards)
    try:
        if not rebuild:
            hit = _load_disk_light_gray_cache(cache_dir, max_open_shards=max_open_shards)
            if hit is not None:
                return hit
        started_at = time.perf_counter()
        samples = _collect_light_cache_samples(config_path, dataset_configs, max_samples=max_samples)
        if not samples:
            return None
        shard_size = max(1, int(train_cfg.get("light_cache_shard_size", 512)))
        worker_count = max(1, int(train_cfg.get("light_cache_num_workers", 8)))
        progress = _light_cache_progress(train_cfg=train_cfg, desc="light-cache build disk", total=len(samples), unit="sample")
        shards: list[dict[str, Any]] = []
        cached_bytes = 0
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for shard_start in range(0, len(samples), shard_size):
                    shard_samples = samples[shard_start : shard_start + shard_size]
                    payloads: list[dict[str, Any] | None] = [None] * len(shard_samples)
                    futures = {
                        executor.submit(_light_cache_sample_payload, sample, max_long_side=max_long_side, train_cfg=train_cfg): index
                        for index, sample in enumerate(shard_samples)
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        payload = future.result()
                        payloads[index] = payload
                        cached_bytes += int(payload["gray"].nbytes)
                        if progress is not None:
                            progress.set_postfix(backend="disk", cached_gb=f"{cached_bytes / float(1024**3):.2f}")
                            progress.update(1)
                    shard_payloads = [item for item in payloads if item is not None]
                    shards.append(_write_light_cache_shard(cache_dir, len(shards), shard_payloads))
        finally:
            if progress is not None:
                progress.close()
        manifest = {
            "version": 1,
            "complete": True,
            "created_at": time.time(),
            "cache_key": cache_key,
            "max_long_side": max_long_side,
            "max_samples": max_samples,
            "max_open_shards": max_open_shards,
            "sample_count": len(samples),
            "cached_gb": cached_bytes / float(1024**3),
            "shards": [{key: value for key, value in shard.items() if key != "bytes"} for shard in shards],
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - started_at
        print(
            f"[cache-build] mode=light_gray_disk samples={len(samples)} shards={len(shards)} "
            f"cached_gb={manifest['cached_gb']:.2f} max_open_shards={max_open_shards} "
            f"elapsed={elapsed:.1f}s dir={cache_dir}",
            file=sys.stderr,
            flush=True,
        )
        return _load_disk_light_gray_cache(cache_dir, announce=False, max_open_shards=max_open_shards)
    finally:
        _release_disk_cache_lock(cache_dir, lock_fd)


def _build_memory_light_gray_cache(
    *,
    config_path: Path,
    dataset_configs: list[str],
    train_cfg: dict[str, Any],
) -> LightGrayCache | None:
    max_long_side = int(train_cfg.get("max_long_side", 512))
    max_samples = max(0, int(train_cfg.get("light_cache_max_samples", 0)))
    started_at = time.perf_counter()
    grays: list[np.ndarray] = []
    boxes: list[list[float]] = []
    sample_ids: list[str] = []
    dataset_ids: list[str] = []
    area_buckets: list[str] = []
    sample_weights: list[float] = []
    supervision_types: list[str] = []
    samples: list[Sample] = []
    progress = _light_cache_progress(
        train_cfg=train_cfg,
        desc="light-cache build memory",
        total=max_samples if max_samples > 0 else None,
        unit="sample",
    )
    cached_bytes = 0
    try:
        for sample in _limit_samples(_iter_training_samples(config_path, dataset_configs=dataset_configs), limit=max_samples):
            gray, box = _load_resized_gray_uint8_and_box(sample, max_long_side)
            grays.append(gray)
            boxes.append(box)
            sample_ids.append(sample.sample_id)
            dataset_ids.append(str(sample.metadata.get("dataset_id", "unknown")))
            area_buckets.append(_sample_area_bucket(sample))
            sample_weights.append(_configured_sample_weight(sample, train_cfg=train_cfg))
            supervision_types.append(sample.supervision_type)
            samples.append(sample)
            cached_bytes += int(gray.nbytes)
            if progress is not None:
                progress.set_postfix(backend="memory", cached_gb=f"{cached_bytes / float(1024**3):.2f}")
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    if not grays:
        return None
    cached_gb = cached_bytes / float(1024**3)
    elapsed = time.perf_counter() - started_at
    dataset_counts: dict[str, int] = {}
    for dataset_id in dataset_ids:
        dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + 1
    print(
        f"[cache-build] mode=light_gray samples={len(grays)} cached_gb={cached_gb:.2f} "
        f"elapsed={elapsed:.1f}s datasets={dataset_counts}",
        file=sys.stderr,
        flush=True,
    )
    return LightGrayCache(
        grays=grays,
        boxes=boxes,
        sample_ids=sample_ids,
        dataset_ids=dataset_ids,
        area_buckets=area_buckets,
        sample_weights=sample_weights,
        supervision_types=supervision_types,
        samples=samples,
        cached_gb=cached_gb,
    )


def _build_light_gray_cache(
    *,
    config_path: Path,
    dataset_configs: list[str],
    train_cfg: dict[str, Any],
) -> LightCache | None:
    if not dataset_configs:
        return None
    backend = str(train_cfg.get("light_cache_backend", "memory")).strip().lower()
    if backend == "auto":
        backend = "disk" if any("rbgt" in str(item).lower() for item in dataset_configs) else "memory"
    if backend in {"disk", "file", "mmap"}:
        return _build_disk_light_gray_cache(config_path=config_path, dataset_configs=dataset_configs, train_cfg=train_cfg)
    return _build_memory_light_gray_cache(config_path=config_path, dataset_configs=dataset_configs, train_cfg=train_cfg)


def _light_cache_batch(
    cache: LightCache,
    indices: list[int],
    *,
    torch: Any,
    F: Any,
    device: str,
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    train_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = cache.grays_for_indices(indices)
    max_h = max(int(gray.shape[0]) for gray in selected)
    max_w = max(int(gray.shape[1]) for gray in selected)
    if all(int(gray.shape[0]) == max_h and int(gray.shape[1]) == max_w for gray in selected):
        gray_np = np.stack(selected).astype(np.float32, copy=False)[:, None]
    else:
        gray_np = np.zeros((len(indices), 1, max_h, max_w), dtype=np.float32)
        for out_index, gray in enumerate(selected):
            h, w = int(gray.shape[0]), int(gray.shape[1])
            gray_np[out_index, 0, :h, :w] = gray.astype(np.float32)
    boxes_np = np.zeros((len(indices), 4), dtype=np.float32)
    for out_index, cache_index in enumerate(indices):
        gray = selected[out_index]
        h, w = int(gray.shape[0]), int(gray.shape[1])
        boxes_np[out_index] = np.asarray(clamp_box_xyxy(cache.boxes[cache_index], width=w, height=h), dtype=np.float32)
    gray_tensor = torch.from_numpy(gray_np).to(device=device, dtype=torch.float32, non_blocking=True)
    boxes = torch.from_numpy(boxes_np).to(device=device, dtype=torch.float32, non_blocking=True)
    image = _ir_prior_stack_batch(
        torch,
        F,
        gray_tensor,
        use_local_contrast=model_cfg.use_local_contrast,
        use_top_hat=model_cfg.use_top_hat,
        use_dog_filter=model_cfg.use_dog_filter,
        use_fft_highpass=model_cfg.use_fft_highpass,
    )
    prior_score = image.max(dim=1).values
    objectness, box_size, box_weight, objectness_weight = _target_from_box_batch(
        torch,
        boxes=boxes,
        height=max_h,
        width=max_w,
        gaussian_sigma=float(target_cfg.get("gaussian_sigma", 2.0)),
        positive_radius=int(target_cfg.get("positive_radius", 1)),
        min_box_side=float(model_cfg.min_box_side),
        hard_negative_weight=float(target_cfg.get("hard_negative_weight", 1.0)),
        hard_negative_percentile=float(target_cfg.get("hard_negative_percentile", 95.0)),
        prior_score=prior_score,
    )
    samples = [cache.samples[index] for index in indices]
    batch = {
        "image": image,
        "objectness": objectness,
        "objectness_weight": objectness_weight,
        "box_size": box_size,
        "box_weight": box_weight,
        "sample_weight": torch.tensor([float(cache.sample_weights[index]) for index in indices], device=device, dtype=torch.float32),
        "sample_ids": [cache.sample_ids[index] for index in indices],
        "dataset_ids": [cache.dataset_ids[index] for index in indices],
        "area_buckets": [cache.area_buckets[index] for index in indices],
        "supervision_types": [cache.supervision_types[index] for index in indices],
        "samples": samples,
        "source": "light_cache_disk" if isinstance(cache, DiskLightGrayCache) else "light_cache",
    }
    dinov3_targetness = _dinov3_targetness_batch(
        samples=samples,
        height=max_h,
        width=max_w,
        train_cfg=train_cfg,
        torch=torch,
        F=F,
        device=device,
    )
    if dinov3_targetness is not None:
        batch["dinov3_targetness"] = dinov3_targetness
    return batch


def _step_optimizer_after_backward(*, optimizer: Any, scaler: Any | None) -> None:
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _teacher_compatible_image(torch: Any, teacher_model: Any, image: Any) -> Any:
    cfg = getattr(teacher_model, "config", None)
    expected = int(getattr(cfg, "input_channels", int(image.shape[1])) or int(image.shape[1]))
    current = int(image.shape[1])
    if expected == current:
        return image
    if expected < current:
        return image[:, :expected]
    pad_shape = (int(image.shape[0]), expected - current, int(image.shape[2]), int(image.shape[3]))
    pad = torch.zeros(pad_shape, dtype=image.dtype, device=image.device)
    return torch.cat([image, pad], dim=1)


def _train_tensor_batch(
    *,
    torch: Any,
    F: Any,
    model: Any,
    optimizer: Any,
    scaler: Any | None,
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str,
    non_blocking: bool,
    use_amp: bool,
    teacher_model: Any | None = None,
    step_optimizer: bool = True,
    zero_grad: bool = True,
    loss_scale: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    step_started = time.perf_counter()
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness = batch["objectness"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_size = batch["box_size"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_weight = batch["box_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        weight = sample_weight.to(device=device, dtype=torch.float32, non_blocking=non_blocking).view(-1, 1, 1, 1)
        objectness_weight = objectness_weight * weight
        box_weight = box_weight * weight
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    with _autocast_context(torch, enabled=use_amp):
        outputs = model(image)
        teacher_outputs = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(_teacher_compatible_image(torch, teacher_model, image))
        loss, loss_parts = _compute_auto_prompt_losses(
            torch=torch,
            F=F,
            outputs=outputs,
            batch={
                **batch,
                "image": image,
                "objectness": objectness,
                "objectness_weight": objectness_weight,
                "box_size": box_size,
                "box_weight": box_weight,
            },
            train_cfg=train_cfg,
            teacher_outputs=teacher_outputs,
        )
    if scaler is not None:
        scaler.scale(loss * float(loss_scale)).backward()
        if step_optimizer:
            _step_optimizer_after_backward(optimizer=optimizer, scaler=scaler)
    else:
        (loss * float(loss_scale)).backward()
        if step_optimizer:
            _step_optimizer_after_backward(optimizer=optimizer, scaler=scaler)
    return (
        float(loss.detach().cpu()),
        float(loss_parts["objectness_loss"].detach().cpu()),
        float(loss_parts["box_loss"].detach().cpu()),
        float(loss_parts["ranking_loss"].detach().cpu()),
        float(loss_parts["distill_loss"].detach().cpu()),
        (time.perf_counter() - step_started) * 1000.0,
    )


def _evaluate_tensor_batch(
    *,
    torch: Any,
    F: Any,
    model: Any,
    teacher_model: Any | None,
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str,
    non_blocking: bool,
    use_amp: bool,
) -> tuple[float, float, float, float, float]:
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness = batch["objectness"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_size = batch["box_size"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_weight = batch["box_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        weight = sample_weight.to(device=device, dtype=torch.float32, non_blocking=non_blocking).view(-1, 1, 1, 1)
        objectness_weight = objectness_weight * weight
        box_weight = box_weight * weight
    with torch.no_grad():
        with _autocast_context(torch, enabled=use_amp):
            outputs = model(image)
            teacher_outputs = teacher_model(_teacher_compatible_image(torch, teacher_model, image)) if teacher_model is not None else None
            loss, loss_parts = _compute_auto_prompt_losses(
                torch=torch,
                F=F,
                outputs=outputs,
                batch={
                    **batch,
                    "image": image,
                    "objectness": objectness,
                    "objectness_weight": objectness_weight,
                    "box_size": box_size,
                    "box_weight": box_weight,
                },
                train_cfg=train_cfg,
                teacher_outputs=teacher_outputs,
            )
    return (
        float(loss.detach().cpu()),
        float(loss_parts["objectness_loss"].detach().cpu()),
        float(loss_parts["box_loss"].detach().cpu()),
        float(loss_parts["ranking_loss"].detach().cpu()),
        float(loss_parts["distill_loss"].detach().cpu()),
    )


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _autotune_light_batch_size(
    *,
    torch: Any,
    F: Any,
    model: Any,
    optimizer: Any,
    cache: LightCache | None,
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    non_blocking: bool,
    use_amp: bool,
    requested: Any,
    max_batch_size: int,
) -> int:
    if cache is None or len(cache) <= 0:
        return 0
    if str(requested).strip().lower() != "auto":
        return max(1, int(requested))
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return min(max(1, int(max_batch_size)), len(cache))
    low, high, best = 1, min(max(1, int(max_batch_size)), len(cache)), 0
    probe_indices = list(range(high))
    while low <= high:
        candidate = (low + high) // 2
        batch = None
        try:
            batch = _light_cache_batch(
                cache,
                probe_indices[:candidate],
                torch=torch,
                F=F,
                device=device,
                target_cfg=target_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
            )
            _train_tensor_batch(
                torch=torch,
                F=F,
                model=model,
                teacher_model=None,
                optimizer=optimizer,
                scaler=None,
                batch=batch,
                train_cfg=train_cfg,
                device=device,
                non_blocking=non_blocking,
                use_amp=use_amp,
                step_optimizer=False,
            )
            optimizer.zero_grad(set_to_none=True)
            del batch
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            best = candidate
            low = candidate + 1
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            if batch is not None:
                del batch
            if _is_oom_error(exc):
                torch.cuda.empty_cache()
                high = candidate - 1
                continue
            raise
    if best <= 0:
        raise RuntimeError(
            "Unable to fit even light_cache_batch_size=1 on the selected CUDA device. "
            "Use a less busy GPU or reduce max_long_side."
        )
    print(f"[batch-autotune] light_cache_batch_size={best} max_candidate={max_batch_size}", file=sys.stderr, flush=True)
    return best


def _sample_light_indices(cache_size: int, *, limit: int, rng: random.Random) -> list[int]:
    if cache_size <= 0:
        return []
    if limit <= 0 or limit >= cache_size:
        indices = list(range(cache_size))
        rng.shuffle(indices)
        return indices
    return rng.sample(range(cache_size), limit)


def _normalized_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for key, raw_weight in value.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0.0:
            continue
        output[str(key)] = weight
        output[str(key).lower()] = weight
    return output


def _weighted_choice(rng: random.Random, items: list[Any], weights: list[float]) -> Any:
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= 0.0:
        return rng.choice(items)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(0.0, float(weight))
        if cumulative >= threshold:
            return item
    return items[-1]


def _sample_balanced_light_indices(
    cache: LightCache,
    train_indices: list[int],
    *,
    limit: int,
    rng: random.Random,
    train_cfg: dict[str, Any],
) -> list[int]:
    domain_weights = _normalized_mapping(train_cfg.get("domain_sampling_weights", {}))
    area_weights = _normalized_mapping(train_cfg.get("area_sampling_weights", {}))
    density_cfg = _density_config(train_cfg)
    density_enabled = bool(density_cfg) and _bool_setting(density_cfg.get("enabled"), default=False) and _bool_setting(
        density_cfg.get("sampling_weighting"),
        default=True,
    )
    if not domain_weights and not area_weights and not density_enabled:
        return [train_indices[position] for position in _sample_light_indices(len(train_indices), limit=limit, rng=rng)]
    sample_count = len(train_indices) if limit <= 0 else int(limit)
    if sample_count <= 0 or not train_indices:
        return []

    by_group: dict[tuple[str, str, str], list[int]] = {}
    group_weights: dict[tuple[str, str, str], float] = {}
    for position, cache_index in enumerate(train_indices):
        dataset_id = str(cache.dataset_ids[cache_index])
        domain_key = dataset_id if dataset_id in domain_weights else dataset_id.lower()
        if domain_weights and domain_key not in domain_weights:
            continue
        area_bucket = cache.area_buckets[cache_index]
        area_key = area_bucket if area_bucket in area_weights else area_bucket.lower()
        if area_weights and area_key not in area_weights:
            continue
        sample = cache.samples[cache_index]
        density_bucket = _sample_density_bucket(sample) if density_enabled else "all"
        group = (dataset_id, area_bucket if area_weights else "all", density_bucket)
        by_group.setdefault(group, []).append(position)
        if group not in group_weights:
            group_weights[group] = (
                (domain_weights.get(domain_key, 1.0) if domain_weights else 1.0)
                * (area_weights.get(area_key, 1.0) if area_weights else 1.0)
                * (_density_weight_from_sample(sample, train_cfg=train_cfg, purpose="sampling") if density_enabled else 1.0)
            )
    if not by_group:
        return [train_indices[position] for position in _sample_light_indices(len(train_indices), limit=limit, rng=rng)]

    group_names = sorted(by_group)
    choice_weights = [group_weights[group] for group in group_names]
    selected_positions: list[int] = []
    for _ in range(sample_count):
        group = _weighted_choice(rng, group_names, choice_weights)
        selected_positions.append(rng.choice(by_group[group]))
    return [train_indices[position] for position in selected_positions]


def _record_batch_counts(batch: dict[str, Any], counts: dict[str, int]) -> None:
    for dataset_id in batch["dataset_ids"]:
        counts[f"dataset:{dataset_id}"] = counts.get(f"dataset:{dataset_id}", 0) + 1
    for area_bucket in batch.get("area_buckets", []):
        counts[f"area_bucket:{area_bucket}"] = counts.get(f"area_bucket:{area_bucket}", 0) + 1
    for sample in batch.get("samples", []):
        counts[f"density_bucket:{_sample_density_bucket(sample)}"] = counts.get(f"density_bucket:{_sample_density_bucket(sample)}", 0) + 1
    for supervision_type in batch["supervision_types"]:
        key = f"supervision:{supervision_type}"
        counts[key] = counts.get(key, 0) + 1


def _split_indices(count: int, *, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(max(0, int(count))))
    if len(indices) <= 1 or validation_ratio <= 0.0:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    validation_count = int(round(len(indices) * min(max(validation_ratio, 0.0), 0.9)))
    validation_count = min(len(indices) - 1, max(1, validation_count))
    validation_indices = sorted(indices[:validation_count])
    train_indices = sorted(indices[validation_count:])
    return train_indices, validation_indices


def _cache_batches_from_indices(indices: list[int], *, batch_size: int) -> Iterator[list[int]]:
    step = max(1, int(batch_size))
    for start in range(0, len(indices), step):
        yield indices[start : start + step]


def _evaluate_cache_selection_metric(
    *,
    torch: Any,
    F: Any,
    model: Any,
    dense_cache: DenseGpuCache | None,
    dense_indices: list[int],
    light_cache: LightCache | None,
    light_indices: list[int],
    batch_size: int,
    light_batch_size: int,
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    non_blocking: bool,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    objectness_sum = 0.0
    box_sum = 0.0
    ranking_sum = 0.0
    distill_sum = 0.0
    batch_count = 0
    sample_count = 0
    max_batches = max(0, int(train_cfg.get("validation_max_batches", 0)))

    def consume(batch: dict[str, Any]) -> bool:
        nonlocal loss_sum, objectness_sum, box_sum, ranking_sum, distill_sum, batch_count, sample_count
        if max_batches > 0 and batch_count >= max_batches:
            return False
        loss_value, objectness_value, box_value, ranking_value, distill_value = _evaluate_tensor_batch(
            torch=torch,
            F=F,
            model=model,
            teacher_model=None,
            batch=batch,
            train_cfg=train_cfg,
            device=device,
            non_blocking=non_blocking,
            use_amp=use_amp,
        )
        loss_sum += loss_value
        objectness_sum += objectness_value
        box_sum += box_value
        ranking_sum += ranking_value
        distill_sum += distill_value
        batch_count += 1
        sample_count += len(batch["sample_ids"])
        return True

    if dense_cache is not None and dense_indices:
        for batch_indices in _cache_batches_from_indices(dense_indices, batch_size=batch_size):
            tensor_indices = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
            if not consume(dense_cache.batch(tensor_indices)):
                break
    if light_cache is not None and light_indices and (max_batches <= 0 or batch_count < max_batches):
        for batch_indices in _cache_batches_from_indices(light_indices, batch_size=max(1, light_batch_size)):
            batch = _light_cache_batch(
                light_cache,
                batch_indices,
                torch=torch,
                F=F,
                device=device,
                target_cfg=target_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
            )
            if not consume(batch):
                break

    if batch_count <= 0:
        return {}
    denom = max(1, batch_count)
    return {
        "val_loss": loss_sum / denom,
        "val_objectness_loss": objectness_sum / denom,
        "val_box_loss": box_sum / denom,
        "val_ranking_loss": ranking_sum / denom,
        "val_heuristic_distill_loss": distill_sum / denom,
        "val_distill_loss": distill_sum / denom,
        "val_batch_count": float(batch_count),
        "val_sample_count": float(sample_count),
    }


def _checkpoint_epoch_record(
    *,
    output_dir: Path,
    epoch: int,
    model: Any,
    model_cfg: AutoPromptModelConfig,
    metadata: dict[str, Any],
    metric_name: str,
    metric_value: float,
) -> dict[str, Any]:
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    save_auto_prompt_checkpoint(checkpoint_path, model, config=model_cfg, metadata={**metadata, "epoch": epoch})
    return {
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
    }


def _finalize_auto_prompt_training(
    *,
    config_path: Path,
    raw: dict[str, Any],
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    model: Any,
    requested_device: str,
    device: str,
    first_epoch_sample_count: int,
    first_epoch_sample_counts: dict[str, int],
    trained_sample_events: int,
    heatmap_samples: list[Sample],
    history: list[dict[str, float]],
    checkpoint_history: list[dict[str, Any]] | None = None,
    selected_checkpoint_path: Path | None = None,
    best_checkpoint_epoch: int | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = {
        "experiment_id": experiment_id,
        "config_path": str(config_path),
        "sample_count": first_epoch_sample_count,
        "sample_counts": first_epoch_sample_counts,
        "trained_sample_events": trained_sample_events,
        "requested_device": requested_device,
        "device": device,
        "target": target_cfg,
        "train": train_cfg,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    save_auto_prompt_checkpoint(checkpoint_path, model, config=model_cfg, metadata=metadata)
    if selected_checkpoint_path is None:
        selected_checkpoint_path = checkpoint_path
    if selected_checkpoint_path != checkpoint_path and not selected_checkpoint_path.exists():
        save_auto_prompt_checkpoint(selected_checkpoint_path, model, config=model_cfg, metadata=metadata)
    checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
    selected_checkpoint_size_bytes = selected_checkpoint_path.stat().st_size if selected_checkpoint_path.exists() else 0
    heatmap_limit = int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8
    heatmap_outputs = _write_training_heatmaps(
        output_dir=output_dir,
        model=model,
        samples=heatmap_samples,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
        limit=heatmap_limit,
        experiment_id=experiment_id,
    )
    summary = {
        **metadata,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "final_checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "selected_checkpoint_size_bytes": selected_checkpoint_size_bytes,
        "best_checkpoint_path": str(selected_checkpoint_path),
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "checkpoint_history": checkpoint_history or [],
        "model": asdict(model_cfg),
        "history": history,
        "final_loss": history[-1]["loss"],
        "heatmaps": heatmap_outputs,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _train_auto_prompt_cached_from_config(
    *,
    config_path: Path,
    raw: dict[str, Any],
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
) -> dict[str, Any]:
    torch, F, _, _ = _require_torch()
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = str(train_cfg.get("device", "cuda"))
    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _ensure_cuda_cache_device(torch, device)

    cache_dtype = _torch_cache_dtype(torch, str(train_cfg.get("cache_dtype", "float16")))
    batch_size = max(1, int(train_cfg.get("batch_size", 1)))
    light_requested_batch = train_cfg.get("light_cache_batch_size", train_cfg.get("stream_batch_size", "auto"))
    light_batch_max = max(1, int(train_cfg.get("light_cache_batch_size_max", train_cfg.get("stream_batch_size_max", 1024))))
    light_samples_per_epoch = max(0, int(train_cfg.get("light_cache_samples_per_epoch", train_cfg.get("stream_samples_per_epoch", 8192))))
    non_blocking = _bool_setting(train_cfg.get("non_blocking"), default=True)
    use_amp = _bool_setting(train_cfg.get("use_amp"), default=False) and str(device).startswith("cuda")
    profile_interval_batches = max(0, int(train_cfg.get("profile_interval_batches", 0)))
    max_steps_per_epoch = max(0, int(train_cfg.get("max_steps_per_epoch", 0)))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    epochs = max(1, int(train_cfg.get("epochs", 1)))
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_started_at = time.perf_counter()
    progress_state_path = _optional_progress_path(
        config_path=config_path,
        output_dir=output_dir,
        train_cfg=train_cfg,
        key="progress_state_path",
        default_name="train_progress_state.json",
    )
    progress_events_path = _optional_progress_path(
        config_path=config_path,
        output_dir=output_dir,
        train_cfg=train_cfg,
        key="progress_events_path",
        default_name="train_progress_events.jsonl",
    )
    _write_training_progress_state(
        state_path=progress_state_path,
        events_path=progress_events_path,
        experiment_id=experiment_id,
        train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
        epoch=0,
        epochs=epochs,
        status="running",
        phase="cache_building",
        started_at=progress_started_at,
    )

    gpu_cache_configs = [str(item) for item in raw.get("gpu_cache_dataset_configs", [])]
    light_cache_configs = [str(item) for item in raw.get("light_cache_dataset_configs", [])]
    validation_light_cache_configs = [
        str(item)
        for item in raw.get("validation_light_cache_dataset_configs", raw.get("validation_dataset_configs", []))
    ]
    dense_cache = _build_dense_gpu_cache(
        torch=torch,
        config_path=config_path,
        dataset_configs=gpu_cache_configs,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        device=device,
        cache_dtype=cache_dtype,
    )
    light_cache = _build_light_gray_cache(config_path=config_path, dataset_configs=light_cache_configs, train_cfg=train_cfg)
    validation_light_cache = _build_light_gray_cache(
        config_path=config_path,
        dataset_configs=validation_light_cache_configs,
        train_cfg=train_cfg,
    )
    if dense_cache is None and light_cache is None:
        raise RuntimeError("No samples were found for auto prompt cached training.")

    model = build_infrared_prompt_estimator(model_cfg).to(device)
    teacher_model, teacher_checkpoint_path = _build_teacher_model_if_requested(
        torch=torch,
        config_path=config_path,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        device=device,
    )
    resume_checkpoint_path, resume_start_epoch, resume_metadata = _load_resume_checkpoint_if_requested(
        torch=torch,
        model=model,
        config_path=config_path,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        device=device,
    )
    init_checkpoint_path: str | None = None
    if resume_checkpoint_path is None:
        init_checkpoint_path = _load_init_checkpoint_if_requested(
            torch=torch,
            model=model,
            config_path=config_path,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            device=device,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = _cuda_grad_scaler(torch) if use_amp else None
    light_batch_size = _autotune_light_batch_size(
        torch=torch,
        F=F,
        model=model,
        optimizer=optimizer,
        cache=light_cache,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        device=device,
        non_blocking=non_blocking,
        use_amp=use_amp,
        requested=light_requested_batch,
        max_batch_size=light_batch_max,
    )

    show_progress = bool(train_cfg.get("show_progress", True))
    progress_backend = str(train_cfg.get("progress_backend", "auto"))
    progress_mininterval = float(train_cfg.get("progress_update_interval_s", 1.0))
    heatmap_limit = int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8
    validation_ratio = float(train_cfg.get("validation_ratio", 0.0))
    validation_seed = int(train_cfg.get("validation_seed", seed + 1009))
    dense_train_indices, dense_val_indices = _split_indices(
        len(dense_cache) if dense_cache is not None else 0,
        validation_ratio=validation_ratio,
        seed=validation_seed,
    )
    light_train_indices, light_val_indices = _split_indices(
        len(light_cache) if light_cache is not None else 0,
        validation_ratio=validation_ratio,
        seed=validation_seed + 17,
    )
    selection_light_cache = validation_light_cache if validation_light_cache is not None else light_cache
    selection_light_indices = (
        list(range(len(validation_light_cache)))
        if validation_light_cache is not None
        else light_val_indices
    )
    checkpoint_interval = max(0, int(train_cfg.get("checkpoint_interval_epochs", 0)))
    select_best_checkpoint = _bool_setting(train_cfg.get("select_best_checkpoint"), default=checkpoint_interval > 0)
    selection_metric_name = str(train_cfg.get("selection_metric", "val_loss")).strip() or "val_loss"
    best_checkpoint_path = output_dir / str(train_cfg.get("best_checkpoint_name", "checkpoint_best.pt"))
    checkpoint_history: list[dict[str, Any]] = _restore_checkpoint_history_from_disk(torch=torch, output_dir=output_dir)
    selected_checkpoint_path, best_checkpoint_epoch, best_metric_name, best_metric_value = _restore_best_checkpoint_state(
        torch=torch,
        best_checkpoint_path=best_checkpoint_path,
        checkpoint_history=checkpoint_history,
    )
    if selected_checkpoint_path is None and resume_checkpoint_path is not None:
        selected_checkpoint_path = Path(resume_checkpoint_path)
    start_epoch = min(max(0, int(resume_start_epoch)), epochs)
    history: list[dict[str, float]] = []
    first_epoch_sample_counts: dict[str, int] = {}
    first_epoch_sample_count = 0
    trained_sample_events = 0
    heatmap_samples: list[Sample] = []
    _write_training_progress_state(
        state_path=progress_state_path,
        events_path=progress_events_path,
        experiment_id=experiment_id,
        train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
        epoch=start_epoch,
        epochs=epochs,
        status="running",
        phase="resuming" if resume_checkpoint_path is not None else "training",
        started_at=progress_started_at,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )
    if resume_checkpoint_path is not None:
        print(
            f"[train-resume] checkpoint={resume_checkpoint_path} start_epoch={start_epoch} epochs={epochs}",
            file=sys.stderr,
            flush=True,
        )
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sum = 0.0
        objectness_sum = 0.0
        box_sum = 0.0
        ranking_sum = 0.0
        distill_sum = 0.0
        batch_count = 0
        pending_accumulation_batches = 0
        epoch_sample_count = 0
        dense_batches = (len(dense_train_indices) + batch_size - 1) // batch_size if dense_cache is not None else 0
        light_train_size = len(light_train_indices) if light_cache is not None else 0
        light_limit = min(light_samples_per_epoch, light_train_size) if light_cache is not None and light_samples_per_epoch > 0 else light_train_size
        light_batches = (light_limit + max(1, light_batch_size) - 1) // max(1, light_batch_size) if light_cache is not None else 0
        progress_total = max_steps_per_epoch if max_steps_per_epoch > 0 else dense_batches + light_batches
        progress = _training_progress_bar(
            desc=f"auto-prompt epoch {epoch + 1}/{epochs}",
            total=progress_total,
            enabled=show_progress,
            backend=progress_backend,
            mininterval=progress_mininterval,
        )

        def consume_batch(batch: dict[str, Any], *, source: str, data_wait_ms: float = 0.0) -> bool:
            nonlocal loss_sum, objectness_sum, box_sum, ranking_sum, distill_sum
            nonlocal batch_count, pending_accumulation_batches, epoch_sample_count, trained_sample_events, heatmap_samples
            if max_steps_per_epoch > 0 and batch_count >= max_steps_per_epoch:
                return False
            zero_grad = pending_accumulation_batches == 0
            pending_accumulation_batches += 1
            step_optimizer = pending_accumulation_batches >= gradient_accumulation_steps
            loss_value, objectness_value, box_value, ranking_value, distill_value, step_ms = _train_tensor_batch(
                torch=torch,
                F=F,
                model=model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                train_cfg=train_cfg,
                device=device,
                non_blocking=non_blocking,
                use_amp=use_amp,
                step_optimizer=step_optimizer,
                zero_grad=zero_grad,
                loss_scale=1.0 / float(gradient_accumulation_steps),
            )
            if step_optimizer:
                pending_accumulation_batches = 0
            batch_count += 1
            batch_sample_count = len(batch["sample_ids"])
            epoch_sample_count += batch_sample_count
            trained_sample_events += batch_sample_count
            loss_sum += loss_value
            objectness_sum += objectness_value
            box_sum += box_value
            ranking_sum += ranking_value
            distill_sum += distill_value
            if epoch == start_epoch:
                _record_batch_counts(batch, first_epoch_sample_counts)
            if len(heatmap_samples) < heatmap_limit:
                needed = heatmap_limit - len(heatmap_samples)
                heatmap_samples.extend(batch["samples"][:needed])
            if profile_interval_batches > 0 and batch_count % profile_interval_batches == 0:
                step_seconds = max(step_ms / 1000.0, 1e-9)
                print(
                    f"[train-profile] epoch={epoch + 1} batch={batch_count} source={source} "
                    f"data_wait_ms={data_wait_ms:.1f} step_ms={step_ms:.1f} "
                    f"samples_per_s={batch_sample_count / step_seconds:.2f} batch_size={batch_sample_count} "
                    f"accum={gradient_accumulation_steps} amp={use_amp}",
                    file=sys.stderr,
                    flush=True,
                )
            if progress is not None:
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    objectness=f"{objectness_value:.4f}",
                    box=f"{box_value:.4f}",
                    rank=f"{ranking_value:.4f}",
                    distill=f"{distill_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    accum=f"{pending_accumulation_batches}/{gradient_accumulation_steps}",
                    source=source,
                )
                progress.update(1)
            return True

        try:
            if dense_cache is not None and dense_train_indices:
                train_tensor = torch.as_tensor(dense_train_indices, device=device, dtype=torch.long)
                order = torch.randperm(len(train_tensor), device=device)
                for start in range(0, len(train_tensor), batch_size):
                    if not consume_batch(dense_cache.batch(train_tensor[order[start : start + batch_size]]), source="gpu_cache"):
                        break
            if light_cache is not None and (max_steps_per_epoch <= 0 or batch_count < max_steps_per_epoch):
                rng = random.Random(seed + epoch)
                light_epoch_indices = _sample_balanced_light_indices(
                    light_cache,
                    light_train_indices,
                    limit=light_samples_per_epoch,
                    rng=rng,
                    train_cfg=train_cfg,
                )
                for start in range(0, len(light_epoch_indices), max(1, light_batch_size)):
                    batch_indices = light_epoch_indices[start : start + max(1, light_batch_size)]
                    data_started = time.perf_counter()
                    batch = _light_cache_batch(
                        light_cache,
                        batch_indices,
                        torch=torch,
                        F=F,
                        device=device,
                        target_cfg=target_cfg,
                        model_cfg=model_cfg,
                        train_cfg=train_cfg,
                    )
                    data_wait_ms = (time.perf_counter() - data_started) * 1000.0
                    if not consume_batch(batch, source="light_cache", data_wait_ms=data_wait_ms):
                        break
            if pending_accumulation_batches > 0:
                _step_optimizer_after_backward(optimizer=optimizer, scaler=scaler)
                pending_accumulation_batches = 0
        finally:
            if progress is not None:
                progress.close()
        if batch_count <= 0:
            raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
        if epoch == start_epoch:
            first_epoch_sample_count = epoch_sample_count
            print(
                f"[train-mix] epoch={epoch + 1} gpu_cache_batches={dense_batches} light_cache_batches={light_batches} "
                f"usable={first_epoch_sample_count}",
                file=sys.stderr,
                flush=True,
            )
        denom = max(1, batch_count)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / denom,
                "objectness_loss": objectness_sum / denom,
                "box_loss": box_sum / denom,
                "ranking_loss": ranking_sum / denom,
                "heuristic_distill_loss": distill_sum / denom,
                "distill_loss": distill_sum / denom,
            }
        )
        history_entry = history[-1]
        epoch_number = epoch + 1
        checkpoint_due = checkpoint_interval > 0 and (epoch_number % checkpoint_interval == 0 or epoch_number == epochs)
        if checkpoint_due:
            validation = _evaluate_cache_selection_metric(
                torch=torch,
                F=F,
                model=model,
                dense_cache=dense_cache,
                dense_indices=dense_val_indices,
                light_cache=selection_light_cache,
                light_indices=selection_light_indices,
                batch_size=batch_size,
                light_batch_size=light_batch_size,
                train_cfg=train_cfg,
                target_cfg=target_cfg,
                model_cfg=model_cfg,
                device=device,
                non_blocking=non_blocking,
                use_amp=use_amp,
            )
            history_entry.update(validation)
            metric_name = selection_metric_name if selection_metric_name in history_entry else ("val_loss" if "val_loss" in history_entry else "loss")
            metric_value = float(history_entry[metric_name])
            checkpoint_metadata = {
                "experiment_id": experiment_id,
                "config_path": str(config_path),
                "epoch": epoch_number,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "target": target_cfg,
                "train": train_cfg,
            }
            checkpoint_record = _checkpoint_epoch_record(
                output_dir=output_dir,
                epoch=epoch_number,
                model=model,
                model_cfg=model_cfg,
                metadata=checkpoint_metadata,
                metric_name=metric_name,
                metric_value=metric_value,
            )
            checkpoint_record.update({key: value for key, value in history_entry.items() if key.startswith("val_")})
            checkpoint_history.append(checkpoint_record)
            if select_best_checkpoint and (best_metric_value is None or metric_value < best_metric_value):
                save_auto_prompt_checkpoint(best_checkpoint_path, model, config=model_cfg, metadata=checkpoint_metadata)
                selected_checkpoint_path = best_checkpoint_path
                best_checkpoint_epoch = epoch_number
                best_metric_name = metric_name
                best_metric_value = metric_value
            print(
                f"[checkpoint] epoch={epoch_number} path={checkpoint_record['checkpoint_path']} "
                f"{metric_name}={metric_value:.6f} best_epoch={best_checkpoint_epoch}",
                file=sys.stderr,
                flush=True,
            )
        _write_training_progress_state(
            state_path=progress_state_path,
            events_path=progress_events_path,
            experiment_id=experiment_id,
            train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
            epoch=epoch_number,
            epochs=epochs,
            status="running",
            phase="training",
            started_at=progress_started_at,
            history_entry=history_entry,
            best_checkpoint_epoch=best_checkpoint_epoch,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
        )

    if not history:
        fallback_loss = _float_or_none(resume_metadata.get("metric_value") if resume_metadata else None)
        if fallback_loss is None:
            fallback_loss = best_metric_value if best_metric_value is not None else 0.0
        history.append(
            {
                "epoch": float(start_epoch),
                "loss": float(fallback_loss),
                "objectness_loss": 0.0,
                "box_loss": 0.0,
                "ranking_loss": 0.0,
                "heuristic_distill_loss": 0.0,
                "distill_loss": 0.0,
            }
        )

    summary = _finalize_auto_prompt_training(
        config_path=config_path,
        raw=raw,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        model=model,
        requested_device=requested_device,
        device=device,
        first_epoch_sample_count=first_epoch_sample_count,
        first_epoch_sample_counts=first_epoch_sample_counts,
        trained_sample_events=trained_sample_events,
        heatmap_samples=heatmap_samples,
        history=history,
        checkpoint_history=checkpoint_history,
        selected_checkpoint_path=selected_checkpoint_path,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        extra_metadata={
            "init_checkpoint": init_checkpoint_path,
            "resume_checkpoint": resume_checkpoint_path,
            "teacher_checkpoint": teacher_checkpoint_path,
            "resume_start_epoch": start_epoch,
            "resume_checkpoint_epoch": resume_start_epoch if resume_checkpoint_path is not None else None,
            "resume_checkpoint_metric_name": resume_metadata.get("metric_name") if resume_metadata else None,
            "resume_checkpoint_metric_value": resume_metadata.get("metric_value") if resume_metadata else None,
            "parameter_count": count_auto_prompt_parameters(model),
            "cache": {
                "mode": "mixed_gpu_light",
                "dense_gpu_samples": len(dense_cache) if dense_cache is not None else 0,
                "dense_gpu_train_samples": len(dense_train_indices),
                "dense_gpu_validation_samples": len(dense_val_indices),
                "dense_gpu_cached_gb": dense_cache.cached_gb if dense_cache is not None else 0.0,
                "light_cache_samples": len(light_cache) if light_cache is not None else 0,
                "light_cache_train_samples": len(light_train_indices),
                "light_cache_validation_samples": len(light_val_indices),
                "explicit_validation_light_cache_samples": len(validation_light_cache) if validation_light_cache is not None else 0,
                "light_cache_cached_gb": light_cache.cached_gb if light_cache is not None else 0.0,
                "explicit_validation_light_cache_cached_gb": validation_light_cache.cached_gb if validation_light_cache is not None else 0.0,
                "light_cache_batch_size": light_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_light_cache_batch_size": light_batch_size * gradient_accumulation_steps,
                "light_cache_samples_per_epoch": light_samples_per_epoch,
            }
        },
    )
    _close_light_cache(light_cache)
    _close_light_cache(validation_light_cache)
    _write_training_progress_state(
        state_path=progress_state_path,
        events_path=progress_events_path,
        experiment_id=experiment_id,
        train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
        epoch=epochs,
        epochs=epochs,
        status="completed",
        phase="finalized",
        started_at=progress_started_at,
        history_entry=history[-1] if history else None,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )
    return summary


def train_auto_prompt_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    raw = _read_yaml(config_path)
    # Enforce the data-role contract before constructing caches or touching a
    # training sample.  The CLI performs the same audit for preflight-only
    # operation, but the library entry point must be safe when called directly.
    from sparksam.protocols.reproduction import audit_auto_prompt_training_config

    protocol_audit = audit_auto_prompt_training_config(config_path, raw)
    torch, F, DataLoader, _ = _require_torch()
    train_cfg = dict(raw.get("train", {}))
    target_cfg = dict(raw.get("target", {}))
    model_cfg = AutoPromptModelConfig(**dict(raw.get("model", {})))
    if raw.get("gpu_cache_dataset_configs") or raw.get("light_cache_dataset_configs"):
        summary = _train_auto_prompt_cached_from_config(
            config_path=config_path,
            raw=raw,
            train_cfg=train_cfg,
            target_cfg=target_cfg,
            model_cfg=model_cfg,
        )
        if protocol_audit.get("strict"):
            summary["reproduction_protocol_audit_at_start"] = protocol_audit
            (Path(summary["output_dir"]) / "train_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return summary
    max_samples = int(train_cfg.get("max_samples", 0))
    max_steps_per_epoch = max(0, int(train_cfg.get("max_steps_per_epoch", 0)))
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    requested_device = str(train_cfg.get("device", "cuda"))
    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    batch_size = max(1, int(train_cfg.get("batch_size", 1)))
    num_workers = max(0, int(train_cfg.get("num_workers", 0)))
    pin_memory = _bool_setting(train_cfg.get("pin_memory"), default=device.startswith("cuda"))
    prefetch_factor = max(1, int(train_cfg.get("prefetch_factor", 2)))
    persistent_workers = _bool_setting(train_cfg.get("persistent_workers"), default=num_workers > 0)
    non_blocking = _bool_setting(train_cfg.get("non_blocking"), default=pin_memory and device.startswith("cuda"))
    use_amp = _bool_setting(train_cfg.get("use_amp"), default=False) and device.startswith("cuda")
    profile_interval_batches = max(0, int(train_cfg.get("profile_interval_batches", 0)))
    heatmap_limit = int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8
    dataset_cls = _build_streaming_dataset_class()
    dataset = dataset_cls(
        config_path=config_path,
        max_long_side=int(train_cfg.get("max_long_side", 512)),
        target_config=target_cfg,
        model_config=model_cfg,
        shuffle_buffer_size=int(train_cfg.get("shuffle_buffer_size", 256)),
        max_samples=max_samples,
        seed=seed,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _collate_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)

    model: Any | None = None
    optimizer: Any | None = None
    scaler: Any | None = None
    init_checkpoint_path: str | None = None
    epochs = max(1, int(train_cfg.get("epochs", 1)))
    show_progress = bool(train_cfg.get("show_progress", True))
    progress_backend = str(train_cfg.get("progress_backend", "auto"))
    progress_mininterval = float(train_cfg.get("progress_update_interval_s", 1.0))
    checkpoint_interval = max(0, int(train_cfg.get("checkpoint_interval_epochs", 0)))
    select_best_checkpoint = _bool_setting(train_cfg.get("select_best_checkpoint"), default=checkpoint_interval > 0)
    selection_metric_name = str(train_cfg.get("selection_metric", "loss")).strip() or "loss"
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_started_at = time.perf_counter()
    progress_state_path = _optional_progress_path(
        config_path=config_path,
        output_dir=output_dir,
        train_cfg=train_cfg,
        key="progress_state_path",
        default_name="train_progress_state.json",
    )
    progress_events_path = _optional_progress_path(
        config_path=config_path,
        output_dir=output_dir,
        train_cfg=train_cfg,
        key="progress_events_path",
        default_name="train_progress_events.jsonl",
    )
    _write_training_progress_state(
        state_path=progress_state_path,
        events_path=progress_events_path,
        experiment_id=experiment_id,
        train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
        epoch=0,
        epochs=epochs,
        status="running",
        phase="training",
        started_at=progress_started_at,
    )
    best_checkpoint_path = output_dir / str(train_cfg.get("best_checkpoint_name", "checkpoint_best.pt"))
    checkpoint_history: list[dict[str, Any]] = []
    selected_checkpoint_path: Path | None = None
    best_checkpoint_epoch: int | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    history: list[dict[str, float]] = []
    first_epoch_sample_counts: dict[str, int] = {}
    first_epoch_sample_count = 0
    trained_sample_events = 0
    heatmap_samples: list[Sample] = []
    for epoch in range(epochs):
        if model is not None:
            model.train()
        loss_sum = 0.0
        objectness_sum = 0.0
        box_sum = 0.0
        ranking_sum = 0.0
        distill_sum = 0.0
        batch_count = 0
        epoch_sample_count = 0
        progress_total = max_steps_per_epoch if max_steps_per_epoch > 0 else None
        if progress_total is None and max_samples > 0:
            progress_total = (max_samples + batch_size - 1) // batch_size
        progress = _training_progress_bar(
            desc=f"auto-prompt epoch {epoch + 1}/{epochs}",
            total=progress_total,
            enabled=show_progress,
            backend=progress_backend,
            mininterval=progress_mininterval,
        )
        try:
            loader_iter = iter(loader)
            data_wait_started = time.perf_counter()
            while True:
                if max_steps_per_epoch > 0 and batch_count >= max_steps_per_epoch:
                    break
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                batch_ready_at = time.perf_counter()
                data_wait_ms = (batch_ready_at - data_wait_started) * 1000.0
                if model is None:
                    model = build_infrared_prompt_estimator(model_cfg).to(device)
                    init_checkpoint_path = _load_init_checkpoint_if_requested(
                        torch=torch,
                        model=model,
                        config_path=config_path,
                        train_cfg=train_cfg,
                        model_cfg=model_cfg,
                        device=device,
                    )
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(train_cfg.get("learning_rate", 3e-4)),
                        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                    )
                    if use_amp:
                        scaler = _cuda_grad_scaler(torch)
                model.train()
                if optimizer is None:
                    raise RuntimeError("Optimizer was not initialized after model creation.")
                step_started = time.perf_counter()
                image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                objectness = batch["objectness"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                box_size = batch["box_size"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                box_weight = batch["box_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(torch, enabled=use_amp):
                    outputs = model(image)
                    loss, loss_parts = _compute_auto_prompt_losses(
                        torch=torch,
                        F=F,
                        outputs=outputs,
                        batch={
                            **batch,
                            "image": image,
                            "objectness": objectness,
                            "objectness_weight": objectness_weight,
                            "box_size": box_size,
                            "box_weight": box_weight,
                        },
                        train_cfg=train_cfg,
                    )
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                step_ms = (time.perf_counter() - step_started) * 1000.0

                loss_value = float(loss.detach().cpu())
                objectness_value = float(loss_parts["objectness_loss"].detach().cpu())
                box_value = float(loss_parts["box_loss"].detach().cpu())
                ranking_value = float(loss_parts["ranking_loss"].detach().cpu())
                distill_value = float(loss_parts["distill_loss"].detach().cpu())
                loss_sum += loss_value
                objectness_sum += objectness_value
                box_sum += box_value
                ranking_sum += ranking_value
                distill_sum += distill_value
                batch_count += 1
                batch_sample_count = len(batch["sample_ids"])
                epoch_sample_count += batch_sample_count
                trained_sample_events += batch_sample_count
                if epoch == 0:
                    for dataset_id in batch["dataset_ids"]:
                        first_epoch_sample_counts[f"dataset:{dataset_id}"] = first_epoch_sample_counts.get(f"dataset:{dataset_id}", 0) + 1
                    for supervision_type in batch["supervision_types"]:
                        key = f"supervision:{supervision_type}"
                        first_epoch_sample_counts[key] = first_epoch_sample_counts.get(key, 0) + 1
                if len(heatmap_samples) < heatmap_limit:
                    needed = heatmap_limit - len(heatmap_samples)
                    heatmap_samples.extend(batch["samples"][:needed])
                if profile_interval_batches > 0 and batch_count % profile_interval_batches == 0:
                    step_seconds = max(step_ms / 1000.0, 1e-9)
                    print(
                        f"[train-profile] epoch={epoch + 1} batch={batch_count} "
                        f"data_wait_ms={data_wait_ms:.1f} step_ms={step_ms:.1f} "
                        f"samples_per_s={batch_sample_count / step_seconds:.2f} "
                        f"batch_size={batch_sample_count} num_workers={num_workers} amp={use_amp}",
                        file=sys.stderr,
                        flush=True,
                    )
                if progress is not None:
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        objectness=f"{objectness_value:.4f}",
                        box=f"{box_value:.4f}",
                        rank=f"{ranking_value:.4f}",
                        distill=f"{distill_value:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                    progress.update(1)
                data_wait_started = time.perf_counter()
        finally:
            if progress is not None:
                progress.close()
        if batch_count <= 0:
            raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
        if epoch == 0:
            first_epoch_sample_count = epoch_sample_count
            print(
                f"[train-load] total usable={first_epoch_sample_count} batches={batch_count}",
                file=sys.stderr,
                flush=True,
            )
        denom = max(1, batch_count)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / denom,
                "objectness_loss": objectness_sum / denom,
                "box_loss": box_sum / denom,
                "ranking_loss": ranking_sum / denom,
                "heuristic_distill_loss": distill_sum / denom,
                "distill_loss": distill_sum / denom,
            }
        )
        history_entry = history[-1]
        epoch_number = epoch + 1
        checkpoint_due = checkpoint_interval > 0 and (epoch_number % checkpoint_interval == 0 or epoch_number == epochs)
        if checkpoint_due and model is not None:
            metric_name = selection_metric_name if selection_metric_name in history_entry else "loss"
            metric_value = float(history_entry[metric_name])
            checkpoint_metadata = {
                "experiment_id": experiment_id,
                "config_path": str(config_path),
                "epoch": epoch_number,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "target": target_cfg,
                "train": train_cfg,
            }
            checkpoint_record = _checkpoint_epoch_record(
                output_dir=output_dir,
                epoch=epoch_number,
                model=model,
                model_cfg=model_cfg,
                metadata=checkpoint_metadata,
                metric_name=metric_name,
                metric_value=metric_value,
            )
            checkpoint_history.append(checkpoint_record)
            if select_best_checkpoint and (best_metric_value is None or metric_value < best_metric_value):
                save_auto_prompt_checkpoint(best_checkpoint_path, model, config=model_cfg, metadata=checkpoint_metadata)
                selected_checkpoint_path = best_checkpoint_path
                best_checkpoint_epoch = epoch_number
                best_metric_name = metric_name
                best_metric_value = metric_value
        _write_training_progress_state(
            state_path=progress_state_path,
            events_path=progress_events_path,
            experiment_id=experiment_id,
            train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
            epoch=epoch_number,
            epochs=epochs,
            status="running",
            phase="training",
            started_at=progress_started_at,
            history_entry=history_entry,
            best_checkpoint_epoch=best_checkpoint_epoch,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
        )

    if model is None:
        raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
    summary = _finalize_auto_prompt_training(
        config_path=config_path,
        raw=raw,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        model=model,
        requested_device=requested_device,
        device=device,
        first_epoch_sample_count=first_epoch_sample_count,
        first_epoch_sample_counts=first_epoch_sample_counts,
        trained_sample_events=trained_sample_events,
        heatmap_samples=heatmap_samples,
        history=history,
        checkpoint_history=checkpoint_history,
        selected_checkpoint_path=selected_checkpoint_path,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        extra_metadata={
            "init_checkpoint": init_checkpoint_path,
            "parameter_count": count_auto_prompt_parameters(model),
        },
    )
    _write_training_progress_state(
        state_path=progress_state_path,
        events_path=progress_events_path,
        experiment_id=experiment_id,
        train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else seed,
        epoch=epochs,
        epochs=epochs,
        status="completed",
        phase="finalized",
        started_at=progress_started_at,
        history_entry=history[-1] if history else None,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )
    if protocol_audit.get("strict"):
        summary["reproduction_protocol_audit_at_start"] = protocol_audit
        (Path(summary["output_dir"]) / "train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def _write_training_heatmaps(
    *,
    output_dir: Path,
    model: Any,
    samples: list[Sample],
    model_config: AutoPromptModelConfig,
    train_config: dict[str, Any],
    device: str,
    limit: int,
    experiment_id: str,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    torch, _, _, _ = _require_torch()
    model.eval()
    records: list[dict[str, str]] = []
    max_long_side = int(train_config.get("max_long_side", 512))
    for sample in samples[:limit]:
        gray, _ = _load_resized_gray_and_box(sample, max_long_side)
        prior = ir_prior_stack(
            gray,
            use_local_contrast=model_config.use_local_contrast,
            use_top_hat=model_config.use_top_hat,
            use_dog_filter=model_config.use_dog_filter,
            use_fft_highpass=model_config.use_fft_highpass,
        )
        with torch.no_grad():
            outputs = model(torch.from_numpy(prior[None]).to(device=device, dtype=torch.float32))
        objectness = torch.sigmoid(outputs["objectness_logits"][0, 0]).detach().cpu().numpy()
        paths = write_heatmap_artifact(
            root=output_dir / "heatmaps",
            experiment_id=experiment_id,
            dataset=str(sample.metadata.get("dataset_id", sample.sequence_id or "dataset")),
            sample_id=sample.sample_id,
            stage="auto_prompt_objectness_train",
            heatmap=objectness,
            image=np.moveaxis(prior[[0, 0, 0]], 0, -1) * 255.0,
            meta={
                "model": "CompactInfraredPromptEstimator",
                "sample_id": sample.sample_id,
                "supervision_type": sample.supervision_type,
                "image_path": str(sample.image_path),
            },
        )
        records.append(paths)
    return records
