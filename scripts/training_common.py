#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparksam.config import load_app_config  # noqa: E402
from sparksam.data import build_dataset_adapter  # noqa: E402
from sparksam.data.masks import sample_mask_array  # noqa: E402
from sparksam.models import ir_prior_stack_from_path  # noqa: E402


@dataclass
class TrainSample:
    image_path: Path
    sample_id: str
    width: int
    height: int
    mask: np.ndarray
    box: list[float] | None
    point: list[float] | None
    dataset_key: str


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.environ.get('RANK', '0')}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    return _rank() == 0


def _setup_distributed() -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.cuda.set_device(_local_rank())
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", _local_rank())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _cuda_device_count_from_visible(value: str) -> int:
    value = value.strip()
    if not value:
        return torch.cuda.device_count()
    return len([item for item in value.split(",") if item.strip()])


def _maybe_launch_distributed(args: argparse.Namespace, train_cfg: dict[str, Any]) -> int | None:
    if args.no_launch or "RANK" in os.environ:
        return None
    if not torch.cuda.is_available():
        return None
    source_cfg = _read_yaml(Path(train_cfg["source_config"]))
    source_execution = source_cfg.get("execution", {}) if isinstance(source_cfg.get("execution"), dict) else {}
    train_execution = train_cfg.get("execution", {}) if isinstance(train_cfg.get("execution"), dict) else {}
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES") or str(
        train_execution.get("cuda_visible_devices", source_execution.get("cuda_visible_devices", ""))
    )
    if cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    nproc = int(os.environ.get("RESPONSE_ADAPTATION_NPROC_PER_NODE", "0") or 0)
    if nproc <= 0:
        nproc = int(train_execution.get("nproc_per_node", 0) or 0)
    if nproc <= 0:
        nproc = _cuda_device_count_from_visible(cuda_visible)
    if args.nproc_per_node:
        nproc = int(args.nproc_per_node)
    if nproc <= 1:
        return None
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--no-launch",
    ]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    cmd += ["--resume", str(args.resume)]
    if args.resume_from is not None:
        cmd += ["--resume-from", str(args.resume_from)]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{SRC_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


def _add_sam2_repo_to_path(source_cfg: dict[str, Any]) -> Path:
    paths = source_cfg.get("paths", {}) if isinstance(source_cfg.get("paths"), dict) else {}
    sam2_cfg = paths.get("sam2", {}) if isinstance(paths.get("sam2"), dict) else {}
    repo_value = str(os.environ.get("SAM2_REPO") or sam2_cfg.get("repo") or PROJECT_ROOT.parent / "sam2")
    repo = Path(os.path.expanduser(os.path.expandvars(repo_value)))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _init_hydra() -> None:
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    if not GlobalHydra.instance().is_initialized():
        initialize_config_module(config_module="sam2_configs", version_base=None)


def _build_sam2_model(model_cfg: dict[str, Any], checkpoint: dict[str, Any], source_cfg: dict[str, Any], device: torch.device, *, train: bool):
    _add_sam2_repo_to_path(source_cfg)
    _init_hydra()
    from sam2.build_sam import build_sam2

    ckpt_path = os.path.expanduser(os.path.expandvars(str(checkpoint.get("path") or model_cfg.get("ckpt"))))
    model = build_sam2(str(model_cfg["cfg"]), str(ckpt_path), device=str(device), mode="train" if train else "eval")
    model.train(train)
    return model


def _dataset_config_payload(
    source_cfg: dict[str, Any],
    dataset_key: str,
    train_cfg: dict[str, Any],
    generated_dir: Path,
    *,
    role: str,
) -> Path:
    datasets = source_cfg.get("datasets", {}) if isinstance(source_cfg.get("datasets"), dict) else {}
    paths = source_cfg.get("paths", {}) if isinstance(source_cfg.get("paths"), dict) else {}
    dataset_entry = datasets.get(dataset_key)
    if not isinstance(dataset_entry, dict) or not isinstance(dataset_entry.get("config"), dict):
        raise KeyError(f"Dataset {dataset_key!r} is missing from source config.")
    dataset_payload = dict(dataset_entry["config"])
    path_overrides = paths.get("datasets", {}) if isinstance(paths.get("datasets"), dict) else {}
    if dataset_key in path_overrides:
        dataset_payload["root"] = str(path_overrides[dataset_key])
    split_policy = train_cfg.get("split_policy", {}) if isinstance(train_cfg.get("split_policy"), dict) else {}
    split_manifest = str(split_policy.get("split_manifest") or dataset_payload.get("split_manifest") or "")
    role_split_names = split_policy.get("role_split_names", {}) if isinstance(split_policy.get("role_split_names"), dict) else {}
    split_name = str(role_split_names.get(role, "") or "")
    if not split_manifest or not split_name:
        raise RuntimeError(
            f"Response Adaptation refuses to load dataset {dataset_key!r} for role {role!r} without explicit split_manifest and split_name."
        )
    split_manifest_path = Path(split_manifest)
    if not split_manifest_path.is_absolute():
        split_manifest_path = PROJECT_ROOT / split_manifest_path
    dataset_payload["split_manifest"] = str(split_manifest_path)
    dataset_payload["split_name"] = split_name
    runtime = {
        "artifact_root": str(train_cfg.get("artifact_root", "artifacts")),
        "reference_results_root": "reference_results",
        "output_name": f"response_dataset_load/{dataset_key}",
        "device": "cpu",
        "num_workers": 0,
        "smoke_test": bool(train_cfg.get("smoke_test", False)),
        "max_samples": int(train_cfg.get("max_samples", 0) or 0),
        "max_images": 0,
        "save_visuals": False,
        "update_reference_results": False,
        "seeds": [42],
    }
    payload = {
        "model": source_cfg.get("model_defaults", {}),
        "dataset": dataset_payload,
        "runtime": runtime,
        "evaluation": source_cfg.get("evaluation_defaults", {}),
        "method": {"name": "response_dataset_loader", "split_role": role, "split_name": split_name},
    }
    path = generated_dir / f"{dataset_key}_{role}_loader.yaml"
    _write_yaml(path, payload)
    return path


def _dataset_config_path(generated_dir: Path, dataset_key: str, *, role: str) -> Path:
    return generated_dir / f"{dataset_key}_{role}_loader.yaml"


def _load_samples(train_cfg: dict[str, Any], source_cfg: dict[str, Any]) -> list[TrainSample]:
    artifact_root = Path(str(train_cfg["artifact_root"]))
    generated_dir = artifact_root / "generated_response_adaptation" / "dataset_loaders"
    output: list[TrainSample] = []
    dataset_keys = [str(item) for item in train_cfg.get("datasets", {}).get("train", [])]
    if dist.is_available() and dist.is_initialized():
        if _is_rank0():
            for dataset_key in dataset_keys:
                _dataset_config_payload(source_cfg, dataset_key, train_cfg, generated_dir, role="train")
        dist.barrier()
    else:
        for dataset_key in dataset_keys:
            _dataset_config_payload(source_cfg, dataset_key, train_cfg, generated_dir, role="train")

    for dataset_key in dataset_keys:
        dataset_config_path = _dataset_config_path(generated_dir, dataset_key, role="train")
        if not dataset_config_path.exists():
            dataset_config_path = _dataset_config_payload(source_cfg, dataset_key, train_cfg, generated_dir, role="train")
        app_config = load_app_config(dataset_config_path)
        loaded = build_dataset_adapter(app_config).load(app_config)
        for sample in loaded.samples:
            mask = sample_mask_array(sample)
            if mask is None:
                continue
            mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
            if mask.ndim != 2:
                continue
            if float(mask.sum()) <= 0.0 and not sample.metadata.get("negative_image", False):
                continue
            output.append(
                TrainSample(
                    image_path=sample.image_path,
                    sample_id=sample.sample_id,
                    width=int(sample.width),
                    height=int(sample.height),
                    mask=mask,
                    box=list(sample.bbox_loose or sample.bbox_tight or []) or None,
                    point=list(sample.point_prompt or []) or None,
                    dataset_key=str(dataset_key),
                )
            )
    if not output:
        raise RuntimeError("Response Adaptation loaded zero train samples.")
    return output


def _shard_samples(samples: list[TrainSample]) -> list[TrainSample]:
    rank = _rank()
    world = _world_size()
    if world <= 1:
        return list(samples)
    padded = list(samples)
    remainder = len(padded) % world
    if remainder:
        padded.extend(padded[: world - remainder])
    return [sample for idx, sample in enumerate(padded) if idx % world == rank]


def _input_tensor(image_rgb: np.ndarray, transforms: Any, device: torch.device) -> torch.Tensor:
    tensor = transforms(image_rgb)
    return tensor[None, ...].to(device=device, non_blocking=True)


def _features_from_image(model: Any, image_tensor: torch.Tensor) -> dict[str, Any]:
    backbone_out = model.forward_image(image_tensor)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    batch_size = image_tensor.shape[0]
    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


def _prepare_prompt(
    sample: TrainSample,
    mode: str,
    transforms: Any,
    device: torch.device,
    *,
    prompt_box: list[float] | None,
    prompt_point: list[float] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    point_coords = None
    point_labels = None
    boxes = None
    if mode in {"point", "box_point"} and prompt_point is not None and len(prompt_point) == 2:
        raw_points = torch.tensor(prompt_point, dtype=torch.float32, device=device).view(1, 1, 2)
        point_coords = transforms.transform_coords(raw_points, normalize=True, orig_hw=(sample.height, sample.width))
        point_labels = torch.ones((1, 1), dtype=torch.int64, device=device)
    if mode in {"box", "box_point"} and prompt_box is not None and len(prompt_box) == 4:
        raw_box = torch.tensor(prompt_box, dtype=torch.float32, device=device).view(1, 4)
        boxes = transforms.transform_boxes(raw_box, normalize=True, orig_hw=(sample.height, sample.width))
    return point_coords, point_labels, boxes


def _decode_prompt(
    model: Any,
    features: dict[str, Any],
    sample: TrainSample,
    mode: str,
    transforms: Any,
    device: torch.device,
    prompt_box: list[float] | None,
    prompt_point: list[float] | None,
) -> dict[str, torch.Tensor] | None:
    point_coords, point_labels, boxes = _prepare_prompt(
        sample,
        mode,
        transforms,
        device,
        prompt_box=prompt_box,
        prompt_point=prompt_point,
    )
    if point_coords is None and boxes is None:
        return None
    concat_points = (point_coords, point_labels) if point_coords is not None else None
    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int64, device=device).repeat(boxes.size(0), 1)
        if concat_points is not None:
            concat_points = (torch.cat([box_coords, concat_points[0]], dim=1), torch.cat([box_labels, concat_points[1]], dim=1))
        else:
            concat_points = (box_coords, box_labels)
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(points=concat_points, boxes=None, masks=None)
    batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=features["image_embed"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=batched_mode,
        high_res_features=[feat for feat in features["high_res_feats"]],
    )
    full_masks = transforms.postprocess_masks(low_res_masks, (sample.height, sample.width))
    return {
        "full_logits": full_masks,
        "low_res_logits": torch.clamp(low_res_masks, -32.0, 32.0),
        "iou": iou_predictions,
        "sparse_embeddings": sparse_embeddings,
    }


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    work = np.asarray(arr, dtype=np.float32)
    min_v = float(work.min()) if work.size else 0.0
    max_v = float(work.max()) if work.size else 0.0
    denom = max(1e-6, max_v - min_v)
    return ((work - min_v) / denom).astype(np.float32)


def _resize_map(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    work = np.asarray(arr, dtype=np.float32)
    if work.ndim == 3:
        work = work[0]
    if work.shape == (height, width):
        return work.astype(np.float32)
    image = Image.fromarray(np.clip(_normalize_map(work) * 255.0, 0.0, 255.0).astype(np.uint8))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return (np.asarray(image.resize((width, height), resample=resample), dtype=np.float32) / 255.0).astype(np.float32)


def _point_heatmap(point: list[float] | None, height: int, width: int, sigma: float) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    if point is None or len(point) != 2:
        return heatmap
    x = float(np.clip(point[0], 0.0, max(0.0, width - 1.0)))
    y = float(np.clip(point[1], 0.0, max(0.0, height - 1.0)))
    yy, xx = np.mgrid[0:height, 0:width]
    denom = 2.0 * max(1e-6, float(sigma) ** 2)
    return np.exp(-(((xx.astype(np.float32) - x) ** 2 + (yy.astype(np.float32) - y) ** 2) / denom)).astype(np.float32)


def _box_mask(box: list[float] | None, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    if box is None or len(box) != 4:
        return mask
    x1, y1, x2, y2 = [float(value) for value in box]
    left = int(np.floor(np.clip(min(x1, x2), 0.0, float(width))))
    right = int(np.ceil(np.clip(max(x1, x2), 0.0, float(width))))
    top = int(np.floor(np.clip(min(y1, y2), 0.0, float(height))))
    bottom = int(np.ceil(np.clip(max(y1, y2), 0.0, float(height))))
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 1.0
    return mask


def _student_prompt_tensor(
    *,
    sample: TrainSample,
    mode: str,
    auto_prompt: Any,
    device: torch.device,
    prompt_sigma: float,
    include_ir_priors: bool,
    include_candidate_prior: bool,
) -> torch.Tensor:
    height, width = int(sample.height), int(sample.width)
    if include_ir_priors:
        channels = [channel for channel in ir_prior_stack_from_path(sample.image_path)]
    else:
        channels = [ir_prior_stack_from_path(sample.image_path)[0]]
    prompt_point = list(auto_prompt.point) if mode in {"point", "box_point"} else None
    prompt_box = list(auto_prompt.box) if mode in {"box", "box_point"} else None
    channels.append(_point_heatmap(prompt_point, height, width, sigma=prompt_sigma))
    channels.append(_box_mask(prompt_box, height, width))
    candidate = getattr(auto_prompt, "objectness", None)
    channels.append(_resize_map(candidate, height, width) if include_candidate_prior and candidate is not None else np.zeros((height, width), dtype=np.float32))
    stacked = np.stack(channels, axis=0).astype(np.float32)
    return torch.from_numpy(stacked)[None, ...].to(device=device, non_blocking=True)


def _boundary_map(prob: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
    return torch.clamp(dilated - eroded, 0.0, 1.0)


def _loss_weights(train_cfg: dict[str, Any]) -> dict[str, float]:
    losses = train_cfg.get("losses", {})
    weights: dict[str, float] = {}
    for name, payload in losses.items():
        if isinstance(payload, dict):
            weights[str(name)] = float(payload.get("weight", 0.0) or 0.0)
    return weights


def _average_gradients(model: Any) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = float(dist.get_world_size())
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)
