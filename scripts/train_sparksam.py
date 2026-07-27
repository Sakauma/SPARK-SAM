#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from sparksam.benchmark.response_guidance_cache import load_teacher_cache  # noqa: E402
from sparksam.config import load_app_config  # noqa: E402
from sparksam.data import build_dataset_adapter  # noqa: E402
from sparksam.data.masks import sample_mask_array  # noqa: E402
from sparksam.models.sam2 import load_image_rgb  # noqa: E402
from sparksam.protocols.reproduction import (  # noqa: E402
    audit_spark_training_config,
    reproduction_protocol_enabled,
    git_record,
    read_lineage,
    resolve_initialization_checkpoint,
    sha256_file,
    write_checkpoint_lineage,
)
from scripts.training_common import (  # noqa: E402
    TrainSample,
    _add_sam2_repo_to_path,
    _average_gradients,
    _boundary_map,
    _build_sam2_model,
    _cleanup_distributed,
    _dataset_config_payload,
    _dice_loss,
    _features_from_image,
    _rank,
    _setup_distributed,
    _shard_samples,
    _world_size,
)
from scripts.response_distillation_losses import _false_alarm_loss, _teacher_mask_kd_loss  # noqa: E402


def _is_rank0():
    return _rank() == 0


def _read_yaml(p: Path):
    x = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(x, dict):
        raise ValueError(f"YAML root must be mapping: {p}")
    return x


def _write_json(p: Path, x: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(p: Path, x: dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(x, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _cuda_count(v: str):
    return len([i for i in str(v).split(",") if i.strip()]) if str(v).strip() else torch.cuda.device_count()


def _maybe_launch(args, cfg):
    if args.no_launch or "RANK" in os.environ or not torch.cuda.is_available():
        return None
    ex = cfg.get("execution", {}) if isinstance(cfg.get("execution"), dict) else {}
    cv = os.environ.get("CUDA_VISIBLE_DEVICES") or str(ex.get("cuda_visible_devices", ""))
    if cv:
        os.environ["CUDA_VISIBLE_DEVICES"] = cv
    n = int(args.nproc_per_node or ex.get("nproc_per_node", 0) or 0) or _cuda_count(cv)
    if n <= 1:
        return None
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={n}",
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--no-launch",
        "--resume",
        str(args.resume),
    ]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.resume_from:
        cmd += ["--resume-from", str(args.resume_from)]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{SRC_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


class JointSelfPromptHead(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        hidden_dim=128,
        token_count=2,
        min_box_side=2.0,
        candidate_count=1,
        prompt_gate_enabled: bool = False,
        box_parameterization: str = "point_centered_width_height",
        box_extent_init_fraction: float = 0.05,
        box_edge_temperature: float = 0.1,
        box_edge_decode_mode: str = "expectation",
        box_edge_outer_quantile: float = 0.1,
        box_occupancy_threshold: float = 0.5,
        point_offset_enabled: bool = False,
        point_offset_max_cell_fraction: float = 0.5,
    ):
        super().__init__()
        self.token_count = int(token_count)
        self.min_box_side = float(min_box_side)
        self.candidate_count = max(1, int(candidate_count))
        self.prompt_gate_enabled = bool(prompt_gate_enabled)
        self.box_parameterization = str(box_parameterization or "point_centered_width_height")
        self.box_edge_temperature = max(float(box_edge_temperature), 1e-6)
        self.box_edge_decode_mode = str(box_edge_decode_mode or "expectation")
        self.box_edge_outer_quantile = float(box_edge_outer_quantile)
        self.box_occupancy_threshold = float(box_occupancy_threshold)
        self.point_offset_enabled = bool(point_offset_enabled)
        self.point_offset_max_cell_fraction = float(point_offset_max_cell_fraction)
        if self.box_edge_decode_mode not in {"expectation", "argmax", "outer_quantile"}:
            raise ValueError(f"Unsupported prompt_head.box_edge_decode_mode={self.box_edge_decode_mode!r}")
        if not 0.0 < self.box_edge_outer_quantile < 0.5:
            raise ValueError("prompt_head.box_edge_outer_quantile must be in (0, 0.5)")
        if not 0.0 < self.box_occupancy_threshold < 1.0:
            raise ValueError("prompt_head.box_occupancy_threshold must be in (0, 1)")
        if not 0.0 < self.point_offset_max_cell_fraction <= 1.0:
            raise ValueError("prompt_head.point_offset_max_cell_fraction must be in (0, 1]")
        self.trunk = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU()
        )
        self.objectness = nn.Conv2d(hidden_dim, 1, 1)
        self.candidate_quality = nn.Conv2d(hidden_dim, 1, 1)
        self.candidate_mask_quality = nn.Conv2d(hidden_dim, 1, 1)
        if self.point_offset_enabled:
            self.point_offsets = nn.Conv2d(hidden_dim, 2, 1)
            nn.init.zeros_(self.point_offsets.weight)
            nn.init.zeros_(self.point_offsets.bias)
        if self.box_parameterization == "independent_center_ltrb":
            self.box_center = nn.Conv2d(hidden_dim, 1, 1)
            self.box_extent = nn.Conv2d(hidden_dim, 4, 1)
            init_fraction = min(max(float(box_extent_init_fraction), 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.box_center.weight)
            nn.init.zeros_(self.box_center.bias)
            nn.init.zeros_(self.box_extent.weight)
            nn.init.constant_(self.box_extent.bias, float(np.log(init_fraction / (1.0 - init_fraction))))
        elif self.box_parameterization == "global_edge_distributions":
            self.box_trunk = nn.Sequential(
                nn.Conv2d(embed_dim + 2, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=8, dilation=8),
                nn.GELU(),
            )
            self.box_global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
            self.box_edges = nn.Conv2d(hidden_dim, 4, 1)
        elif self.box_parameterization == "global_box_occupancy":
            self.box_trunk = nn.Sequential(
                nn.Conv2d(embed_dim + 2, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=8, dilation=8),
                nn.GELU(),
            )
            self.box_global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
            self.box_occupancy = nn.Conv2d(hidden_dim, 1, 1)
        elif self.box_parameterization == "point_centered_width_height":
            self.box_size = nn.Conv2d(hidden_dim, 2, 1)
        else:
            raise ValueError(f"Unsupported prompt_head.box_parameterization={self.box_parameterization!r}")
        if self.prompt_gate_enabled:
            self.candidate_gate = nn.Conv2d(hidden_dim, 1, 1)
            nn.init.zeros_(self.candidate_gate.weight)
            nn.init.zeros_(self.candidate_gate.bias)
        self.token_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_dim, self.token_count * embed_dim))

    def forward(self, image_embed, image_size: int, temperature: float, max_box_fraction: float):
        feat = self.trunk(image_embed)
        logits = self.objectness(feat)
        b, _, h, w = logits.shape
        prob = torch.softmax(logits.flatten(2) / max(float(temperature), 1e-6), dim=-1).view(b, 1, h, w)
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=prob.dtype, device=prob.device), torch.arange(w, dtype=prob.dtype, device=prob.device), indexing="ij"
        )
        xn = ((prob[:, 0] * (xx + 0.5)).sum((1, 2)) / float(w)).clamp(0, 1)
        yn = ((prob[:, 0] * (yy + 0.5)).sum((1, 2)) / float(h)).clamp(0, 1)
        soft_point = torch.stack([xn, yn], -1).unsqueeze(1) * float(image_size)
        box_center_logits = None
        box_center_prob = None
        box_edge_logits = None
        box_occupancy_logits = None
        if self.box_parameterization == "independent_center_ltrb":
            box_center_logits = self.box_center(feat)
            box_center_prob = torch.softmax(
                box_center_logits.flatten(2) / max(float(temperature), 1e-6), dim=-1
            ).view(b, 1, h, w)
            box_x = ((box_center_prob[:, 0] * (xx + 0.5)).sum((1, 2)) / float(w)).clamp(0, 1)
            box_y = ((box_center_prob[:, 0] * (yy + 0.5)).sum((1, 2)) / float(h)).clamp(0, 1)
            box_center_xy = torch.stack([box_x, box_y], -1) * float(image_size)
            extent = torch.sigmoid(self.box_extent(feat))
            extent = (box_center_prob * extent).sum((2, 3)) * float(image_size) * float(max_box_fraction)
            left, top, right, bottom = extent.unbind(dim=1)
            cx, cy = box_center_xy.unbind(dim=1)
            box = torch.stack(
                [
                    (cx - left - self.min_box_side * 0.5).clamp(0, image_size - 1),
                    (cy - top - self.min_box_side * 0.5).clamp(0, image_size - 1),
                    (cx + right + self.min_box_side * 0.5).clamp(0, image_size - 1),
                    (cy + bottom + self.min_box_side * 0.5).clamp(0, image_size - 1),
                ],
                -1,
            )
        elif self.box_parameterization == "global_edge_distributions":
            coords = torch.stack(
                [
                    ((xx + 0.5) / float(w)).expand(b, -1, -1),
                    ((yy + 0.5) / float(h)).expand(b, -1, -1),
                ],
                dim=1,
            )
            box_feat = self.box_trunk(torch.cat([image_embed, coords], dim=1))
            box_feat = box_feat + self.box_global_context(box_feat)
            box_edge_logits = self.box_edges(box_feat)
            left_logits = torch.logsumexp(box_edge_logits[:, 0], dim=1)
            top_logits = torch.logsumexp(box_edge_logits[:, 1], dim=2)
            right_logits = torch.logsumexp(box_edge_logits[:, 2], dim=1)
            bottom_logits = torch.logsumexp(box_edge_logits[:, 3], dim=2)
            x_coords = (torch.arange(w, dtype=prob.dtype, device=prob.device) + 0.5) / float(w) * float(image_size)
            y_coords = (torch.arange(h, dtype=prob.dtype, device=prob.device) + 0.5) / float(h) * float(image_size)
            def decode_edge(edge_logits, edge_coords, outer_side):
                edge_prob = torch.softmax(edge_logits / self.box_edge_temperature, dim=1)
                if self.box_edge_decode_mode == "expectation":
                    return (edge_prob * edge_coords).sum(dim=1)
                if self.box_edge_decode_mode == "argmax":
                    return edge_coords[edge_prob.argmax(dim=1)]
                quantile = self.box_edge_outer_quantile if outer_side == "low" else 1.0 - self.box_edge_outer_quantile
                edge_index = (edge_prob.cumsum(dim=1) >= quantile).to(torch.int64).argmax(dim=1)
                return edge_coords[edge_index]

            left = decode_edge(left_logits, x_coords, "low")
            top = decode_edge(top_logits, y_coords, "low")
            right = decode_edge(right_logits, x_coords, "high")
            bottom = decode_edge(bottom_logits, y_coords, "high")
            half_min = self.min_box_side * 0.5
            box = torch.stack(
                [
                    (torch.minimum(left, right) - half_min).clamp(0, image_size - 1),
                    (torch.minimum(top, bottom) - half_min).clamp(0, image_size - 1),
                    (torch.maximum(left, right) + half_min).clamp(0, image_size - 1),
                    (torch.maximum(top, bottom) + half_min).clamp(0, image_size - 1),
                ],
                dim=1,
            )
        elif self.box_parameterization == "global_box_occupancy":
            coords = torch.stack(
                [
                    ((xx + 0.5) / float(w)).expand(b, -1, -1),
                    ((yy + 0.5) / float(h)).expand(b, -1, -1),
                ],
                dim=1,
            )
            box_feat = self.box_trunk(torch.cat([image_embed, coords], dim=1))
            box_feat = box_feat + self.box_global_context(box_feat)
            box_occupancy_logits = self.box_occupancy(box_feat)
            occupancy = torch.sigmoid(box_occupancy_logits[:, 0])
            support = occupancy >= self.box_occupancy_threshold
            flat_peak = occupancy.flatten(1).argmax(dim=1)
            fallback_x = flat_peak % w
            fallback_y = flat_peak // w
            x_support = support.any(dim=1)
            y_support = support.any(dim=2)
            has_support = support.flatten(1).any(dim=1)
            x_index = torch.arange(w, device=occupancy.device).view(1, w)
            y_index = torch.arange(h, device=occupancy.device).view(1, h)
            left_index = torch.where(x_support, x_index, w).min(dim=1).values
            right_index = torch.where(x_support, x_index, -1).max(dim=1).values
            top_index = torch.where(y_support, y_index, h).min(dim=1).values
            bottom_index = torch.where(y_support, y_index, -1).max(dim=1).values
            left_index = torch.where(has_support, left_index, fallback_x)
            right_index = torch.where(has_support, right_index, fallback_x)
            top_index = torch.where(has_support, top_index, fallback_y)
            bottom_index = torch.where(has_support, bottom_index, fallback_y)
            box = torch.stack(
                [
                    left_index.to(occupancy.dtype) / float(w) * float(image_size),
                    top_index.to(occupancy.dtype) / float(h) * float(image_size),
                    (right_index.to(occupancy.dtype) + 1.0) / float(w) * float(image_size),
                    (bottom_index.to(occupancy.dtype) + 1.0) / float(h) * float(image_size),
                ],
                dim=1,
            ).clamp(0, image_size - 1)
        else:
            size = F.softplus(self.box_size(feat))
            wp = self.min_box_side + (prob[:, 0] * size[:, 0]).sum((1, 2)) * float(image_size) * float(max_box_fraction)
            hp = self.min_box_side + (prob[:, 0] * size[:, 1]).sum((1, 2)) * float(image_size) * float(max_box_fraction)
            cx = soft_point[:, 0, 0]
            cy = soft_point[:, 0, 1]
            box = torch.stack(
                [
                    (cx - 0.5 * wp).clamp(0, image_size - 1),
                    (cy - 0.5 * hp).clamp(0, image_size - 1),
                    (cx + 0.5 * wp).clamp(0, image_size - 1),
                    (cy + 0.5 * hp).clamp(0, image_size - 1),
                ],
                -1,
            )
        k = min(self.candidate_count, h * w)
        objectness_scores, top_idx = torch.topk(torch.sigmoid(logits.flatten(2))[:, 0], k=k, dim=1)
        quality_flat = self.candidate_quality(feat).flatten(2)[:, 0]
        candidate_logits = torch.gather(quality_flat, 1, top_idx)
        mask_quality_flat = self.candidate_mask_quality(feat).flatten(2)[:, 0]
        candidate_mask_quality_logits = torch.gather(mask_quality_flat, 1, top_idx)
        candidate_offsets = None
        if self.point_offset_enabled:
            offset_flat = self.point_offsets(feat).flatten(2)
            candidate_offsets = torch.gather(offset_flat, 2, top_idx[:, None].expand(-1, 2, -1)).transpose(1, 2)
            candidate_offsets = torch.tanh(candidate_offsets) * self.point_offset_max_cell_fraction
        candidate_gate_logits = None
        if self.prompt_gate_enabled:
            gate_flat = self.candidate_gate(feat).flatten(2)[:, 0]
            candidate_gate_logits = torch.gather(gate_flat, 1, top_idx)
        order = torch.argsort(candidate_logits, dim=1, descending=True)
        top_idx = torch.gather(top_idx, 1, order)
        objectness_scores = torch.gather(objectness_scores, 1, order)
        candidate_logits = torch.gather(candidate_logits, 1, order)
        candidate_mask_quality_logits = torch.gather(candidate_mask_quality_logits, 1, order)
        if candidate_offsets is not None:
            candidate_offsets = torch.gather(candidate_offsets, 1, order[:, :, None].expand(-1, -1, 2))
        if candidate_gate_logits is not None:
            candidate_gate_logits = torch.gather(candidate_gate_logits, 1, order)
        top_y = torch.div(top_idx, w, rounding_mode="floor").to(dtype=prob.dtype)
        top_x = (top_idx % w).to(dtype=prob.dtype)
        if candidate_offsets is not None:
            top_x = top_x + candidate_offsets[:, :, 0]
            top_y = top_y + candidate_offsets[:, :, 1]
        top_points = torch.stack([(top_x + 0.5) / float(w) * float(image_size), (top_y + 0.5) / float(h) * float(image_size)], -1).clamp(
            0, image_size - 1
        )
        tokens = self.token_head(feat).view(b, self.token_count, image_embed.shape[1])
        out = {
            "objectness_logits": logits,
            "objectness_prob": prob,
            "soft_point_coords": soft_point,
            "point_coords": top_points,
            "point_scores": objectness_scores,
            "candidate_logits": candidate_logits,
            "candidate_mask_quality_logits": candidate_mask_quality_logits,
            "box_coords": box,
            "residual_tokens": tokens,
        }
        if box_center_logits is not None:
            out["box_center_logits"] = box_center_logits
            out["box_center_prob"] = box_center_prob
        if box_edge_logits is not None:
            out["box_edge_logits"] = box_edge_logits
        if box_occupancy_logits is not None:
            out["box_occupancy_logits"] = box_occupancy_logits
        if candidate_offsets is not None:
            out["point_offsets"] = candidate_offsets
        if candidate_gate_logits is not None:
            out["candidate_gate_logits"] = candidate_gate_logits
            out["candidate_gate_values"] = torch.sigmoid(candidate_gate_logits)
        return out


def _scale_box_coords(box_coords, scale: float, image_size: int):
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("prompt_head.prompt_box_scale must be positive")
    center = 0.5 * (box_coords[:, :2] + box_coords[:, 2:])
    half_size = 0.5 * (box_coords[:, 2:] - box_coords[:, :2]) * scale
    return torch.cat([center - half_size, center + half_size], dim=1).clamp(0, image_size - 1)


class SPARKSAM(nn.Module):
    def __init__(
        self,
        sam2_model,
        token_count=2,
        min_box_side=2.0,
        max_box_fraction=0.05,
        temperature=0.05,
        candidate_count=1,
        decoder_point_count=0,
        candidate_mask_count=0,
        decoder_mode="top_points",
        mask_select_temperature=1.0,
        local_prompt_token_count=0,
        mask_selector_score_source="candidate_logits",
        mask_selector_sam_iou_weight=1.0,
        prompt_gate_enabled: bool = False,
        prompt_gate_strength: float = 0.5,
        box_parameterization: str = "point_centered_width_height",
        box_extent_init_fraction: float = 0.05,
        box_edge_temperature: float = 0.1,
        box_edge_decode_mode: str = "expectation",
        box_edge_outer_quantile: float = 0.1,
        box_occupancy_threshold: float = 0.5,
        point_offset_enabled: bool = False,
        point_offset_max_cell_fraction: float = 0.5,
        prompt_box_scale: float = 1.0,
        logit_calibration_enabled: bool = False,
        logit_calibration_max_abs_bias: float = 2.0,
        dense_mask_refinement_enabled: bool = False,
        dense_mask_refinement_hidden_dim: int = 32,
        dense_mask_refinement_max_abs_residual: float = 2.0,
        highres_prompt_refinement_enabled: bool = False,
        highres_prompt_refinement_hidden_dim: int = 32,
        highres_prompt_refinement_scale: float = 1.0,
        highres_prompt_recenter_box: bool = True,
        highres_prompt_candidate_mode: str = "replace",
        highres_prompt_base_candidate_count: int = 4,
    ):
        super().__init__()
        self.sam2_model = sam2_model
        d = int(getattr(sam2_model.sam_prompt_encoder, "embed_dim", 256))
        self.prompt_gate_enabled = bool(prompt_gate_enabled)
        self.prompt_gate_strength = float(prompt_gate_strength or 0.0)
        self.logit_calibration_enabled = bool(logit_calibration_enabled)
        self.logit_calibration_max_abs_bias = float(logit_calibration_max_abs_bias or 2.0)
        self.dense_mask_refinement_enabled = bool(dense_mask_refinement_enabled)
        self.dense_mask_refinement_max_abs_residual = float(dense_mask_refinement_max_abs_residual or 2.0)
        self.highres_prompt_refinement_enabled = bool(highres_prompt_refinement_enabled)
        self.highres_prompt_refinement_scale = float(highres_prompt_refinement_scale or 1.0)
        self.highres_prompt_recenter_box = bool(highres_prompt_recenter_box)
        self.highres_prompt_candidate_mode = str(highres_prompt_candidate_mode or "replace")
        if self.highres_prompt_candidate_mode not in {"replace", "supplement"}:
            raise ValueError(
                "prompt_head.highres_prompt_candidate_mode must be 'replace' or 'supplement'"
            )
        self.highres_prompt_base_candidate_count = max(0, int(highres_prompt_base_candidate_count or 0))
        self.min_box_side = float(min_box_side)
        self.prompt_box_scale = float(prompt_box_scale)
        if self.prompt_box_scale <= 0.0:
            raise ValueError("prompt_head.prompt_box_scale must be positive")
        self.prompt_head = JointSelfPromptHead(
            d,
            token_count=token_count,
            min_box_side=min_box_side,
            candidate_count=candidate_count,
            prompt_gate_enabled=self.prompt_gate_enabled,
            box_parameterization=box_parameterization,
            box_extent_init_fraction=box_extent_init_fraction,
            box_edge_temperature=box_edge_temperature,
            box_edge_decode_mode=box_edge_decode_mode,
            box_edge_outer_quantile=box_edge_outer_quantile,
            box_occupancy_threshold=box_occupancy_threshold,
            point_offset_enabled=point_offset_enabled,
            point_offset_max_cell_fraction=point_offset_max_cell_fraction,
        )
        self.max_box_fraction = float(max_box_fraction)
        self.temperature = float(temperature)
        self.candidate_count = max(1, int(candidate_count))
        self.decoder_point_count = max(0, int(decoder_point_count or 0))
        self.candidate_mask_count = max(0, int(candidate_mask_count or 0))
        self.decoder_mode = str(decoder_mode or "top_points")
        self.mask_select_temperature = max(float(mask_select_temperature or 1.0), 1e-6)
        self.local_prompt_token_count = max(0, int(local_prompt_token_count or 0))
        self.mask_selector_score_source = str(mask_selector_score_source or "candidate_logits")
        self.mask_selector_sam_iou_weight = float(mask_selector_sam_iou_weight or 1.0)
        if self.local_prompt_token_count > 0:
            self.local_prompt_projector = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        if self.logit_calibration_enabled:
            hidden = max(32, d // 4)
            self.logit_calibration_head = nn.Sequential(nn.Conv2d(d, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
            nn.init.zeros_(self.logit_calibration_head[-1].weight)
            nn.init.zeros_(self.logit_calibration_head[-1].bias)
        if self.dense_mask_refinement_enabled:
            high_res_channels = max(16, d // 8)
            hidden = max(8, int(dense_mask_refinement_hidden_dim or 32))
            self.dense_mask_refinement_head = nn.Sequential(
                nn.Conv2d(high_res_channels + 1, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1)
            )
            nn.init.zeros_(self.dense_mask_refinement_head[-1].weight)
            nn.init.zeros_(self.dense_mask_refinement_head[-1].bias)
        if self.highres_prompt_refinement_enabled:
            high_res_channels = max(16, d // 8)
            hidden = max(8, int(highres_prompt_refinement_hidden_dim or 32))
            self.highres_prompt_refinement_head = nn.Sequential(
                nn.Conv2d(high_res_channels, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
            nn.init.zeros_(self.highres_prompt_refinement_head[-1].weight)
            nn.init.zeros_(self.highres_prompt_refinement_head[-1].bias)

    @property
    def image_size(self):
        return int(getattr(self.sam2_model, "image_size", 1024))

    def _gate_scale(self, p, m: int, gate_start: int = 0):
        if not self.prompt_gate_enabled or "candidate_gate_values" not in p:
            return None
        gates = p["candidate_gate_values"][:, gate_start : gate_start + m].unsqueeze(-1)
        return (1.0 + self.prompt_gate_strength * (gates - 0.5)).clamp(0.25, 1.75)

    def _local_prompt_tokens(self, feats, point_coords, p=None, gate_start: int = 0):
        if self.local_prompt_token_count <= 0:
            return None
        m = max(1, min(self.local_prompt_token_count, point_coords.shape[1]))
        pts = point_coords[:, :m]
        grid = (pts / float(self.image_size)) * 2.0 - 1.0
        grid = grid.view(grid.shape[0], m, 1, 2)
        sampled = (
            F.grid_sample(feats["image_embed"], grid, mode="bilinear", padding_mode="border", align_corners=False)
            .squeeze(-1)
            .transpose(1, 2)
        )
        tokens = self.local_prompt_projector(sampled)
        scale = self._gate_scale(p, m, gate_start) if p is not None else None
        return tokens if scale is None else tokens * scale

    def _apply_highres_prompt_refinement(self, feats, prompt):
        high_res = feats["high_res_feats"][0]
        residual = self.highres_prompt_refinement_head(high_res)
        base_logits = F.interpolate(
            prompt["objectness_logits"], size=residual.shape[-2:], mode="bilinear", align_corners=False
        )
        fused_logits = base_logits + self.highres_prompt_refinement_scale * residual
        batch_size, _, height, width = fused_logits.shape
        candidate_count = min(self.candidate_count, height * width)
        candidate_logits, flat_indices = torch.topk(
            fused_logits.flatten(2)[:, 0], k=candidate_count, dim=1
        )
        grid_y = torch.div(flat_indices, width, rounding_mode="floor").to(dtype=fused_logits.dtype)
        grid_x = (flat_indices % width).to(dtype=fused_logits.dtype)
        highres_points = torch.stack(
            [
                (grid_x + 0.5) / float(width) * float(self.image_size),
                (grid_y + 0.5) / float(height) * float(self.image_size),
            ],
            dim=-1,
        ).clamp(0.0, float(self.image_size) - 1.0)
        highres_candidate_logits = candidate_logits
        highres_point_scores = torch.sigmoid(highres_candidate_logits)
        if self.highres_prompt_candidate_mode == "supplement":
            base_points = prompt["point_coords"]
            base_point_scores = prompt.get("point_scores")
            base_candidate_logits = prompt.get("candidate_logits")
            base_mask_quality_logits = prompt.get("candidate_mask_quality_logits", base_candidate_logits)
            base_keep = min(
                self.highres_prompt_base_candidate_count,
                int(base_points.shape[1]),
                max(0, candidate_count - 1),
            )
            highres_keep = candidate_count - base_keep
            if base_candidate_logits is None:
                base_candidate_logits = torch.logit(base_point_scores.clamp(1e-6, 1.0 - 1e-6))
            if base_point_scores is None:
                base_point_scores = torch.sigmoid(base_candidate_logits)
            if base_mask_quality_logits is None:
                base_mask_quality_logits = base_candidate_logits
            points = torch.cat([base_points[:, :base_keep], highres_points[:, :highres_keep]], dim=1)
            candidate_logits = torch.cat(
                [base_candidate_logits[:, :base_keep], highres_candidate_logits[:, :highres_keep]], dim=1
            )
            point_scores = torch.cat(
                [base_point_scores[:, :base_keep], highres_point_scores[:, :highres_keep]], dim=1
            )
            mask_quality_logits = torch.cat(
                [base_mask_quality_logits[:, :base_keep], highres_candidate_logits[:, :highres_keep]], dim=1
            )
        else:
            points = highres_points
            candidate_logits = highres_candidate_logits
            point_scores = highres_point_scores
            mask_quality_logits = highres_candidate_logits
        prompt["objectness_logits_lowres"] = prompt["objectness_logits"]
        prompt["objectness_logits"] = fused_logits
        prompt["objectness_prob"] = torch.softmax(fused_logits.flatten(2), dim=-1).view(
            batch_size, 1, height, width
        )
        prompt["point_coords"] = points
        prompt["point_scores"] = point_scores
        prompt["candidate_logits"] = candidate_logits
        prompt["candidate_mask_quality_logits"] = mask_quality_logits
        prompt["highres_prompt_residual"] = residual
        prompt["highres_point_coords"] = highres_points
        prompt["highres_candidate_logits"] = highres_candidate_logits
        if "candidate_gate_logits" in prompt:
            prompt["candidate_gate_logits"] = candidate_logits
            prompt["candidate_gate_values"] = torch.sigmoid(candidate_logits)
        if self.highres_prompt_recenter_box:
            original_box = prompt["box_coords"]
            box_width = (original_box[:, 2] - original_box[:, 0]).clamp_min(self.min_box_side)
            box_height = (original_box[:, 3] - original_box[:, 1]).clamp_min(self.min_box_side)
            center_x = highres_points[:, 0, 0]
            center_y = highres_points[:, 0, 1]
            prompt["box_coords"] = torch.stack(
                [
                    center_x - 0.5 * box_width,
                    center_y - 0.5 * box_height,
                    center_x + 0.5 * box_width,
                    center_y + 0.5 * box_height,
                ],
                dim=1,
            ).clamp(0.0, float(self.image_size) - 1.0)
        return prompt

    def _decode(self, feats, p, point_coords, gate_start: int = 0):
        b = point_coords.shape[0]
        box = p["box_coords"].reshape(-1, 2, 2)
        bl = torch.tensor([[2, 3]], dtype=torch.int64, device=point_coords.device).repeat(b, 1)
        pl = torch.ones((b, point_coords.shape[1]), dtype=torch.int64, device=point_coords.device)
        points = (torch.cat([box, point_coords], 1), torch.cat([bl, pl], 1))
        sparse, dense = self.sam2_model.sam_prompt_encoder(points=points, boxes=None, masks=None)
        scale = self._gate_scale(p, point_coords.shape[1], gate_start)
        if scale is not None and sparse.shape[1] >= 2 + point_coords.shape[1]:
            sparse = torch.cat(
                [sparse[:, :2], sparse[:, 2 : 2 + point_coords.shape[1]] * scale, sparse[:, 2 + point_coords.shape[1] :]], dim=1
            )
        extras = [p["residual_tokens"]]
        local_tokens = self._local_prompt_tokens(feats, point_coords, p, gate_start)
        if local_tokens is not None:
            extras.append(local_tokens)
        sparse = torch.cat([sparse, *extras], 1)
        return self.sam2_model.sam_mask_decoder(
            image_embeddings=feats["image_embed"],
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[x for x in feats["high_res_feats"]],
        )[:2]

    def _match_candidate_width(self, t, k: int, fill: float = 0.0):
        if t is None or not torch.is_tensor(t) or t.ndim < 2 or int(t.shape[1]) == int(k):
            return t
        if int(t.shape[1]) > int(k):
            return t[:, :k, ...]
        pad_shape = list(t.shape)
        pad_shape[1] = int(k) - int(t.shape[1])
        pad = torch.full(pad_shape, float(fill), dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    def _apply_fixed_prompt(self, p, fixed_prompt):
        if not fixed_prompt:
            return p
        device = p["point_coords"].device
        dtype = p["point_coords"].dtype
        if fixed_prompt.get("point_coords") is not None:
            pts = fixed_prompt["point_coords"].to(device=device, dtype=dtype)
            if pts.ndim != 3 or pts.shape[-1] != 2:
                raise ValueError(f"fixed_prompt point_coords must be BxKx2, got {tuple(pts.shape)}")
            p["point_coords"] = pts.clamp(0, self.image_size - 1)
            k = int(pts.shape[1])
            for name in [
                "point_scores",
                "candidate_logits",
                "candidate_mask_quality_logits",
                "candidate_gate_logits",
                "candidate_gate_values",
                "candidate_mask_selector_logits",
            ]:
                if name in p:
                    p[name] = self._match_candidate_width(p[name], k)
        if fixed_prompt.get("box_coords") is not None:
            box = fixed_prompt["box_coords"].to(device=device, dtype=dtype)
            if box.ndim != 2 or box.shape[-1] != 4:
                raise ValueError(f"fixed_prompt box_coords must be Bx4, got {tuple(box.shape)}")
            p["box_coords"] = box.clamp(0, self.image_size - 1)
        if fixed_prompt.get("residual_tokens") is not None:
            rt = fixed_prompt["residual_tokens"].to(device=device, dtype=p["residual_tokens"].dtype)
            if rt.ndim != 3 or rt.shape[0] != p["residual_tokens"].shape[0] or rt.shape[-1] != p["residual_tokens"].shape[-1]:
                raise ValueError(
                    f"fixed_prompt residual_tokens must be BxTxD compatible with {tuple(p['residual_tokens'].shape)}, got {tuple(rt.shape)}"
                )
            p["residual_tokens"] = rt
        elif bool(fixed_prompt.get("zero_residual_tokens", False)):
            p["residual_tokens"] = torch.zeros_like(p["residual_tokens"])
        p["fixed_prompt_prior_used"] = torch.ones((p["point_coords"].shape[0],), device=device, dtype=torch.float32)
        return p

    def forward(self, image_tensor, return_candidate_masks: bool = False, fixed_prompt=None, return_decode_context: bool = False):
        feats = _features_from_image(self.sam2_model, image_tensor)
        p = self.prompt_head(feats["image_embed"], self.image_size, self.temperature, self.max_box_fraction)
        if self.prompt_box_scale != 1.0:
            p["box_coords"] = _scale_box_coords(p["box_coords"], self.prompt_box_scale, self.image_size)
        if self.highres_prompt_refinement_enabled and fixed_prompt is None:
            p = self._apply_highres_prompt_refinement(feats, p)
        p = self._apply_fixed_prompt(p, fixed_prompt)
        k = p["point_coords"].shape[1]
        use_k = k if self.decoder_point_count <= 0 else max(1, min(self.decoder_point_count, k))
        need_masks = (self.decoder_mode in {"topk_mask_select", "topk_mask_argmax"}) or bool(return_candidate_masks)
        low, iou = self._decode(feats, p, p["point_coords"][:, :use_k])
        if self.logit_calibration_enabled:
            bias = self.logit_calibration_head(feats["image_embed"]).clamp(
                -self.logit_calibration_max_abs_bias, self.logit_calibration_max_abs_bias
            )
            if bias.shape[-2:] != low.shape[-2:]:
                bias = F.interpolate(bias, size=low.shape[-2:], mode="bilinear", align_corners=False)
            low = low + bias
            p["logit_calibration_bias"] = bias
        if self.dense_mask_refinement_enabled:
            high_res = feats["high_res_feats"][0]
            if high_res.shape[-2:] != low.shape[-2:]:
                high_res = F.interpolate(high_res, size=low.shape[-2:], mode="bilinear", align_corners=False)
            residual = self.dense_mask_refinement_head(torch.cat([high_res, low], dim=1)).clamp(
                -self.dense_mask_refinement_max_abs_residual,
                self.dense_mask_refinement_max_abs_residual,
            )
            low = low + residual
            p["dense_mask_refinement_residual"] = residual
        p.update(
            {
                "low_res_logits": torch.clamp(low, -32, 32),
                "iou": iou,
                "decoder_point_count_used": use_k,
                "decoder_mode": self.decoder_mode,
                "local_prompt_token_count_used": min(self.local_prompt_token_count, p["point_coords"].shape[1])
                if self.local_prompt_token_count > 0
                else 0,
            }
        )
        if need_masks and self.candidate_mask_count > 0 and k > 0:
            m = max(1, min(self.candidate_mask_count, k))
            lows = []
            ious = []
            for j in range(m):
                low_j, iou_j = self._decode(feats, p, p["point_coords"][:, j : j + 1], gate_start=j)
                lows.append(torch.clamp(low_j[:, 0], -32, 32))
                ious.append(iou_j.float().mean(1))
            cand_masks = torch.stack(lows, dim=1)
            cand_ious = torch.stack(ious, dim=1)
            p["candidate_mask_logits"] = cand_masks
            p["candidate_mask_iou"] = cand_ious
            p["candidate_mask_count_used"] = m
            selector_scores = p["candidate_logits"]
            if self.mask_selector_score_source in {"mask_quality", "candidate_mask_quality", "candidate_mask_quality_logits"}:
                selector_scores = p.get("candidate_mask_quality_logits", selector_scores)
            elif self.mask_selector_score_source in {"sam_iou", "decoder_iou", "candidate_mask_iou"}:
                selector_scores = cand_ious
            elif self.mask_selector_score_source in {"mask_quality_plus_sam_iou", "candidate_mask_quality_plus_sam_iou"}:
                selector_scores = (
                    p.get("candidate_mask_quality_logits", selector_scores)[:, :m] + self.mask_selector_sam_iou_weight * cand_ious
                )
            p["candidate_mask_selector_logits"] = selector_scores[:, :m]
            if self.decoder_mode == "topk_mask_select":
                weights = torch.softmax(p["candidate_mask_selector_logits"] / self.mask_select_temperature, dim=1)
                selected = (weights[:, :, None, None] * cand_masks).sum(dim=1, keepdim=True)
                p["low_res_logits"] = torch.clamp(selected, -32, 32)
                p["mask_select_weights"] = weights
                p["iou"] = (weights * cand_ious).sum(dim=1, keepdim=True)
            elif self.decoder_mode == "topk_mask_argmax":
                idx = p["candidate_mask_selector_logits"].argmax(dim=1)
                gather = idx[:, None, None, None].expand(-1, 1, cand_masks.shape[-2], cand_masks.shape[-1])
                selected = cand_masks.gather(1, gather)
                p["low_res_logits"] = torch.clamp(selected, -32, 32)
                p["mask_select_argmax"] = idx
                p["iou"] = cand_ious.gather(1, idx[:, None])
        if return_decode_context:
            p["_decode_context"] = feats
        return p


def _load_samples_for_role(cfg, source_cfg, role, return_ledgers=False):
    root = Path(str(cfg["artifact_root"]))
    gen = root / "generated_sparksam" / "dataset_loaders"
    keys = [str(x) for x in cfg.get("datasets", {}).get(role, [])]
    if not keys:
        raise RuntimeError(f"SPARK-SAM has no datasets for role={role!r}")
    if dist.is_available() and dist.is_initialized():
        if _is_rank0():
            for k in keys:
                _dataset_config_payload(source_cfg, k, cfg, gen, role=role)
        dist.barrier()
    else:
        for k in keys:
            _dataset_config_payload(source_cfg, k, cfg, gen, role=role)
    out = []
    access_ledgers = []
    for k in keys:
        loader_config = load_app_config(gen / f"{k}_{role}_loader.yaml")
        loaded = build_dataset_adapter(loader_config).load(loader_config)
        access_ledgers.append(loaded.manifest.to_dict())
        for s in loaded.samples:
            m = sample_mask_array(s)
            if m is None:
                continue
            m = (np.asarray(m, dtype=np.float32) > 0.5).astype(np.float32)
            if m.ndim != 2 or (float(m.sum()) <= 0 and not s.metadata.get("negative_image", False)):
                continue
            out.append(
                TrainSample(
                    s.image_path,
                    s.sample_id,
                    int(s.width),
                    int(s.height),
                    m,
                    list(s.bbox_loose or s.bbox_tight or []) or None,
                    list(s.point_prompt or []) or None,
                    str(k),
                )
            )
    if not out:
        raise RuntimeError(f"SPARK-SAM loaded zero samples for role={role!r}")
    return (out, access_ledgers) if return_ledgers else out


def _batchify(samples, batch_size):
    return [samples[i : i + max(1, int(batch_size))] for i in range(0, len(samples), max(1, int(batch_size)))]


def _np_to_bchw(items, device):
    a = np.stack([np.asarray(x, dtype=np.float32) for x in items], 0)
    if a.ndim == 3:
        a = a[:, None]
    return torch.from_numpy(a).to(device=device, non_blocking=True).float()


def _resize(x, size):
    return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _np_to_bchw_resized(items, size, device):
    tensors = []
    for x in items:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 3:
            if a.shape[0] == 1:
                a = a[0]
            elif a.shape[-1] == 1:
                a = a[..., 0]
            else:
                a = a[0]
        if a.ndim != 2:
            raise ValueError(f"Expected 2D mask/prob array after squeeze, got shape={a.shape}")
        t = torch.from_numpy(a)[None, None].to(device=device, non_blocking=True).float()
        tensors.append(_resize(t, size)[0])
    return torch.stack(tensors, 0).clamp(0, 1)


def _resolve_project_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class DenseObjectnessCache:
    def __init__(self, manifest_path: Path, manifest: dict[str, Any]):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.entries = {}
        base = manifest_path.parent
        for e in manifest.get("entries", []):
            if not isinstance(e, dict):
                continue
            key = (str(e.get("dataset", "")), str(e.get("sample_id", "")))
            rel = e.get("path") or e.get("npz")
            if key[0] and key[1] and rel:
                path = Path(str(rel))
                self.entries[key] = path if path.is_absolute() else (base / path).resolve()

    def get(self, dataset: str, sample_id: str):
        path = self.entries.get((str(dataset), str(sample_id)))
        if path is None or not path.exists():
            return None
        with np.load(path) as data:
            if "objectness_prob" in data:
                arr = np.asarray(data["objectness_prob"], dtype=np.float32)
            elif "objectness_logits" in data:
                arr = 1.0 / (1.0 + np.exp(-np.asarray(data["objectness_logits"], dtype=np.float32)))
            else:
                raise KeyError(f"Dense prompt-teacher cache entry lacks objectness map: {path}")
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32, copy=False)


def _load_prompt_teacher_dense_cache(cfg):
    spec = cfg.get("prompt_teacher_dense_objectness_cache")
    if not isinstance(spec, dict) or not spec.get("manifest"):
        return None
    manifest_path = _resolve_project_path(spec.get("manifest"))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DenseObjectnessCache(manifest_path, payload)


class CalibrationResponseMaskCache:
    def __init__(self, manifest_path: Path, manifest: dict[str, Any]):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.entries = {}
        self.required = bool(manifest.get("required", True))
        base = manifest_path.parent
        for e in manifest.get("entries", manifest.get("records", [])):
            if not isinstance(e, dict):
                continue
            key = (str(e.get("dataset", "")), str(e.get("sample_id", "")))
            rel = e.get("path") or e.get("npz") or e.get("record_path")
            if key[0] and key[1] and rel:
                path = Path(str(rel))
                self.entries[key] = path if path.is_absolute() else (base / path).resolve()

    def get(self, dataset: str, sample_id: str):
        path = self.entries.get((str(dataset), str(sample_id)))
        if path is None or not path.exists():
            return None
        with np.load(path, allow_pickle=True) as data:
            if "teacher_prob" in data:
                arr = np.asarray(data["teacher_prob"], dtype=np.float32)
            elif "calibration_response_prob" in data:
                arr = np.asarray(data["calibration_response_prob"], dtype=np.float32)
            elif "prob" in data:
                arr = np.asarray(data["prob"], dtype=np.float32)
            elif "teacher_logits" in data:
                arr = 1.0 / (1.0 + np.exp(-np.asarray(data["teacher_logits"], dtype=np.float32)))
            elif "calibration_response_logits" in data:
                arr = 1.0 / (1.0 + np.exp(-np.asarray(data["calibration_response_logits"], dtype=np.float32)))
            else:
                raise KeyError(f"Calibration response cache entry lacks teacher_prob/logits: {path}")
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float32, copy=False)


def _load_calibration_response_mask_cache(cfg):
    spec = cfg.get("calibration_response_cache")
    if not isinstance(spec, dict) or not spec.get("manifest"):
        return None
    manifest_path = _resolve_project_path(spec.get("manifest"))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "required" not in payload:
        payload["required"] = bool(spec.get("required", True))
    return CalibrationResponseMaskCache(manifest_path, payload)


class FixedPromptPrior:
    def __init__(self, manifest_path: Path, manifest: dict[str, Any], spec: dict[str, Any]):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.spec = spec
        self.required = bool(spec.get("required", manifest.get("required", True)))
        self.residual_token_mode = str(spec.get("residual_token_mode", manifest.get("residual_token_mode", "manifest")) or "manifest")
        self.point_count = int(spec.get("point_count", manifest.get("point_count", 0)) or 0)
        self.entries = {}
        self.any_role_entries = {}
        self.missing = []
        for e in manifest.get("entries", manifest.get("records", [])):
            if not isinstance(e, dict):
                continue
            dataset = str(e.get("dataset", "") or e.get("dataset_key", ""))
            sample_id = str(e.get("sample_id", ""))
            role = str(e.get("role", "") or "")
            if not dataset or not sample_id:
                continue
            self.entries[(role, dataset, sample_id)] = e
            self.any_role_entries.setdefault((dataset, sample_id), e)

    def _lookup(self, s, role: str = ""):
        key = (str(role or ""), str(s.dataset_key), str(s.sample_id))
        rec = self.entries.get(key) or self.any_role_entries.get((str(s.dataset_key), str(s.sample_id)))
        if rec is None and self.required:
            raise RuntimeError(f"Fixed prompt prior missing {role or '*'} {s.dataset_key}/{s.sample_id} in {self.manifest_path}")
        return rec

    def _points_internal(self, rec, s, image_size: int):
        pts = rec.get("point_coords_internal") or rec.get("points_internal") or rec.get("point_coords") or rec.get("points")
        if pts is None:
            pts = rec.get("point_coords_original") or rec.get("points_original")
            if pts is None:
                raise RuntimeError(f"Fixed prompt prior record has no point coords: {s.dataset_key}/{s.sample_id}")
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            arr[:, 0] = arr[:, 0] / max(1.0, float(s.width)) * float(image_size)
            arr[:, 1] = arr[:, 1] / max(1.0, float(s.height)) * float(image_size)
        else:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if self.point_count > 0:
            arr = arr[: self.point_count]
        if arr.shape[0] <= 0:
            raise RuntimeError(f"Fixed prompt prior has zero points: {s.dataset_key}/{s.sample_id}")
        return np.clip(arr, 0, float(image_size) - 1.0).astype(np.float32, copy=False)

    def _box_internal(self, rec, s, image_size: int):
        box = rec.get("box_coords_internal") or rec.get("box_internal") or rec.get("box_coords") or rec.get("box")
        if box is None:
            box = rec.get("box_coords_original") or rec.get("box_original") or rec.get("prompt_box")
            if box is None:
                raise RuntimeError(f"Fixed prompt prior record has no box coords: {s.dataset_key}/{s.sample_id}")
            arr = np.asarray(box, dtype=np.float32).reshape(4)
            arr = np.asarray(
                [
                    arr[0] / max(1.0, float(s.width)) * float(image_size),
                    arr[1] / max(1.0, float(s.height)) * float(image_size),
                    arr[2] / max(1.0, float(s.width)) * float(image_size),
                    arr[3] / max(1.0, float(s.height)) * float(image_size),
                ],
                dtype=np.float32,
            )
        else:
            arr = np.asarray(box, dtype=np.float32).reshape(4)
        return np.clip(arr, 0, float(image_size) - 1.0).astype(np.float32, copy=False)

    def batch_for(self, batch, device, image_size: int, role: str = ""):
        records = [self._lookup(s, role=role) for s in batch]
        if any(r is None for r in records):
            return None
        pts = [self._points_internal(r, s, image_size) for r, s in zip(records, batch)]
        max_k = max(int(p.shape[0]) for p in pts)
        pts = [np.concatenate([p, np.repeat(p[-1:], max_k - p.shape[0], axis=0)], axis=0) if p.shape[0] < max_k else p for p in pts]
        boxes = [self._box_internal(r, s, image_size) for r, s in zip(records, batch)]
        out = {
            "point_coords": torch.tensor(np.stack(pts, 0), dtype=torch.float32, device=device),
            "box_coords": torch.tensor(np.stack(boxes, 0), dtype=torch.float32, device=device),
            "manifest": str(self.manifest_path),
        }
        if self.residual_token_mode in {"zero", "zeros", "disabled", "none"}:
            out["zero_residual_tokens"] = True
        elif self.residual_token_mode in {"manifest", "artifact", "fixed"}:
            toks = []
            for r, s in zip(records, batch):
                rt = r.get("residual_tokens")
                if rt is None:
                    if self.required:
                        raise RuntimeError(
                            f"Fixed prompt prior residual_tokens missing for {s.dataset_key}/{s.sample_id}; set residual_token_mode=zero or prompt_head to proceed"
                        )
                    toks = []
                    break
                toks.append(np.asarray(rt, dtype=np.float32))
            if toks:
                out["residual_tokens"] = torch.tensor(np.stack(toks, 0), dtype=torch.float32, device=device)
        return out

    def summary(self):
        roles = sorted({k[0] for k in self.entries.keys() if k[0]})
        datasets = sorted({k[1] for k in self.entries.keys()})
        return {
            "manifest": str(self.manifest_path),
            "entry_count": len(self.entries),
            "roles": roles,
            "datasets": datasets,
            "required": self.required,
            "residual_token_mode": self.residual_token_mode,
            "point_count": self.point_count,
        }


def _load_fixed_prompt_prior_from_config(cfg):
    spec = cfg.get("fixed_prompt_prior")
    if not isinstance(spec, dict) or not spec.get("manifest"):
        return None
    manifest_path = _resolve_project_path(spec.get("manifest"))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return FixedPromptPrior(manifest_path, payload, spec)


def _dense_maps_to_bchw_resized(maps, size, device):
    tensors = []
    for arr in maps:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        t = torch.from_numpy(a)[None, None].to(device=device, non_blocking=True).float()
        tensors.append(_resize(t, size)[0])
    return torch.stack(tensors, 0).clamp(0, 1)


def _prompt_teacher_dense_objectness_loss(student_logits, target_prob, loss_cfg):
    target = target_prob.clamp(0, 1)
    temp = max(float(loss_cfg.get("temperature", 0.25) or 0.25), 1e-6)
    eps = 1e-6
    s_logp = F.log_softmax(student_logits.flatten(1) / temp, dim=1)
    t = target.flatten(1) + eps
    t = t / t.sum(dim=1, keepdim=True).clamp_min(eps)
    kl = F.kl_div(s_logp, t, reduction="batchmean") * (temp * temp)
    mse = F.mse_loss(torch.sigmoid(student_logits), target)
    return float(loss_cfg.get("kl_weight", 1.0) or 1.0) * kl + float(loss_cfg.get("prob_mse_weight", 0.5) or 0.5) * mse, {
        "prompt_teacher_dense_kl": float(kl.detach().cpu()),
        "prompt_teacher_dense_prob_mse": float(mse.detach().cpu()),
    }


def _set_trainable(module, trainable: bool):
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def _trainable_params(module):
    return [param for param in module.parameters() if param.requires_grad]


def _module_map(model):
    modules = {
        "image_encoder": model.sam2_model.image_encoder,
        "prompt_encoder": model.sam2_model.sam_prompt_encoder,
        "mask_decoder": model.sam2_model.sam_mask_decoder,
        "prompt_head": model.prompt_head,
    }
    if hasattr(model, "local_prompt_projector"):
        modules["local_prompt_projector"] = model.local_prompt_projector
    if hasattr(model, "logit_calibration_head"):
        modules["logit_calibration_head"] = model.logit_calibration_head
    if hasattr(model, "dense_mask_refinement_head"):
        modules["dense_mask_refinement_head"] = model.dense_mask_refinement_head
    if hasattr(model, "highres_prompt_refinement_head"):
        modules["highres_prompt_refinement_head"] = model.highres_prompt_refinement_head
    return modules


def _apply_module_policy(model, train_cfg):
    policy = train_cfg.get("module_policy") if isinstance(train_cfg, dict) else None
    if not isinstance(policy, dict):
        return None
    modules = _module_map(model)
    unknown = set(policy) - set(modules)
    if unknown:
        raise RuntimeError(f"Unknown train.module_policy modules for this model: {sorted(unknown)}")
    resolved = {}
    for name, module in modules.items():
        state = str(policy.get(name, "frozen")).strip().lower()
        trainable = state in {"train", "trainable"}
        if state not in {"train", "trainable", "freeze", "frozen"}:
            raise RuntimeError(f"Invalid module policy {name}={state!r}")
        _set_trainable(module, trainable)
        resolved[name] = "train" if trainable else "frozen"
    return resolved


def _apply_prompt_head_trainable_prefixes(model, train_cfg):
    prefixes = train_cfg.get("prompt_head_trainable_prefixes") if isinstance(train_cfg, dict) else None
    if prefixes is None:
        return None
    prefixes = [str(value) for value in prefixes]
    matched = []
    for name, param in model.prompt_head.named_parameters():
        trainable = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        param.requires_grad = bool(trainable)
        if trainable:
            matched.append(name)
    if not matched:
        raise RuntimeError(f"prompt_head_trainable_prefixes matched no parameters: {prefixes}")
    return {"prefixes": prefixes, "matched_parameters": matched}


def _score(records, device):
    return torch.tensor(
        [float(r.get("teacher_iou", r.get("sam2_feedback", 0.0)) or 0.0) for r in records], dtype=torch.float32, device=device
    ).view(-1, 1)


def _task_grounded_hard_mask_iou_target(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    prediction = (probability.detach() >= float(threshold)).float()
    ground_truth = (target.detach() >= 0.5).float()
    intersection = (prediction * ground_truth).flatten(1).sum(1)
    union = ((prediction + ground_truth) > 0.5).float().flatten(1).sum(1)
    score = torch.where(union > 0.0, intersection / union.clamp_min(1.0), torch.ones_like(union))
    return score.view(-1, 1)


def _mask_quality_loss(
    iou_prediction: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    records: list[dict[str, Any]],
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_type = str(cfg.get("type", "mask_score_or_iou_mse") or "mask_score_or_iou_mse")
    predicted_quality = iou_prediction.float().mean(1, keepdim=True)
    target_threshold = float(cfg.get("target_threshold", 0.5) or 0.5)
    if loss_type in {"mask_score_or_iou_mse", "teacher_score_mse", "teacher_iou_mse"}:
        target_quality = _score(records, device)
        target_source = 0.0
    elif loss_type == "task_grounded_hard_mask_iou_mse":
        target_quality = _task_grounded_hard_mask_iou_target(probability, target, target_threshold)
        target_source = 1.0
    else:
        raise RuntimeError(f"Unsupported SPARK-SAM iou_distillation loss type: {loss_type!r}")
    loss = F.mse_loss(predicted_quality, target_quality)
    details = {
        "mask_quality_predicted_mean": float(predicted_quality.detach().mean().cpu()),
        "mask_quality_target_mean": float(target_quality.detach().mean().cpu()),
        "mask_quality_absolute_error": float((predicted_quality.detach() - target_quality.detach()).abs().mean().cpu()),
        "mask_quality_target_threshold": target_threshold,
        "mask_quality_target_is_gt_iou": target_source,
    }
    return loss, details


def _prompt_targets(records, samples, image_size, grid_hw, device, sigma, box_source="teacher_prompt", point_source="teacher_prompt"):
    h, w = grid_hw
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    yy = yy.float()
    xx = xx.float()
    hs = []
    pts = []
    boxes = []
    for r, s in zip(records, samples):
        if str(point_source) in {"sample_gt", "gt_point", "ground_truth_point"}:
            p = s.point or r.get("prompt_point") or [s.width * 0.5, s.height * 0.5]
        elif str(point_source) in {"teacher_prompt", "response_teacher", "teacher"}:
            p = r.get("prompt_point") or s.point or [s.width * 0.5, s.height * 0.5]
        else:
            raise ValueError(f"Unsupported prompt_head.target_point_source={point_source!r}")
        if str(box_source) in {"sample_union", "gt_union", "ground_truth_union"}:
            box = s.box or r.get("prompt_box")
        elif str(box_source) in {"teacher_prompt", "response_teacher", "teacher"}:
            box = r.get("prompt_box") or s.box
        else:
            raise ValueError(f"Unsupported prompt_head.target_box_source={box_source!r}")
        box = box or [p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1]
        px = float(p[0]) / max(1.0, float(s.width))
        py = float(p[1]) / max(1.0, float(s.height))
        gx = px * w
        gy = py * h
        hs.append(torch.exp(-(((xx + 0.5 - gx) ** 2 + (yy + 0.5 - gy) ** 2) / (2 * float(sigma) ** 2))))
        pts.append([px * image_size, py * image_size])
        boxes.append(
            [
                float(box[0]) / max(1.0, float(s.width)) * image_size,
                float(box[1]) / max(1.0, float(s.height)) * image_size,
                float(box[2]) / max(1.0, float(s.width)) * image_size,
                float(box[3]) / max(1.0, float(s.height)) * image_size,
            ]
        )
    return {
        "objectness": torch.stack(hs).unsqueeze(1).clamp(0, 1),
        "points": torch.tensor(pts, dtype=torch.float32, device=device).view(-1, 1, 2),
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
    }


def _balanced_focal(logits, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1 - prob)
    at = torch.where(target > 0.5, torch.full_like(target, float(alpha)), torch.full_like(target, 1 - float(alpha)))
    return (at * torch.pow(1 - pt, float(gamma)) * bce).mean()


def _ranking(logits, target, margin=1.0, k=32):
    f = logits.flatten(1)
    t = target.flatten(1)
    pos = (f * t).sum(1) / t.sum(1).clamp_min(1)
    neg = f.masked_fill(~(t < 0.05), -1e6).topk(min(int(k), f.shape[1]), dim=1).values.mean(1)
    return F.relu(float(margin) - pos + neg).mean()


def _nearest_point_l1(pred_points, target_points, scale):
    # pred_points: BxKx2, target_points: Bx1x2. Min-over-candidates keeps KPrompt supervised by Prompt Estimator selected point without forcing every candidate to collapse.
    dist = torch.abs(pred_points / scale - target_points / scale).mean(dim=-1)
    return dist.min(dim=1).values.mean()


def _box_geometry_loss(pred_box, target_box, scale, loss_cfg):
    pred = pred_box / float(scale)
    target = target_box / float(scale)
    smooth_l1 = F.smooth_l1_loss(pred, target)

    px1, py1, px2, py2 = pred.unbind(dim=1)
    tx1, ty1, tx2, ty2 = target.unbind(dim=1)
    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    pred_area = (px2 - px1).clamp_min(1e-6) * (py2 - py1).clamp_min(1e-6)
    target_area = (tx2 - tx1).clamp_min(1e-6) * (ty2 - ty1).clamp_min(1e-6)
    union = pred_area + target_area - inter
    iou = inter / union.clamp_min(1e-6)
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    enclosing = (cx2 - cx1).clamp_min(1e-6) * (cy2 - cy1).clamp_min(1e-6)
    giou = iou - (enclosing - union) / enclosing.clamp_min(1e-6)
    giou_loss = (1.0 - giou).mean()

    containment = torch.stack(
        [
            F.relu(px1 - tx1),
            F.relu(py1 - ty1),
            F.relu(tx2 - px2),
            F.relu(ty2 - py2),
        ],
        dim=1,
    ).mean()
    area_ratio = torch.abs(torch.log(pred_area / target_area.clamp_min(1e-6))).mean()
    pred_center = torch.stack([(px1 + px2) * 0.5, (py1 + py2) * 0.5], dim=1)
    target_center = torch.stack([(tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5], dim=1)
    center = F.smooth_l1_loss(pred_center, target_center)

    total = (
        float(loss_cfg.get("smooth_l1_weight", 1.0) or 0.0) * smooth_l1
        + float(loss_cfg.get("giou_weight", 0.0) or 0.0) * giou_loss
        + float(loss_cfg.get("containment_weight", 0.0) or 0.0) * containment
        + float(loss_cfg.get("area_weight", 0.0) or 0.0) * area_ratio
        + float(loss_cfg.get("center_weight", 0.0) or 0.0) * center
    )
    return total, {
        "prompt_box_smooth_l1": smooth_l1,
        "prompt_box_giou": giou_loss,
        "prompt_box_containment": containment,
        "prompt_box_area_log_ratio": area_ratio,
        "prompt_box_center": center,
    }


def _box_edge_distribution_loss(edge_logits, target_box, image_size):
    if edge_logits is None:
        return target_box.sum() * 0.0
    _, _, height, width = edge_logits.shape
    left_logits = torch.logsumexp(edge_logits[:, 0], dim=1)
    top_logits = torch.logsumexp(edge_logits[:, 1], dim=2)
    right_logits = torch.logsumexp(edge_logits[:, 2], dim=1)
    bottom_logits = torch.logsumexp(edge_logits[:, 3], dim=2)
    x_targets = (target_box[:, [0, 2]] / float(image_size) * float(width)).floor().long().clamp(0, width - 1)
    y_targets = (target_box[:, [1, 3]] / float(image_size) * float(height)).floor().long().clamp(0, height - 1)
    return 0.25 * (
        F.cross_entropy(left_logits, x_targets[:, 0])
        + F.cross_entropy(top_logits, y_targets[:, 0])
        + F.cross_entropy(right_logits, x_targets[:, 1])
        + F.cross_entropy(bottom_logits, y_targets[:, 1])
    )


def _box_occupancy_loss(occupancy_logits, target_box, image_size):
    if occupancy_logits is None:
        return target_box.sum() * 0.0
    batch, _, height, width = occupancy_logits.shape
    x0 = (target_box[:, 0] / float(image_size) * float(width)).floor().long().clamp(0, width - 1)
    y0 = (target_box[:, 1] / float(image_size) * float(height)).floor().long().clamp(0, height - 1)
    x1 = (target_box[:, 2] / float(image_size) * float(width)).ceil().long().clamp(1, width)
    y1 = (target_box[:, 3] / float(image_size) * float(height)).ceil().long().clamp(1, height)
    yy = torch.arange(height, device=target_box.device).view(1, height, 1)
    xx = torch.arange(width, device=target_box.device).view(1, 1, width)
    target = ((xx >= x0[:, None, None]) & (xx < x1[:, None, None]) & (yy >= y0[:, None, None]) & (yy < y1[:, None, None])).to(
        occupancy_logits.dtype
    )
    logits = occupancy_logits[:, 0]
    positive = target.sum(dim=(1, 2)).clamp_min(1.0)
    negative = float(height * width) - positive
    positive_weight = (negative / positive).clamp(1.0, 64.0)[:, None, None]
    weights = torch.where(target > 0.5, positive_weight, torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction="none").mean(dim=(1, 2))
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2))
    dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0)
    return (bce + dice).mean()


def _hit_point_box_consistency_loss(pred_box, pred_points, batch, scale):
    """Expand the predicted box to contain candidates that actually hit GT.

    Candidate coordinates come from a discrete top-k operation, so they are treated as
    fixed targets here. Gradients flow only into the box head. On multi-component masks,
    multiple hit candidates encourage one union box instead of a primary-component box.
    Samples with no hitting candidate contribute zero rather than pulling the box toward
    a known false candidate.
    """
    losses = []
    points = pred_points.detach()
    for index, sample in enumerate(batch):
        gt = np.asarray(sample.mask, dtype=np.float32) > 0.5
        selected = []
        for point in points[index]:
            ox = int(np.clip(round(float(point[0]) / float(scale) * float(sample.width)), 0, int(sample.width) - 1))
            oy = int(np.clip(round(float(point[1]) / float(scale) * float(sample.height)), 0, int(sample.height) - 1))
            if bool(gt[oy, ox]):
                selected.append(point)
        if not selected:
            continue
        hit_points = torch.stack(selected, dim=0) / float(scale)
        box = pred_box[index] / float(scale)
        violations = torch.stack(
            [
                F.relu(box[0] - hit_points[:, 0]),
                F.relu(hit_points[:, 0] - box[2]),
                F.relu(box[1] - hit_points[:, 1]),
                F.relu(hit_points[:, 1] - box[3]),
            ],
            dim=1,
        )
        losses.append(violations.mean())
    return torch.stack(losses).mean() if losses else pred_box.sum() * 0.0


def _candidate_rank_targets(pred_points, target_points, batch, scale, device):
    labels = []
    hit_rows = []
    dist_rows = []
    pts = pred_points.detach().cpu().numpy()
    tgt = target_points.detach().cpu().numpy()[:, 0]
    for bi, s in enumerate(batch):
        gt = np.asarray(s.mask, dtype=np.float32) > 0.5
        hits = []
        dists = []
        for p in pts[bi]:
            ox = int(np.clip(round(float(p[0]) / float(scale) * float(s.width)), 0, int(s.width) - 1))
            oy = int(np.clip(round(float(p[1]) / float(scale) * float(s.height)), 0, int(s.height) - 1))
            h = bool(gt[oy, ox])
            hits.append(1.0 if h else 0.0)
            dists.append(abs(float(p[0]) - float(tgt[bi, 0])) + abs(float(p[1]) - float(tgt[bi, 1])))
        hit_arr = np.asarray(hits, dtype=np.float32)
        dist_arr = np.asarray(dists, dtype=np.float32)
        if hit_arr.max() > 0:
            masked = np.where(hit_arr > 0, dist_arr, 1e9)
            label = int(masked.argmin())
        else:
            label = int(dist_arr.argmin())
        labels.append(label)
        hit_rows.append(hit_arr)
        dist_rows.append(dist_arr / max(float(scale), 1.0))
    return (
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(np.stack(hit_rows, 0), dtype=torch.float32, device=device),
        torch.tensor(np.stack(dist_rows, 0), dtype=torch.float32, device=device),
    )


def _counterfactual_negative_points(out, batch, scale, device, min_gt_distance_px=0.0):
    candidate_points = out["point_coords"]
    candidate_logits = out["candidate_logits"].detach()
    dummy_target = candidate_points[:, :1].detach()
    _, hit_mask, _ = _candidate_rank_targets(candidate_points, dummy_target, batch, scale, device)
    background_mask = hit_mask <= 0.5
    point_np = candidate_points.detach().cpu().numpy()
    distance_rows = []
    for batch_index, sample in enumerate(batch):
        foreground_yx = np.argwhere(np.asarray(sample.mask, dtype=np.float32) > 0.5)
        distances = []
        for point in point_np[batch_index]:
            original_x = float(point[0]) / max(float(scale), 1.0) * float(sample.width)
            original_y = float(point[1]) / max(float(scale), 1.0) * float(sample.height)
            if len(foreground_yx) == 0:
                distances.append(float("inf"))
            else:
                dx = foreground_yx[:, 1].astype(np.float32) - original_x
                dy = foreground_yx[:, 0].astype(np.float32) - original_y
                distances.append(float(np.sqrt(dx * dx + dy * dy).min()))
        distance_rows.append(distances)
    gt_distance_px = torch.tensor(distance_rows, dtype=torch.float32, device=device)
    far_background = background_mask & (gt_distance_px >= float(min_gt_distance_px))
    has_far_background = far_background.any(dim=1)
    eligible_mask = torch.where(has_far_background[:, None], far_background, background_mask)
    masked_logits = candidate_logits.masked_fill(~eligible_mask, -1e9)
    indices = masked_logits.argmax(dim=1)
    gathered = candidate_points.detach().gather(
        1, indices[:, None, None].expand(-1, 1, candidate_points.shape[-1])
    )
    candidate_available = background_mask.any(dim=1)
    if not bool(candidate_available.all()):
        gathered = gathered.clone()
        for batch_index, sample in enumerate(batch):
            if bool(candidate_available[batch_index]):
                continue
            gt = np.asarray(sample.mask, dtype=np.float32) > 0.5
            background_yx = np.argwhere(~gt)
            if len(background_yx) == 0:
                fallback_y, fallback_x = 0, 0
            else:
                foreground_yx = np.argwhere(gt)
                if len(foreground_yx) == 0:
                    fallback_y, fallback_x = [int(value) for value in background_yx[0]]
                else:
                    center = foreground_yx.mean(axis=0, keepdims=True)
                    farthest = int(np.square(background_yx - center).sum(axis=1).argmax())
                    fallback_y, fallback_x = [int(value) for value in background_yx[farthest]]
            gathered[batch_index, 0, 0] = float(fallback_x) / max(1.0, float(sample.width)) * float(scale)
            gathered[batch_index, 0, 1] = float(fallback_y) / max(1.0, float(sample.height)) * float(scale)
    selected_scores = candidate_logits.gather(1, indices[:, None])
    selected_distance = gt_distance_px.gather(1, indices[:, None])
    return gathered, indices, selected_scores, selected_distance, candidate_available.float(), has_far_background.float(), hit_mask


def _counterfactual_dense_background_points(out, batch, scale, device, min_gt_distance_px=8.0):
    dense_logits = out["objectness_logits"].detach()[:, 0]
    batch_size, height, width = dense_logits.shape
    selected_points = []
    selected_scores = []
    selected_distances = []
    valid_rows = []
    for batch_index, sample in enumerate(batch):
        foreground_yx = np.argwhere(np.asarray(sample.mask, dtype=np.float32) > 0.5)
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        original_x = (grid_x + 0.5) / float(width) * float(sample.width)
        original_y = (grid_y + 0.5) / float(height) * float(sample.height)
        if len(foreground_yx) == 0:
            distance = np.full((height, width), np.inf, dtype=np.float32)
        else:
            distance = np.full((height, width), np.inf, dtype=np.float32)
            for foreground_y, foreground_x in foreground_yx:
                current = np.sqrt(
                    np.square(original_x - float(foreground_x)) + np.square(original_y - float(foreground_y))
                )
                distance = np.minimum(distance, current)
        valid = distance >= float(min_gt_distance_px)
        valid_rows.append(float(valid.any()))
        score_map = dense_logits[batch_index].float().cpu().numpy()
        if valid.any():
            flat_index = int(np.where(valid, score_map, -np.inf).argmax())
        else:
            flat_index = int(distance.argmax())
        grid_row, grid_col = np.unravel_index(flat_index, (height, width))
        selected_points.append(
            [
                (float(grid_col) + 0.5) / float(width) * float(scale),
                (float(grid_row) + 0.5) / float(height) * float(scale),
            ]
        )
        selected_scores.append(float(score_map[grid_row, grid_col]))
        selected_distances.append(float(distance[grid_row, grid_col]))
    return (
        torch.tensor(selected_points, dtype=out["point_coords"].dtype, device=device).view(batch_size, 1, 2),
        torch.tensor(selected_scores, dtype=torch.float32, device=device).view(batch_size, 1),
        torch.tensor(selected_distances, dtype=torch.float32, device=device).view(batch_size, 1),
        torch.tensor(valid_rows, dtype=torch.float32, device=device),
    )


def _detach_decode_context(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_decode_context(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_decode_context(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_decode_context(item) for item in value)
    return value


def _counterfactual_prompt_quality_loss(model, out, target, batch, cfg, device):
    if "_decode_context" not in out:
        raise RuntimeError("Counterfactual quality calibration requires return_decode_context=True")
    selection_source = str(cfg.get("selection_source", "dense_objectness_far_background") or "dense_objectness_far_background")
    min_gt_distance_px = float(cfg.get("min_gt_distance_px", 8.0) or 0.0)
    indices = None
    if selection_source == "dense_objectness_far_background":
        negative_points, selected_scores, selected_distance, far_available = _counterfactual_dense_background_points(
            out, batch, float(model.image_size), device, min_gt_distance_px=min_gt_distance_px
        )
        candidate_available = torch.ones_like(far_available)
        selected_hit_rate = torch.zeros((), dtype=torch.float32, device=device)
    elif selection_source == "topk_high_score_background":
        negative_points, indices, selected_scores, selected_distance, candidate_available, far_available, hit_mask = (
            _counterfactual_negative_points(
                out, batch, float(model.image_size), device, min_gt_distance_px=min_gt_distance_px
            )
        )
        selected_hit_rate = hit_mask.gather(1, indices[:, None]).mean()
    else:
        raise RuntimeError(f"Unsupported counterfactual selection_source={selection_source!r}")
    original_box = out["box_coords"].detach()
    negative_box_scale = float(cfg.get("negative_box_scale", 0.25) or 0.25)
    minimum_box_side = float(cfg.get("minimum_box_side", 4.0) or 4.0)
    width = ((original_box[:, 2] - original_box[:, 0]) * negative_box_scale).clamp_min(minimum_box_side)
    height = ((original_box[:, 3] - original_box[:, 1]) * negative_box_scale).clamp_min(minimum_box_side)
    center_x = negative_points[:, 0, 0]
    center_y = negative_points[:, 0, 1]
    negative_box = torch.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        dim=1,
    ).clamp(0.0, float(model.image_size) - 1.0)
    residual_tokens = out["residual_tokens"].detach()
    residual_source = str(cfg.get("residual_token_source", "batch_derangement") or "batch_derangement")
    if residual_source == "batch_derangement":
        residual_tokens = torch.roll(residual_tokens, shifts=1, dims=0) if residual_tokens.shape[0] > 1 else torch.zeros_like(residual_tokens)
    elif residual_source == "zero":
        residual_tokens = torch.zeros_like(residual_tokens)
    elif residual_source != "original":
        raise RuntimeError(f"Unsupported counterfactual residual_token_source={residual_source!r}")
    negative_prompt = {
        "box_coords": negative_box.detach(),
        "residual_tokens": residual_tokens,
    }
    if indices is not None and "candidate_gate_values" in out:
        negative_prompt["candidate_gate_values"] = out["candidate_gate_values"].detach().gather(
            1, indices[:, None]
        )
    decode_context = _detach_decode_context(out["_decode_context"])
    decoder = model.sam2_model.sam_mask_decoder
    decoder_states = [(parameter, bool(parameter.requires_grad)) for parameter in decoder.parameters()]
    quality_parameter_count = 0
    for name, parameter in decoder.named_parameters():
        quality_trainable = bool(parameter.requires_grad) and (
            name == "iou_prediction_head" or name.startswith("iou_prediction_head.")
        )
        parameter.requires_grad_(quality_trainable)
        if quality_trainable:
            quality_parameter_count += int(parameter.numel())
    if quality_parameter_count <= 0:
        raise RuntimeError("Counterfactual calibration found no trainable SAM2 iou_prediction_head parameters")
    local_states = []
    if hasattr(model, "local_prompt_projector"):
        local_states = [(parameter, bool(parameter.requires_grad)) for parameter in model.local_prompt_projector.parameters()]
        for parameter, _ in local_states:
            parameter.requires_grad_(False)
    try:
        negative_logits, negative_iou = model._decode(decode_context, negative_prompt, negative_points.detach())
    finally:
        for parameter, state in decoder_states:
            parameter.requires_grad_(state)
        for parameter, state in local_states:
            parameter.requires_grad_(state)
    target_threshold = float(cfg.get("target_threshold", 0.2) or 0.2)
    decoded_target_quality = _task_grounded_hard_mask_iou_target(
        torch.sigmoid(negative_logits.detach()), target, target_threshold
    )
    target_mode = str(cfg.get("target_mode", "decoded_hard_mask_iou") or "decoded_hard_mask_iou")
    if target_mode == "decoded_hard_mask_iou":
        target_quality = decoded_target_quality
        prompt_failure_supervision = 0.0
    elif target_mode == "prompt_miss_zero":
        target_quality = torch.zeros_like(decoded_target_quality)
        prompt_failure_supervision = 1.0
    else:
        raise RuntimeError(f"Unsupported counterfactual target_mode={target_mode!r}")
    predicted_quality = negative_iou.float().mean(1, keepdim=True)
    loss = F.mse_loss(predicted_quality, target_quality)
    return loss, {
        "counterfactual_quality_predicted_mean": float(predicted_quality.detach().mean().cpu()),
        "counterfactual_quality_target_mean": float(target_quality.detach().mean().cpu()),
        "counterfactual_quality_absolute_error": float(
            (predicted_quality.detach() - target_quality.detach()).abs().mean().cpu()
        ),
        "counterfactual_failure_target_rate": float((target_quality.detach() < 0.2).float().mean().cpu()),
        "counterfactual_decoded_quality_target_mean": float(decoded_target_quality.detach().mean().cpu()),
        "counterfactual_decoded_failure_rate": float(
            (decoded_target_quality.detach() < 0.2).float().mean().cpu()
        ),
        "counterfactual_prompt_failure_supervision": prompt_failure_supervision,
        "counterfactual_candidate_available_rate": float(candidate_available.mean().detach().cpu()),
        "counterfactual_far_candidate_available_rate": float(far_available.mean().detach().cpu()),
        "counterfactual_selected_candidate_score_mean": float(selected_scores.mean().detach().cpu()),
        "counterfactual_selected_gt_distance_px_mean": float(selected_distance.mean().detach().cpu()),
        "counterfactual_selected_candidate_hit_rate": float(selected_hit_rate.detach().cpu()),
        "counterfactual_quality_head_parameter_count": int(quality_parameter_count),
        "counterfactual_target_threshold": target_threshold,
        "counterfactual_min_gt_distance_px": min_gt_distance_px,
        "counterfactual_negative_box_scale": negative_box_scale,
        "counterfactual_residual_is_deranged": float(residual_source == "batch_derangement"),
        "counterfactual_uses_dense_objectness": float(selection_source == "dense_objectness_far_background"),
    }


def _soft_iou_from_logits(mask_logits, target):
    prob = torch.sigmoid(mask_logits)
    tgt = target.expand(-1, prob.shape[1], -1, -1)
    inter = (prob * tgt).flatten(2).sum(-1)
    union = (prob + tgt - prob * tgt).flatten(2).sum(-1).clamp_min(1e-6)
    return inter / union


def _candidate_mask_quality_target(mask_logits, target, mode="soft_iou", thresholds=(0.3,), fa_penalty=0.0):
    soft_iou = _soft_iou_from_logits(mask_logits, target)
    mode = str(mode or "soft_iou")
    if mode in {"fa_aware_soft_iou", "soft_iou_fa_aware", "fa_aware_soft_iou_penalty"}:
        prob = torch.sigmoid(mask_logits)
        tgt = target.expand(-1, prob.shape[1], -1, -1)
        fp = (prob * (1.0 - tgt)).flatten(2).sum(-1)
        prob_area = prob.flatten(2).sum(-1).clamp_min(1e-6)
        fp_ratio = fp / prob_area
        return (soft_iou - float(fa_penalty) * fp_ratio).clamp(0, 1).detach(), soft_iou.detach()
    if mode in {"hard_iou", "hard_iou_sweep", "threshold_iou", "threshold_iou_sweep", "fa_aware_hard_iou_sweep"}:
        prob = torch.sigmoid(mask_logits)
        tgt = target.expand(-1, prob.shape[1], -1, -1)
        scores = []
        for thr in thresholds:
            pred = (prob >= float(thr)).float()
            inter = (pred * tgt).flatten(2).sum(-1)
            union = (pred + tgt - pred * tgt).flatten(2).sum(-1).clamp_min(1e-6)
            hard_iou = inter / union
            fp = (pred * (1.0 - tgt)).flatten(2).sum(-1)
            pred_area = pred.flatten(2).sum(-1).clamp_min(1e-6)
            fp_ratio = fp / pred_area
            scores.append((hard_iou - float(fa_penalty) * fp_ratio).clamp(0, 1))
        return torch.stack(scores, 0).max(dim=0).values.detach(), soft_iou.detach()
    return soft_iou.detach(), soft_iou.detach()


def _candidate_mask_aware_losses(
    candidate_logits, candidate_mask_logits, target, margin=0.25, target_mode="soft_iou", target_thresholds=(0.3,), fa_penalty=0.0
):
    m = min(candidate_logits.shape[1], candidate_mask_logits.shape[1])
    logits = candidate_logits[:, :m]
    masks = candidate_mask_logits[:, :m]
    quality_target, soft_iou = _candidate_mask_quality_target(masks, target, target_mode, target_thresholds, fa_penalty)
    labels = quality_target.argmax(dim=1)
    ce = F.cross_entropy(logits, labels)
    pos = logits.gather(1, labels[:, None])
    neg = logits.masked_fill(F.one_hot(labels, m).bool(), -1e6).max(dim=1, keepdim=True).values
    pair = F.relu(float(margin) - pos + neg).mean()
    reg = F.mse_loss(torch.sigmoid(logits), quality_target.clamp(0, 1))
    best_mask = masks.gather(1, labels[:, None, None, None].expand(-1, 1, masks.shape[-2], masks.shape[-1])).squeeze(1)
    oracle_target = target[:, 0]
    oracle_bce = F.binary_cross_entropy_with_logits(best_mask, oracle_target) + _dice_loss(best_mask[:, None], target)
    top1_iou = soft_iou[:, 0].mean()
    best_iou = soft_iou.max(dim=1).values.mean()
    gap = (soft_iou.max(dim=1).values - soft_iou[:, 0]).mean()
    best_quality = quality_target.max(dim=1).values.mean()
    top1_quality = quality_target[:, 0].mean()
    details = {
        "candidate_mask_top1_soft_iou_train": float(top1_iou.detach().cpu()),
        "candidate_mask_best_soft_iou_train": float(best_iou.detach().cpu()),
        "candidate_mask_oracle_gap_train": float(gap.detach().cpu()),
        "candidate_mask_top1_quality_train": float(top1_quality.detach().cpu()),
        "candidate_mask_best_quality_train": float(best_quality.detach().cpu()),
        "candidate_mask_quality_regression": float(reg.detach().cpu()),
        "candidate_mask_count_train": int(m),
    }
    return ce, pair, reg, oracle_bce, details


def _candidate_score_losses(candidate_logits, pred_points, target_points, batch, scale, device, margin=1.0, point_scores=None):
    labels, hit_mask, dist = _candidate_rank_targets(pred_points, target_points, batch, scale, device)
    ce = F.cross_entropy(candidate_logits, labels)
    pos = candidate_logits.gather(1, labels[:, None])
    neg = candidate_logits.masked_fill(F.one_hot(labels, candidate_logits.shape[1]).bool(), -1e6).max(dim=1, keepdim=True).values
    pair = F.relu(float(margin) - pos + neg).mean()
    # Multi-positive supervision keeps all GT-hitting candidates competitive on hard datasets.
    # If none of the current top-K hits GT, fall back to the nearest prompt-estimator-selected point label.
    multi_target = hit_mask.clone()
    no_hit = multi_target.sum(dim=1, keepdim=True) <= 0
    multi_target = torch.where(no_hit, F.one_hot(labels, candidate_logits.shape[1]).float(), multi_target)
    multi_bce = F.binary_cross_entropy_with_logits(candidate_logits, multi_target)
    coverage = F.softplus(1.0 - torch.logsumexp(candidate_logits.masked_fill(multi_target <= 0, -1e6), dim=1)).mean()
    objectness_bce = torch.zeros((), device=device)
    if point_scores is not None:
        objectness_bce = F.binary_cross_entropy(point_scores.clamp(1e-4, 1 - 1e-4), multi_target)
    top1_hit = hit_mask[:, 0].mean()
    any_hit = (hit_mask.max(dim=1).values > 0).float().mean()
    label_hit = hit_mask.gather(1, labels[:, None]).mean()
    mean_hit = hit_mask.sum(dim=1).mean()
    details = {
        "candidate_top1_hit_train": float(top1_hit.detach().cpu()),
        "candidate_any_hit_train": float(any_hit.detach().cpu()),
        "candidate_label_hit_train": float(label_hit.detach().cpu()),
        "candidate_mean_hit_count_train": float(mean_hit.detach().cpu()),
        "candidate_multi_positive_bce": float(multi_bce.detach().cpu()),
        "candidate_coverage_loss": float(coverage.detach().cpu()),
        "candidate_objectness_bce": float(objectness_bce.detach().cpu()),
    }
    return ce, pair, multi_bce, coverage, objectness_bce, details


def _point_gate_loss(gate_logits, pred_points, target_points, batch, scale, device):
    labels, hit_mask, dist = _candidate_rank_targets(pred_points, target_points, batch, scale, device)
    multi_target = hit_mask.clone()
    no_hit = multi_target.sum(dim=1, keepdim=True) <= 0
    multi_target = torch.where(no_hit, F.one_hot(labels, gate_logits.shape[1]).float(), multi_target)
    pos = (multi_target > 0).float()
    neg = 1.0 - pos
    pos_n = pos.sum().clamp_min(1.0)
    neg_n = neg.sum().clamp_min(1.0)
    weights = 0.5 * pos / pos_n + 0.5 * neg / neg_n
    bce = F.binary_cross_entropy_with_logits(gate_logits, multi_target, reduction="none")
    gate_prob = torch.sigmoid(gate_logits)
    entropy = -(gate_prob.clamp(1e-5, 1 - 1e-5).log() * gate_prob + (1 - gate_prob).clamp(1e-5, 1 - 1e-5).log() * (1 - gate_prob)).mean()
    top1_gate = gate_prob[:, 0].mean()
    hit_gate = (gate_prob * multi_target).sum() / multi_target.sum().clamp_min(1.0)
    bg_gate = (gate_prob * (1.0 - multi_target)).sum() / (1.0 - multi_target).sum().clamp_min(1.0)
    details = {
        "point_gate_top1_mean": float(top1_gate.detach().cpu()),
        "point_gate_hit_mean": float(hit_gate.detach().cpu()),
        "point_gate_bg_mean": float(bg_gate.detach().cpu()),
        "point_gate_entropy": float(entropy.detach().cpu()),
    }
    return (bce * weights).sum() + 0.02 * entropy, details


def _foreground_recall_loss(logits, target, cfg):
    if not isinstance(cfg, dict):
        return torch.zeros((), device=logits.device), {}
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    positive_mass = target.sum(dim=dims).clamp_min(1.0)
    positive_bce = (F.softplus(-logits) * target).sum(dim=dims) / positive_mass
    alpha = float(cfg.get("tversky_fp_weight", 0.3) or 0.3)
    beta = float(cfg.get("tversky_fn_weight", 0.7) or 0.7)
    smooth = float(cfg.get("smooth", 1.0) or 1.0)
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - prob) * target).sum(dim=dims)
    tversky = 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    positive_bce_weight = float(cfg.get("positive_bce_weight", 1.0) or 0.0)
    tversky_weight = float(cfg.get("tversky_weight", 1.0) or 0.0)
    loss = positive_bce_weight * positive_bce.mean() + tversky_weight * tversky.mean()
    return loss, {
        "foreground_positive_bce": float(positive_bce.mean().detach().cpu()),
        "foreground_tversky": float(tversky.mean().detach().cpu()),
        "foreground_tversky_fp_weight": alpha,
        "foreground_tversky_fn_weight": beta,
    }


def _calibration_band_loss(logits, target, cfg):
    if not isinstance(cfg, dict):
        return torch.zeros((), device=logits.device)
    thresholds = cfg.get("thresholds", [0.10, 0.18, 0.30])
    width = float(cfg.get("band_width", 0.08) or 0.08)
    band_gain = float(cfg.get("band_gain", 2.0) or 2.0)
    if isinstance(thresholds, (int, float, str)):
        thresholds = [float(thresholds)]
    prob = torch.sigmoid(logits.detach())
    weight = torch.ones_like(target)
    for t in thresholds:
        weight = weight + band_gain * torch.exp(-torch.abs(prob - float(t)) / max(width, 1e-4))
    bg_gain = float(cfg.get("background_gain", 0.5) or 0.5)
    weight = weight * (1.0 + bg_gain * (1.0 - target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def _schedule_multiplier(cfg, epoch: int, name: str):
    sched = cfg.get("loss_schedule", {}) if isinstance(cfg.get("loss_schedule"), dict) else {}
    prompt_names = set(str(x) for x in sched.get("prompt_loss_names", []))
    if name not in prompt_names:
        return 1.0
    warm = int(sched.get("prompt_full_weight_epochs", 30) or 30)
    end = int(sched.get("prompt_decay_end_epoch", 60) or 60)
    late = float(sched.get("prompt_late_multiplier", 0.4) or 0.4)
    if int(epoch) <= warm:
        return 1.0
    late_by_name = sched.get("late_multiplier_by_name", {}) if isinstance(sched.get("late_multiplier_by_name", {}), dict) else {}
    late = float(late_by_name.get(name, late))
    if int(epoch) >= end:
        return late
    a = (float(epoch) - warm) / max(1.0, float(end - warm))
    return (1.0 - a) + a * late


def _train_one_batch(
    model, batch, cache, transforms, device, cfg, dense_cache=None, calibration_response_cache=None, fixed_prompt_prior=None, epoch: int = 1
):
    lc = cfg.get("losses", {}) if isinstance(cfg.get("losses"), dict) else {}
    w = {n: float(v.get("weight", 0) or 0) for n, v in lc.items() if isinstance(v, dict)}
    records = []
    if cache is not None:
        for s in batch:
            e = cache.entry_for(dataset=s.dataset_key, sample_id=s.sample_id, prompt_mode="box_point")
            if e is None:
                raise RuntimeError(f"SPARK-SAM cache missing box_point for {s.dataset_key}/{s.sample_id}")
            records.append(cache.load_record(e))
    else:
        records = [{} for _ in batch]
    need_candidate_masks = any(
        w.get(n, 0) > 0
        for n in [
            "candidate_mask_aware_score_loss",
            "candidate_mask_aware_pairwise_loss",
            "candidate_mask_quality_regression_loss",
            "candidate_oracle_mask_response_loss",
        ]
    )
    image_tensor = torch.stack([transforms(load_image_rgb(s.image_path)) for s in batch], 0).to(device=device, non_blocking=True)
    fixed_prompt = fixed_prompt_prior.batch_for(batch, device, model.image_size, role="train") if fixed_prompt_prior is not None else None
    out = model(
        image_tensor,
        return_candidate_masks=need_candidate_masks,
        fixed_prompt=fixed_prompt,
        return_decode_context=w.get("counterfactual_iou_calibration", 0) > 0,
    )
    logits = out["low_res_logits"]
    prob = torch.sigmoid(logits)
    zero = torch.zeros((), device=device)
    target = _np_to_bchw_resized([r.get("gt_mask", s.mask) for r, s in zip(records, batch)], logits.shape[-2:], device)
    teacher = None
    calibration_response_used = False
    needs_teacher = any(w.get(n, 0) > 0 for n in ["teacher_mask_distillation", "boundary_distillation", "iou_distillation"])
    if cache is not None:
        teacher_items = [r["teacher_prob"] for r in records]
        if calibration_response_cache is not None:
            calibration_response_maps = []
            missing = []
            for s in batch:
                arr = calibration_response_cache.get(s.dataset_key, s.sample_id)
                if arr is None:
                    missing.append(f"{s.dataset_key}/{s.sample_id}")
                else:
                    calibration_response_maps.append(arr)
            if missing:
                if bool(getattr(calibration_response_cache, "required", True)):
                    raise RuntimeError(f"Calibration response mask cache missing {len(missing)} samples, first={missing[0]}")
            elif len(calibration_response_maps) == len(batch):
                teacher_items = calibration_response_maps
                calibration_response_used = True
        teacher = _np_to_bchw_resized(teacher_items, logits.shape[-2:], device)
    elif needs_teacher:
        raise RuntimeError("SPARK-SAM config enables KD/teacher-dependent losses but has no teacher_cache.")
    else:
        teacher = target.detach()
    tkd = zero
    if w.get("teacher_mask_distillation", 0) > 0:
        tkd = _teacher_mask_kd_loss(logits, prob, teacher, target, lc.get("teacher_mask_distillation", {}))
    gt = F.binary_cross_entropy_with_logits(logits, target) + _dice_loss(logits, target)
    bkd = zero
    if w.get("boundary_distillation", 0) > 0:
        bkd = F.mse_loss(_boundary_map(prob), _boundary_map(teacher))
    fa = zero
    fd = {}
    if w.get("false_alarm_penalty", 0) > 0:
        fa, fd = _false_alarm_loss(prob, target, teacher, lc.get("false_alarm_penalty", {}))
    skd = zero
    skd_details = {}
    if w.get("iou_distillation", 0) > 0:
        skd, skd_details = _mask_quality_loss(
            out["iou"], prob, target, records, device, lc.get("iou_distillation", {})
        )
    counterfactual_quality = zero
    counterfactual_quality_details = {}
    if w.get("counterfactual_iou_calibration", 0) > 0:
        counterfactual_quality, counterfactual_quality_details = _counterfactual_prompt_quality_loss(
            model, out, target, batch, lc.get("counterfactual_iou_calibration", {}), device
        )
    pc = cfg.get("prompt_head", {}) if isinstance(cfg.get("prompt_head"), dict) else {}
    pt = _prompt_targets(
        records,
        batch,
        model.image_size,
        out["objectness_logits"].shape[-2:],
        device,
        float(pc.get("target_gaussian_sigma", 2.0) or 2.0),
        box_source=str(pc.get("target_box_source", "teacher_prompt") or "teacher_prompt"),
        point_source=str(pc.get("target_point_source", "teacher_prompt") or "teacher_prompt"),
    )
    obj = _balanced_focal(
        out["objectness_logits"], pt["objectness"], float(pc.get("focal_alpha", 0.75) or 0.75), float(pc.get("focal_gamma", 2.0) or 2.0)
    )
    scale = float(model.image_size)
    box, box_details = _box_geometry_loss(
        out["box_coords"], pt["boxes"], scale, lc.get("prompt_box_loss", {})
    )
    box_edge_ce = _box_edge_distribution_loss(out.get("box_edge_logits"), pt["boxes"], scale)
    box = box + float(lc.get("prompt_box_loss", {}).get("edge_ce_weight", 0.0) or 0.0) * box_edge_ce
    box_details["prompt_box_edge_ce"] = box_edge_ce
    box_occupancy = _box_occupancy_loss(out.get("box_occupancy_logits"), pt["boxes"], scale)
    box = box + float(lc.get("prompt_box_loss", {}).get("occupancy_weight", 0.0) or 0.0) * box_occupancy
    box_details["prompt_box_occupancy"] = box_occupancy
    point = _nearest_point_l1(out["point_coords"], pt["points"], scale)
    point_box = _hit_point_box_consistency_loss(out["box_coords"], out["point_coords"], batch, scale)
    rank = _ranking(
        out["objectness_logits"],
        pt["objectness"],
        float(pc.get("ranking_margin", 1.0) or 1.0),
        int(pc.get("ranking_negative_top_k", 32) or 32),
    )
    cand_ce = torch.zeros((), device=device)
    cand_pair = torch.zeros((), device=device)
    cand_multi = torch.zeros((), device=device)
    cand_cov = torch.zeros((), device=device)
    cand_obj = torch.zeros((), device=device)
    cand_mask_ce = torch.zeros((), device=device)
    cand_mask_pair = torch.zeros((), device=device)
    cand_mask_reg = torch.zeros((), device=device)
    cand_oracle_mask = torch.zeros((), device=device)
    gate_loss = torch.zeros((), device=device)
    calib_loss = torch.zeros((), device=device)
    cand_details = {}
    if "candidate_logits" in out:
        cand_ce, cand_pair, cand_multi, cand_cov, cand_obj, cand_details = _candidate_score_losses(
            out["candidate_logits"],
            out["point_coords"],
            pt["points"],
            batch,
            scale,
            device,
            float(pc.get("candidate_ranking_margin", 1.0) or 1.0),
            point_scores=out.get("point_scores"),
        )
        if "candidate_gate_logits" in out:
            gate_loss, gate_details = _point_gate_loss(
                out["candidate_gate_logits"], out["point_coords"], pt["points"], batch, scale, device
            )
            cand_details.update(gate_details)
        if "candidate_mask_logits" in out:
            selector_logits = out.get("candidate_mask_selector_logits", out["candidate_logits"])
            cm_cfg = (
                lc.get("candidate_mask_aware_score_loss", {}) if isinstance(lc.get("candidate_mask_aware_score_loss", {}), dict) else {}
            )
            thr_cfg = cm_cfg.get("target_thresholds", pc.get("candidate_mask_quality_target_thresholds", [0.3]))
            if isinstance(thr_cfg, (int, float, str)):
                thr_cfg = [float(thr_cfg)]
            cm_ce, cm_pair, cm_reg, cm_oracle, cm_details = _candidate_mask_aware_losses(
                selector_logits,
                out["candidate_mask_logits"],
                target,
                float(pc.get("candidate_mask_ranking_margin", 0.25) or 0.25),
                target_mode=str(cm_cfg.get("target_mode", pc.get("candidate_mask_quality_target_mode", "soft_iou"))),
                target_thresholds=[float(x) for x in thr_cfg],
                fa_penalty=float(
                    cm_cfg.get("target_false_alarm_penalty", pc.get("candidate_mask_quality_target_false_alarm_penalty", 0.0)) or 0.0
                ),
            )
            cand_mask_ce, cand_mask_pair, cand_mask_reg, cand_oracle_mask = cm_ce, cm_pair, cm_reg, cm_oracle
            cand_details.update(cm_details)
    dense = torch.zeros((), device=device)
    dense_details = {}
    dense_cfg = (
        lc.get("prompt_teacher_dense_objectness_distillation", {})
        if isinstance(lc.get("prompt_teacher_dense_objectness_distillation", {}), dict)
        else {}
    )
    if dense_cache is not None and float(dense_cfg.get("weight", 0) or 0) > 0:
        maps = []
        missing = []
        for s in batch:
            arr = dense_cache.get(s.dataset_key, s.sample_id)
            if arr is None:
                missing.append(f"{s.dataset_key}/{s.sample_id}")
            else:
                maps.append(arr)
        if missing:
            required = bool((cfg.get("prompt_teacher_dense_objectness_cache") or {}).get("required", True))
            if required:
                raise RuntimeError(f"Dense prompt-teacher objectness cache missing {len(missing)} samples, first={missing[0]}")
        if len(maps) == len(batch):
            dense_target = _dense_maps_to_bchw_resized(maps, out["objectness_logits"].shape[-2:], device)
            dense, dense_details = _prompt_teacher_dense_objectness_loss(out["objectness_logits"], dense_target, dense_cfg)
    foreground_recall, foreground_recall_details = _foreground_recall_loss(logits, target, lc.get("foreground_recall_loss", {}))
    calib_loss = _calibration_band_loss(logits, target, lc.get("logit_calibration_band_loss", {}))
    weighted = {
        "teacher_mask_distillation": w.get("teacher_mask_distillation", 1)
        * _schedule_multiplier(cfg, epoch, "teacher_mask_distillation")
        * tkd,
        "gt_segmentation": w.get("gt_segmentation", 0.5) * _schedule_multiplier(cfg, epoch, "gt_segmentation") * gt,
        "boundary_distillation": w.get("boundary_distillation", 0.2) * _schedule_multiplier(cfg, epoch, "boundary_distillation") * bkd,
        "false_alarm_penalty": w.get("false_alarm_penalty", 0.2) * _schedule_multiplier(cfg, epoch, "false_alarm_penalty") * fa,
        "iou_distillation": w.get("iou_distillation", 0.1) * _schedule_multiplier(cfg, epoch, "iou_distillation") * skd,
        "counterfactual_iou_calibration": w.get("counterfactual_iou_calibration", 0.0)
        * _schedule_multiplier(cfg, epoch, "counterfactual_iou_calibration")
        * counterfactual_quality,
        "prompt_objectness_distillation": w.get("prompt_objectness_distillation", 0.3)
        * _schedule_multiplier(cfg, epoch, "prompt_objectness_distillation")
        * obj,
        "prompt_box_loss": w.get("prompt_box_loss", 0.1) * _schedule_multiplier(cfg, epoch, "prompt_box_loss") * box,
        "prompt_point_box_consistency_loss": w.get("prompt_point_box_consistency_loss", 0.0)
        * _schedule_multiplier(cfg, epoch, "prompt_point_box_consistency_loss")
        * point_box,
        "prompt_point_loss": w.get("prompt_point_loss", 0.1) * _schedule_multiplier(cfg, epoch, "prompt_point_loss") * point,
        "prompt_ranking_loss": w.get("prompt_ranking_loss", 0.35) * _schedule_multiplier(cfg, epoch, "prompt_ranking_loss") * rank,
        "candidate_score_loss": w.get("candidate_score_loss", 0) * _schedule_multiplier(cfg, epoch, "candidate_score_loss") * cand_ce,
        "candidate_pairwise_ranking_loss": w.get("candidate_pairwise_ranking_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_pairwise_ranking_loss")
        * cand_pair,
        "candidate_multi_positive_loss": w.get("candidate_multi_positive_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_multi_positive_loss")
        * cand_multi,
        "candidate_coverage_loss": w.get("candidate_coverage_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_coverage_loss")
        * cand_cov,
        "candidate_objectness_loss": w.get("candidate_objectness_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_objectness_loss")
        * cand_obj,
        "candidate_mask_aware_score_loss": w.get("candidate_mask_aware_score_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_mask_aware_score_loss")
        * cand_mask_ce,
        "candidate_mask_aware_pairwise_loss": w.get("candidate_mask_aware_pairwise_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_mask_aware_pairwise_loss")
        * cand_mask_pair,
        "candidate_mask_quality_regression_loss": w.get("candidate_mask_quality_regression_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_mask_quality_regression_loss")
        * cand_mask_reg,
        "candidate_oracle_mask_response_loss": w.get("candidate_oracle_mask_response_loss", 0)
        * _schedule_multiplier(cfg, epoch, "candidate_oracle_mask_response_loss")
        * cand_oracle_mask,
        "prompt_teacher_dense_objectness_distillation": w.get("prompt_teacher_dense_objectness_distillation", 0)
        * _schedule_multiplier(cfg, epoch, "prompt_teacher_dense_objectness_distillation")
        * dense,
        "point_gate_loss": w.get("point_gate_loss", 0) * _schedule_multiplier(cfg, epoch, "point_gate_loss") * gate_loss,
        "foreground_recall_loss": w.get("foreground_recall_loss", 0)
        * _schedule_multiplier(cfg, epoch, "foreground_recall_loss")
        * foreground_recall,
        "logit_calibration_band_loss": w.get("logit_calibration_band_loss", 0)
        * _schedule_multiplier(cfg, epoch, "logit_calibration_band_loss")
        * calib_loss,
    }
    total = sum(weighted.values())
    metrics = {
        "loss": float(total.detach().cpu()),
        "teacher_mask_kd": float(tkd.detach().cpu()),
        "gt": float(gt.detach().cpu()),
        "boundary_kd": float(bkd.detach().cpu()),
        "fa": float(fa.detach().cpu()),
        "iou_kd": float(skd.detach().cpu()),
        **skd_details,
        "counterfactual_iou_calibration": float(counterfactual_quality.detach().cpu()),
        **counterfactual_quality_details,
        "prompt_objectness": float(obj.detach().cpu()),
        "prompt_box": float(box.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in box_details.items()},
        "prompt_point": float(point.detach().cpu()),
        "prompt_ranking": float(rank.detach().cpu()),
        "candidate_score_ce": float(cand_ce.detach().cpu()),
        "candidate_pairwise": float(cand_pair.detach().cpu()),
        "candidate_multi_positive": float(cand_multi.detach().cpu()),
        "candidate_coverage": float(cand_cov.detach().cpu()),
        "candidate_objectness": float(cand_obj.detach().cpu()),
        "candidate_mask_aware_score": float(cand_mask_ce.detach().cpu()),
        "candidate_mask_aware_pairwise": float(cand_mask_pair.detach().cpu()),
        "candidate_mask_quality_regression_loss": float(cand_mask_reg.detach().cpu()),
        "candidate_oracle_mask_response": float(cand_oracle_mask.detach().cpu()),
        "point_gate_loss": float(gate_loss.detach().cpu()),
        "logit_calibration_band_loss": float(calib_loss.detach().cpu()),
        "foreground_recall_loss": float(foreground_recall.detach().cpu()),
        **foreground_recall_details,
        "prompt_teacher_dense_objectness": float(dense.detach().cpu()),
        "prompt_schedule_multiplier": float(_schedule_multiplier(cfg, epoch, "prompt_objectness_distillation")),
        "dense_schedule_multiplier": float(_schedule_multiplier(cfg, epoch, "prompt_teacher_dense_objectness_distillation")),
        "calibration_response_mask_used": float(1.0 if calibration_response_used else 0.0),
        "fixed_prompt_prior_used": float(1.0 if fixed_prompt_prior is not None else 0.0),
        "candidate_count": int(out["point_coords"].shape[1]),
        "decoder_point_count_used": int(out.get("decoder_point_count_used", out["point_coords"].shape[1])),
        "local_prompt_token_count_used": int(out.get("local_prompt_token_count_used", 0)),
        "mean_top1_prompt_score": float(out.get("point_scores", torch.zeros(1, device=device))[:, 0].mean().detach().cpu())
        if "point_scores" in out
        else 0.0,
        "mean_top1_candidate_logit": float(out.get("candidate_logits", torch.zeros(1, 1, device=device))[:, 0].mean().detach().cpu())
        if "candidate_logits" in out
        else 0.0,
        "mean_top1_mask_quality_logit": float(
            out.get("candidate_mask_quality_logits", torch.zeros(1, 1, device=device))[:, 0].mean().detach().cpu()
        )
        if "candidate_mask_quality_logits" in out
        else 0.0,
        **cand_details,
        **dense_details,
        "batch_size": len(batch),
        **{f"fa_{k}": v for k, v in fd.items()},
    }
    return total, metrics


def _save(path, model, opt, scaler, epoch, step, cfg, config_path=None, protocol_audit=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    losses = cfg.get("losses", {}) if isinstance(cfg.get("losses", {}), dict) else {}
    dense_cfg = (
        losses.get("prompt_teacher_dense_objectness_distillation", {})
        if isinstance(losses.get("prompt_teacher_dense_objectness_distillation", {}), dict)
        else {}
    )
    protocol = str(
        cfg.get("checkpoint_protocol")
        or (
            "sparksam_single_checkpoint_dense_prompt_guidance_v1"
            if float(dense_cfg.get("weight", 0) or 0) > 0
            else "sparksam_single_checkpoint_v1"
        )
    )
    torch.save(
        {
            "protocol": protocol,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(step),
            "config": cfg,
            "inference_contract": {
                "input": "image",
                "output": "mask",
                "forbidden": ["prompt_teacher", "prompt_student", "SAM2 Large", "teacher_cache", "external_prompt"],
            },
        },
        path,
    )
    if config_path is not None and protocol_audit is not None:
        write_checkpoint_lineage(
            path,
            config_path=config_path,
            cfg=cfg,
            protocol_audit=protocol_audit,
            extra={
                "epoch": int(epoch),
                "global_step": int(step),
                "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            },
        )


def _load_resume(path, model, opt, scaler, device, load_optimizer: bool = True, strict: bool = True):
    p = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(p["model_state_dict"], strict=bool(strict))
    if _is_rank0() and (missing or unexpected):
        print(
            json.dumps(
                {"resume_load_strict": bool(strict), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}, ensure_ascii=False
            ),
            flush=True,
        )
    if not load_optimizer:
        return 1, 0
    opt.load_state_dict(p.get("optimizer_state_dict", {}))
    if p.get("scaler_state_dict"):
        scaler.load_state_dict(p["scaler_state_dict"])
    return int(p.get("epoch", 0)) + 1, int(p.get("global_step", 0))


def _listify(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def _matches_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _load_prompt_prior_from_config(cfg, model, device):
    spec = cfg.get("prompt_prior", {}) if isinstance(cfg.get("prompt_prior", {}), dict) else {}
    ckpt = spec.get("checkpoint") or spec.get("path")
    if not ckpt:
        return {}
    prefixes = _listify(spec.get("load_modules"), ["prompt_head", "local_prompt_projector"])
    path = Path(str(ckpt))
    payload = torch.load(path, map_location=device)
    source = payload.get("model_state_dict", payload)
    if not isinstance(source, dict):
        raise RuntimeError(f"Prompt prior checkpoint has no state dict: {path}")
    current = model.state_dict()
    selected = {}
    skipped = []
    for name, tensor in source.items():
        if not _matches_prefix(str(name), prefixes):
            continue
        if name not in current:
            skipped.append({"key": str(name), "reason": "missing_in_target"})
            continue
        if tuple(current[name].shape) != tuple(tensor.shape):
            skipped.append(
                {
                    "key": str(name),
                    "reason": "shape_mismatch",
                    "source_shape": list(tensor.shape),
                    "target_shape": list(current[name].shape),
                }
            )
            continue
        selected[name] = tensor
    if not selected and bool(spec.get("required", True)):
        raise RuntimeError(f"Prompt prior loaded zero tensors from {path} with prefixes={prefixes}")
    merged = dict(current)
    merged.update(selected)
    model.load_state_dict(merged, strict=True)
    summary = {
        "checkpoint": str(path),
        "load_modules": prefixes,
        "matched_key_count": len(selected),
        "skipped_key_count": len(skipped),
        "matched_keys": sorted(selected.keys()),
        "skipped_keys": skipped[:50],
    }
    if _is_rank0():
        print(json.dumps({"prompt_prior_init": summary}, ensure_ascii=False), flush=True)
    return summary


def train(args):
    config_path = Path(args.config).resolve()
    cfg = _read_yaml(config_path)
    protocol_audit = audit_spark_training_config(config_path, cfg)
    strict_clean = reproduction_protocol_enabled(cfg)
    if strict_clean and args.resume_from is not None:
        raise RuntimeError("Clean protocol forbids --resume-from; use initialization.selection_lock or same-stage --resume auto.")
    if strict_clean and args.max_steps:
        raise RuntimeError("Clean paper runs forbid --max-steps; use a separate non-paper diagnostic config.")
    clean_train_dir = Path(str(cfg["artifact_root"])) / "train"
    if strict_clean and args.resume == "none" and (clean_train_dir / "checkpoint_last_sparksam.pt").exists():
        raise RuntimeError("Refusing to overwrite a clean run checkpoint; use --resume auto or a new artifact root.")
    if args.preflight_only:
        print(json.dumps(protocol_audit, ensure_ascii=False, indent=2), flush=True)
        return 0
    launched = _maybe_launch(args, cfg)
    if launched is not None:
        return int(launched)
    source = _read_yaml(Path(str(cfg["source_config"])))
    repo = _add_sam2_repo_to_path(source)
    device = _setup_distributed()
    train_cfg0 = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    base_seed = int(train_cfg0.get("seed", 42) or 42)
    seed = base_seed + _rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from sam2.utils.transforms import SAM2Transforms

    root = Path(str(cfg["artifact_root"]))
    train_dir = root / "train"
    samples = _load_samples_for_role(cfg, source, "train")
    debug_overfit = cfg.get("debug_overfit", {}) if isinstance(cfg.get("debug_overfit", {}), dict) else {}
    random.Random(int(debug_overfit.get("sample_seed", base_seed) or base_seed)).shuffle(samples)
    max_train_samples = int(debug_overfit.get("max_train_samples", 0) or 0)
    if max_train_samples > 0:
        samples = samples[:max_train_samples]
    full_train_sample_count = len(samples)
    samples = _shard_samples(samples)
    bs = max(1, int(cfg.get("train", {}).get("batch_size", 4) or 4))
    if args.max_steps:
        samples = samples[: max(1, args.max_steps * bs)]
    teacher_cache_spec = cfg.get("teacher_cache")
    expected_cache_splits = ["train"]
    cache = (
        load_teacher_cache(
            teacher_cache_spec,
            expected_datasets=[str(x) for x in cfg.get("datasets", {}).get("train", [])],
            expected_prompt_modes=["box_point"],
            expected_splits=expected_cache_splits,
        )
        if isinstance(teacher_cache_spec, dict)
        else None
    )
    dense_cache = _load_prompt_teacher_dense_cache(cfg)
    calibration_response_cache = _load_calibration_response_mask_cache(cfg)
    fixed_prompt_prior = _load_fixed_prompt_prior_from_config(cfg)
    sam = _build_sam2_model(cfg["student"]["model"], cfg["student"]["checkpoint"], source, device, train=True)
    hc = cfg.get("prompt_head", {}) if isinstance(cfg.get("prompt_head"), dict) else {}
    model = SPARKSAM(
        sam,
        token_count=int(hc.get("learned_sparse_prompt_tokens", 2) or 2),
        min_box_side=float(hc.get("min_box_side", 2.0) or 2.0),
        max_box_fraction=float(hc.get("max_box_fraction", 0.05) or 0.05),
        temperature=float(hc.get("soft_argmax_temperature", 0.05) or 0.05),
        candidate_count=int(hc.get("candidate_count", hc.get("point_budget", 1)) or 1),
        decoder_point_count=int(hc.get("decoder_point_count", 0) or 0),
        candidate_mask_count=int(hc.get("candidate_mask_count", 0) or 0),
        decoder_mode=str(hc.get("decoder_mode", "top_points") or "top_points"),
        mask_select_temperature=float(hc.get("mask_select_temperature", 1.0) or 1.0),
        local_prompt_token_count=int(hc.get("local_prompt_token_count", 0) or 0),
        mask_selector_score_source=str(hc.get("mask_selector_score_source", "candidate_logits") or "candidate_logits"),
        mask_selector_sam_iou_weight=float(hc.get("mask_selector_sam_iou_weight", 1.0) or 1.0),
        prompt_gate_enabled=bool(hc.get("prompt_gate_enabled", False)),
        prompt_gate_strength=float(hc.get("prompt_gate_strength", 0.5) or 0.5),
        box_parameterization=str(hc.get("box_parameterization", "point_centered_width_height") or "point_centered_width_height"),
        box_extent_init_fraction=float(hc.get("box_extent_init_fraction", 0.05) or 0.05),
        box_edge_temperature=float(hc.get("box_edge_temperature", 0.1) or 0.1),
        box_edge_decode_mode=str(hc.get("box_edge_decode_mode", "expectation") or "expectation"),
        box_edge_outer_quantile=float(hc.get("box_edge_outer_quantile", 0.1) or 0.1),
        box_occupancy_threshold=float(hc.get("box_occupancy_threshold", 0.5) or 0.5),
        point_offset_enabled=bool(hc.get("point_offset_enabled", False)),
        point_offset_max_cell_fraction=float(hc.get("point_offset_max_cell_fraction", 0.5) or 0.5),
        prompt_box_scale=float(hc.get("prompt_box_scale", 1.0) or 1.0),
        logit_calibration_enabled=bool(hc.get("logit_calibration_enabled", False)),
        logit_calibration_max_abs_bias=float(hc.get("logit_calibration_max_abs_bias", 2.0) or 2.0),
        dense_mask_refinement_enabled=bool(hc.get("dense_mask_refinement_enabled", False)),
        dense_mask_refinement_hidden_dim=int(hc.get("dense_mask_refinement_hidden_dim", 32) or 32),
        dense_mask_refinement_max_abs_residual=float(hc.get("dense_mask_refinement_max_abs_residual", 2.0) or 2.0),
        highres_prompt_refinement_enabled=bool(hc.get("highres_prompt_refinement_enabled", False)),
        highres_prompt_refinement_hidden_dim=int(hc.get("highres_prompt_refinement_hidden_dim", 32) or 32),
        highres_prompt_refinement_scale=float(hc.get("highres_prompt_refinement_scale", 1.0) or 1.0),
        highres_prompt_recenter_box=bool(hc.get("highres_prompt_recenter_box", True)),
        highres_prompt_candidate_mode=str(hc.get("highres_prompt_candidate_mode", "replace") or "replace"),
        highres_prompt_base_candidate_count=int(hc.get("highres_prompt_base_candidate_count", 4) or 0),
    ).to(device)
    prompt_prior_init = _load_prompt_prior_from_config(cfg, model, device)
    initialization_checkpoint, initialization_summary = resolve_initialization_checkpoint(cfg, config_path=config_path)
    if initialization_checkpoint is not None:
        init_payload = torch.load(initialization_checkpoint, map_location=device)
        init_state = init_payload.get("model_state_dict", init_payload)
        init_cfg = cfg.get("initialization", {}) if isinstance(cfg.get("initialization"), dict) else {}
        strict_init = bool(init_cfg.get("strict", True))
        allowed_missing_prefixes = tuple(str(value) for value in init_cfg.get("allowed_missing_key_prefixes", []))
        if strict_init and allowed_missing_prefixes:
            missing, unexpected = model.load_state_dict(init_state, strict=False)
            invalid_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "Strict initialization failed outside explicitly allowed new-module keys: "
                    f"missing={invalid_missing}, unexpected={list(unexpected)}"
                )
        else:
            missing, unexpected = model.load_state_dict(init_state, strict=strict_init)
        initialization_summary = {**initialization_summary, "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        if _is_rank0():
            print(json.dumps({"clean_initialization": initialization_summary}, ensure_ascii=False), flush=True)
    tc = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    lr = float(tc.get("learning_rate", 1e-5) or 1e-5)
    enc_lr = lr * float(tc.get("image_encoder_lr_multiplier", 0.1) or 0.1)
    resolved_module_policy = _apply_module_policy(model, tc)
    prompt_head_prefix_policy = _apply_prompt_head_trainable_prefixes(model, tc)
    freeze = set()
    if resolved_module_policy is None:
        freeze = set(str(x) for x in (tc.get("freeze_modules", []) or tc.get("freeze", []) or []))
        if "image_encoder" in freeze:
            _set_trainable(model.sam2_model.image_encoder, False)
        if "prompt_encoder" in freeze or "sam_prompt_encoder" in freeze:
            _set_trainable(model.sam2_model.sam_prompt_encoder, False)
        if "mask_decoder" in freeze or "sam_mask_decoder" in freeze:
            _set_trainable(model.sam2_model.sam_mask_decoder, False)
        if "prompt_head" in freeze:
            _set_trainable(model.prompt_head, False)
        if "prompt_head_except_gate" in freeze:
            _set_trainable(model.prompt_head, False)
            if hasattr(model.prompt_head, "candidate_gate"):
                _set_trainable(model.prompt_head.candidate_gate, True)
        if hasattr(model, "local_prompt_projector") and "local_prompt_projector" in freeze:
            _set_trainable(model.local_prompt_projector, False)
        if hasattr(model, "logit_calibration_head") and "logit_calibration_head" in freeze:
            _set_trainable(model.logit_calibration_head, False)
        if hasattr(model, "dense_mask_refinement_head") and "dense_mask_refinement_head" in freeze:
            _set_trainable(model.dense_mask_refinement_head, False)
        if hasattr(model, "highres_prompt_refinement_head") and "highres_prompt_refinement_head" in freeze:
            _set_trainable(model.highres_prompt_refinement_head, False)
    else:
        freeze = {name for name, state in resolved_module_policy.items() if state == "frozen"}
    param_groups = []
    image_params = _trainable_params(model.sam2_model.image_encoder)
    if image_params:
        param_groups.append({"params": image_params, "lr": enc_lr})
    prompt_enc_params = _trainable_params(model.sam2_model.sam_prompt_encoder)
    if prompt_enc_params:
        param_groups.append({"params": prompt_enc_params, "lr": lr * float(tc.get("prompt_encoder_lr_multiplier", 1.0) or 1.0)})
    mask_decoder = model.sam2_model.sam_mask_decoder
    mask_decoder_lr = lr * float(tc.get("mask_decoder_lr_multiplier", 1.0) or 1.0)
    quality_lr_multiplier = float(tc.get("mask_quality_head_lr_multiplier", 1.0) or 1.0)
    quality_ids = {
        id(parameter)
        for name, parameter in mask_decoder.named_parameters()
        if parameter.requires_grad and (name == "iou_prediction_head" or name.startswith("iou_prediction_head."))
    }
    mask_dec_params = [
        parameter for parameter in mask_decoder.parameters() if parameter.requires_grad and id(parameter) not in quality_ids
    ]
    quality_params = [
        parameter for parameter in mask_decoder.parameters() if parameter.requires_grad and id(parameter) in quality_ids
    ]
    if mask_dec_params:
        param_groups.append({"params": mask_dec_params, "lr": mask_decoder_lr})
    if quality_params:
        param_groups.append({"params": quality_params, "lr": mask_decoder_lr * quality_lr_multiplier})
    prompt_head_params = _trainable_params(model.prompt_head)
    if prompt_head_params:
        param_groups.append({"params": prompt_head_params, "lr": lr * float(tc.get("prompt_head_lr_multiplier", 1.0) or 1.0)})
    if hasattr(model, "local_prompt_projector"):
        local_params = _trainable_params(model.local_prompt_projector)
        if local_params:
            param_groups.append({"params": local_params, "lr": lr * float(tc.get("local_prompt_lr_multiplier", 1.0) or 1.0)})
    if hasattr(model, "logit_calibration_head"):
        calib_params = _trainable_params(model.logit_calibration_head)
        if calib_params:
            param_groups.append({"params": calib_params, "lr": lr * float(tc.get("logit_calibration_lr_multiplier", 1.0) or 1.0)})
    if hasattr(model, "dense_mask_refinement_head"):
        refine_params = _trainable_params(model.dense_mask_refinement_head)
        if refine_params:
            param_groups.append({"params": refine_params, "lr": lr * float(tc.get("dense_mask_refinement_lr_multiplier", 1.0) or 1.0)})
    if hasattr(model, "highres_prompt_refinement_head"):
        highres_prompt_params = _trainable_params(model.highres_prompt_refinement_head)
        if highres_prompt_params:
            param_groups.append(
                {
                    "params": highres_prompt_params,
                    "lr": lr * float(tc.get("highres_prompt_refinement_lr_multiplier", 1.0) or 1.0),
                }
            )
    if not param_groups:
        raise RuntimeError("SPARK-SAM train config froze all trainable parameter groups.")
    opt = torch.optim.AdamW(param_groups, weight_decay=float(tc.get("weight_decay", 0.01) or 0.01))
    amp = bool(tc.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch = 1
    step = 0
    resume = args.resume_from or (
        (train_dir / "checkpoint_last_sparksam.pt") if args.resume == "auto" and (train_dir / "checkpoint_last_sparksam.pt").exists() else None
    )
    if resume:
        if strict_clean:
            resume_lineage = read_lineage(Path(resume), required=True, expected_split_manifest=cfg["reproduction_protocol"]["split_manifest"])
            if str((resume_lineage.get("config", {}) or {}).get("sha256", "")) != sha256_file(config_path):
                raise RuntimeError("Clean resume checkpoint was produced by a different config.")
            if str((resume_lineage.get("code", {}) or {}).get("revision", "")) != str(git_record().get("revision", "")):
                raise RuntimeError("Clean resume checkpoint was produced by a different code revision.")
            if str(resume_lineage.get("stage", "")) != str(cfg["reproduction_protocol"].get("stage", "")) or str(
                resume_lineage.get("ablation", "")
            ) != str(cfg["reproduction_protocol"].get("ablation", "")):
                raise RuntimeError("Clean resume checkpoint belongs to a different stage or ablation.")
        start_epoch, step = _load_resume(
            Path(resume),
            model,
            opt,
            scaler,
            device,
            load_optimizer=bool(tc.get("resume_optimizer", True)),
            strict=bool(tc.get("resume_strict", True)),
        )
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
    accum = max(1, int(tc.get("gradient_accumulation_steps", 1) or 1))
    epochs = int(tc.get("epochs", 60) or 60)
    interval = max(1, int(tc.get("checkpoint_interval_epochs", 5) or 5))
    if _is_rank0():
        _write_yaml(train_dir / "sparksam_train_config.yaml", cfg)
        _write_json(train_dir / "reproduction_protocol_audit.json", protocol_audit)
        _write_json(
            train_dir / "runtime_start.json",
            {
                "time": time.time(),
                "world_size": _world_size(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "batch_size_per_rank": bs,
                "effective_batch_per_step": bs * max(1, _world_size()) * accum,
                "amp": amp,
                "freeze_modules": sorted(freeze),
                "module_policy": resolved_module_policy,
                "prompt_head_trainable_prefixes": prompt_head_prefix_policy,
                "initialization": initialization_summary,
                "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "optimizer_lrs": [float(g["lr"]) for g in param_groups],
                "sam2_repo": str(repo),
                "teacher_cache_audit": (cache.audit if cache is not None else None),
                "teacher_cache_loaded": bool(cache is not None),
                "prompt_teacher_dense_objectness_cache": (str(dense_cache.manifest_path) if dense_cache is not None else ""),
                "calibration_response_mask_cache": (str(calibration_response_cache.manifest_path) if calibration_response_cache is not None else ""),
                "fixed_prompt_prior": (fixed_prompt_prior.summary() if fixed_prompt_prior is not None else None),
                "prompt_teacher_dense_aux": (
                    "enabled_dense_prompt_teacher_objectness_v1_1"
                    if dense_cache is not None
                    else "disabled_no_dense_prompt_teacher_objectness_cache_in_sparksam_v1"
                ),
                "debug_overfit": debug_overfit,
                "full_train_sample_count": int(full_train_sample_count),
                "rank_train_sample_count": int(len(samples)),
            },
        )
        if prompt_prior_init:
            _write_json(train_dir / "prompt_prior_init.json", prompt_prior_init)
    try:
        opt.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, epochs + 1):
            batches = _batchify(samples, bs)
            random.Random(seed + epoch).shuffle(batches)
            epoch_loss = 0.0
            seen = 0
            for i, batch in enumerate(batches, 1):
                with torch.amp.autocast("cuda", enabled=amp):
                    loss, metrics = _train_one_batch(
                        model,
                        batch,
                        cache,
                        transforms,
                        device,
                        cfg,
                        dense_cache=dense_cache,
                        calibration_response_cache=calibration_response_cache,
                        fixed_prompt_prior=fixed_prompt_prior,
                        epoch=epoch,
                    )
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(f"Non-finite SPARK-SAM loss epoch={epoch} batch={i}: {metrics}")
                scaler.scale(loss / accum).backward()
                if i % accum == 0 or i == len(batches):
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.get("grad_clip_norm", 1.0) or 1.0))
                    _average_gradients(model)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                else:
                    grad_norm = torch.tensor(0.0)
                epoch_loss += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
                if _is_rank0() and (i == 1 or i % 10 == 0):
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "rank": _rank(),
                                "batch_index": i,
                                "batches": len(batches),
                                "global_step": step,
                                "grad_norm": float(grad_norm),
                                **metrics,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if args.max_steps and i >= args.max_steps:
                    break
            if _is_rank0():
                print(
                    json.dumps({"epoch": epoch, "mean_loss": epoch_loss / max(1, seen), "global_step": step}, ensure_ascii=False),
                    flush=True,
                )
                _save(train_dir / "checkpoint_last_sparksam.pt", model, opt, scaler, epoch, step, cfg, config_path, protocol_audit)
                if epoch % interval == 0 or epoch == epochs:
                    _save(train_dir / f"checkpoint_epoch_{epoch}_sparksam.pt", model, opt, scaler, epoch, step, cfg, config_path, protocol_audit)
                    if not reproduction_protocol_enabled(cfg):
                        shutil.copy2(train_dir / "checkpoint_last_sparksam.pt", train_dir / "checkpoint_selected_sparksam.pt")
            if args.max_steps:
                break
    finally:
        _cleanup_distributed()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--nproc-per-node", type=int, default=0)
    ap.add_argument("--resume", choices=["auto", "none"], default="auto")
    ap.add_argument("--resume-from", type=Path)
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
