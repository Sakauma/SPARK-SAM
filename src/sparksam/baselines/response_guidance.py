from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import BenchmarkMethodProtocol, InferenceMode
from ..data.auto_prompt import HEURISTIC_IR_AUTO_PROMPT_PROTOCOL, generate_heuristic_ir_auto_prompt_from_path
from ..data.masks import sample_mask_array
from ..data.prompt_synthesis import (
    MASK_DERIVED_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL,
    clamp_box_xyxy,
)
from ..data.sample import Sample
from ..evaluation.image_metrics import mask_iou
from ..models import (
    LEARNED_IR_AUTO_PROMPT_PROTOCOL,
    SAM2ModelAdapter,
    apply_mask_feedback,
    box_calibration_metadata,
    calibrate_box_from_results,
    candidate_metadata,
    feedback_metadata,
    load_auto_prompt_model,
    load_image_rgb,
    make_scaled_boxes,
    predict_learned_auto_prompt_from_path,
    prompt_reranker_config_from_dict,
    rank_prompt_candidates,
)
from ..evaluation.heatmaps import write_heatmap_artifact


def box_to_mask(box: list[float], height: int, width: int) -> np.ndarray:
    # BBox baseline 把 prompt box 直接当作预测 mask，用来衡量“框形状本身”的上限/偏差。
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box_xyxy(box, width, height)]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class BaseMethod:
    # 所有 baseline 都暴露统一接口：单样本和批量样本。
    # runner 会根据 inference_mode 选择对应评估协议。
    name = "base"
    family = "base"
    inference_mode = InferenceMode.PROMPTED

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        return {sample.sample_id: self.predict_sample(sample) for sample in samples}


def _predict_prompted_sam2_samples(
    samples: List[Sample],
    *,
    adapter: SAM2ModelAdapter,
    inference_mode: InferenceMode,
    build_prompt: Callable[[Sample], tuple[Dict[str, Any], Dict[str, Any]]],
    build_prediction: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not samples:
        return {}
    kwargs_and_prompts = [build_prompt(sample) for sample in samples]
    kwargs_list = [item[0] for item in kwargs_and_prompts]
    prompts = [item[1] for item in kwargs_and_prompts]
    image_paths = [str(sample.image_path) for sample in samples]

    def predict_group(indices: List[int]) -> Dict[str, Dict[str, Any]]:
        group_samples = [samples[idx] for idx in indices]
        group_kwargs = [kwargs_list[idx] for idx in indices]
        group_prompts = [prompts[idx] for idx in indices]
        image_rgb = load_image_rgb(group_samples[0].image_path)
        results = adapter.predict_prompts_for_image(
            image_rgb,
            boxes=[kwargs.get("box") for kwargs in group_kwargs],
            points=[kwargs.get("points") for kwargs in group_kwargs],
            point_labels=[kwargs.get("point_labels") for kwargs in group_kwargs],
            multimask_output=inference_mode == InferenceMode.MULTI_MASK,
        )
        if len(results) != len(group_samples):
            raise RuntimeError(f"Prompt prediction returned {len(results)} results for {len(group_samples)} samples.")
        return {
            sample.sample_id: build_prediction(result, prompt)
            for sample, result, prompt in zip(group_samples, results, group_prompts)
        }

    if len(set(image_paths)) == len(image_paths):
        images = [load_image_rgb(sample.image_path) for sample in samples]
        results = adapter.predict_images(
            images,
            boxes=[kwargs.get("box") for kwargs in kwargs_list],
            points=[kwargs.get("points") for kwargs in kwargs_list],
            point_labels=[kwargs.get("point_labels") for kwargs in kwargs_list],
            multimask_output=inference_mode == InferenceMode.MULTI_MASK,
        )
        if len(results) != len(samples):
            raise RuntimeError(f"Batch prediction returned {len(results)} results for {len(samples)} samples.")
        return {
            sample.sample_id: build_prediction(result, prompt)
            for sample, result, prompt in zip(samples, results, prompts)
        }

    image_groups: Dict[str, List[int]] = {}
    for idx, image_path in enumerate(image_paths):
        image_groups.setdefault(image_path, []).append(idx)

    predictions: Dict[str, Dict[str, Any]] = {}
    for indices in image_groups.values():
        predictions.update(predict_group(indices))
    return predictions


class BBoxRectMaskBaseline(BaseMethod):
    name = "BBoxRectMaskBaseline"
    family = "baseline"
    inference_mode = InferenceMode.BOX

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        box = sample.bbox_loose or sample.bbox_tight or [0.0, 0.0, 1.0, 1.0]
        return {"mask": box_to_mask(box, sample.height, sample.width), "score": 1.0}


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"External prediction frame_id must be a safe relative path, got {value!r}.")
    return path


def _resolve_prediction_root(config: AppConfig, raw_root: str) -> Path:
    env_root = os.environ.get("EXTERNAL_PREDICTION_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    path = Path(raw_root).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        config.root / path,
        config.root.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_existing_path(config: AppConfig, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        config.config_path.parent / path,
        config.root / path,
        config.root.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_output_path(config: AppConfig, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (config.root / path).resolve()


class ExternalPredictionMaskBaseline(BaseMethod):
    name = "ExternalPredictionMaskBaseline"
    family = "external_prediction"
    inference_mode = InferenceMode.SINGLE_MASK

    def __init__(self, config: AppConfig):
        self._configuration_error: str | None = None
        raw_root = config.method.get("prediction_root")
        if not raw_root:
            self._configuration_error = "external_prediction_mask requires method.prediction_root."
            self.prediction_root = Path()
        else:
            self.prediction_root = _resolve_prediction_root(config, str(raw_root))
        self.dataset_id = str(config.method.get("prediction_dataset_id") or config.dataset.dataset_id)
        self.prediction_dataset_dir = self._prediction_dataset_dir_from_config(config)
        self.prediction_suffix = str(config.method.get("prediction_suffix", ".png"))
        if not self.prediction_suffix.startswith("."):
            self.prediction_suffix = f".{self.prediction_suffix}"
        self.threshold = float(config.method.get("prediction_threshold", 0.5))
        self.model_name = str(config.method.get("external_model_name") or config.method.get("name") or "external_prediction")
        if self._configuration_error:
            self._latency_by_frame_id: Dict[str, float] = {}
            self._latency_by_sample_id: Dict[str, float] = {}
            self._prediction_by_sample_id: Dict[str, Path] = {}
        else:
            self._latency_by_frame_id, self._latency_by_sample_id, self._prediction_by_sample_id = self._load_prediction_manifest()

    @property
    def _dataset_dir(self) -> Path:
        return self.prediction_root / self.prediction_dataset_dir

    def _prediction_dataset_dir_from_config(self, config: AppConfig) -> Path:
        raw_dir = config.method.get("prediction_dataset_dir")
        raw_mapping = config.method.get("prediction_dataset_dir_by_dataset_id")
        if raw_dir is None and isinstance(raw_mapping, dict):
            raw_dir = raw_mapping.get(self.dataset_id) or raw_mapping.get(str(config.dataset.dataset_id))
        value = self.dataset_id if raw_dir is None else str(raw_dir)
        return _safe_relative_path(value)

    def _resolve_manifest_prediction_path(self, value: object) -> Path | None:
        if value is None:
            return None
        raw = Path(str(value))
        if raw.is_absolute():
            return raw
        candidate = self._dataset_dir / raw
        if candidate.exists():
            return candidate
        return (Path.cwd() / raw).resolve()

    def _load_prediction_manifest(self) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Path]]:
        manifest_path = self._dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            return {}, {}, {}
        latency_by_frame_id: Dict[str, float] = {}
        latency_by_sample_id: Dict[str, float] = {}
        prediction_by_sample_id: Dict[str, Path] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                frame_id = record.get("frame_id")
                sample_id = record.get("sample_id")
                latency_ms = record.get("latency_ms")
                if frame_id is not None and latency_ms is not None:
                    latency_by_frame_id[str(frame_id)] = float(latency_ms)
                if sample_id is not None:
                    if latency_ms is not None:
                        latency_by_sample_id[str(sample_id)] = float(latency_ms)
                    prediction_path = self._resolve_manifest_prediction_path(record.get("prediction_path"))
                    if prediction_path is not None:
                        prediction_by_sample_id[str(sample_id)] = prediction_path
        return latency_by_frame_id, latency_by_sample_id, prediction_by_sample_id

    def _prediction_path(self, frame_id: str, sample_id: str | None = None) -> Path:
        if sample_id is not None:
            manifest_path = self._prediction_by_sample_id.get(sample_id)
            if manifest_path is not None and manifest_path.exists():
                return manifest_path
        rel = _safe_relative_path(frame_id)
        appended = self._dataset_dir / Path(f"{rel.as_posix()}{self.prediction_suffix}")
        replaced = self._dataset_dir / rel.with_suffix(self.prediction_suffix)
        candidates = [appended]
        if replaced != appended:
            candidates.append(replaced)
        if rel.suffix:
            candidates.append(self._dataset_dir / rel)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        candidate_list = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"External prediction mask not found for frame_id={frame_id!r}. Tried: {candidate_list}")

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        if self._configuration_error:
            raise ValueError(self._configuration_error)
        prediction_path = self._prediction_path(sample.frame_id, sample.sample_id)
        with Image.open(prediction_path) as image:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
        if arr.size and float(arr.max()) > 1.0:
            arr = arr / 255.0
        mask = (arr > self.threshold).astype(np.float32)
        payload: Dict[str, Any] = {
            "mask": mask,
            "score": 1.0,
            "prompt": {
                "protocol": "external_prediction_import_v1",
                "source": "external_prediction",
                "box_variant": "none",
            },
            "metadata": {
                "ExternalPredictionModel": self.model_name,
                "ExternalPredictionDatasetId": self.dataset_id,
                "ExternalPredictionDatasetDir": self.prediction_dataset_dir.as_posix(),
                "ExternalPredictionPath": str(prediction_path),
                "ExternalPredictionRoot": str(self.prediction_root),
                "ExternalPredictionThreshold": self.threshold,
            },
        }
        latency_ms = self._latency_by_sample_id.get(sample.sample_id, self._latency_by_frame_id.get(sample.frame_id))
        if latency_ms is not None:
            payload["LatencyMs"] = latency_ms
        return payload


class PretrainedPromptedSAM2(BaseMethod):
    name = "PretrainedPromptedSAM2"
    family = "sam2"

    def __init__(self, adapter: SAM2ModelAdapter, prompt_mode: InferenceMode, box_variant: str = "loose"):
        self.adapter = adapter
        self.inference_mode = prompt_mode
        self.box_variant = box_variant

    def _sample_box(self, sample: Sample) -> list[float] | None:
        # 默认使用 loose box；tight box 仅用于 protocol ablation。
        if self.box_variant == "tight":
            return sample.bbox_tight or sample.bbox_loose
        return sample.bbox_loose or sample.bbox_tight

    def _prompt_payload(self, sample: Sample, *, box: list[float] | None = None, point: list[float] | None = None) -> Dict[str, Any]:
        # prompt 元数据会写入每行结果，保证论文表能追溯 box/point 是如何从 GT mask 派生的。
        prompt_generation = dict(sample.metadata.get("prompt_generation", {}))
        if box is not None and point is not None:
            protocol = (
                MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL
                if self.box_variant == "tight"
                else MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL
            )
        elif box is not None:
            protocol = (
                MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL
                if self.box_variant == "tight"
                else MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL
            )
        elif point is not None:
            protocol = MASK_DERIVED_CENTROID_POINT_PROTOCOL
        else:
            protocol = prompt_generation.get("protocol", "unknown")
        payload: Dict[str, Any] = {
            **prompt_generation,
            "protocol": protocol,
            "box_variant": self.box_variant if box is not None else None,
            "box": box,
            "point": point,
        }
        return payload

    def _predict_kwargs_and_prompt(self, sample: Sample) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # pretrained prompted SAM2 按 inference_mode 传入 box、point 或 box+point；它不是 no-prompt。
        kwargs: Dict[str, Any] = {"multimask_output": self.inference_mode == InferenceMode.MULTI_MASK}
        box = self._sample_box(sample)
        point = sample.point_prompt
        if self.inference_mode == InferenceMode.BOX:
            kwargs["box"] = box
            prompt = self._prompt_payload(sample, box=box)
        elif self.inference_mode == InferenceMode.POINT:
            kwargs["points"] = np.array([point], dtype=np.float32) if point else None
            kwargs["point_labels"] = np.array([1], dtype=np.int32) if point else None
            prompt = self._prompt_payload(sample, point=point)
        elif self.inference_mode == InferenceMode.BOX_POINT:
            kwargs["box"] = box
            kwargs["points"] = np.array([point], dtype=np.float32) if point else None
            kwargs["point_labels"] = np.array([1], dtype=np.int32) if point else None
            prompt = self._prompt_payload(sample, box=box, point=point)
        elif self.inference_mode in {InferenceMode.SINGLE_MASK, InferenceMode.MULTI_MASK}:
            kwargs["box"] = box
            prompt = self._prompt_payload(sample, box=box)
        else:
            raise ValueError(f"Unsupported pretrained prompted SAM2 mode: {self.inference_mode.value}")
        return kwargs, prompt

    def _prediction_from_result(self, result: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
        # SAM2 可能返回多个候选 mask；论文主表取分数最高的候选，保持确定性。
        masks = result["masks"]
        scores = result["scores"]
        best_idx = int(np.argmax(scores))
        return {"mask": masks[best_idx].astype(np.float32), "score": float(scores[best_idx]), "prompt": prompt}

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        image_rgb = load_image_rgb(sample.image_path)
        kwargs, prompt = self._predict_kwargs_and_prompt(sample)
        result = self.adapter.predict_image(image_rgb, **kwargs)
        return self._prediction_from_result(result, prompt)

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        # 同一图像内的多实例 prompt 共享一次 SAM2 image embedding。
        return _predict_prompted_sam2_samples(
            samples,
            adapter=self.adapter,
            inference_mode=self.inference_mode,
            build_prompt=self._predict_kwargs_and_prompt,
            build_prediction=self._prediction_from_result,
        )


class NoPromptAutoMaskSAM2(BaseMethod):
    name = "NoPromptAutoMaskSAM2"
    family = "sam2"
    inference_mode = InferenceMode.NO_PROMPT_AUTO_MASK

    def __init__(self, adapter: SAM2ModelAdapter):
        self.adapter = adapter

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        # 自动掩码模式不接受外部 prompt；runner 会按 image-level 与 GT instances 做匹配。
        image_rgb = load_image_rgb(sample.image_path)
        raw_masks = self.adapter.predict_auto_masks(image_rgb)
        instances = []
        for idx, item in enumerate(raw_masks):
            segmentation = item.get("segmentation")
            if segmentation is None:
                continue
            mask = np.asarray(segmentation, dtype=np.float32)
            instances.append(
                {
                    "instance_id": f"{sample.frame_id}::pred::{idx}",
                    "mask": mask,
                    "score": float(item.get("predicted_iou", item.get("stability_score", 1.0))),
                }
            )
        runtime = getattr(getattr(self.adapter, "config", None), "runtime", None)
        generator_kwargs = dict(getattr(self.adapter, "auto_mask_generator_kwargs", {}))
        points_per_batch = int(generator_kwargs.get("points_per_batch", getattr(runtime, "auto_mask_points_per_batch", 64)))
        return {
            "instances": instances,
            "auto_mask_points_per_batch": points_per_batch,
            "auto_mask_generator": generator_kwargs,
        }


class HeuristicAutoPromptedSAM2(BaseMethod):
    name = "HeuristicAutoPromptedSAM2"
    family = "sam2"

    def __init__(
        self,
        adapter: SAM2ModelAdapter,
        config: AppConfig | None = None,
        prompt_mode: InferenceMode = InferenceMode.POINT,
        *,
        use_negative_ring: bool = False,
        top_k: int = 1,
        protocol: str = HEURISTIC_IR_AUTO_PROMPT_PROTOCOL,
        point_rule: str = "highest_scoring_ir_response_peak",
        box_rule: str = "connected_response_component_around_primary_peak",
    ):
        self.adapter = adapter
        self.inference_mode = prompt_mode
        self.use_negative_ring = use_negative_ring
        method_config = getattr(config, "method", {}) if config is not None else {}
        self.top_k = int(method_config.get("prompt_top_k", top_k))
        self.nms_radius = int(method_config.get("prompt_nms_radius", method_config.get("prompt_candidate_nms_radius", 8)))
        self.response_threshold = float(method_config.get("prompt_response_threshold", 0.1))
        self.box_threshold = float(method_config.get("prompt_box_threshold", 0.55))
        self.fallback_half_side = int(method_config.get("prompt_fallback_half_side", 4))
        self.protocol = str(method_config.get("prompt_generation_protocol", protocol))
        self.point_rule = str(method_config.get("point_rule", point_rule))
        self.box_rule = str(method_config.get("box_rule", box_rule))

    def _auto_prompt(self, sample: Sample) -> Dict[str, Any]:
        auto_prompt = generate_heuristic_ir_auto_prompt_from_path(
            sample.image_path,
            top_k=self.top_k,
            nms_radius=self.nms_radius,
            response_threshold=self.response_threshold,
            box_threshold=self.box_threshold,
            fallback_half_side=self.fallback_half_side,
            negative_ring=self.use_negative_ring,
        )
        prompt = {
            **auto_prompt.metadata,
            "protocol": self.protocol,
            "box_variant": "heuristic",
            "point_rule": self.point_rule,
            "box_rule": self.box_rule,
        }
        return prompt

    def _predict_kwargs_and_prompt(self, sample: Sample) -> tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._auto_prompt(sample)
        kwargs: Dict[str, Any] = {"multimask_output": self.inference_mode == InferenceMode.MULTI_MASK}
        if self.inference_mode == InferenceMode.POINT:
            kwargs["points"] = np.array([prompt["point"]], dtype=np.float32)
            kwargs["point_labels"] = np.array([1], dtype=np.int32)
        elif self.inference_mode == InferenceMode.BOX:
            kwargs["box"] = prompt["box"]
        elif self.inference_mode == InferenceMode.BOX_POINT:
            kwargs["box"] = prompt["box"]
            kwargs["points"] = np.array(prompt["points"], dtype=np.float32)
            kwargs["point_labels"] = np.array(prompt["point_labels"], dtype=np.int32)
        else:
            raise ValueError(f"Unsupported heuristic auto prompt mode: {self.inference_mode.value}")
        return kwargs, prompt

    def _prediction_from_result(self, result: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
        masks = result["masks"]
        scores = result["scores"]
        best_idx = int(np.argmax(scores))
        return {"mask": masks[best_idx].astype(np.float32), "score": float(scores[best_idx]), "prompt": prompt}

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        image_rgb = load_image_rgb(sample.image_path)
        kwargs, prompt = self._predict_kwargs_and_prompt(sample)
        result = self.adapter.predict_image(image_rgb, **kwargs)
        return self._prediction_from_result(result, prompt)

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        return _predict_prompted_sam2_samples(
            samples,
            adapter=self.adapter,
            inference_mode=self.inference_mode,
            build_prompt=self._predict_kwargs_and_prompt,
            build_prediction=self._prediction_from_result,
        )


class LearnedAutoPromptedSAM2(BaseMethod):
    name = "LearnedAutoPromptedSAM2"
    family = "sam2"

    def __init__(
        self,
        adapter: SAM2ModelAdapter,
        config: AppConfig,
        prompt_mode: InferenceMode,
        *,
        use_negative_ring: bool = False,
        rerank_candidates: bool = False,
        calibrate_box: bool = False,
        box_calibration_policy: str = "always",
    ):
        self.adapter = adapter
        self.config = config
        self.inference_mode = prompt_mode
        self.use_negative_ring = use_negative_ring
        self.rerank_candidates = bool(rerank_candidates)
        self.calibrate_box = bool(calibrate_box)
        self.box_calibration_policy = str(box_calibration_policy or "always").strip().lower()
        self.device = str(config.method.get("prompt_device") or getattr(config.runtime, "device", "cpu"))
        self.min_box_side = float(config.method.get("prompt_min_box_side", 2.0))
        self.negative_ring_offset = float(config.method.get("prompt_negative_ring_offset", 4.0))
        self.top_k = int(config.method.get("prompt_top_k", 1))
        self.point_budget = int(config.method.get("prompt_point_budget", 1))
        self.response_threshold = float(config.method.get("prompt_response_threshold", 0.0))
        self.nms_radius = int(config.method.get("prompt_nms_radius", 4))
        self.border_suppression_px = int(config.method.get("prompt_border_suppression_px", 0))
        self.use_local_contrast = bool(config.method.get("prompt_use_local_contrast", True))
        self.use_top_hat = bool(config.method.get("prompt_use_top_hat", True))
        self.use_dog_filter = bool(config.method.get("prompt_use_dog_filter", False))
        self.use_fft_highpass = bool(config.method.get("prompt_use_fft_highpass", False))
        self.fuse_filter_candidates = bool(config.method.get("prompt_fuse_filter_candidates", False))
        self.filter_candidate_weight = float(config.method.get("prompt_filter_candidate_weight", 0.5))
        self.prompt_inference_dtype = str(config.method.get("prompt_inference_dtype", "float32"))
        raw_prompt_train_seed = config.method.get("prompt_train_seed")
        self.prompt_train_seed = None if raw_prompt_train_seed is None else int(raw_prompt_train_seed)
        reranker_payload = config.method.get("prompt_reranker", {})
        reranker_payload = dict(reranker_payload) if isinstance(reranker_payload, dict) else {}
        reranker_payload.setdefault("min_box_side", self.min_box_side)
        self._reranker_config = prompt_reranker_config_from_dict(reranker_payload)
        heatmap_config = config.method.get("heatmaps", {})
        heatmap_config = heatmap_config if isinstance(heatmap_config, dict) else {}
        self._heatmap_enabled = bool(heatmap_config.get("enabled", False))
        self._heatmap_limit = max(0, int(heatmap_config.get("sample_limit", 0)))
        self._heatmap_sample_ids = {str(item) for item in heatmap_config.get("sample_ids", [])}
        raw_heatmap_root = heatmap_config.get("root")
        fallback_heatmap_root = getattr(config, "output_dir", config.root) / "heatmaps"
        self._heatmap_root = _resolve_output_path(config, str(raw_heatmap_root)) if raw_heatmap_root else fallback_heatmap_root / "heatmaps"
        self._heatmap_experiment_id = str(heatmap_config.get("experiment_id") or getattr(config.runtime, "output_name", "learned_auto_prompt"))
        self._heatmap_dataset_id = str(heatmap_config.get("dataset_id") or getattr(getattr(config, "dataset", None), "dataset_id", "dataset"))
        self._saved_heatmaps = 0
        raw_checkpoint = (
            config.method.get("prompt_checkpoint")
            or config.method.get("auto_prompt_checkpoint")
            or config.method.get("learned_auto_prompt_checkpoint")
        )
        self._configuration_error: str | None = None
        self._checkpoint_path = Path()
        if not raw_checkpoint:
            self._configuration_error = (
                "learned auto prompt baselines require method.prompt_checkpoint "
                "or method.auto_prompt_checkpoint."
            )
        else:
            self._checkpoint_path = _resolve_existing_path(config, str(raw_checkpoint))
            if not self._checkpoint_path.exists():
                self._configuration_error = f"learned auto prompt checkpoint not found: {self._checkpoint_path}"
        self._prompt_model: Any | None = None
        self._checkpoint_metadata: Dict[str, Any] = {}

    def _ensure_prompt_model(self):
        if self._configuration_error:
            raise ValueError(self._configuration_error)
        if self._prompt_model is None:
            self._prompt_model, self._checkpoint_metadata = load_auto_prompt_model(
                self._checkpoint_path,
                device=self.device,
                inference_dtype=self.prompt_inference_dtype,
            )
        return self._prompt_model

    def _auto_prompt_object(self, sample: Sample):
        model = self._ensure_prompt_model()
        return predict_learned_auto_prompt_from_path(
            model=model,
            image_path=sample.image_path,
            device=self.device,
            negative_ring=self.use_negative_ring,
            min_box_side=self.min_box_side,
            negative_ring_offset=self.negative_ring_offset,
            top_k=self.top_k,
            point_budget=self.point_budget,
            response_threshold=self.response_threshold,
            nms_radius=self.nms_radius,
            border_suppression_px=self.border_suppression_px,
            use_local_contrast=self.use_local_contrast,
            use_top_hat=self.use_top_hat,
            use_dog_filter=self.use_dog_filter,
            use_fft_highpass=self.use_fft_highpass,
            fuse_filter_candidates=self.fuse_filter_candidates,
            filter_candidate_weight=self.filter_candidate_weight,
        )

    def _prompt_from_auto_prompt(self, sample: Sample, auto_prompt) -> Dict[str, Any]:
        heatmap_paths = self._maybe_write_heatmap(sample, auto_prompt)
        prompt = {
            **auto_prompt.metadata,
            "protocol": LEARNED_IR_AUTO_PROMPT_PROTOCOL,
            "box_variant": "learned",
            "point_rule": "argmax_learned_objectness",
            "box_rule": "learned_local_box_size",
            "checkpoint_path": str(self._checkpoint_path),
            "prompt_train_seed": self.prompt_train_seed,
            "checkpoint_protocol": self._checkpoint_metadata.get("protocol", LEARNED_IR_AUTO_PROMPT_PROTOCOL),
            "objectness_map_id": heatmap_paths.get("raw", ""),
            "objectness_heatmap_overlay": heatmap_paths.get("overlay", ""),
        }
        return prompt

    def _auto_prompt(self, sample: Sample) -> Dict[str, Any]:
        return self._prompt_from_auto_prompt(sample, self._auto_prompt_object(sample))

    def _maybe_write_heatmap(self, sample: Sample, auto_prompt) -> dict[str, str]:
        if not self._heatmap_enabled:
            return {}
        selected = sample.sample_id in self._heatmap_sample_ids or sample.frame_id in self._heatmap_sample_ids
        if not selected:
            if self._heatmap_limit <= 0 or self._saved_heatmaps >= self._heatmap_limit:
                return {}
        paths = write_heatmap_artifact(
            root=self._heatmap_root,
            experiment_id=self._heatmap_experiment_id,
            dataset=self._heatmap_dataset_id,
            sample_id=sample.sample_id,
            stage="learned_auto_prompt_objectness",
            heatmap=auto_prompt.objectness,
            image=sample.image_path,
            meta={
                "model": self.name,
                "checkpoint_path": str(self._checkpoint_path),
                "prompt_train_seed": self.prompt_train_seed,
                "prompt_mode": self.inference_mode.value,
                "point": auto_prompt.point,
                "box": auto_prompt.box,
                "candidate_score": auto_prompt.metadata.get("candidate_score"),
                "candidate_points": auto_prompt.metadata.get("candidate_points"),
                "primary_border_distance_px": auto_prompt.metadata.get("primary_border_distance_px"),
            },
        )
        self._saved_heatmaps += 1
        return paths

    def _point_labels_for_box(self, point: list[float], box: list[float], *, width: int, height: int) -> tuple[list[list[float]], list[int]]:
        points = [[float(point[0]), float(point[1])]]
        labels = [1]
        if not self.use_negative_ring:
            return points, labels
        x1, y1, x2, y2 = [float(value) for value in box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        offset = float(self.negative_ring_offset)
        raw_points = [
            [cx, y1 - offset],
            [cx, y2 + offset],
            [x1 - offset, cy],
            [x2 + offset, cy],
        ]
        max_x = max(0.0, float(width - 1))
        max_y = max(0.0, float(height - 1))
        points.extend([[float(min(max(0.0, x), max_x)), float(min(max(0.0, y), max_y))] for x, y in raw_points])
        labels.extend([0] * len(raw_points))
        return points, labels

    def _kwargs_from_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"multimask_output": self.inference_mode == InferenceMode.MULTI_MASK}
        if self.inference_mode == InferenceMode.POINT:
            kwargs["points"] = np.array(prompt["points"], dtype=np.float32)
            kwargs["point_labels"] = np.array(prompt["point_labels"], dtype=np.int32)
        elif self.inference_mode == InferenceMode.BOX:
            kwargs["box"] = prompt["box"]
        elif self.inference_mode == InferenceMode.BOX_POINT:
            kwargs["box"] = prompt["box"]
            kwargs["points"] = np.array(prompt["points"], dtype=np.float32)
            kwargs["point_labels"] = np.array(prompt["point_labels"], dtype=np.int32)
        else:
            raise ValueError(f"Unsupported learned auto prompt mode: {self.inference_mode.value}")
        return kwargs

    def _apply_rerank_to_prompt(self, prompt: Dict[str, Any], rerank_result) -> Dict[str, Any]:
        selected = rerank_result.selected
        ranked_points = [
            [float(item.point[0]), float(item.point[1]), float(item.final_score)]
            for item in rerank_result.candidates
        ]
        prompt.update(
            {
                "point": [float(selected.point[0]), float(selected.point[1])],
                "points": [[float(selected.point[0]), float(selected.point[1])]],
                "point_labels": [1],
                "candidate_score": float(selected.final_score),
                "candidate_points": ranked_points,
                "candidate_rank": int(selected.index),
                "candidate_count": len(rerank_result.candidates),
                "point_rule": "multi_cue_reranked_learned_objectness",
                "rerank_policy": rerank_result.policy,
                "rerank_selected_index": int(selected.index),
                "rerank_prior_score": float(selected.prior_score),
                "rerank_feedback_score": selected.feedback_score,
                "rerank_objectness_score": float(selected.objectness),
                "rerank_learned_score": selected.learned_score,
                "rerank_candidates_json": json.dumps(candidate_metadata(rerank_result.candidates), ensure_ascii=False),
                "rerank_feedback_json": json.dumps(feedback_metadata(selected.feedback), ensure_ascii=False),
            }
        )
        return prompt

    def _predict_prompt_results(
        self,
        image_rgb: np.ndarray,
        *,
        image_is_set: bool,
        boxes: list[Any] | None = None,
        points: list[np.ndarray | None] | None = None,
        point_labels: list[np.ndarray | None] | None = None,
        multimask_output: bool = False,
    ) -> list[Dict[str, Any]]:
        if image_is_set:
            return self.adapter.predict_current_image_prompts(
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
        return self.adapter.predict_prompts_for_image(
            image_rgb,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

    def _predict_single_prompt_result(
        self,
        image_rgb: np.ndarray,
        *,
        image_is_set: bool,
        box: Any = None,
        points: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        multimask_output: bool = False,
    ) -> Dict[str, Any]:
        if image_is_set:
            return self.adapter.predict_current_image_prompts(
                boxes=[box],
                points=[points],
                point_labels=[point_labels],
                multimask_output=multimask_output,
            )[0]
        return self.adapter.predict_image(
            image_rgb,
            box=box,
            points=points,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

    def _attach_eval_profile(self, prediction: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(getattr(self.config, "runtime", None), "profile_eval", False)):
            return prediction
        metadata = dict(prediction.get("metadata", {}))
        metadata.update(profile)
        prediction["metadata"] = metadata
        return prediction

    def _rerank_prompt(
        self,
        sample: Sample,
        image_rgb: np.ndarray,
        auto_prompt,
        prompt: Dict[str, Any],
        *,
        image_is_set: bool = False,
    ):
        rerank_result = rank_prompt_candidates(auto_prompt, sample.image_path, self._reranker_config)
        feedback_limit = max(0, int(self._reranker_config.max_feedback_candidates))
        feedback_candidates = rerank_result.candidates[:feedback_limit]
        if self._reranker_config.use_mask_feedback and feedback_candidates:
            feedback_results = self._predict_prompt_results(
                image_rgb,
                image_is_set=image_is_set,
                points=[np.array([candidate.point], dtype=np.float32) for candidate in feedback_candidates],
                point_labels=[np.array([1], dtype=np.int32) for _ in feedback_candidates],
                multimask_output=False,
            )
            rerank_result = apply_mask_feedback(
                rerank_result.candidates,
                image_path=sample.image_path,
                sam_results=feedback_results,
                config=self._reranker_config,
            )
        return self._apply_rerank_to_prompt(prompt, rerank_result), rerank_result

    def _box_calibration_components(
        self,
        *,
        sample: Sample,
        image_rgb: np.ndarray,
        auto_prompt,
        rerank_result,
        image_is_set: bool = False,
    ) -> tuple[Any, bool, Dict[str, Any] | None, float]:
        height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        selected = rerank_result.selected
        boxes = make_scaled_boxes(
            point=selected.point,
            base_box=auto_prompt.box,
            image_width=width,
            image_height=height,
            min_box_side=self.min_box_side,
            scales=self._reranker_config.box_scales,
        )
        if not boxes:
            boxes = [(1.0, auto_prompt.box)]
        points: list[np.ndarray | None] = []
        point_labels: list[np.ndarray | None] = []
        for _, box in boxes:
            if self.inference_mode == InferenceMode.BOX:
                points.append(None)
                point_labels.append(None)
                continue
            candidate_points, candidate_labels = self._point_labels_for_box(selected.point, box, width=width, height=height)
            points.append(np.array(candidate_points, dtype=np.float32))
            point_labels.append(np.array(candidate_labels, dtype=np.int32))
        sam_results = self._predict_prompt_results(
            image_rgb,
            image_is_set=image_is_set,
            boxes=[box for _, box in boxes],
            points=points,
            point_labels=point_labels,
            multimask_output=False,
        )
        calibration = calibrate_box_from_results(
            boxes=boxes,
            image_path=sample.image_path,
            point=selected.point,
            sam_results=sam_results,
            config=self._reranker_config,
        )
        point_feedback_score = 0.0 if selected.feedback_score is None else float(selected.feedback_score)
        point_result = selected.sam_result
        apply_box = True
        if self.box_calibration_policy == "gated":
            apply_box = (
                point_feedback_score >= float(self._reranker_config.box_enable_min_point_feedback_score)
                and
                float(calibration.selected.score) >= point_feedback_score + float(self._reranker_config.box_enable_margin)
                and float(calibration.selected.score) >= float(self._reranker_config.box_enable_min_score)
            )
            if not apply_box:
                if point_result is None:
                    point_result = self._predict_single_prompt_result(
                        image_rgb,
                        image_is_set=image_is_set,
                        points=np.array([[float(selected.point[0]), float(selected.point[1])]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=False,
                    )
        return calibration, apply_box, point_result, point_feedback_score

    def _prediction_from_box_calibration(
        self,
        *,
        prompt: Dict[str, Any],
        rerank_result,
        calibration,
        apply_box: bool,
        point_result: Dict[str, Any] | None,
        point_feedback_score: float,
        image_rgb: np.ndarray,
    ) -> Dict[str, Any]:
        height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        selected = rerank_result.selected
        if not apply_box:
            if point_result is None:
                raise RuntimeError("Gated box calibration skipped but no point prediction is available.")
            prompt.update(
                {
                    "box": None,
                    "box_variant": "gated_box_skipped",
                    "box_rule": "multi_cue_rerank_then_gated_box_skipped",
                    "box_width": None,
                    "box_height": None,
                    "box_calibration_policy": "gated_sam2_feedback",
                    "box_calibration_applied": 0.0,
                    "box_calibration_scale": float(calibration.selected.scale),
                    "box_calibration_score": float(calibration.selected.score),
                    "box_calibration_point_feedback_score": point_feedback_score,
                    "box_calibration_margin": float(self._reranker_config.box_enable_margin),
                    "box_calibration_min_point_feedback_score": float(self._reranker_config.box_enable_min_point_feedback_score),
                    "box_calibration_candidate_count": len(calibration.candidates),
                    "box_calibration_candidates_json": json.dumps(box_calibration_metadata(calibration.candidates), ensure_ascii=False),
                }
            )
            return self._prediction_from_result(point_result, prompt)
        selected_box = calibration.selected.box
        final_points, final_labels = self._point_labels_for_box(selected.point, selected_box, width=width, height=height)
        prompt.update(
            {
                "box": selected_box,
                "points": final_points,
                "point_labels": final_labels,
                "box_variant": "calibrated_multi_scale",
                "box_rule": "multi_cue_rerank_then_sam2_feedback_box_calibration",
                "box_width": float(selected_box[2] - selected_box[0]),
                "box_height": float(selected_box[3] - selected_box[1]),
                "box_calibration_policy": "gated_sam2_feedback" if self.box_calibration_policy == "gated" else calibration.policy,
                "box_calibration_applied": 1.0,
                "box_calibration_scale": float(calibration.selected.scale),
                "box_calibration_score": float(calibration.selected.score),
                "box_calibration_point_feedback_score": point_feedback_score,
                "box_calibration_margin": float(self._reranker_config.box_enable_margin),
                "box_calibration_min_point_feedback_score": float(self._reranker_config.box_enable_min_point_feedback_score),
                "box_calibration_candidate_count": len(calibration.candidates),
                "box_calibration_candidates_json": json.dumps(box_calibration_metadata(calibration.candidates), ensure_ascii=False),
            }
        )
        return self._prediction_from_result(calibration.selected.sam_result, prompt)

    def _calibrated_box_prediction(
        self,
        *,
        sample: Sample,
        image_rgb: np.ndarray,
        auto_prompt,
        prompt: Dict[str, Any],
        rerank_result,
        image_is_set: bool = False,
    ) -> Dict[str, Any]:
        calibration, apply_box, point_result, point_feedback_score = self._box_calibration_components(
            sample=sample,
            image_rgb=image_rgb,
            auto_prompt=auto_prompt,
            rerank_result=rerank_result,
            image_is_set=image_is_set,
        )
        return self._prediction_from_box_calibration(
            prompt=prompt,
            rerank_result=rerank_result,
            calibration=calibration,
            apply_box=apply_box,
            point_result=point_result,
            point_feedback_score=point_feedback_score,
            image_rgb=image_rgb,
        )

    def _prediction_from_rerank_result(
        self,
        *,
        image_rgb: np.ndarray,
        prompt: Dict[str, Any],
        rerank_result,
        image_is_set: bool = False,
    ) -> Dict[str, Any]:
        selected_result = rerank_result.selected.sam_result
        if selected_result is not None and self.inference_mode == InferenceMode.POINT:
            return self._prediction_from_result(selected_result, prompt)
        result = self._predict_single_prompt_result(
            image_rgb,
            image_is_set=image_is_set,
            **self._kwargs_from_prompt(prompt),
        )
        return self._prediction_from_result(result, prompt)

    def _predict_kwargs_and_prompt(self, sample: Sample) -> tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._auto_prompt(sample)
        return self._kwargs_from_prompt(prompt), prompt

    def _prediction_from_result(self, result: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
        masks = result["masks"]
        scores = result["scores"]
        best_idx = int(np.argmax(scores))
        return {"mask": masks[best_idx].astype(np.float32), "score": float(scores[best_idx]), "prompt": prompt}

    def _adapter_profile_snapshot(self) -> dict[str, int]:
        profile = getattr(self.adapter, "profile_counters", None)
        if not callable(profile):
            return {}
        return {key: int(value) for key, value in profile().items()}

    @staticmethod
    def _profile_delta(before: dict[str, int], after: dict[str, int], key: str) -> int:
        return int(after.get(key, 0)) - int(before.get(key, 0))

    def _predict_reranked_image_group(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        if not samples:
            return {}
        group_start = time.perf_counter()
        image_start = time.perf_counter()
        representative = samples[0]
        image_rgb = load_image_rgb(representative.image_path)
        image_load_ms = (time.perf_counter() - image_start) * 1000.0

        auto_start = time.perf_counter()
        auto_prompt = self._auto_prompt_object(representative)
        auto_prompt_ms = (time.perf_counter() - auto_start) * 1000.0

        adapter_before = self._adapter_profile_snapshot()
        set_image_start = time.perf_counter()
        self.adapter.set_image(image_rgb)
        sam_set_image_ms = (time.perf_counter() - set_image_start) * 1000.0

        prompt = self._prompt_from_auto_prompt(representative, auto_prompt)
        rerank_start = time.perf_counter()
        prompt, rerank_result = self._rerank_prompt(
            representative,
            image_rgb,
            auto_prompt,
            prompt,
            image_is_set=True,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        predictions: Dict[str, Dict[str, Any]] = {}
        calibration_ms = 0.0
        final_prompt_ms = 0.0
        if self.calibrate_box:
            calibration_start = time.perf_counter()
            calibration, apply_box, point_result, point_feedback_score = self._box_calibration_components(
                sample=representative,
                image_rgb=image_rgb,
                auto_prompt=auto_prompt,
                rerank_result=rerank_result,
                image_is_set=True,
            )
            calibration_ms = (time.perf_counter() - calibration_start) * 1000.0
            for idx, sample in enumerate(samples):
                sample_prompt = prompt if idx == 0 else self._apply_rerank_to_prompt(
                    self._prompt_from_auto_prompt(sample, auto_prompt),
                    rerank_result,
                )
                predictions[sample.sample_id] = self._prediction_from_box_calibration(
                    prompt=sample_prompt,
                    rerank_result=rerank_result,
                    calibration=calibration,
                    apply_box=apply_box,
                    point_result=point_result,
                    point_feedback_score=point_feedback_score,
                    image_rgb=image_rgb,
                )
        else:
            final_start = time.perf_counter()
            selected_result = rerank_result.selected.sam_result
            if selected_result is not None and self.inference_mode == InferenceMode.POINT:
                final_result = selected_result
            else:
                final_result = self._predict_single_prompt_result(
                    image_rgb,
                    image_is_set=True,
                    **self._kwargs_from_prompt(prompt),
                )
            final_prompt_ms = (time.perf_counter() - final_start) * 1000.0
            for idx, sample in enumerate(samples):
                sample_prompt = prompt if idx == 0 else self._apply_rerank_to_prompt(
                    self._prompt_from_auto_prompt(sample, auto_prompt),
                    rerank_result,
                )
                predictions[sample.sample_id] = self._prediction_from_result(final_result, sample_prompt)

        adapter_after = self._adapter_profile_snapshot()
        group_elapsed_ms = (time.perf_counter() - group_start) * 1000.0
        profile = {
            "EvalOptimizationPath": "image_grouped_rerank_v1",
            "EvalImageGroupSize": float(len(samples)),
            "EvalImageLoadMs": float(image_load_ms),
            "EvalAutoPromptMs": float(auto_prompt_ms),
            "EvalSamSetImageMs": float(sam_set_image_ms),
            "EvalRerankMs": float(rerank_ms),
            "EvalCalibrationMs": float(calibration_ms),
            "EvalFinalPromptMs": float(final_prompt_ms),
            "EvalGroupLatencyMs": float(group_elapsed_ms),
            "EvalSamSetImageCalls": float(self._profile_delta(adapter_before, adapter_after, "set_image")),
            "EvalSamPromptPredictCalls": float(self._profile_delta(adapter_before, adapter_after, "prompt_predict")),
            "EvalFeedbackCandidateCount": float(
                max(0, min(int(self._reranker_config.max_feedback_candidates), len(rerank_result.candidates)))
                if self._reranker_config.use_mask_feedback
                else 0
            ),
        }
        return {sample_id: self._attach_eval_profile(prediction, profile) for sample_id, prediction in predictions.items()}

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        image_rgb = load_image_rgb(sample.image_path)
        if self.rerank_candidates or self.calibrate_box:
            auto_prompt = self._auto_prompt_object(sample)
            prompt = self._prompt_from_auto_prompt(sample, auto_prompt)
            prompt, rerank_result = self._rerank_prompt(sample, image_rgb, auto_prompt, prompt)
            if self.calibrate_box:
                return self._calibrated_box_prediction(
                    sample=sample,
                    image_rgb=image_rgb,
                    auto_prompt=auto_prompt,
                    prompt=prompt,
                    rerank_result=rerank_result,
                )
            return self._prediction_from_rerank_result(image_rgb=image_rgb, prompt=prompt, rerank_result=rerank_result)
        kwargs, prompt = self._predict_kwargs_and_prompt(sample)
        result = self.adapter.predict_image(image_rgb, **kwargs)
        return self._prediction_from_result(result, prompt)

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        if self.rerank_candidates or self.calibrate_box:
            predictions: Dict[str, Dict[str, Any]] = {}
            image_groups: Dict[str, List[Sample]] = {}
            for sample in samples:
                image_groups.setdefault(str(sample.image_path), []).append(sample)
            for group in image_groups.values():
                predictions.update(self._predict_reranked_image_group(group))
            return predictions
        return _predict_prompted_sam2_samples(
            samples,
            adapter=self.adapter,
            inference_mode=self.inference_mode,
            build_prompt=self._predict_kwargs_and_prompt,
            build_prediction=self._prediction_from_result,
        )


def _resize_mask_nearest_local(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape == (height, width):
        return arr
    src_h, src_w = arr.shape
    if src_h <= 0 or src_w <= 0:
        return np.zeros((height, width), dtype=np.float32)
    y_idx = np.minimum((np.arange(height) * (src_h / float(height))).astype(np.int64), src_h - 1)
    x_idx = np.minimum((np.arange(width) * (src_w / float(width))).astype(np.int64), src_w - 1)
    return arr[y_idx[:, None], x_idx[None, :]].astype(np.float32)


class LearnedTopKGTIoUOracleSAM2(LearnedAutoPromptedSAM2):
    name = "LearnedTopKGTIoUOracleSAM2"
    family = "sam2_oracle_prompt"

    def __init__(self, adapter: SAM2ModelAdapter, config: AppConfig, *, multimask_oracle: bool = False):
        super().__init__(adapter, config, prompt_mode=InferenceMode.POINT)
        self.multimask_oracle = bool(multimask_oracle)

    @staticmethod
    def _candidate_points(prompt: Dict[str, Any]) -> list[list[float]]:
        raw_candidates = prompt.get("candidate_points")
        candidates: list[list[float]] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                score = float(item[2]) if len(item) >= 3 else float(prompt.get("candidate_score", 0.0) or 0.0)
                candidates.append([float(item[0]), float(item[1]), score])
        if candidates:
            return candidates
        point = prompt.get("point")
        if isinstance(point, list) and len(point) >= 2:
            return [[float(point[0]), float(point[1]), float(prompt.get("candidate_score", 0.0) or 0.0)]]
        return []

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        gt_mask = sample_mask_array(sample)
        if gt_mask is None:
            raise ValueError("sam2_learned_topk_gt_iou_oracle_prompt requires mask supervision.")
        image_rgb = load_image_rgb(sample.image_path)
        auto_prompt = self._auto_prompt_object(sample)
        prompt = self._prompt_from_auto_prompt(sample, auto_prompt)
        candidates = self._candidate_points(prompt)
        if not candidates:
            raise ValueError(f"No learned prompt candidates were produced for sample_id={sample.sample_id!r}.")
        results = self.adapter.predict_prompts_for_image(
            image_rgb,
            points=[np.array([[point[0], point[1]]], dtype=np.float32) for point in candidates],
            point_labels=[np.array([1], dtype=np.int32) for _ in candidates],
            multimask_output=self.multimask_oracle,
        )
        best: dict[str, Any] | None = None
        for candidate_index, (candidate, result) in enumerate(zip(candidates, results)):
            masks = np.asarray(result["masks"], dtype=np.float32)
            scores = np.asarray(result["scores"], dtype=np.float32)
            for mask_index, mask in enumerate(masks):
                aligned_mask = _resize_mask_nearest_local(mask, int(sample.height), int(sample.width))
                iou = mask_iou(aligned_mask, gt_mask)
                record = {
                    "candidate_index": candidate_index,
                    "mask_index": mask_index,
                    "candidate": candidate,
                    "mask": aligned_mask,
                    "sam_score": float(scores[mask_index]) if mask_index < len(scores) else 0.0,
                    "gt_iou": float(iou),
                }
                if best is None or (record["gt_iou"], record["sam_score"]) > (best["gt_iou"], best["sam_score"]):
                    best = record
        if best is None:
            raise RuntimeError(f"No SAM2 masks were returned for sample_id={sample.sample_id!r}.")
        selected = best["candidate"]
        prompt.update(
            {
                "protocol": "learned_topk_gt_iou_oracle_v1",
                "source": "gt_selection_over_learned_candidates",
                "point": [float(selected[0]), float(selected[1])],
                "points": [[float(selected[0]), float(selected[1])]],
                "point_labels": [1],
                "candidate_score": float(selected[2]),
                "candidate_rank": int(best["candidate_index"]),
                "candidate_count": len(candidates),
                "oracle_policy": "best_candidate_mask_by_gt_iou",
                "oracle_multimask": bool(self.multimask_oracle),
                "oracle_selected_mask_index": int(best["mask_index"]),
                "oracle_selected_gt_iou": float(best["gt_iou"]),
                "oracle_selected_sam_score": float(best["sam_score"]),
            }
        )
        return {"mask": best["mask"].astype(np.float32), "score": float(best["sam_score"]), "prompt": prompt}

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        return {sample.sample_id: self.predict_sample(sample) for sample in samples}


CANONICAL_BASELINE_NAMES = (
    "bbox_rect",
    "external_prediction_mask",
    "sam2_pretrained_box_prompt",
    "sam2_pretrained_tight_box_prompt",
    "sam2_pretrained_point_prompt",
    "sam2_pretrained_box_point_prompt",
    "sam2_no_prompt_auto_mask",
    "sam2_heuristic_auto_point_prompt",
    "sam2_heuristic_auto_box_prompt",
    "sam2_heuristic_auto_box_point_prompt",
    "sam2_heuristic_auto_box_point_neg_prompt",
    "sam2_cc_peak_auto_point_prompt",
    "sam2_cc_peak_auto_box_point_prompt",
    "sam2_learned_auto_point_prompt",
    "sam2_learned_auto_box_prompt",
    "sam2_learned_auto_box_point_prompt",
    "sam2_learned_auto_box_point_neg_prompt",
    "sam2_learned_auto_point_rerank_prompt",
    "sam2_learned_auto_box_point_calibrated_prompt",
    "sam2_learned_auto_box_point_calibrated_neg_prompt",
    "sam2_learned_auto_box_point_gated_prompt",
    "sam2_learned_topk_gt_iou_oracle_prompt",
    "sam2_learned_topk_multimask_gt_iou_oracle_prompt",
)


def build_baseline_registry(config: AppConfig) -> Dict[str, BenchmarkMethodProtocol]:
    adapter = SAM2ModelAdapter(config)
    return {
        "bbox_rect": BBoxRectMaskBaseline(),
        "external_prediction_mask": ExternalPredictionMaskBaseline(config),
        "sam2_pretrained_box_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX),
        "sam2_pretrained_tight_box_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX, box_variant="tight"),
        "sam2_pretrained_point_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.POINT),
        "sam2_pretrained_box_point_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_no_prompt_auto_mask": NoPromptAutoMaskSAM2(adapter),
        "sam2_heuristic_auto_point_prompt": HeuristicAutoPromptedSAM2(adapter, config, prompt_mode=InferenceMode.POINT),
        "sam2_heuristic_auto_box_prompt": HeuristicAutoPromptedSAM2(adapter, config, prompt_mode=InferenceMode.BOX),
        "sam2_heuristic_auto_box_point_prompt": HeuristicAutoPromptedSAM2(adapter, config, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_heuristic_auto_box_point_neg_prompt": HeuristicAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.BOX_POINT,
            use_negative_ring=True,
        ),
        "sam2_cc_peak_auto_point_prompt": HeuristicAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.POINT,
            protocol="cc_peak_ir_prompt_v1",
            point_rule="thresholded_connected_component_peak",
            box_rule="thresholded_connected_component_around_peak",
        ),
        "sam2_cc_peak_auto_box_point_prompt": HeuristicAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.BOX_POINT,
            protocol="cc_peak_ir_prompt_v1",
            point_rule="thresholded_connected_component_peak",
            box_rule="thresholded_connected_component_around_peak",
        ),
        "sam2_learned_auto_point_prompt": LearnedAutoPromptedSAM2(adapter, config, prompt_mode=InferenceMode.POINT),
        "sam2_learned_auto_box_prompt": LearnedAutoPromptedSAM2(adapter, config, prompt_mode=InferenceMode.BOX),
        "sam2_learned_auto_box_point_prompt": LearnedAutoPromptedSAM2(adapter, config, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_learned_auto_box_point_neg_prompt": LearnedAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.BOX_POINT,
            use_negative_ring=True,
        ),
        "sam2_learned_auto_point_rerank_prompt": LearnedAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.POINT,
            rerank_candidates=True,
        ),
        "sam2_learned_auto_box_point_calibrated_prompt": LearnedAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.BOX_POINT,
            rerank_candidates=True,
            calibrate_box=True,
        ),
        "sam2_learned_auto_box_point_calibrated_neg_prompt": LearnedAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.BOX_POINT,
            use_negative_ring=True,
            rerank_candidates=True,
            calibrate_box=True,
        ),
        "sam2_learned_auto_box_point_gated_prompt": LearnedAutoPromptedSAM2(
            adapter,
            config,
            prompt_mode=InferenceMode.BOX_POINT,
            rerank_candidates=True,
            calibrate_box=True,
            box_calibration_policy="gated",
        ),
        "sam2_learned_topk_gt_iou_oracle_prompt": LearnedTopKGTIoUOracleSAM2(
            adapter,
            config,
            multimask_oracle=False,
        ),
        "sam2_learned_topk_multimask_gt_iou_oracle_prompt": LearnedTopKGTIoUOracleSAM2(
            adapter,
            config,
            multimask_oracle=True,
        ),
    }
