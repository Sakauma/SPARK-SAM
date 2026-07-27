from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import ModelCapabilities


def _require_module(module_name: str, package_name: str):
    # 依赖缺失时给出面向 benchmark 环境的错误，而不是暴露底层 import 栈。
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Required SAM2 runtime dependency is not available: {package_name}. "
            f"Install the benchmark runtime dependencies and verify your PyTorch/CUDA environment."
        ) from exc


def check_sam2_runtime(repo: Path) -> None:
    # 这里只检查运行 SAM2 必需的 Python 依赖和本地 repo 路径；CUDA 可用性由 PyTorch 推理时判断。
    _require_module("torch", "torch>=2.5.1")
    _require_module("torchvision", "torchvision>=0.20.1")
    _require_module("hydra", "hydra-core>=1.3.2")
    _require_module("iopath", "iopath>=0.1.10")
    if not repo.exists():
        raise RuntimeError(f"SAM2_REPO does not exist: {repo}")


def load_image_rgb(path: Path) -> np.ndarray:
    # 遥感红外图常是单通道。SAM2 image predictor 需要 RGB，因此这里做 min-max 归一化后复制三通道。
    raw = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    min_v = float(raw.min())
    max_v = float(raw.max())
    denom = max(1e-6, max_v - min_v)
    norm = ((raw.astype(np.float32) - min_v) / denom * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


class SAM2ModelAdapter:
    """Unified adapter for local SAM2 repo, config, checkpoint, and inference APIs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo = config.sam2_repo
        self.model: Any | None = None
        self.image_predictor: Any | None = None
        self.auto_mask_generator: Any | None = None
        self.auto_mask_generator_kwargs: dict[str, Any] = {}
        self._profile_counters: dict[str, int] = {"set_image": 0, "prompt_predict": 0, "set_image_batch": 0}
        self.capabilities = ModelCapabilities(
            supports_auto_mask=True,
        )

    def ensure_loaded(self) -> None:
        # SAM2 权重和 predictor 延迟加载，避免 dry-run、配置检查和分析阶段占用显存。
        if self.model is not None:
            return
        check_sam2_runtime(self.repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        build_sam2 = self._resolve_symbol("sam2.build_sam", "build_sam2")
        image_predictor_cls = self._resolve_symbol("sam2.sam2_image_predictor", "SAM2ImagePredictor")
        if not GlobalHydra.instance().is_initialized():
            initialize_config_module(config_module="sam2_configs", version_base=None)
        ckpt_path = self._resolve_checkpoint_path()
        model_kwargs = {"device": self.config.runtime.device}
        try:
            model = build_sam2(self.config.model.cfg, str(ckpt_path), **model_kwargs)
        except TypeError:
            model = build_sam2(self.config.model.cfg, str(ckpt_path), mode="eval", **model_kwargs)
        model.eval()
        self.model = model
        self.image_predictor = image_predictor_cls(model)
        self.auto_mask_generator = self._build_auto_mask_generator(model)

    def _resolve_symbol(self, module_name: str, symbol_name: str) -> Any:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)

    def _resolve_checkpoint_path(self) -> Path:
        # 允许 YAML 写绝对路径、相对 SAM2 repo 的路径，或相对 benchmark 项目的路径。
        ckpt = Path(os.path.expanduser(os.path.expandvars(self.config.model.ckpt)))
        if ckpt.is_absolute():
            return ckpt
        repo_candidate = self.repo / ckpt
        if repo_candidate.exists():
            return repo_candidate
        root_candidate = self.config.root / ckpt
        if root_candidate.exists():
            return root_candidate
        return ckpt

    def _build_auto_mask_generator(self, model: Any | None = None) -> Any | None:
        # 自动掩码类在不同版本 SAM2 中模块名不同；找不到时 no-prompt 模式会显式报错。
        if model is None:
            model = self.model
        if model is None:
            raise RuntimeError("SAM2 model did not initialize.")
        runtime = getattr(self.config, "runtime", None)
        configured = getattr(runtime, "auto_mask_generator", {}) or {}
        if not isinstance(configured, dict):
            raise ValueError("runtime.auto_mask_generator must be a mapping when provided.")
        generator_kwargs: dict[str, Any] = {
            "points_per_batch": int(getattr(runtime, "auto_mask_points_per_batch", 64)),
        }
        allowed_keys = {
            "points_per_side",
            "points_per_batch",
            "pred_iou_thresh",
            "stability_score_thresh",
            "stability_score_offset",
            "mask_threshold",
            "box_nms_thresh",
            "crop_n_layers",
            "crop_nms_thresh",
            "crop_overlap_ratio",
            "crop_n_points_downscale_factor",
            "min_mask_region_area",
            "output_mode",
            "use_m2m",
            "multimask_output",
        }
        generator_kwargs.update({key: value for key, value in configured.items() if key in allowed_keys and value is not None})
        candidates = [
            ("sam2.automatic_mask_generator", "SAM2AutomaticMaskGenerator"),
            ("sam2.sam2_automatic_mask_generator", "SAM2AutomaticMaskGenerator"),
        ]
        for module_name, symbol_name in candidates:
            try:
                cls = self._resolve_symbol(module_name, symbol_name)
                signature = inspect.signature(cls.__init__)
                accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
                if accepts_kwargs:
                    resolved_kwargs = generator_kwargs
                else:
                    resolved_kwargs = {key: value for key, value in generator_kwargs.items() if key in signature.parameters}
                self.auto_mask_generator_kwargs = dict(resolved_kwargs)
                return cls(model, **resolved_kwargs)
            except Exception:
                continue
        return None

    def _require_image_predictor(self) -> Any:
        self.ensure_loaded()
        if self.image_predictor is None:
            raise RuntimeError("SAM2 image predictor did not initialize.")
        return self.image_predictor

    def reset_profile_counters(self) -> None:
        self._profile_counters = {"set_image": 0, "prompt_predict": 0, "set_image_batch": 0}

    def profile_counters(self) -> dict[str, int]:
        return dict(getattr(self, "_profile_counters", {}))

    def set_image(self, image_rgb: np.ndarray) -> None:
        image_predictor = self._require_image_predictor()
        image_predictor.set_image(image_rgb)
        counters = getattr(self, "_profile_counters", None)
        if counters is not None:
            counters["set_image"] = int(counters.get("set_image", 0)) + 1

    @staticmethod
    def _prediction_payload(masks: Any, scores: Any, logits: Any) -> dict[str, Any]:
        return {
            "masks": np.asarray(masks, dtype=np.float32),
            "scores": np.asarray(scores, dtype=np.float32),
            "logits": np.asarray(logits, dtype=np.float32),
        }

    @staticmethod
    def _prompt_count(
        *,
        boxes: list[Optional[Iterable[float]]] | None,
        points: list[Optional[np.ndarray]] | None,
        point_labels: list[Optional[np.ndarray]] | None,
    ) -> int:
        lengths = [len(value) for value in (boxes, points, point_labels) if value is not None]
        if not lengths:
            return 1
        if len(set(lengths)) != 1:
            raise ValueError(f"Prompt batch fields have mismatched lengths: {lengths!r}.")
        return lengths[0]

    def predict_image(
        self,
        image_rgb: np.ndarray,
        *,
        box: Optional[Iterable[float]] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> dict[str, Any]:
        # 单图 prompted SAM2 推理。所有 box/point 坐标都已由上层按原图像素坐标生成。
        self.set_image(image_rgb)
        return self.predict_current_image_prompts(
            boxes=[box],
            points=[points],
            point_labels=[point_labels],
            multimask_output=multimask_output,
        )[0]

    def predict_prompts_for_image(
        self,
        image_rgb: np.ndarray,
        *,
        boxes: list[Optional[Iterable[float]]] | None = None,
        points: list[Optional[np.ndarray]] | None = None,
        point_labels: list[Optional[np.ndarray]] | None = None,
        multimask_output: bool = False,
    ) -> list[dict[str, Any]]:
        # 同一张图的多个 prompt 共享一次 image embedding，避免重复 set_image。
        self.set_image(image_rgb)
        return self.predict_current_image_prompts(
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

    def predict_current_image_prompts(
        self,
        *,
        boxes: list[Optional[Iterable[float]]] | None = None,
        points: list[Optional[np.ndarray]] | None = None,
        point_labels: list[Optional[np.ndarray]] | None = None,
        multimask_output: bool = False,
    ) -> list[dict[str, Any]]:
        # 调用方已经 set_image 后可用这个接口继续复用同一 image embedding。
        image_predictor = self._require_image_predictor()
        prompt_count = self._prompt_count(boxes=boxes, points=points, point_labels=point_labels)
        results: list[dict[str, Any]] = []
        for idx in range(prompt_count):
            box = None if boxes is None or boxes[idx] is None else np.array(boxes[idx], dtype=np.float32)
            masks, scores, logits = image_predictor.predict(
                box=box,
                point_coords=None if points is None else points[idx],
                point_labels=None if point_labels is None else point_labels[idx],
                multimask_output=multimask_output,
                return_logits=True,
            )
            counters = getattr(self, "_profile_counters", None)
            if counters is not None:
                counters["prompt_predict"] = int(counters.get("prompt_predict", 0)) + 1
            results.append(self._prediction_payload(masks, scores, logits))
        return results

    def predict_images(
        self,
        image_rgbs: list[np.ndarray],
        *,
        boxes: list[Optional[Iterable[float]]] | None = None,
        points: list[Optional[np.ndarray]] | None = None,
        point_labels: list[Optional[np.ndarray]] | None = None,
        multimask_output: bool = False,
    ) -> list[dict[str, Any]]:
        # 优先使用 SAM2 的 batch API；本地 checkout 不支持时回退为逐图预测。
        if not image_rgbs:
            return []
        image_predictor = self._require_image_predictor()
        if len(image_rgbs) == 1 or not hasattr(image_predictor, "set_image_batch") or not hasattr(image_predictor, "predict_batch"):
            return [
                self.predict_image(
                    image_rgbs[idx],
                    box=None if boxes is None else boxes[idx],
                    points=None if points is None else points[idx],
                    point_labels=None if point_labels is None else point_labels[idx],
                    multimask_output=multimask_output,
                )
                for idx in range(len(image_rgbs))
            ]
        box_batch = None if boxes is None else [None if box is None else np.array(box, dtype=np.float32) for box in boxes]
        point_batch = None if points is None else points
        label_batch = None if point_labels is None else point_labels
        image_predictor.set_image_batch(image_rgbs)
        counters = getattr(self, "_profile_counters", None)
        if counters is not None:
            counters["set_image_batch"] = int(counters.get("set_image_batch", 0)) + 1
            counters["set_image"] = int(counters.get("set_image", 0)) + len(image_rgbs)
        masks, scores, logits = image_predictor.predict_batch(
            point_coords_batch=point_batch,
            point_labels_batch=label_batch,
            box_batch=box_batch,
            multimask_output=multimask_output,
            return_logits=True,
        )
        if counters is not None:
            counters["prompt_predict"] = int(counters.get("prompt_predict", 0)) + len(image_rgbs)
        return [
            self._prediction_payload(masks[idx], scores[idx], logits[idx])
            for idx in range(len(image_rgbs))
        ]

    def predict_auto_masks(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        # no-prompt 模式返回的是多个候选 instance，后续由 image-level evaluator 统一匹配 GT。
        self.ensure_loaded()
        if self.auto_mask_generator is None:
            raise RuntimeError("Automatic mask generation is not available in the local SAM2 checkout.")
        return list(self.auto_mask_generator.generate(image_rgb))
