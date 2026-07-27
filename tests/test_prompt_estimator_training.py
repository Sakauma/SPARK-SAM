import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

import sparksam.training.prompt_estimator as auto_prompt_module
from sparksam.data.sample import Sample
from sparksam.training import train_auto_prompt_from_config


def _write_dataset(root: Path) -> Path:
    data_root = root / "data"
    images = data_root / "images"
    masks = data_root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for idx in range(2):
        image = np.zeros((16, 16), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        image[6:8, 6 + idx : 8 + idx] = 255
        mask[6:8, 6 + idx : 8 + idx] = 255
        Image.fromarray(image).save(images / f"{idx}.png")
        Image.fromarray(mask).save(masks / f"{idx}.png")

    config_dir = root / "configs"
    config_dir.mkdir()
    dataset_config = config_dir / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "model": {"model_id": "dummy", "cfg": "dummy.yaml", "ckpt": "dummy.pt"},
                "dataset": {
                    "dataset_id": "synthetic",
                    "adapter": "generic_image_mask",
                    "root": "data",
                    "images_dir": "images",
                    "masks_dir": "masks",
                    "modality": "ir",
                    "mask_mode": "binary",
                },
                "runtime": {
                    "artifact_root": "artifacts",
                    "reference_results_root": "reference_results",
                    "output_name": "synthetic",
                    "device": "cpu",
                    "max_samples": 0,
                    "max_images": 0,
                    "save_visuals": False,
                    "update_reference_results": False,
                    "seeds": [42],
                },
                "evaluation": {
                    "benchmark_version": "test",
                    "track": "mask_prompted_segmentation",
                    "protocol": "test",
                    "inference_mode": "box",
                    "prompt_policy": {
                        "name": "default_box_gt",
                        "prompt_type": "box",
                        "prompt_source": "gt",
                        "prompt_budget": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_config


class AutoPromptTrainingTests(unittest.TestCase):
    def test_train_auto_prompt_from_config_writes_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {
                            "device": "cpu",
                            "epochs": 1,
                            "batch_size": 1,
                            "learning_rate": 0.001,
                            "objectness_loss": "balanced_focal",
                            "ranking_loss_weight": 0.1,
                            "heuristic_distill_weight": 0.05,
                            "max_long_side": 16,
                            "max_samples": 2,
                            "show_progress": False,
                        },
                        "model": {"hidden_channels": 4},
                        "target": {"gaussian_sigma": 1.0, "positive_radius": 1},
                    }
                ),
                encoding="utf-8",
            )

            summary = train_auto_prompt_from_config(train_config)

            checkpoint = Path(summary["checkpoint_path"])
            summary_path = Path(summary["output_dir"]) / "train_summary.json"
            self.assertTrue(checkpoint.exists())
            self.assertTrue(summary_path.exists())
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["sample_count"], 2)
            self.assertIn("final_loss", saved)
            self.assertIn("ranking_loss", saved["history"][0])
            self.assertIn("heuristic_distill_loss", saved["history"][0])
            self.assertEqual(len(saved["heatmaps"]), 2)
            self.assertTrue(Path(saved["heatmaps"][0]["overlay"]).exists())

    def test_train_auto_prompt_reports_line_progress_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_progress",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {
                            "device": "cpu",
                            "epochs": 1,
                            "batch_size": 1,
                            "learning_rate": 0.001,
                            "max_long_side": 16,
                            "max_samples": 2,
                            "show_progress": True,
                            "progress_backend": "line",
                            "progress_update_interval_s": 0.0,
                        },
                        "model": {"hidden_channels": 4},
                        "target": {"gaussian_sigma": 1.0, "positive_radius": 1},
                        "heatmaps": {"sample_limit": 0},
                    }
                ),
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                summary = train_auto_prompt_from_config(train_config)

            progress_text = stderr.getvalue()
            self.assertIn("[train-progress] auto-prompt epoch 1/1", progress_text)
            self.assertIn("2/2 batch", progress_text)
            self.assertEqual(summary["final_loss"], summary["history"][-1]["loss"])

    def test_train_auto_prompt_saves_interval_and_best_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt_checkpoints.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_checkpoints",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {
                            "device": "cpu",
                            "epochs": 2,
                            "batch_size": 1,
                            "learning_rate": 0.001,
                            "max_long_side": 16,
                            "max_samples": 2,
                            "checkpoint_interval_epochs": 1,
                            "select_best_checkpoint": True,
                            "selection_metric": "loss",
                            "show_progress": False,
                        },
                        "model": {"hidden_channels": 4},
                        "target": {"gaussian_sigma": 1.0, "positive_radius": 1},
                        "heatmaps": {"sample_limit": 0},
                    }
                ),
                encoding="utf-8",
            )

            summary = train_auto_prompt_from_config(train_config)

            output_dir = Path(summary["output_dir"])
            self.assertTrue((output_dir / "checkpoint_epoch_001.pt").exists())
            self.assertTrue((output_dir / "checkpoint_epoch_002.pt").exists())
            self.assertTrue((output_dir / "checkpoint_best.pt").exists())
            self.assertEqual(len(summary["checkpoint_history"]), 2)
            self.assertEqual(Path(summary["selected_checkpoint_path"]).name, "checkpoint_best.pt")
            self.assertEqual(Path(summary["best_checkpoint_path"]).name, "checkpoint_best.pt")
            self.assertIn(summary["best_checkpoint_epoch"], [1, 2])

    def test_streaming_training_does_not_preconsume_full_adapter(self):
        class FakeStreamingAdapter:
            adapter_name = "fake_stream"
            notes = ""

            def __init__(self, image_path: Path) -> None:
                self.image_path = image_path
                self.yielded = 0
                self.to_called_before_first_yield = None

            def iter_samples(self, config, *, shard_id=0, num_shards=1):
                for index in range(100):
                    if index == 0:
                        self.to_called_before_first_yield = model_state["to_called"]
                    self.yielded += 1
                    yield Sample(
                        image_path=self.image_path,
                        sample_id=f"sample_{index}",
                        frame_id=f"frame_{index}",
                        sequence_id="seq",
                        frame_index=index,
                        temporal_key=f"frame_{index}",
                        width=16,
                        height=16,
                        category="target",
                        target_scale="small",
                        device_source="synthetic",
                        annotation_protocol_flag="bbox",
                        supervision_type="bbox",
                        bbox_tight=[6.0, 6.0, 8.0, 8.0],
                        bbox_loose=[5.0, 5.0, 9.0, 9.0],
                        point_prompt=[7.0, 7.0],
                    )

        torch, _, _, _ = auto_prompt_module._require_torch()
        model_state = {"to_called": False}

        class RecordingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def to(self, *args, **kwargs):
                model_state["to_called"] = True
                return super().to(*args, **kwargs)

            def forward(self, image):
                batch, _, height, width = image.shape
                objectness = self.weight + torch.zeros((batch, 1, height, width), device=image.device, dtype=image.dtype)
                box_size = self.weight + torch.ones((batch, 2, height, width), device=image.device, dtype=image.dtype)
                return {"objectness_logits": objectness, "box_size": box_size}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            image_path = root / "data" / "images" / "0.png"
            train_config = root / "auto_prompt_stream.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_stream",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {
                            "device": "cpu",
                            "epochs": 1,
                            "batch_size": 1,
                            "learning_rate": 0.001,
                            "max_long_side": 16,
                            "max_steps_per_epoch": 1,
                            "shuffle_buffer_size": 0,
                            "show_progress": False,
                        },
                        "model": {"hidden_channels": 4},
                        "target": {"gaussian_sigma": 1.0, "positive_radius": 1},
                        "heatmaps": {"sample_limit": 0},
                    }
                ),
                encoding="utf-8",
            )
            fake_adapter = FakeStreamingAdapter(image_path)

            with (
                patch.object(auto_prompt_module, "build_dataset_adapter", return_value=fake_adapter),
                patch.object(auto_prompt_module, "build_infrared_prompt_estimator", return_value=RecordingModel()),
            ):
                summary = train_auto_prompt_from_config(train_config)

            self.assertEqual(fake_adapter.yielded, 1)
            self.assertFalse(fake_adapter.to_called_before_first_yield)
            self.assertTrue(model_state["to_called"])
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["trained_sample_events"], 1)

    def test_iter_training_samples_passes_shard_to_adapter(self):
        class FakeShardAdapter:
            adapter_name = "fake_shard"
            notes = ""

            def __init__(self, image_path: Path) -> None:
                self.image_path = image_path
                self.calls = []

            def iter_samples(self, config, *, shard_id=0, num_shards=1):
                self.calls.append((shard_id, num_shards))
                for index in range(6):
                    if index % num_shards != shard_id:
                        continue
                    yield Sample(
                        image_path=self.image_path,
                        sample_id=f"sample_{index}",
                        frame_id=f"frame_{index}",
                        sequence_id="seq",
                        frame_index=index,
                        temporal_key=f"frame_{index}",
                        width=16,
                        height=16,
                        category="target",
                        target_scale="small",
                        device_source="synthetic",
                        annotation_protocol_flag="bbox",
                        supervision_type="bbox",
                        bbox_tight=[6.0, 6.0, 8.0, 8.0],
                        bbox_loose=[5.0, 5.0, 9.0, 9.0],
                        point_prompt=[7.0, 7.0],
                    )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            image_path = root / "data" / "images" / "0.png"
            train_config = root / "auto_prompt_stream.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_stream",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {"device": "cpu"},
                    }
                ),
                encoding="utf-8",
            )
            fake_adapter = FakeShardAdapter(image_path)

            with patch.object(auto_prompt_module, "build_dataset_adapter", return_value=fake_adapter):
                samples = list(auto_prompt_module._iter_training_samples(train_config, shard_id=1, num_shards=3))

            self.assertEqual(fake_adapter.calls, [(1, 3)])
            self.assertEqual([sample.sample_id for sample in samples], ["sample_1", "sample_4"])

    def test_train_auto_prompt_loader_options_are_safe_without_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt_loader_options.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_loader_options",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {
                            "device": "cpu",
                            "epochs": 1,
                            "batch_size": 1,
                            "learning_rate": 0.001,
                            "max_long_side": 16,
                            "max_samples": 1,
                            "num_workers": 0,
                            "prefetch_factor": 4,
                            "persistent_workers": True,
                            "pin_memory": False,
                            "use_amp": True,
                            "profile_interval_batches": 1,
                            "show_progress": False,
                        },
                        "model": {"hidden_channels": 4},
                        "target": {"gaussian_sigma": 1.0, "positive_radius": 1},
                        "heatmaps": {"sample_limit": 0},
                    }
                ),
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                summary = train_auto_prompt_from_config(train_config)

            self.assertEqual(summary["sample_count"], 1)
            self.assertIn("[train-profile] epoch=1 batch=1", stderr.getvalue())

    def test_torch_batch_prior_and_target_shapes(self):
        torch, F, _, _ = auto_prompt_module._require_torch()
        gray = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
        gray[0, :, 6:8, 6:8] = 255.0
        gray[1, :, 4:7, 8:11] = 255.0
        prior = auto_prompt_module._ir_prior_stack_batch(torch, F, gray)
        boxes = torch.tensor([[6.0, 6.0, 8.0, 8.0], [8.0, 4.0, 11.0, 7.0]], dtype=torch.float32)

        objectness, box_size, box_weight, objectness_weight = auto_prompt_module._target_from_box_batch(
            torch,
            boxes=boxes,
            height=16,
            width=16,
            gaussian_sigma=1.0,
            positive_radius=1,
            min_box_side=2.0,
            hard_negative_weight=2.0,
            hard_negative_percentile=95.0,
            prior_score=prior.max(dim=1).values,
        )

        self.assertEqual(tuple(prior.shape), (2, 3, 16, 16))
        self.assertGreaterEqual(float(prior.min()), 0.0)
        self.assertLessEqual(float(prior.max()), 1.0)
        self.assertEqual(tuple(objectness.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(box_size.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(box_weight.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(objectness_weight.shape), (2, 1, 16, 16))
        self.assertGreater(float(box_weight.sum()), 0.0)

    def test_light_gray_cache_batch_generates_gpu_style_targets_on_cpu(self):
        torch, F, _, _ = auto_prompt_module._require_torch()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt_light.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_light",
                        "output_root": str(root / "outputs"),
                        "light_cache_dataset_configs": [str(dataset_config)],
                        "train": {"device": "cpu", "max_long_side": 16},
                    }
                ),
                encoding="utf-8",
            )
            cache = auto_prompt_module._build_light_gray_cache(
                config_path=train_config,
                dataset_configs=[str(dataset_config)],
                train_cfg={"max_long_side": 16},
            )
            self.assertIsNotNone(cache)
            assert cache is not None
            batch = auto_prompt_module._light_cache_batch(
                cache,
                [0, 1],
                torch=torch,
                F=F,
                device="cpu",
                target_cfg={"gaussian_sigma": 1.0, "positive_radius": 1},
                model_cfg=auto_prompt_module.AutoPromptModelConfig(hidden_channels=4),
            )

        self.assertEqual(tuple(batch["image"].shape), (2, 3, 16, 16))
        self.assertEqual(tuple(batch["objectness"].shape), (2, 1, 16, 16))
        self.assertEqual(batch["source"], "light_cache")

    def test_train_tensor_batch_gradient_accumulation_matches_effective_batch(self):
        torch, F, _, _ = auto_prompt_module._require_torch()

        class TinyPromptModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))
                self.bias = torch.nn.Parameter(torch.tensor(-0.1))

            def forward(self, image):
                logits = self.weight * image[:, :1] + self.bias
                box_size = torch.zeros(
                    (image.shape[0], 2, image.shape[2], image.shape[3]),
                    dtype=image.dtype,
                    device=image.device,
                )
                return {"objectness_logits": logits, "box_size": box_size}

        image = torch.linspace(0.0, 1.0, steps=4 * 3 * 4 * 4, dtype=torch.float32).reshape(
            4,
            3,
            4,
            4,
        )
        objectness = torch.zeros((4, 1, 4, 4), dtype=torch.float32)
        objectness[:, :, 1:3, 1:3] = 1.0
        batch = {
            "image": image,
            "objectness": objectness,
            "objectness_weight": torch.ones((4, 1, 4, 4), dtype=torch.float32),
            "box_size": torch.zeros((4, 2, 4, 4), dtype=torch.float32),
            "box_weight": torch.ones((4, 1, 4, 4), dtype=torch.float32),
        }
        train_cfg = {
            "objectness_loss": "weighted_bce",
            "box_loss_weight": 0.0,
            "ranking_loss_weight": 0.0,
            "heuristic_distill_weight": 0.0,
        }

        full_model = TinyPromptModel()
        accum_model = TinyPromptModel()
        accum_model.load_state_dict(full_model.state_dict())
        full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
        accum_optimizer = torch.optim.SGD(accum_model.parameters(), lr=0.1)

        auto_prompt_module._train_tensor_batch(
            torch=torch,
            F=F,
            model=full_model,
            optimizer=full_optimizer,
            scaler=None,
            batch=batch,
            train_cfg=train_cfg,
            device="cpu",
            non_blocking=False,
            use_amp=False,
        )
        for start in (0, 2):
            micro_batch = {key: value[start : start + 2] for key, value in batch.items()}
            auto_prompt_module._train_tensor_batch(
                torch=torch,
                F=F,
                model=accum_model,
                optimizer=accum_optimizer,
                scaler=None,
                batch=micro_batch,
                train_cfg=train_cfg,
                device="cpu",
                non_blocking=False,
                use_amp=False,
                step_optimizer=start == 2,
                zero_grad=start == 0,
                loss_scale=0.5,
            )

        self.assertTrue(torch.allclose(full_model.weight, accum_model.weight, atol=1e-7))
        self.assertTrue(torch.allclose(full_model.bias, accum_model.bias, atol=1e-7))

    def test_cached_training_requires_cuda_device(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt_gpu_cache.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit_gpu_cache",
                        "output_root": str(root / "outputs"),
                        "gpu_cache_dataset_configs": [str(dataset_config)],
                        "train": {"device": "cpu", "epochs": 1},
                        "model": {"hidden_channels": 4},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "requires a CUDA training device"):
                train_auto_prompt_from_config(train_config)


if __name__ == "__main__":
    unittest.main()
