import tempfile
import unittest
from pathlib import Path

from sparksam.models.prompt_estimator import (
    AutoPromptModelConfig,
    _require_torch,
    build_infrared_prompt_estimator,
    load_auto_prompt_model,
    save_auto_prompt_checkpoint,
)
from sparksam.training.prompt_estimator import _build_teacher_model_if_requested


def _parameter_count(model) -> int:
    return sum(int(param.numel()) for param in model.parameters())


class LearnedAutoPromptModelTests(unittest.TestCase):
    def test_context_aware_keeps_output_contract_and_is_larger_than_small(self):
        torch, _, _ = _require_torch()
        x = torch.randn(2, 3, 16, 16)
        compact_model = build_infrared_prompt_estimator(AutoPromptModelConfig(hidden_channels=8))
        context_model = build_infrared_prompt_estimator(AutoPromptModelConfig(architecture="context_aware", hidden_channels=8))

        outputs = context_model(x)

        self.assertEqual(getattr(context_model, "model_name"), "ContextAwareInfraredPromptEstimator")
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(outputs["box_size"].shape), (2, 2, 16, 16))
        self.assertEqual(tuple(outputs["confidence_logits"].shape), (2, 1))
        self.assertGreater(_parameter_count(context_model), _parameter_count(compact_model) * 3)

    def test_high_resolution_feature_pyramid_keeps_output_contract_and_extends_feature_pyramid(self):
        torch, _, _ = _require_torch()
        x = torch.randn(2, 3, 16, 16)
        pyramid_model = build_infrared_prompt_estimator(AutoPromptModelConfig(architecture="feature_pyramid", hidden_channels=8, fpn_channels=12))
        high_resolution_model = build_infrared_prompt_estimator(
            AutoPromptModelConfig(architecture="high_resolution_feature_pyramid", hidden_channels=8, fpn_channels=12)
        )

        outputs = high_resolution_model(x)

        self.assertEqual(getattr(high_resolution_model, "model_name"), "HighResolutionFeaturePyramidInfraredPromptEstimator")
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(outputs["box_size"].shape), (2, 2, 16, 16))
        self.assertEqual(tuple(outputs["confidence_logits"].shape), (2, 1))
        self.assertGreater(_parameter_count(high_resolution_model), _parameter_count(pyramid_model))

    def test_context_aware_checkpoint_round_trip_preserves_outputs(self):
        torch, _, _ = _require_torch()
        torch.manual_seed(7)
        config = AutoPromptModelConfig(architecture="context_aware", hidden_channels=8)
        model = build_infrared_prompt_estimator(config)
        model.eval()
        x = torch.randn(1, 3, 12, 12)
        with torch.no_grad():
            expected = model(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            save_auto_prompt_checkpoint(checkpoint_path, model, config=config, metadata={"unit": True})
            loaded, info = load_auto_prompt_model(checkpoint_path)
            with torch.no_grad():
                actual = loaded(x)

        self.assertEqual(getattr(loaded, "model_name"), "ContextAwareInfraredPromptEstimator")
        self.assertEqual(info["config"]["architecture"], "context_aware")
        self.assertTrue(info["metadata"]["unit"])
        self.assertTrue(torch.allclose(expected["objectness_logits"], actual["objectness_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(expected["box_size"], actual["box_size"], atol=1e-6))
        self.assertTrue(torch.allclose(expected["confidence_logits"], actual["confidence_logits"], atol=1e-6))

    def test_high_resolution_feature_pyramid_checkpoint_round_trip_preserves_outputs(self):
        torch, _, _ = _require_torch()
        torch.manual_seed(11)
        config = AutoPromptModelConfig(architecture="high_resolution_feature_pyramid", hidden_channels=8, fpn_channels=12)
        model = build_infrared_prompt_estimator(config)
        model.eval()
        x = torch.randn(1, 3, 12, 12)
        with torch.no_grad():
            expected = model(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            save_auto_prompt_checkpoint(checkpoint_path, model, config=config, metadata={"unit": True})
            loaded, info = load_auto_prompt_model(checkpoint_path)
            with torch.no_grad():
                actual = loaded(x)

        self.assertEqual(getattr(loaded, "model_name"), "HighResolutionFeaturePyramidInfraredPromptEstimator")
        self.assertEqual(info["config"]["architecture"], "high_resolution_feature_pyramid")
        self.assertTrue(info["metadata"]["unit"])
        self.assertTrue(torch.allclose(expected["objectness_logits"], actual["objectness_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(expected["box_size"], actual["box_size"], atol=1e-6))
        self.assertTrue(torch.allclose(expected["confidence_logits"], actual["confidence_logits"], atol=1e-6))

    def test_teacher_distillation_loader_allows_cross_architecture_teacher(self):
        torch, _, _ = _require_torch()
        teacher_config = AutoPromptModelConfig(architecture="feature_pyramid", hidden_channels=8, fpn_channels=12, depth=2)
        student_config = AutoPromptModelConfig(architecture="compact", hidden_channels=4)
        teacher = build_infrared_prompt_estimator(teacher_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "teacher.pt"
            save_auto_prompt_checkpoint(checkpoint_path, teacher, config=teacher_config)
            loaded, loaded_path = _build_teacher_model_if_requested(
                torch=torch,
                config_path=root / "train.yaml",
                train_cfg={
                    "teacher_loss_weight": 0.05,
                    "teacher_checkpoint": str(checkpoint_path),
                    "teacher_checkpoint_required": True,
                },
                model_cfg=student_config,
                device="cpu",
            )

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded_path, str(checkpoint_path))
        self.assertEqual(getattr(loaded, "model_name"), "FeaturePyramidInfraredPromptEstimator")


if __name__ == "__main__":
    unittest.main()
