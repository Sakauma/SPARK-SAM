import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sparksam.baselines.response_guidance import LearnedAutoPromptedSAM2, LearnedTopKGTIoUOracleSAM2
from sparksam.core.interfaces import InferenceMode
from sparksam.data.sample import Sample
from sparksam.models import (
    LEARNED_IR_AUTO_PROMPT_PROTOCOL,
    AutoPromptModelConfig,
    LearnedAutoPrompt,
    build_infrared_prompt_estimator,
    decode_auto_prompt,
    ir_prior_stack_from_path,
    load_auto_prompt_model,
    save_auto_prompt_checkpoint,
)


class DummySAM2Adapter:
    def __init__(self):
        self.kwargs = None
        self.image_batch_calls = 0
        self.prompt_batch_calls = 0

    def predict_image(self, image_rgb, **kwargs):
        self.kwargs = kwargs
        return {"masks": np.ones((1, 8, 8), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)}

    def predict_images(self, image_rgbs, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.image_batch_calls += 1
        self.kwargs = {"boxes": boxes, "points": points, "point_labels": point_labels, "multimask_output": multimask_output}
        return [{"masks": np.ones((1, 8, 8), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)} for _ in image_rgbs]

    def predict_prompts_for_image(self, image_rgb, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.prompt_batch_calls += 1
        self.kwargs = {"boxes": boxes, "points": points, "point_labels": point_labels, "multimask_output": multimask_output}
        prompt_count = len(boxes or points or point_labels or [])
        return [{"masks": np.ones((1, 8, 8), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)} for _ in range(prompt_count)]


class GatedBoxSAM2Adapter:
    def __init__(self):
        self.prompt_batch_calls = 0

    def predict_image(self, image_rgb, **kwargs):
        mask = np.zeros((1, 8, 8), dtype=np.float32)
        mask[:, 3:5, 3:5] = 1.0
        return {"masks": mask, "scores": np.array([1.0], dtype=np.float32)}

    def predict_prompts_for_image(self, image_rgb, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.prompt_batch_calls += 1
        if boxes is None:
            count = len(points or [])
            mask = np.zeros((1, 8, 8), dtype=np.float32)
            mask[:, 3:5, 3:5] = 1.0
            return [{"masks": mask.copy(), "scores": np.array([1.0], dtype=np.float32)} for _ in range(count)]
        mask = np.ones((1, 8, 8), dtype=np.float32)
        return [{"masks": mask.copy(), "scores": np.array([0.1], dtype=np.float32)} for _ in range(len(boxes))]


class TopKOracleSAM2Adapter:
    def __init__(self):
        self.kwargs = {}

    def predict_prompts_for_image(self, image_rgb, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        del image_rgb, boxes, point_labels
        self.kwargs = {"points": points, "multimask_output": multimask_output}
        masks = []
        scores = []
        for idx, _ in enumerate(points or []):
            mask = np.zeros((1, 8, 8), dtype=np.float32)
            if idx == 0:
                mask[:, 0:2, 0:2] = 1.0
                score = 0.9
            else:
                mask[:, 3:5, 3:5] = 1.0
                score = 0.1
            masks.append(mask)
            scores.append(np.array([score], dtype=np.float32))
        return [{"masks": mask, "scores": score} for mask, score in zip(masks, scores)]


def _sample(path: Path, sample_id: str = "sample") -> Sample:
    return Sample(
        image_path=path,
        sample_id=sample_id,
        frame_id="sample",
        sequence_id="seq",
        frame_index=0,
        temporal_key="sample",
        width=8,
        height=8,
        category="target",
        target_scale="small",
        device_source="test",
        annotation_protocol_flag="mask",
        supervision_type="mask",
    )


def _config(checkpoint_path: Path, method_overrides: dict | None = None):
    config = type("Config", (), {})()
    config.method = {"prompt_checkpoint": str(checkpoint_path), "prompt_device": "cpu"}
    if method_overrides:
        config.method.update(method_overrides)
    config.runtime = type("Runtime", (), {"device": "cpu"})()
    config.root = checkpoint_path.parent
    config.config_path = checkpoint_path.parent / "config.yaml"
    return config


class LearnedAutoPromptTests(unittest.TestCase):
    def test_ir_prior_stack_from_path_has_three_channels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            image = np.zeros((8, 8), dtype=np.uint8)
            image[3:5, 3:5] = 255
            Image.fromarray(image).save(path)

            prior = ir_prior_stack_from_path(path)

            self.assertEqual(prior.shape, (3, 8, 8))
            self.assertGreaterEqual(float(prior.min()), 0.0)
            self.assertLessEqual(float(prior.max()), 1.0)

    def test_decode_auto_prompt_uses_argmax_and_local_box_size(self):
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        logits[0, 2, 3] = 10.0
        box_size = np.zeros((2, 4, 5), dtype=np.float32)
        box_size[0, :, :] = 4.0
        box_size[1, :, :] = 2.0

        prompt = decode_auto_prompt(
            objectness_logits=logits,
            box_size=box_size,
            image_width=5,
            image_height=4,
            negative_ring=True,
        )

        self.assertEqual(prompt.point, [3.0, 2.0])
        self.assertEqual(prompt.metadata["protocol"], LEARNED_IR_AUTO_PROMPT_PROTOCOL)
        self.assertEqual(prompt.metadata["negative_point_count"], 4)
        self.assertEqual(len(prompt.points), 5)

    def test_decode_auto_prompt_supports_topk_nms_and_threshold_metadata(self):
        logits = np.zeros((1, 8, 8), dtype=np.float32)
        logits[0, 2, 2] = 8.0
        logits[0, 2, 3] = 7.0
        logits[0, 6, 6] = 6.0
        box_size = np.ones((2, 8, 8), dtype=np.float32) * 2.0

        prompt = decode_auto_prompt(
            objectness_logits=logits,
            box_size=box_size,
            image_width=8,
            image_height=8,
            top_k=3,
            point_budget=2,
            nms_radius=2,
            response_threshold=0.1,
        )

        self.assertEqual(prompt.metadata["candidate_top_k"], 3)
        self.assertEqual(prompt.metadata["candidate_nms_radius"], 2)
        self.assertEqual(prompt.metadata["positive_point_count"], 2)
        self.assertEqual(prompt.points[0], [2.0, 2.0])
        self.assertEqual(prompt.points[1], [6.0, 6.0])
        self.assertEqual(prompt.metadata["candidate_points"][0][:2], [2.0, 2.0])

    def test_decode_auto_prompt_can_suppress_border_candidates(self):
        logits = np.zeros((1, 8, 8), dtype=np.float32)
        logits[0, 0, 7] = 10.0
        logits[0, 4, 4] = 8.0
        box_size = np.ones((2, 8, 8), dtype=np.float32) * 2.0

        prompt = decode_auto_prompt(
            objectness_logits=logits,
            box_size=box_size,
            image_width=8,
            image_height=8,
            border_suppression_px=1,
        )

        self.assertEqual(prompt.point, [4.0, 4.0])
        self.assertEqual(prompt.metadata["border_suppression_px"], 1)
        self.assertGreater(prompt.metadata["primary_border_distance_px"], 0)

    def test_checkpoint_roundtrip_loads_model_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            cfg = AutoPromptModelConfig(hidden_channels=4)
            model = build_infrared_prompt_estimator(cfg)

            save_auto_prompt_checkpoint(checkpoint, model, config=cfg, metadata={"tag": "unit"})
            loaded, meta = load_auto_prompt_model(checkpoint, device="cpu")

            self.assertIsNotNone(loaded)
            self.assertEqual(meta["protocol"], LEARNED_IR_AUTO_PROMPT_PROTOCOL)
            self.assertEqual(meta["metadata"]["tag"], "unit")

    def test_learned_auto_prompted_sam2_passes_prompt_to_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            save_auto_prompt_checkpoint(checkpoint, build_infrared_prompt_estimator(AutoPromptModelConfig(hidden_channels=4)))
            adapter = DummySAM2Adapter()
            method = LearnedAutoPromptedSAM2(adapter, _config(checkpoint), prompt_mode=InferenceMode.BOX_POINT, use_negative_ring=True)

            pred = method.predict_sample(_sample(image_path))

            self.assertIn("box", adapter.kwargs)
            self.assertEqual(adapter.kwargs["points"].shape[1], 2)
            self.assertIn(0, adapter.kwargs["point_labels"].tolist())
            self.assertEqual(pred["prompt"]["source"], "learned_auto_prompt")
            self.assertEqual(pred["prompt"]["protocol"], LEARNED_IR_AUTO_PROMPT_PROTOCOL)

    def test_learned_auto_prompted_batch_reuses_single_image_embedding_for_same_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            save_auto_prompt_checkpoint(checkpoint, build_infrared_prompt_estimator(AutoPromptModelConfig(hidden_channels=4)))
            adapter = DummySAM2Adapter()
            method = LearnedAutoPromptedSAM2(adapter, _config(checkpoint), prompt_mode=InferenceMode.BOX_POINT)

            predictions = method.predict_samples([_sample(image_path, "sample_0"), _sample(image_path, "sample_1")])

            self.assertEqual(sorted(predictions), ["sample_0", "sample_1"])
            self.assertEqual(adapter.prompt_batch_calls, 1)
            self.assertEqual(adapter.image_batch_calls, 0)
            self.assertEqual(len(adapter.kwargs["boxes"]), 2)

    def test_learned_auto_prompted_sam2_records_calibration_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            save_auto_prompt_checkpoint(checkpoint, build_infrared_prompt_estimator(AutoPromptModelConfig(hidden_channels=4)))
            adapter = DummySAM2Adapter()
            prompt = LearnedAutoPrompt(
                point=[3.0, 3.0],
                box=[2.0, 2.0, 5.0, 5.0],
                points=[[3.0, 3.0]],
                point_labels=[1],
                metadata={
                    "source": "learned_auto_prompt",
                    "protocol": LEARNED_IR_AUTO_PROMPT_PROTOCOL,
                    "point": [3.0, 3.0],
                    "box": [2.0, 2.0, 5.0, 5.0],
                    "points": [[3.0, 3.0]],
                    "point_labels": [1],
                    "candidate_score": 0.8,
                    "candidate_points": [[3.0, 3.0, 0.8], [5.0, 5.0, 0.7]],
                    "candidate_rank": 0,
                    "candidate_count": 2,
                    "candidate_top_k": 2,
                    "candidate_nms_radius": 4,
                    "negative_point_count": 0,
                    "box_width": 3.0,
                    "box_height": 3.0,
                },
                objectness=np.zeros((8, 8), dtype=np.float32),
            )
            config = _config(
                checkpoint,
                {
                    "prompt_reranker": {
                        "use_frequency": False,
                        "max_feedback_candidates": 1,
                        "box_scales": [1.0, 2.0],
                    }
                },
            )
            method = LearnedAutoPromptedSAM2(
                adapter,
                config,
                prompt_mode=InferenceMode.BOX_POINT,
                use_negative_ring=True,
                rerank_candidates=True,
                calibrate_box=True,
            )

            with patch("sparksam.baselines.response_guidance.predict_learned_auto_prompt_from_path", return_value=prompt):
                pred = method.predict_sample(_sample(image_path))

            self.assertEqual(adapter.prompt_batch_calls, 2)
            self.assertEqual(pred["prompt"]["box_variant"], "calibrated_multi_scale")
            self.assertEqual(pred["prompt"]["box_calibration_candidate_count"], 2)
            self.assertEqual(pred["prompt"]["rerank_policy"], "prior_multi_cue_sam2_mask_feedback")
            self.assertIn(0, pred["prompt"]["point_labels"])

    def test_learned_auto_prompted_sam2_gated_box_can_fall_back_to_point_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            image = np.zeros((8, 8), dtype=np.uint8)
            image[3:5, 3:5] = 255
            Image.fromarray(image).save(image_path)
            save_auto_prompt_checkpoint(checkpoint, build_infrared_prompt_estimator(AutoPromptModelConfig(hidden_channels=4)))
            adapter = GatedBoxSAM2Adapter()
            prompt = LearnedAutoPrompt(
                point=[4.0, 4.0],
                box=[2.0, 2.0, 6.0, 6.0],
                points=[[4.0, 4.0]],
                point_labels=[1],
                metadata={
                    "source": "learned_auto_prompt",
                    "protocol": LEARNED_IR_AUTO_PROMPT_PROTOCOL,
                    "point": [4.0, 4.0],
                    "box": [2.0, 2.0, 6.0, 6.0],
                    "points": [[4.0, 4.0]],
                    "point_labels": [1],
                    "candidate_score": 0.9,
                    "candidate_points": [[4.0, 4.0, 0.9]],
                    "candidate_rank": 0,
                    "candidate_count": 1,
                    "candidate_top_k": 1,
                    "candidate_nms_radius": 4,
                    "negative_point_count": 0,
                    "box_width": 4.0,
                    "box_height": 4.0,
                },
                objectness=np.zeros((8, 8), dtype=np.float32),
            )
            config = _config(
                checkpoint,
                {
                    "prompt_reranker": {
                        "use_frequency": False,
                        "max_feedback_candidates": 1,
                        "box_scales": [1.0, 2.0],
                        "box_enable_margin": 0.02,
                    }
                },
            )
            method = LearnedAutoPromptedSAM2(
                adapter,
                config,
                prompt_mode=InferenceMode.BOX_POINT,
                rerank_candidates=True,
                calibrate_box=True,
                box_calibration_policy="gated",
            )

            with patch("sparksam.baselines.response_guidance.predict_learned_auto_prompt_from_path", return_value=prompt):
                pred = method.predict_sample(_sample(image_path))

            self.assertEqual(adapter.prompt_batch_calls, 2)
            self.assertEqual(pred["prompt"]["box_variant"], "gated_box_skipped")
            self.assertEqual(pred["prompt"]["box_calibration_policy"], "gated_sam2_feedback")
            self.assertEqual(pred["prompt"]["box_calibration_applied"], 0.0)
            self.assertEqual(pred["prompt"]["box_calibration_min_point_feedback_score"], 0.0)
            self.assertEqual(float((pred["mask"] > 0.5).sum()), 4.0)

    def test_topk_gt_iou_oracle_selects_candidate_by_mask_iou(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"placeholder")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            adapter = TopKOracleSAM2Adapter()
            method = LearnedTopKGTIoUOracleSAM2(adapter, _config(checkpoint))
            sample = _sample(image_path)
            sample.mask_array = np.zeros((8, 8), dtype=np.float32)
            sample.mask_array[3:5, 3:5] = 1.0
            prompt = LearnedAutoPrompt(
                point=[1.0, 1.0],
                box=[0.0, 0.0, 2.0, 2.0],
                points=[[1.0, 1.0]],
                point_labels=[1],
                metadata={
                    "source": "learned_auto_prompt",
                    "candidate_score": 0.9,
                    "candidate_points": [[1.0, 1.0, 0.9], [4.0, 4.0, 0.1]],
                    "candidate_count": 2,
                    "candidate_top_k": 2,
                },
                objectness=np.zeros((8, 8), dtype=np.float32),
            )

            with patch.object(method, "_auto_prompt_object", return_value=prompt):
                predictions = method.predict_samples([sample])
            pred = predictions[sample.sample_id]

            self.assertFalse(adapter.kwargs["multimask_output"])
            self.assertEqual(pred["prompt"]["candidate_rank"], 1)
            self.assertEqual(pred["prompt"]["oracle_selected_mask_index"], 0)
            self.assertAlmostEqual(pred["prompt"]["oracle_selected_gt_iou"], 1.0)
            self.assertAlmostEqual(pred["score"], 0.1, places=6)
            self.assertEqual(float((pred["mask"] > 0.5).sum()), 4.0)


if __name__ == "__main__":
    unittest.main()
