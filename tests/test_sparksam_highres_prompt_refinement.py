from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from scripts.train_sparksam import SPARKSAM


class _FakePromptEncoder(nn.Module):
    embed_dim = 32


class _FakeSam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam_prompt_encoder = _FakePromptEncoder()
        self.image_size = 16


class _FixedResidual(nn.Module):
    def forward(self, high_res: torch.Tensor) -> torch.Tensor:
        residual = torch.zeros(
            (high_res.shape[0], 1, high_res.shape[2], high_res.shape[3]),
            dtype=high_res.dtype,
            device=high_res.device,
        )
        residual[:, :, 6, 5] = 10.0
        return residual


def _model() -> SPARKSAM:
    return SPARKSAM(
        _FakeSam(),
        candidate_count=4,
        highres_prompt_refinement_enabled=True,
        highres_prompt_refinement_hidden_dim=16,
        highres_prompt_recenter_box=True,
    )


def _prompt() -> dict[str, torch.Tensor]:
    return {
        "objectness_logits": torch.zeros((1, 1, 2, 2)),
        "box_coords": torch.tensor([[2.0, 2.0, 6.0, 6.0]]),
        "candidate_mask_quality_logits": torch.zeros((1, 4)),
    }


class HighResolutionPromptRefinementTests(unittest.TestCase):
    def test_high_resolution_peak_controls_top_candidate_and_recenters_box(self) -> None:
        model = _model()
        model.highres_prompt_refinement_head = _FixedResidual()
        output = model._apply_highres_prompt_refinement(
            {"high_res_feats": [torch.zeros((1, 16, 8, 8))]},
            _prompt(),
        )
        torch.testing.assert_close(output["point_coords"][0, 0], torch.tensor([11.0, 13.0]))
        torch.testing.assert_close(output["box_coords"][0], torch.tensor([9.0, 11.0, 13.0, 15.0]))
        self.assertEqual(tuple(output["objectness_logits"].shape), (1, 1, 8, 8))
        self.assertEqual(tuple(output["candidate_logits"].shape), (1, 4))

    def test_dense_objectness_loss_reaches_new_high_resolution_head(self) -> None:
        model = _model()
        output = model._apply_highres_prompt_refinement(
            {"high_res_feats": [torch.randn((1, 16, 8, 8))]},
            _prompt(),
        )
        (output["objectness_logits"] - 1.0).square().mean().backward()
        final_layer = model.highres_prompt_refinement_head[-1]
        self.assertIsNotNone(final_layer.weight.grad)
        self.assertGreater(float(final_layer.weight.grad.abs().sum()), 0.0)

    def test_supplement_mode_preserves_base_top4_and_appends_highres_top1(self) -> None:
        model = SPARKSAM(
            _FakeSam(),
            candidate_count=8,
            decoder_point_count=5,
            highres_prompt_refinement_enabled=True,
            highres_prompt_refinement_hidden_dim=16,
            highres_prompt_recenter_box=False,
            highres_prompt_candidate_mode="supplement",
            highres_prompt_base_candidate_count=4,
        )
        model.highres_prompt_refinement_head = _FixedResidual()
        prompt = _prompt()
        prompt.update(
            {
                "point_coords": torch.tensor(
                    [[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0], [9.0, 9.0], [11.0, 11.0], [13.0, 13.0], [15.0, 15.0]]]
                ),
                "point_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]),
                "candidate_logits": torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0]]),
                "candidate_mask_quality_logits": torch.tensor(
                    [[4.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0]]
                ),
            }
        )
        original_box = prompt["box_coords"].clone()
        output = model._apply_highres_prompt_refinement(
            {"high_res_feats": [torch.zeros((1, 16, 8, 8))]},
            prompt,
        )
        torch.testing.assert_close(
            output["point_coords"][0, :4],
            torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]]),
        )
        torch.testing.assert_close(output["point_coords"][0, 4], torch.tensor([11.0, 13.0]))
        torch.testing.assert_close(output["box_coords"], original_box)
        self.assertEqual(tuple(output["point_coords"].shape), (1, 8, 2))
        self.assertEqual(tuple(output["candidate_logits"].shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
