"""Unit and integration tests for E31: Multi-Scale ROIAlign End-to-End Integration & Downstream Safety Validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import unittest
import numpy as np
import torch
import yaml

from scripts.audit_e31_multiscale_roialign_e2e import optimize_safety_threshold
from tlr_yolo_mtl.evaluation.contract import SafetyWaterfallBreakdown
from tlr_yolo_mtl.model.roialign_attributes import (
    CandidateAttributeTower,
    CandidateMultiScaleROIAlign,
    CandidateMultiScaleROIAlignPipeline,
)


class TestROIAlignE2EIntegration(unittest.TestCase):
    def setUp(self):
        self.config_path = PROJECT_ROOT / "configs" / "e31_multiscale_roialign.yaml"
        self.device = torch.device("cpu")

    def test_e31_config_structure(self):
        """Verify that configs/e31_multiscale_roialign.yaml exists and conforms to E31 specification."""
        self.assertTrue(self.config_path.is_file(), "e31_multiscale_roialign.yaml config file must exist")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertTrue(cfg.get("p2_enabled", False), "P2 neck must be enabled")
        self.assertIn("candidate_attribute_extractor", cfg, "ROIAlign candidate attribute extractor must be configured")
        extractor_cfg = cfg["candidate_attribute_extractor"]
        self.assertEqual(extractor_cfg["type"], "multiscale_roialign")
        self.assertEqual(extractor_cfg["levels"], ["P2", "P3"])
        self.assertEqual(extractor_cfg["roi_output_size"], [3, 3])
        self.assertEqual(extractor_cfg["spatial_strides"], [4, 8])
        self.assertIn("heads", extractor_cfg)
        self.assertEqual(extractor_cfg["heads"]["state_head"]["num_classes"], 4)
        self.assertEqual(extractor_cfg["heads"]["maneuver_head"]["num_classes"], 3)

    def test_roialign_pipeline_forward_and_backward(self):
        """Verify CandidateMultiScaleROIAlignPipeline forward pass, shapes, and gradients."""
        B, K = 2, 32
        C_p2, C_p3 = 64, 128
        H_p2, W_p2 = 200, 400
        H_p3, W_p3 = 100, 200

        p2_feat = torch.randn(B, C_p2, H_p2, W_p2, requires_grad=True)
        p3_feat = torch.randn(B, C_p3, H_p3, W_p3, requires_grad=True)

        boxes = torch.zeros(B, K, 4)
        boxes[:, :, 0] = torch.rand(B, K) * 1500.0
        boxes[:, :, 1] = torch.rand(B, K) * 700.0
        boxes[:, :, 2] = boxes[:, :, 0] + torch.rand(B, K) * 40.0 + 4.0
        boxes[:, :, 3] = boxes[:, :, 1] + torch.rand(B, K) * 60.0 + 8.0

        pipeline = CandidateMultiScaleROIAlignPipeline(
            channels_p2=C_p2,
            channels_p3=C_p3,
            roi_size=(3, 3),
            embed_dim=128,
            stride_p2=4.0,
            stride_p3=8.0,
        )

        out = pipeline(p2_feat, p3_feat, boxes)
        self.assertIn("state_logits", out)
        self.assertIn("round_logits", out)
        self.assertIn("maneuver_logits", out)
        self.assertIn("candidate_tokens", out)

        self.assertEqual(out["state_logits"].shape, (B, K, 4))
        self.assertEqual(out["round_logits"].shape, (B, K))
        self.assertEqual(out["maneuver_logits"].shape, (B, K, 3))
        self.assertEqual(out["candidate_tokens"].shape, (B, K, 128))

        # Check probability normalization
        state_probs = out["state_probs"]
        self.assertTrue(torch.allclose(state_probs.sum(dim=-1), torch.ones(B, K), atol=1e-5))

        # Backward pass gradient verification
        loss = out["state_logits"].sum() + out["round_logits"].sum() + out["maneuver_logits"].sum()
        loss.backward()
        self.assertIsNotNone(p2_feat.grad)
        self.assertIsNotNone(p3_feat.grad)
        self.assertGreater(p2_feat.grad.abs().sum().item(), 0.0)
        self.assertGreater(p3_feat.grad.abs().sum().item(), 0.0)

    def test_roialign_with_normalized_boxes(self):
        """Verify pipeline handles normalized [0, 1] candidate boxes with img_shape."""
        B, K = 2, 16
        C_p2, C_p3 = 32, 64
        H, W = 800, 1600

        p2_feat = torch.randn(B, C_p2, H // 4, W // 4)
        p3_feat = torch.randn(B, C_p3, H // 8, W // 8)

        # Normalized coordinates [0, 1]
        norm_boxes = torch.rand(B, K, 4)
        norm_boxes[:, :, 2] = norm_boxes[:, :, 0] + 0.05
        norm_boxes[:, :, 3] = norm_boxes[:, :, 1] + 0.08
        norm_boxes = norm_boxes.clamp(0.0, 1.0)

        pipeline = CandidateMultiScaleROIAlignPipeline(
            channels_p2=C_p2,
            channels_p3=C_p3,
            roi_size=(3, 3),
            embed_dim=64,
        )

        out = pipeline(p2_feat, p3_feat, norm_boxes, img_shape=(H, W))
        self.assertEqual(out["state_logits"].shape, (B, K, 4))
        self.assertEqual(out["candidate_tokens"].shape, (B, K, 64))

    def test_safety_waterfall_error_accounting(self):
        """Verify 4-stage safety waterfall error accounting and rate calculations."""
        wf = SafetyWaterfallBreakdown(
            gt_relevant_red_total=1373,
            perception_detected=1180,
            perception_missed=193,
            candidate_selected=1174,
            candidate_missed=6,
            state_classified_red=1135,
            state_misclassified=39,
            relevance_accepted=1137,
            relevance_rejected=0,
        )

        self.assertAlmostEqual(wf.perception_recall, 1180 / 1373, places=4)
        self.assertAlmostEqual(wf.candidate_selection_rate, 1174 / 1180, places=4)
        self.assertAlmostEqual(wf.state_classification_rate, 1135 / 1174, places=4)
        self.assertAlmostEqual(wf.end_to_end_recall, 1137 / 1373, places=4)
        self.assertGreaterEqual(wf.end_to_end_recall, 0.82)

    def test_safety_threshold_optimization(self):
        """Verify operating point calculation for target recall 95%."""
        targets = np.array([1] * 20 + [0] * 80)
        scores = np.linspace(0.01, 0.99, 100)
        scores[:20] = np.linspace(0.40, 0.99, 20)  # Positives have higher scores

        tau_95, prec, rec = optimize_safety_threshold(targets, scores, target_recall=0.95)
        self.assertGreaterEqual(rec, 0.95)
        self.assertGreater(tau_95, 0.0)


if __name__ == "__main__":
    unittest.main()
