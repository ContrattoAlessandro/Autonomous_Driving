"""Unit and regression tests for E30: B4-Isolated Causal Assigner Validation."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.audit_e30_b4_isolated_tal_causality import (
    CausalMetricDecomposition,
    decompose_metric,
    load_model_with_custom_arrow_pool,
    optimize_safety_threshold,
)
from tlr_yolo_mtl.evaluation.contract import SafetyWaterfallBreakdown
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig, attach_unified_relevance_head

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestB4IsolatedCausality(unittest.TestCase):
    def setUp(self):
        self.config_path = PROJECT_ROOT / "configs" / "b4_isolated_k_arrow_16.yaml"
        self.device = torch.device("cpu")

    def test_b4_isolated_config_structure(self):
        """Verify that configs/b4_isolated_k_arrow_16.yaml exists and conforms to E30 specification."""
        self.assertTrue(self.config_path.is_file(), "b4_isolated_k_arrow_16.yaml config file must exist")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertTrue(cfg.get("p2_enabled", False), "P2 neck must be enabled")
        self.assertEqual(cfg["architecture"]["max_arrows"], 16, "Isolated candidate arrow budget must be 16")
        self.assertEqual(cfg["architecture"]["max_traffic_lights"], 32, "TL candidate budget must be 32")
        self.assertIn("tal_assigner", cfg, "TAL assigner configuration must be specified")
        self.assertEqual(cfg["tal_assigner"]["mode"], "scale_adaptive", "NWD-TAL must be scale-adaptive")
        self.assertAlmostEqual(cfg["tal_assigner"]["lambda_nwd"], 0.5, places=2)

    def test_model_candidate_pool_shapes(self):
        """Verify that model with K_Arrow=16 creates correctly shaped candidate tensors."""
        model_cfg_path = PROJECT_ROOT / "configs" / "model" / "tlr_yolo11n_p2.yaml"
        wrapper = build_detection_model(model_cfg_path)
        arch_cfg = UnifiedHeadConfig(
            token_dim=64,
            token_feature_dim=32,
            attention_heads=4,
            max_traffic_lights=16,
            max_arrows=8,
        )
        attach_unified_relevance_head(wrapper, config=arch_cfg)
        model = wrapper.model.to(self.device).eval()

        dummy_img = torch.randn(2, 3, 256, 512)
        with torch.no_grad():
            preds = model(dummy_img)

        if isinstance(preds, tuple):
            decoded, raw = preds
        elif isinstance(preds, dict):
            raw = preds
        else:
            raw = {}

        self.assertIn("traffic_candidate_boxes", raw)
        self.assertIn("arrow_candidate_boxes", raw)
        self.assertEqual(raw["traffic_candidate_boxes"].shape[1], 16)
        self.assertEqual(raw["arrow_candidate_boxes"].shape[1], 8)
        self.assertEqual(raw["relevance_logits"].shape[2], 16)

    def test_causal_decomposition_math(self):
        """Verify mathematical decomposition logic for assigner vs arrow pool contributions."""
        # Case 1: Pure Assigner Gain (e.g. Sub-4px recall)
        # B2=0.084, B4-iso=0.4446, B4-full=0.4446
        d1 = decompose_metric("Sub-4px Recall", 0.084, 0.4446, 0.4446)
        self.assertAlmostEqual(d1.delta_assigner, 0.3606, places=4)
        self.assertAlmostEqual(d1.delta_arrow_pool, 0.0, places=4)
        self.assertAlmostEqual(d1.assigner_share_pct, 100.0, places=2)
        self.assertAlmostEqual(d1.arrow_pool_share_pct, 0.0, places=2)
        self.assertTrue(d1.is_assigner_dominant)

        # Case 2: Arrow Pool Dominant (e.g. Arrow token recall)
        # B2=0.884, B4-iso=0.884, B4-full=0.9502
        d2 = decompose_metric("Arrow Token Recall", 0.884, 0.884, 0.9502)
        self.assertAlmostEqual(d2.delta_assigner, 0.0, places=4)
        self.assertAlmostEqual(d2.delta_arrow_pool, 0.0662, places=4)
        self.assertAlmostEqual(d2.assigner_share_pct, 0.0, places=2)
        self.assertAlmostEqual(d2.arrow_pool_share_pct, 100.0, places=2)
        self.assertFalse(d2.is_assigner_dominant)

        # Case 3: Shared Contribution (e.g. Relevance AUPRC)
        # B2=0.85, B4-iso=0.89, B4-full=0.91 (Total delta 0.06, assigner 0.04 -> 66.7%, arrow 0.02 -> 33.3%)
        d3 = decompose_metric("Relevance AUPRC", 0.85, 0.89, 0.91)
        self.assertAlmostEqual(d3.delta_total, 0.06, places=4)
        self.assertAlmostEqual(d3.assigner_share_pct, 66.6666, places=1)
        self.assertAlmostEqual(d3.arrow_pool_share_pct, 33.3333, places=1)

    def test_safety_threshold_optimization(self):
        """Verify operating point threshold calculation for target recalls."""
        targets = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        # Target recall 100% (4/4) -> threshold should include score 0.6
        tau, prec, rec = optimize_safety_threshold(targets, scores, target_recall=1.00)
        self.assertAlmostEqual(rec, 1.00, places=2)
        self.assertGreaterEqual(tau, 0.5)
        self.assertGreaterEqual(prec, 0.4)

        # Target recall 75% (3/4)
        tau_75, prec_75, rec_75 = optimize_safety_threshold(targets, scores, target_recall=0.75)
        self.assertGreaterEqual(rec_75, 0.75)
        self.assertGreaterEqual(tau_75, 0.5)

    def test_safety_waterfall_monotonicity(self):
        """Verify that SafetyWaterfallBreakdown maintains survival monotonicity."""
        wf = SafetyWaterfallBreakdown(
            gt_relevant_red_total=100,
            perception_detected=95,
            perception_missed=5,
            candidate_selected=92,
            candidate_missed=3,
            state_classified_red=88,
            state_misclassified=4,
            relevance_accepted=75,
            relevance_rejected=13,
        )
        self.assertLessEqual(wf.relevance_accepted, wf.state_classified_red)
        self.assertLessEqual(wf.state_classified_red, wf.candidate_selected)
        self.assertLessEqual(wf.candidate_selected, wf.perception_detected)
        self.assertLessEqual(wf.perception_detected, wf.gt_relevant_red_total)
        self.assertAlmostEqual(wf.end_to_end_recall, 0.75, places=4)


if __name__ == "__main__":
    unittest.main()
