"""Unit and regression tests for E14: Post-P2 Scale Recall & TAL Assigner Starvation Audit."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import torch

from scripts.audit_post_p2_assigner_scale import (
    compute_iou_matrix,
    compute_nwd_matrix,
    load_p2_model_and_criterion,
)
from tlr_yolo_mtl.evaluation.metrics import AREA_BUCKETS, SIDE_BUCKETS
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.training.losses import TLRMultiTaskCriterion


class TestPostP2AssignerAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.p2_model_config = self.project_root / "configs" / "model" / "tlr_yolo11n_p2.yaml"
        self.p2_train_config = self.project_root / "configs" / "tlr_yolo11s_champion_v4.yaml"

    def test_iou_matrix_calculation(self) -> None:
        # Perfectly overlapping boxes
        b1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        b2 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        iou = compute_iou_matrix(b1, b2)
        self.assertEqual(iou.shape, (1, 1))
        self.assertAlmostEqual(float(iou[0, 0].item()), 1.0, places=4)

        # Disjoint boxes
        b3 = torch.tensor([[50.0, 50.0, 70.0, 70.0]])
        iou_disjoint = compute_iou_matrix(b1, b3)
        self.assertAlmostEqual(float(iou_disjoint[0, 0].item()), 0.0, places=4)

        # Partial overlap
        b4 = torch.tensor([[20.0, 20.0, 40.0, 40.0]])
        iou_partial = compute_iou_matrix(b1, b4)
        # inter = 10*10 = 100, union = 400 + 400 - 100 = 700, iou = 1/7 ~= 0.142857
        self.assertAlmostEqual(float(iou_partial[0, 0].item()), 1.0 / 7.0, places=4)

    def test_nwd_matrix_continuous_behavior_on_tiny_boxes(self) -> None:
        # Two tiny 4x4 boxes slightly separated
        b1 = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
        b2 = torch.tensor([[15.0, 10.0, 19.0, 14.0]])  # zero IoU, 1px apart

        iou = compute_iou_matrix(b1, b2)
        nwd = compute_nwd_matrix(b1, b2, constant=12.0)

        self.assertAlmostEqual(float(iou[0, 0].item()), 0.0, places=4)
        self.assertGreater(float(nwd[0, 0].item()), 0.5)  # Continuous positive gradient signal

    def test_4_level_pyramid_strides_and_anchor_counts(self) -> None:
        wrapper = build_detection_model(self.p2_model_config)
        config = UnifiedHeadConfig(max_traffic_lights=32, max_arrows=16)
        attach_unified_relevance_head(wrapper, config=config)
        model = wrapper.model
        detect = model.model[-1]

        strides = tuple(int(s) for s in detect.stride.tolist())
        self.assertEqual(strides, (4, 8, 16, 32))

        h, w = 800, 1600
        anchors_p2 = (h // 4) * (w // 4)  # 80,000
        anchors_p3 = (h // 8) * (w // 8)  # 20,000
        anchors_p4 = (h // 16) * (w // 16)  # 5,000
        anchors_p5 = (h // 32) * (w // 32)  # 1,250
        total_anchors = anchors_p2 + anchors_p3 + anchors_p4 + anchors_p5

        self.assertEqual(anchors_p2, 80000)
        self.assertEqual(total_anchors, 106250)

    def test_load_p2_model_and_criterion(self) -> None:
        device = torch.device("cpu")
        model, criterion, cfg = load_p2_model_and_criterion(self.p2_train_config, device)
        self.assertIsInstance(criterion, TLRMultiTaskCriterion)
        self.assertTrue(cfg.get("p2_enabled", False))
        detect = model.model[-1]
        self.assertEqual(tuple(int(s) for s in detect.stride.tolist()), (4, 8, 16, 32))

    def test_scale_buckets_integrity(self) -> None:
        self.assertIn("<32", AREA_BUCKETS)
        self.assertIn("32-64", AREA_BUCKETS)
        self.assertIn(">512", AREA_BUCKETS)
        self.assertIn("<4", SIDE_BUCKETS)
        self.assertIn("4-6", SIDE_BUCKETS)
        self.assertIn(">12", SIDE_BUCKETS)


if __name__ == "__main__":
    unittest.main()
