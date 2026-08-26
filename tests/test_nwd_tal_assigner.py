"""Unit and regression tests for E15: Tiny-Aware / NWD-Aware TaskAlignedAssigner Metric."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import yaml
import torch
from torch import nn

from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.training.engine import build_multitask_criterion
from tlr_yolo_mtl.training.losses import (
    IgnoreAwareDetectionLoss,
    TLRMultiTaskCriterion,
)
from tlr_yolo_mtl.training.tal import (
    NWDAwareTaskAlignedAssigner,
    TaskAlignedAssigner,
    build_task_aligned_assigner,
    compute_nwd_similarity,
)


class TestNWDTALAssigner(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.p2_model_config = self.project_root / "configs" / "model" / "tlr_yolo11n_p2.yaml"
        self.b4_train_config = self.project_root / "configs" / "tlr_yolo11s_champion_v4.yaml"

    def test_nwd_similarity_identical_boxes(self) -> None:
        b1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        b2 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        sim = compute_nwd_similarity(b1, b2, constant=12.0)
        self.assertEqual(sim.shape, (1,))
        self.assertAlmostEqual(float(sim[0].item()), 1.0, places=5)

    def test_nwd_similarity_distance_decay(self) -> None:
        b_ref = torch.tensor([[100.0, 100.0, 110.0, 120.0]])
        b_near = torch.tensor([[102.0, 100.0, 112.0, 120.0]])
        b_mid = torch.tensor([[110.0, 100.0, 120.0, 120.0]])
        b_far = torch.tensor([[200.0, 100.0, 210.0, 120.0]])

        sim_near = float(compute_nwd_similarity(b_ref, b_near, constant=12.0)[0].item())
        sim_mid = float(compute_nwd_similarity(b_ref, b_mid, constant=12.0)[0].item())
        sim_far = float(compute_nwd_similarity(b_ref, b_far, constant=12.0)[0].item())

        self.assertGreater(sim_near, sim_mid)
        self.assertGreater(sim_mid, sim_far)
        self.assertGreater(sim_far, 0.0)
        self.assertLessEqual(sim_near, 1.0)

    def test_nwd_similarity_zero_overlap_tiny_boxes(self) -> None:
        # 4x4 tiny box at (10, 10) to (14, 14) and (16, 10) to (20, 14) [2px apart]
        b1 = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
        b2 = torch.tensor([[16.0, 10.0, 20.0, 14.0]])

        sim = compute_nwd_similarity(b1, b2, constant=12.0)
        self.assertGreater(float(sim[0].item()), 0.5)

    def test_standard_tal_starves_zero_iou_vs_nwd_tal_recovery(self) -> None:
        # Create a batch with 1 image, 1 GT box, and candidate anchors
        # GT box is tiny (3x6 px) at (100, 100) -> [98.5, 97.0, 101.5, 103.0]
        bs = 1
        num_anchors = 16
        num_classes = 2

        # Create anchor points in a 4x4 grid around (100, 100) spaced by 4px (stride 4)
        x_coords = torch.tensor([94.0, 98.0, 102.0, 106.0])
        y_coords = torch.tensor([94.0, 98.0, 102.0, 106.0])
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        anc_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (16, 2)

        gt_labels = torch.zeros((bs, 1, 1), dtype=torch.float32)  # class 0
        gt_bboxes = torch.tensor([[[98.5, 97.0, 101.5, 103.0]]], dtype=torch.float32)  # (1, 1, 4)
        mask_gt = torch.ones((bs, 1, 1), dtype=torch.bool)

        # Predicted boxes: slightly shifted tiny boxes such that IoU with GT is strictly 0.0
        # e.g., predicted boxes at (x+3, y+3)
        pd_bboxes = torch.zeros((bs, num_anchors, 4), dtype=torch.float32)
        for i in range(num_anchors):
            cx, cy = float(anc_points[i, 0]), float(anc_points[i, 1])
            # 3x6 predicted box shifted by 5px
            pd_bboxes[0, i] = torch.tensor([cx + 4.0, cy + 4.0, cx + 7.0, cy + 10.0])

        # Scores: positive detection confidence for class 0
        pd_scores = torch.full((bs, num_anchors, num_classes), 0.8, dtype=torch.float32)

        # 1. Standard TAL
        std_assigner = TaskAlignedAssigner(
            topk=4,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
            stride=[4, 8, 16, 32],
        )
        _, _, target_scores_std, fg_mask_std, _ = std_assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

        # Standard TAL gets 0 IoU, so alignment score collapses to 0 -> fg_mask is all False
        self.assertEqual(int(fg_mask_std.sum().item()), 0)
        self.assertEqual(float(target_scores_std.sum().item()), 0.0)

        # 2. NWD-Aware TAL
        nwd_assigner = NWDAwareTaskAlignedAssigner(
            topk=4,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
            stride=[4, 8, 16, 32],
            nwd_weight=0.5,
            nwd_constant=12.0,
            area_threshold=64.0,
            mode="scale_adaptive",
        )
        _, _, target_scores_nwd, fg_mask_nwd, _ = nwd_assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

        # NWD-Aware TAL maintains positive NWD overlap -> recovers positive foreground anchors
        self.assertGreater(int(fg_mask_nwd.sum().item()), 0)
        self.assertGreater(float(target_scores_nwd.sum().item()), 0.0)

    def test_scale_adaptive_preserves_large_box_iou(self) -> None:
        # For large GT box (area >> 64 px^2, e.g. 50x50 = 2500 px^2)
        bs = 1
        num_anchors = 4
        num_classes = 2

        anc_points = torch.tensor([[100.0, 100.0], [110.0, 100.0], [100.0, 110.0], [110.0, 110.0]])
        gt_labels = torch.zeros((bs, 1, 1), dtype=torch.float32)
        gt_bboxes = torch.tensor([[[80.0, 80.0, 130.0, 130.0]]], dtype=torch.float32)  # area = 2500
        mask_gt = torch.ones((bs, 1, 1), dtype=torch.bool)

        pd_bboxes = torch.tensor([[[85.0, 85.0, 125.0, 125.0]] * num_anchors], dtype=torch.float32)
        pd_scores = torch.full((bs, num_anchors, num_classes), 0.9, dtype=torch.float32)

        std_assigner = TaskAlignedAssigner(
            topk=2, num_classes=num_classes, alpha=0.5, beta=6.0, stride=[4, 8, 16, 32]
        )
        nwd_assigner = NWDAwareTaskAlignedAssigner(
            topk=2,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
            stride=[4, 8, 16, 32],
            nwd_weight=0.5,
            nwd_constant=12.0,
            area_threshold=64.0,
            mode="scale_adaptive",
        )

        _, _, target_scores_std, fg_mask_std, _ = std_assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )
        _, _, target_scores_nwd, fg_mask_nwd, _ = nwd_assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

        # Foreground masks and target scores should match identically for large objects
        self.assertTrue(torch.equal(fg_mask_std, fg_mask_nwd))
        self.assertTrue(torch.allclose(target_scores_std, target_scores_nwd, atol=1e-4))

    def test_build_task_aligned_assigner_factory(self) -> None:
        a_std = build_task_aligned_assigner(assigner_type="standard", topk=10, num_classes=2)
        self.assertIsInstance(a_std, TaskAlignedAssigner)
        self.assertNotIsInstance(a_std, NWDAwareTaskAlignedAssigner)

        a_nwd = build_task_aligned_assigner(
            assigner_type="nwd",
            topk=10,
            num_classes=2,
            nwd_weight=0.6,
            nwd_constant=15.0,
            area_threshold=128.0,
        )
        self.assertIsInstance(a_nwd, NWDAwareTaskAlignedAssigner)
        self.assertAlmostEqual(a_nwd.nwd_weight, 0.6)
        self.assertAlmostEqual(a_nwd.nwd_constant, 15.0)
        self.assertAlmostEqual(a_nwd.area_threshold, 128.0)

        with self.assertRaises(ValueError):
            build_task_aligned_assigner(assigner_type="invalid_type")

    def test_ignore_aware_loss_with_nwd_tal(self) -> None:
        wrapper = build_detection_model(self.p2_model_config)
        config = UnifiedHeadConfig(max_traffic_lights=32, max_arrows=32)
        attach_unified_relevance_head(wrapper, config=config)
        model = wrapper.model

        loss_std = IgnoreAwareDetectionLoss(model, tal_assigner_type="standard")
        self.assertIsInstance(loss_std.assigner, TaskAlignedAssigner)
        self.assertNotIsInstance(loss_std.assigner, NWDAwareTaskAlignedAssigner)

        loss_nwd = IgnoreAwareDetectionLoss(
            model,
            tal_assigner_type="nwd",
            tal_assigner_config={"nwd_weight": 0.5, "nwd_constant": 12.0, "area_threshold": 64.0},
        )
        self.assertIsInstance(loss_nwd.assigner, NWDAwareTaskAlignedAssigner)
        self.assertAlmostEqual(loss_nwd.assigner.nwd_weight, 0.5)

    def test_tlr_multitask_criterion_integration(self) -> None:
        wrapper = build_detection_model(self.p2_model_config)
        config = UnifiedHeadConfig(max_traffic_lights=32, max_arrows=32)
        attach_unified_relevance_head(wrapper, config=config)
        model = wrapper.model

        criterion = TLRMultiTaskCriterion(
            model,
            tal_assigner_type="nwd",
            tal_assigner_config={"nwd_weight": 0.5, "nwd_constant": 12.0, "area_threshold": 64.0},
        )
        self.assertEqual(criterion.tal_assigner_type, "nwd")
        self.assertIsInstance(criterion.traffic.assigner, NWDAwareTaskAlignedAssigner)

    def test_b4_config_loading_and_criterion_building(self) -> None:
        with open(self.b4_train_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertTrue(cfg.get("p2_enabled", False))
        self.assertIn("tal_assigner", cfg)
        self.assertIn(cfg["tal_assigner"]["type"], ("nwd", "nwd_aware_tal"))

        wrapper = build_detection_model(self.p2_model_config)
        arch_cfg = {
            k: v for k, v in cfg.get("architecture", {}).items()
            if k in UnifiedHeadConfig.__dataclass_fields__
        }
        attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(**arch_cfg))
        model = wrapper.model

        criterion = build_multitask_criterion(model, cfg)
        self.assertIsInstance(criterion.traffic.assigner, NWDAwareTaskAlignedAssigner)
        self.assertAlmostEqual(criterion.traffic.assigner.nwd_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
