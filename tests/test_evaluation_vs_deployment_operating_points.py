"""Unit and regression tests for E37: Rigorous Separation of Evaluation AP and Deployment Operating Points."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e37_evaluation_vs_deployment import (
    ConfidenceSweepPoint,
    NmsIouSweepPoint,
    format_markdown_report,
)
from tlr_yolo_mtl.evaluation.metrics import (
    FINE_AREA_BUCKETS,
    FINE_SIDE_BUCKETS,
    _get_bucket_name,
    compute_detection_and_attribute_map,
    compute_granular_scale_metrics,
)


class TestEvaluationVsDeploymentOperatingPoints(unittest.TestCase):
    def test_fine_scale_bucket_definitions(self):
        """Verify FINE_SIDE_BUCKETS and FINE_AREA_BUCKETS partition scale ranges."""
        self.assertIn("<8", FINE_SIDE_BUCKETS)
        self.assertIn("8-16", FINE_SIDE_BUCKETS)
        self.assertIn("16-32", FINE_SIDE_BUCKETS)
        self.assertIn(">32", FINE_SIDE_BUCKETS)

        self.assertEqual(FINE_SIDE_BUCKETS["<8"], (0.0, 8.0))
        self.assertEqual(FINE_SIDE_BUCKETS["8-16"], (8.0, 16.0))
        self.assertEqual(FINE_SIDE_BUCKETS["16-32"], (16.0, 32.0))
        self.assertEqual(FINE_SIDE_BUCKETS[">32"], (32.0, float("inf")))

        # Test bucket retrieval
        self.assertEqual(_get_bucket_name(3.5, FINE_SIDE_BUCKETS), "<8")
        self.assertEqual(_get_bucket_name(7.99, FINE_SIDE_BUCKETS), "<8")
        self.assertEqual(_get_bucket_name(8.0, FINE_SIDE_BUCKETS), "8-16")
        self.assertEqual(_get_bucket_name(15.5, FINE_SIDE_BUCKETS), "8-16")
        self.assertEqual(_get_bucket_name(16.0, FINE_SIDE_BUCKETS), "16-32")
        self.assertEqual(_get_bucket_name(31.9, FINE_SIDE_BUCKETS), "16-32")
        self.assertEqual(_get_bucket_name(32.0, FINE_SIDE_BUCKETS), ">32")
        self.assertEqual(_get_bucket_name(100.0, FINE_SIDE_BUCKETS), ">32")

    def test_compute_detection_and_attribute_map_fine_scale(self):
        """Verify fine-grained scale APs are computed in compute_detection_and_attribute_map."""
        img_h, img_w = 800, 1600
        # Create 4 GT boxes with different sizes:
        # Box 0: width=4px, height=4px -> side=4px (<8px)
        # Box 1: width=10px, height=10px -> side=10px (8-16px)
        # Box 2: width=20px, height=20px -> side=20px (16-32px)
        # Box 3: width=40px, height=40px -> side=40px (>32px)
        def box(cx_px, cy_px, w_px, h_px):
            return np.array([
                (cx_px - w_px / 2) / img_w,
                (cy_px - h_px / 2) / img_h,
                (cx_px + w_px / 2) / img_w,
                (cy_px + h_px / 2) / img_h,
            ])

        gt_boxes = np.stack([
            box(100, 100, 4, 4),
            box(200, 200, 10, 10),
            box(300, 300, 20, 20),
            box(400, 400, 40, 40),
        ])
        gt_classes = np.array([0, 0, 0, 0], dtype=np.int64)

        # High quality predictions matching all boxes
        pred_boxes = np.copy(gt_boxes)
        pred_scores = np.array([0.95, 0.90, 0.85, 0.80])
        pred_classes = np.array([0, 0, 0, 0], dtype=np.int64)

        res = compute_detection_and_attribute_map(
            pred_boxes_list=[pred_boxes],
            pred_scores_list=[pred_scores],
            pred_classes_list=[pred_classes],
            gt_boxes_list=[gt_boxes],
            gt_classes_list=[gt_classes],
            image_shape=(img_h, img_w),
        )

        self.assertIn("ap_tl_sub8px", res)
        self.assertIn("ap_tl_8_16px", res)
        self.assertIn("ap_tl_16_32px", res)
        self.assertIn("ap_tl_gt32px", res)

        self.assertGreaterEqual(res["ap_tl_sub8px"], 0.95)
        self.assertGreaterEqual(res["ap_tl_8_16px"], 0.95)
        self.assertGreaterEqual(res["ap_tl_16_32px"], 0.95)
        self.assertGreaterEqual(res["ap_tl_gt32px"], 0.95)

    def test_compute_granular_scale_metrics_fine_buckets(self):
        """Verify compute_granular_scale_metrics returns fine_side_buckets and fine_area_buckets."""
        img_h, img_w = 800, 1600
        gt_boxes = np.array([[0.1, 0.1, 0.104, 0.108]])  # ~6.4 x 6.4 px
        gt_classes = np.array([0], dtype=np.int64)

        pred_boxes = np.array([[0.1, 0.1, 0.104, 0.108]])
        pred_scores = np.array([0.85])
        pred_classes = np.array([0], dtype=np.int64)

        res = compute_granular_scale_metrics(
            pred_boxes_list=[pred_boxes],
            pred_scores_list=[pred_scores],
            pred_classes_list=[pred_classes],
            gt_boxes_list=[gt_boxes],
            gt_classes_list=[gt_classes],
            target_class=0,
            image_shape=(img_h, img_w),
            conf_threshold=0.05,
        )

        self.assertIn("fine_side_buckets", res)
        self.assertIn("fine_area_buckets", res)
        self.assertIn("<8", res["fine_side_buckets"])
        self.assertEqual(res["fine_side_buckets"]["<8"]["n_gt"], 1)
        self.assertEqual(res["fine_side_buckets"]["<8"]["n_tp"], 1)

    def test_markdown_report_formatting(self):
        """Verify format_markdown_report generates valid markdown with complete sections."""
        conf_sweep = [
            ConfidenceSweepPoint(
                conf_threshold=0.001,
                map50=0.85,
                map50_95=0.55,
                ap_tl_50=0.75,
                ap_arrow_50=0.95,
                ap_small=0.72,
                ap_medium=0.90,
                ap_tl_sub8px=0.30,
                ap_tl_8_16px=0.68,
                ap_tl_16_32px=0.88,
                ap_tl_gt32px=0.96,
                state_acc=0.94,
                relevance_auprc=0.91,
            ),
            ConfidenceSweepPoint(
                conf_threshold=0.25,
                map50=0.78,
                map50_95=0.50,
                ap_tl_50=0.65,
                ap_arrow_50=0.92,
                ap_small=0.45,
                ap_medium=0.87,
                ap_tl_sub8px=0.10,
                ap_tl_8_16px=0.52,
                ap_tl_16_32px=0.82,
                ap_tl_gt32px=0.95,
                state_acc=0.95,
                relevance_auprc=0.92,
            ),
        ]
        nms_sweep = [
            NmsIouSweepPoint(
                iou_threshold=0.50,
                conf_threshold=0.001,
                map50=0.85,
                map50_95=0.55,
                ap_tl_50=0.75,
                ap_arrow_50=0.95,
                ap_small=0.72,
                ap_tl_sub8px=0.30,
                ap_tl_8_16px=0.68,
                ap_tl_16_32px=0.88,
            )
        ]
        eval_bench = {"ap_tl_sub8px": 0.30, "ap_tl_8_16px": 0.68, "ap_tl_16_32px": 0.88}
        deploy_bench = {"ap_tl_sub8px": 0.10, "ap_tl_8_16px": 0.52, "ap_tl_16_32px": 0.82}

        report = format_markdown_report(
            conf_sweep, nms_sweep, eval_bench, deploy_bench, "best_composite.pt"
        )
        self.assertIn("# Ticket E37 Diagnostic & Empirical Audit", report)
        self.assertIn("Confidence Threshold Sensitivity Matrix", report)
        self.assertIn("Scale-Stratified Perception Floor Comparison", report)
        self.assertIn("PASSED", report)


if __name__ == "__main__":
    unittest.main()
