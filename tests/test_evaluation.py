from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tlr_yolo_mtl.evaluation.calibration import fit_temperature
from tlr_yolo_mtl.evaluation.matching import greedy_iou_match, pairwise_iou
from tlr_yolo_mtl.evaluation.metrics import (
    binary_average_precision,
    binary_classification_metrics,
    multiclass_confusion_matrix,
    multiclass_metrics,
    multilabel_metrics,
    threshold_for_minimum_recall,
    validation_selection_score,
)


class EvaluationTests(unittest.TestCase):
    def test_greedy_match_retains_prediction_indices(self) -> None:
        predicted = [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]]
        targets = [[0, 0, 10, 10], [20, 20, 30, 30]]
        matches, false_positives, missed = greedy_iou_match(
            predicted, [0.9, 0.8, 0.7], targets
        )
        self.assertEqual(
            [(value.prediction_index, value.target_index) for value in matches],
            [(0, 0), (2, 1)],
        )
        self.assertEqual(false_positives, [1])
        self.assertEqual(missed, [])

    def test_pairwise_iou(self) -> None:
        values = pairwise_iou([[0, 0, 2, 2]], [[1, 1, 3, 3]])
        self.assertAlmostEqual(float(values[0, 0]), 1 / 7)

    def test_binary_metrics_and_average_precision(self) -> None:
        targets = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.1]
        self.assertAlmostEqual(binary_average_precision(targets, scores), (1 + 2 / 3) / 2)
        metrics = binary_classification_metrics(targets, scores, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 1, 0))

    def test_multiclass_and_multilabel_metrics(self) -> None:
        confusion = multiclass_confusion_matrix([0, 1, 2, 2], [0, 1, 1, 2], classes=3)
        metrics = multiclass_metrics(confusion)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        arrows = multilabel_metrics(
            [[1, 1, 0], [0, 0, 1]],
            [[0.9, 0.8, 0.1], [0.2, 0.1, 0.7]],
        )
        self.assertEqual(arrows["exact_match_accuracy"], 1.0)

    def test_recall_constrained_threshold(self) -> None:
        selected = threshold_for_minimum_recall(
            [1, 1, 0, 0], [0.9, 0.6, 0.7, 0.1], minimum_recall=1.0
        )
        self.assertEqual(selected["threshold"], 0.6)
        self.assertEqual(selected["recall"], 1.0)

    def test_validation_selection_score(self) -> None:
        score = validation_selection_score(
            {
                "traffic_light_tiny_ap": 1.0,
                "state_macro_f1": 1.0,
                "pictogram_macro_f1": 1.0,
                "arrow_ap": 1.0,
                "relevance_auprc": 1.0,
            }
        )
        self.assertAlmostEqual(score, 1.0)

    def test_temperature_search_does_not_worsen_nll(self) -> None:
        logits = torch.tensor([5.0, 4.0, -3.0, -5.0])
        targets = torch.tensor([1, 0, 0, 0])
        fit = fit_temperature(logits, targets)
        self.assertTrue(math.isfinite(fit.temperature))
        self.assertLessEqual(fit.loss_after, fit.loss_before + 1e-6)

    def test_evaluate_validation_epoch_structure(self) -> None:
        from tlr_yolo_mtl.evaluation.evaluator import evaluate_validation_epoch

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Linear(1, 1)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                b = x.shape[0]
                decoded = torch.zeros((b, 6, 10), device=x.device)
                decoded[:, 4, :5] = 0.8
                decoded[:, 5, 5:] = 0.8
                raw = {
                    "traffic_candidate_boxes": torch.zeros((b, 32, 4), device=x.device),
                    "traffic_candidate_valid": torch.ones((b, 32), device=x.device, dtype=torch.bool),
                    "traffic_candidate_scores": torch.ones((b, 32), device=x.device) * 0.8,
                    "traffic_candidate_indices": torch.zeros((b, 32), device=x.device, dtype=torch.long),
                    "relevance_logits": torch.zeros((b, 1, 32), device=x.device),
                    "state_logits": torch.zeros((b, 4, 10), device=x.device),
                    "round_logits": torch.zeros((b, 1, 10), device=x.device),
                    "maneuver_logits": torch.zeros((b, 3, 10), device=x.device),
                    "ego_lane_logits": torch.zeros((b, 1, 10), device=x.device),
                    "dense_local_relevance_logits": torch.zeros((b, 1, 10), device=x.device),
                    "attention_enabled_flag": torch.tensor(1.0, device=x.device),
                }
                return decoded, raw

        batch = {
            "img": torch.zeros((2, 3, 32, 32)),
            "object_batch_idx": torch.tensor([0, 1]),
            "object_cls": torch.tensor([[0.0], [1.0]]),
            "object_bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]),
            "object_state": torch.tensor([0, -1]),
            "object_round": torch.tensor([1, -1]),
            "object_maneuver": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            "object_relevance": torch.tensor([1, -1]),
            "object_ego_lane": torch.tensor([-1, 1]),
            "unified_detection_valid": torch.tensor([True, True]),
            "traffic_relevance_valid": torch.tensor([True, False]),
        }
        val_loader = [batch]
        model = DummyModel()
        result = evaluate_validation_epoch(
            model,
            val_loader,  # type: ignore
            device=torch.device("cpu"),
            amp_enabled=False,
            granular_scale_metrics=True,
        )
        self.assertIn("selection_score", result)
        self.assertIn("detection", result)
        self.assertIn("relevance", result)
        self.assertIn("attributes", result)
        self.assertIn("granular_scale", result)
        self.assertEqual(result["samples_evaluated"], 2)

    def test_granular_scale_metrics_computation(self) -> None:
        from tlr_yolo_mtl.evaluation.metrics import (
            AREA_BUCKETS,
            SIDE_BUCKETS,
            compute_granular_scale_metrics,
        )

        # Image size 800 x 1600
        # GT 1: Tiny TL (area = 20 px^2, w=4, h=5) -> area in "<32", min_side in "4-6"
        # GT 2: Medium TL (area = 200 px^2, w=10, h=20) -> area in "128-256", min_side in "8-12"
        # Pred 1: Exactly matches GT 1 (high score 0.95)
        # Pred 2: Slightly shifted match to GT 2 (score 0.85, dx=2px, dy=1px)
        # Pred 3: FP box in area >512 (w=30, h=30 -> area=900)

        # In normalized coordinates [x1, y1, x2, y2]
        # GT 1: cx=100, cy=100, w=4, h=5 -> x1=98/1600, y1=97.5/800, x2=102/1600, y2=102.5/800
        gt_boxes = np.array(
            [
                [98.0 / 1600.0, 97.5 / 800.0, 102.0 / 1600.0, 102.5 / 800.0],
                [300.0 / 1600.0, 200.0 / 800.0, 310.0 / 1600.0, 220.0 / 800.0],
            ],
            dtype=float,
        )
        gt_classes = np.array([0, 0], dtype=np.int64)

        pred_boxes = np.array(
            [
                [98.0 / 1600.0, 97.5 / 800.0, 102.0 / 1600.0, 102.5 / 800.0],
                [302.0 / 1600.0, 201.0 / 800.0, 312.0 / 1600.0, 221.0 / 800.0],
                [500.0 / 1600.0, 500.0 / 800.0, 530.0 / 1600.0, 530.0 / 800.0],
            ],
            dtype=float,
        )
        pred_scores = np.array([0.95, 0.85, 0.70], dtype=float)
        pred_classes = np.array([0, 0, 0], dtype=np.int64)

        metrics = compute_granular_scale_metrics(
            [pred_boxes],
            [pred_scores],
            [pred_classes],
            [gt_boxes],
            [gt_classes],
            image_shape=(800, 1600),
            conf_threshold=0.50,
        )

        self.assertIn("area_buckets", metrics)
        self.assertIn("side_buckets", metrics)

        # Check <32 bucket
        tiny_area = metrics["area_buckets"]["<32"]
        self.assertEqual(tiny_area["n_gt"], 1)
        self.assertEqual(tiny_area["n_tp"], 1)
        self.assertEqual(tiny_area["recall"], 1.0)
        self.assertAlmostEqual(tiny_area["ap50"], 1.0, delta=0.01)
        self.assertAlmostEqual(tiny_area["mean_dr"], 0.0, places=4)

        # Check 128-256 bucket
        med_area = metrics["area_buckets"]["128-256"]
        self.assertEqual(med_area["n_gt"], 1)
        self.assertEqual(med_area["n_tp"], 1)
        self.assertAlmostEqual(med_area["mean_dx"], 2.0, places=3)
        self.assertAlmostEqual(med_area["mean_dy"], 1.0, places=3)

        # Check >512 bucket (has 0 GT, 1 FP)
        large_area = metrics["area_buckets"][">512"]
        self.assertEqual(large_area["n_gt"], 0)
        self.assertEqual(large_area["n_fp"], 1)
        self.assertEqual(large_area["recall"], 0.0)

        # Check side buckets
        side_4_6 = metrics["side_buckets"]["4-6"]
        self.assertEqual(side_4_6["n_gt"], 1)
        self.assertEqual(side_4_6["n_tp"], 1)

    def test_pairwise_nwd_and_greedy_nwd(self) -> None:
        from tlr_yolo_mtl.evaluation.matching import greedy_nwd_match, pairwise_nwd

        boxes_a = [[10.0, 10.0, 20.0, 30.0], [50.0, 50.0, 60.0, 70.0]]
        boxes_b = [[11.0, 10.5, 21.0, 30.5], [100.0, 100.0, 110.0, 120.0]]

        nwd_matrix = pairwise_nwd(boxes_a, boxes_b, constant=12.0)
        self.assertEqual(nwd_matrix.shape, (2, 2))
        self.assertGreater(nwd_matrix[0, 0], 0.8)  # close boxes have high NWD
        self.assertLess(nwd_matrix[0, 1], 0.01)   # far boxes have near 0 NWD

        scores = [0.95, 0.80]
        matches, un_preds, un_tgts = greedy_nwd_match(boxes_a, scores, boxes_b, nwd_threshold=0.5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].prediction_index, 0)
        self.assertEqual(matches[0].target_index, 0)
        self.assertEqual(un_preds, [1])
        self.assertEqual(un_tgts, [1])

    def test_pairwise_and_greedy_center_distance(self) -> None:
        from tlr_yolo_mtl.evaluation.matching import (
            greedy_center_distance_match,
            pairwise_center_distance,
        )

        boxes_a = [[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 60.0, 60.0]]  # centers: (15, 15), (55, 55)
        boxes_b = [[12.0, 12.0, 22.0, 22.0], [50.0, 50.0, 60.0, 60.0]]  # centers: (17, 17), (55, 55)

        distances = pairwise_center_distance(boxes_a, boxes_b)
        self.assertEqual(distances.shape, (2, 2))
        self.assertAlmostEqual(distances[0, 0], np.sqrt(8.0), places=3)
        self.assertAlmostEqual(distances[1, 1], 0.0, places=3)

        matches, un_preds, un_tgts = greedy_center_distance_match(
            boxes_a, [0.9, 0.8], boxes_b, max_distance_px=5.0
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(un_preds, [])
        self.assertEqual(un_tgts, [])

    def test_fixed_topk_candidates(self) -> None:
        from tlr_yolo_mtl.model.unified import fixed_topk_candidates

        scores = torch.tensor([[0.1, 0.9, 0.4, 0.8, 0.2]])
        indices, selected_scores, valid = fixed_topk_candidates(scores, k=3, threshold=0.3)

        self.assertEqual(indices.shape, (1, 3))
        self.assertEqual(indices[0].tolist(), [1, 3, 2])  # indices of 0.9, 0.8, 0.4
        self.assertTrue(torch.allclose(selected_scores[0], torch.tensor([0.9, 0.8, 0.4])))
        self.assertEqual(valid[0].tolist(), [True, True, True])

        # Test with threshold filtering out lowest
        indices, selected_scores, valid = fixed_topk_candidates(scores, k=3, threshold=0.5)
        self.assertEqual(valid[0].tolist(), [True, True, False])  # 0.4 is below 0.5


if __name__ == "__main__":
    unittest.main()

