"""Unit and integration tests for Conformal Temperature Scaling & Conformal Risk Control Tau95 (Ticket 08)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from tlr_yolo_mtl.deployment.postprocess import postprocess_multitask_outputs
from tlr_yolo_mtl.evaluation.calibration import (
    ConformalRiskController,
    ConformalSafetyGate,
    ConformalStatePredictor,
    ConformalThresholdResult,
    MultiTaskTemperatureCalibrator,
    TemperatureFit,
    apply_temperature,
    compute_brier_multiclass,
    compute_classwise_ece,
    compute_maximum_calibration_error,
    compute_multiclass_ece,
    compute_nll,
    fit_temperature,
)


class TestConformalTemperatureScaling(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def test_metric_computations_binary_and_multiclass(self) -> None:
        # Binary case
        y_bin = np.array([1, 1, 0, 0], dtype=np.int64)
        p_bin_perfect = np.array([0.99, 0.95, 0.05, 0.01])
        p_bin_bad = np.array([0.10, 0.20, 0.80, 0.90])

        nll_perfect = compute_nll(y_bin, p_bin_perfect)
        nll_bad = compute_nll(y_bin, p_bin_bad)
        self.assertLess(nll_perfect, nll_bad)
        self.assertGreater(nll_perfect, 0.0)

        ece = compute_multiclass_ece(y_bin, p_bin_perfect, bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

        # Multi-class case (4 classes: Red, Yellow, Green, Off)
        y_multi = np.array([0, 1, 2, 3], dtype=np.int64)
        p_multi = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.10, 0.70, 0.10, 0.10],
            [0.05, 0.05, 0.80, 0.10],
            [0.05, 0.05, 0.10, 0.80],
        ])
        nll_multi = compute_nll(y_multi, p_multi)
        self.assertLess(nll_multi, 1.0)

        multi_ece = compute_multiclass_ece(y_multi, p_multi, bins=5)
        self.assertGreaterEqual(multi_ece, 0.0)
        self.assertLessEqual(multi_ece, 0.5)

        class_ece = compute_classwise_ece(y_multi, p_multi, num_classes=4, bins=5)
        self.assertEqual(len(class_ece), 4)
        for c in range(4):
            self.assertGreaterEqual(class_ece[c], 0.0)

        mce = compute_maximum_calibration_error(y_multi, p_multi, bins=5)
        self.assertGreaterEqual(mce, 0.0)

        brier = compute_brier_multiclass(y_multi, p_multi)
        self.assertGreaterEqual(brier, 0.0)
        self.assertLess(brier, 1.0)

    def test_temperature_fitting_improves_or_preserves_nll(self) -> None:
        # Overconfident logits on 4-class problem: high logits but 25% misclassification rate
        n_samples = 500
        pred_labels = torch.randint(0, 4, (n_samples,))
        one_hot = F.one_hot(pred_labels, num_classes=4).float()
        # High magnitude logits
        overconfident_logits = one_hot * 6.0 + torch.randn(n_samples, 4) * 0.5
        
        # Ground truth targets match predictions for 75% of samples, but differ on 25%
        true_labels = pred_labels.clone()
        corrupt_indices = torch.randperm(n_samples)[: int(0.25 * n_samples)]
        true_labels[corrupt_indices] = (true_labels[corrupt_indices] + torch.randint(1, 4, (len(corrupt_indices),))) % 4

        fit = fit_temperature(overconfident_logits, true_labels, compute_diagnostics=True)

        self.assertTrue(math.isfinite(fit.temperature))
        self.assertGreater(fit.temperature, 1.0)  # Overconfident models with errors require T > 1.0
        self.assertLessEqual(fit.loss_after, fit.loss_before + 1e-5)
        self.assertEqual(fit.valid_samples, n_samples)
        self.assertIsNotNone(fit.ece_before)
        self.assertIsNotNone(fit.ece_after)
        self.assertLessEqual(fit.ece_after, fit.ece_before + 1e-4)

    def test_multitask_temperature_calibrator_fit_and_apply(self) -> None:
        calibrator = MultiTaskTemperatureCalibrator()

        state_logits = torch.randn(200, 4) * 2.0
        state_targets = torch.randint(0, 4, (200,))

        rel_logits = torch.randn(200) * 2.0
        rel_targets = torch.randint(0, 2, (200,))

        fits = calibrator.fit(
            state_logits=state_logits,
            state_targets=state_targets,
            relevance_logits=rel_logits,
            relevance_targets=rel_targets,
        )

        self.assertIn("state", fits)
        self.assertIn("relevance", fits)
        self.assertGreater(calibrator.state_temperature, 0.0)
        self.assertGreater(calibrator.relevance_temperature, 0.0)

        # Apply calibration
        test_state_logits = torch.randn(2, 4, 10)
        cal_state_logits = calibrator.calibrate_state_logits(test_state_logits)
        self.assertEqual(cal_state_logits.shape, (2, 4, 10))

        cal_state_probs = calibrator.calibrate_state_probabilities(test_state_logits)
        self.assertTrue(torch.allclose(cal_state_probs.sum(dim=1), torch.ones(2, 10)))

        test_rel_logits = torch.randn(2, 1, 10)
        cal_rel_logits = calibrator.calibrate_relevance_logits(test_rel_logits)
        cal_rel_probs = calibrator.calibrate_relevance_probabilities(test_rel_logits)
        self.assertTrue((cal_rel_probs >= 0.0).all() and (cal_rel_probs <= 1.0).all())

    def test_calibrator_serialization(self) -> None:
        calibrator = MultiTaskTemperatureCalibrator(
            state_temperature=1.234,
            relevance_temperature=1.056,
        )
        d = calibrator.to_dict()
        restored = MultiTaskTemperatureCalibrator.from_dict(d)
        self.assertAlmostEqual(restored.state_temperature, 1.234)
        self.assertAlmostEqual(restored.relevance_temperature, 1.056)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            calibrator.save(tmp_path)
            loaded = MultiTaskTemperatureCalibrator.load(tmp_path)
            self.assertAlmostEqual(loaded.state_temperature, 1.234)
            self.assertAlmostEqual(loaded.relevance_temperature, 1.056)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_conformal_state_predictor_coverage_and_sets(self) -> None:
        n_cal = 600
        n_test = 600
        num_classes = 4

        # Generate realistic softmax distributions
        logits_cal = torch.randn(n_cal, num_classes) * 1.5
        targets_cal = torch.randint(0, num_classes, (n_cal,))
        # Boost true class slightly
        for i in range(n_cal):
            logits_cal[i, targets_cal[i]] += 2.0
        probs_cal = F.softmax(logits_cal, dim=-1).numpy()

        logits_test = torch.randn(n_test, num_classes) * 1.5
        targets_test = torch.randint(0, num_classes, (n_test,))
        for i in range(n_test):
            logits_test[i, targets_test[i]] += 2.0
        probs_test = F.softmax(logits_test, dim=-1).numpy()

        for method in ("lac", "aps"):
            predictor = ConformalStatePredictor(method=method)
            predictor.fit(probs_cal, targets_cal)

            q_95 = predictor.get_quantile(alpha=0.05)
            self.assertGreater(q_95, 0.0)

            # Predict sets on test split
            eval_res = predictor.evaluate_prediction_sets(probs_test, targets_test, alpha=0.05)
            self.assertGreaterEqual(eval_res["empirical_coverage"], 0.90, f"Method {method} coverage must exceed 90% for alpha=0.05")
            self.assertGreaterEqual(eval_res["avg_set_size"], 1.0)
            self.assertLessEqual(eval_res["avg_set_size"], 4.0)
            self.assertEqual(eval_res["empty_rate"], 0.0)

    def test_conformal_risk_controller_safety_threshold_monotonicity(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 500
        # Positive scores centered around 0.85, negative around 0.20
        pos_scores = rng.beta(8, 2, size=n_samples)
        neg_scores = rng.beta(2, 8, size=n_samples)
        scores = np.concatenate([pos_scores, neg_scores])
        targets = np.concatenate([np.ones(n_samples), np.zeros(n_samples)]).astype(int)

        res_90 = ConformalRiskController.solve_safety_threshold(scores, targets, target_recall=0.90)
        res_95 = ConformalRiskController.solve_safety_threshold(scores, targets, target_recall=0.95)
        res_975 = ConformalRiskController.solve_safety_threshold(scores, targets, target_recall=0.975)

        self.assertTrue(res_90.guarantee_satisfied)
        self.assertTrue(res_95.guarantee_satisfied)
        self.assertTrue(res_975.guarantee_satisfied)

        # Monotonicity: higher recall constraint implies lower or equal threshold
        self.assertGreaterEqual(res_90.fitted_threshold, res_95.fitted_threshold)
        self.assertGreaterEqual(res_95.fitted_threshold, res_975.fitted_threshold)

        # Calibration recall must meet target constraint
        self.assertGreaterEqual(res_975.calibration_recall, 0.975)

        # Serialization
        d = res_975.to_dict()
        restored = ConformalThresholdResult.from_dict(d)
        self.assertEqual(restored.target_recall, res_975.target_recall)
        self.assertEqual(restored.fitted_threshold, res_975.fitted_threshold)

    def test_conformal_safety_gate_runtime_evaluation(self) -> None:
        gate = ConformalSafetyGate(
            tau_nominal=0.25,
            tau_safety_95=0.35,
            tau_safety_975=0.20,
            red_prob_threshold=0.40,
        )

        # Batch of 2 images, 3 traffic light candidates
        # Candidate 0: Clear relevant red (Red prob 0.90, Rel prob 0.80 -> joint 0.72)
        # Candidate 1: Borderline red (Red prob 0.60, Rel prob 0.35 -> joint 0.21)
        # Candidate 2: Green light (Red prob 0.05, Rel prob 0.90 -> joint 0.045)
        state_probs = torch.tensor([
            [
                [0.90, 0.60, 0.05],  # Red (class 0)
                [0.05, 0.10, 0.05],  # Yellow
                [0.03, 0.20, 0.85],  # Green
                [0.02, 0.10, 0.05],  # Off
            ],
            [
                [0.10, 0.95, 0.02],
                [0.10, 0.02, 0.02],
                [0.70, 0.02, 0.90],
                [0.10, 0.01, 0.06],
            ]
        ])  # [2, 4, 3]

        relevance_probs = torch.tensor([
            [[0.80, 0.35, 0.90]],
            [[0.20, 0.85, 0.10]],
        ])  # [2, 1, 3]

        detection_scores = torch.tensor([
            [0.85, 0.50, 0.90],
            [0.40, 0.80, 0.70],
        ])  # [2, 3]

        safety_eval = gate.evaluate_safety_gate(state_probs, relevance_probs, detection_scores)

        self.assertIn("is_safety_95_certified", safety_eval)
        self.assertIn("is_safety_975_certified", safety_eval)
        self.assertIn("emergency_brake_trigger", safety_eval)

        # Image 0 Candidate 0 must be safety certified
        self.assertTrue(safety_eval["is_safety_95_certified"][0, 0])
        self.assertTrue(safety_eval["is_safety_975_certified"][0, 0])
        self.assertTrue(safety_eval["emergency_brake_trigger"][0, 0])

        # Image 0 Candidate 1: joint score 0.21 >= tau_safety_975 (0.20), but < tau_safety_95 (0.35)
        self.assertFalse(safety_eval["is_safety_95_certified"][0, 1])
        self.assertTrue(safety_eval["is_safety_975_certified"][0, 1])

        # Image 0 Candidate 2: Green light -> must not trigger red safety cert
        self.assertFalse(safety_eval["is_safety_95_certified"][0, 2])
        self.assertFalse(safety_eval["is_safety_975_certified"][0, 2])

    def test_postprocess_multitask_outputs_with_calibration_and_gate(self) -> None:
        b, n_anchors = 1, 20
        detection = torch.zeros(b, 6, n_anchors)
        detection[:, 0:4, :] = torch.tensor([100.0, 100.0, 20.0, 40.0]).view(1, 4, 1)
        detection[:, 4, :5] = 0.80  # Class 0: TL
        detection[:, 5, 5:10] = 0.80  # Class 1: Arrow

        states = torch.randn(b, 4, n_anchors)
        rounds = torch.randn(b, 1, n_anchors)
        maneuvers = torch.randn(b, 3, n_anchors)
        ego_lane = torch.randn(b, 1, n_anchors)
        traffic_candidates = torch.arange(8, dtype=torch.long).unsqueeze(0)
        traffic_candidate_valid = torch.ones(b, 8, dtype=torch.bool)
        arrow_candidates = torch.arange(8, 16, dtype=torch.long).unsqueeze(0)
        arrow_candidate_valid = torch.ones(b, 8, dtype=torch.bool)
        relevance = torch.randn(b, 1, 8)
        attention = torch.zeros(b, 4, 8, 8)

        outputs_11 = (
            detection,
            states,
            rounds,
            maneuvers,
            ego_lane,
            traffic_candidates,
            traffic_candidate_valid,
            arrow_candidates,
            arrow_candidate_valid,
            relevance,
            attention,
        )

        gate = ConformalSafetyGate(
            tau_nominal=0.25,
            tau_safety_95=0.35,
            tau_safety_975=0.20,
        )

        result = postprocess_multitask_outputs(
            outputs_11,
            traffic_confidence=0.25,
            arrow_confidence=0.25,
            temperature_state=1.20,
            temperature_relevance=1.10,
            conformal_safety_gate=gate,
        )

        self.assertIn("traffic_lights", result)
        self.assertIn("road_arrows", result)
        tl = result["traffic_lights"]
        self.assertIn("state_probabilities", tl)
        self.assertIn("relevance_probabilities", tl)
        self.assertIn("is_safety_975_certified", tl)
        self.assertIn("emergency_brake_trigger", tl)


if __name__ == "__main__":
    unittest.main()
