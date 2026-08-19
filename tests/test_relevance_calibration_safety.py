"""Tests for E19 Relevance Calibration and Safety Operating Points Pipeline."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_relevance_safety import (
    SampleRecord,
    compute_nll,
    compute_reliability_curve,
    deterministic_split_flag,
    optimize_f1_threshold,
    optimize_safety_threshold,
    run_e19_calibration,
)
from tlr_yolo_mtl.evaluation.calibration import apply_temperature, fit_temperature
from tlr_yolo_mtl.evaluation.metrics import expected_calibration_error, brier_score


def test_deterministic_split_flag():
    # Determinism check
    img_ids = [f"image_{i:04d}.png" for i in range(500)]
    split_1 = [deterministic_split_flag(img_id) for img_id in img_ids]
    split_2 = [deterministic_split_flag(img_id) for img_id in img_ids]
    assert split_1 == split_2, "Split assignment must be strictly deterministic"

    # 50/50 Balance check
    n_cal = sum(split_1)
    ratio = n_cal / len(img_ids)
    assert 0.40 <= ratio <= 0.60, f"Split ratio {ratio:.2f} should be approximately 50%"

    # Salt sensitivity
    split_other = [deterministic_split_flag(img_id, salt="other_salt") for img_id in img_ids]
    assert split_1 != split_other, "Different salts should produce distinct hash splits"


def test_reliability_curve_and_nll():
    targets = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.95, 0.6])

    curve = compute_reliability_curve(targets, probs, bins=5)
    assert len(curve["bin_centers"]) == 5
    assert len(curve["bin_accs"]) == 5
    assert sum(curve["bin_counts"]) == len(targets)
    assert np.allclose(curve["edges"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    nll = compute_nll(targets, probs)
    assert nll > 0.0
    # Perfect predictions have lower NLL
    perfect_probs = np.array([0.99 if y == 1 else 0.01 for y in targets])
    assert compute_nll(targets, perfect_probs) < nll


def test_safety_threshold_optimization():
    targets = np.array([1] * 100 + [0] * 100)
    # Positive scores centered around 0.8, negatives around 0.3
    rng = np.random.default_rng(42)
    scores_pos = rng.uniform(0.5, 0.95, size=100)
    scores_neg = rng.uniform(0.05, 0.6, size=100)
    scores = np.concatenate([scores_pos, scores_neg])

    # Tier 1 (90%)
    tau_90, prec_90, rec_90 = optimize_safety_threshold(targets, scores, target_recall=0.90)
    assert rec_90 >= 0.90, f"Recall {rec_90:.3f} must satisfy constraint >= 0.90"
    assert prec_90 > 0.50

    # Tier 2 (95%)
    tau_95, prec_95, rec_95 = optimize_safety_threshold(targets, scores, target_recall=0.95)
    assert rec_95 >= 0.95, f"Recall {rec_95:.3f} must satisfy constraint >= 0.95"

    # Tier 3 (97.5%)
    tau_975, prec_975, rec_975 = optimize_safety_threshold(targets, scores, target_recall=0.975)
    assert rec_975 >= 0.975, f"Recall {rec_975:.3f} must satisfy constraint >= 0.975"

    # Threshold hierarchy: higher recall constraint requires lower or equal decision threshold
    assert tau_90 >= tau_95 >= tau_975, f"Thresholds must be monotonic: tau_90={tau_90}, tau_95={tau_95}, tau_975={tau_975}"


def test_f1_threshold_optimization():
    targets = np.array([1] * 50 + [0] * 50)
    scores = np.array([0.8] * 50 + [0.2] * 50)
    tau, f1, prec, rec = optimize_f1_threshold(targets, scores)
    assert 0.2 < tau < 0.8
    assert f1 == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_e19_calibration_pipeline(tmp_path):
    # Construct synthetic sample records
    rng = np.random.default_rng(123)
    records: list[SampleRecord] = []

    for i in range(400):
        img_id = f"test_img_{i:04d}"
        split = "cal" if deterministic_split_flag(img_id) else "eval"
        target = rng.choice([0, 1], p=[0.5, 0.5])
        # Overconfident logits
        raw_logit = float(rng.normal(loc=(3.0 if target == 1 else -3.0), scale=1.5))
        prob = 1.0 / (1.0 + math.exp(-raw_logit))
        is_red = (rng.random() < 0.4)
        detected = (rng.random() < 0.95)
        pred_st = 0 if (is_red and rng.random() < 0.9) else 1

        records.append(
            SampleRecord(
                image_id=img_id,
                split_group=split,
                gt_target=int(target),
                uncal_prob=prob,
                raw_logit=raw_logit,
                is_red=is_red,
                is_directional=(rng.random() < 0.3),
                has_arrows=(rng.random() < 0.5),
                area_px=float(rng.uniform(20, 600)),
                area_bucket="64-128",
                detected=detected,
                pred_state=pred_st,
                det_score=0.85 if detected else 0.0,
            )
        )

    results = run_e19_calibration(records, tmp_path)

    assert "temperature_optimal" in results
    assert results["temperature_optimal"] > 0.0
    assert "safety_thresholds" in results
    assert "evaluation_split_metrics" in results
    assert "safety_waterfall_evaluation" in results

    eval_wf = results["safety_waterfall_evaluation"]["operating_regimes"]
    for reg_key, reg_data in eval_wf.items():
        total_rr = reg_data["total_relevant_red"]
        s1 = reg_data["stage1_perception_miss"]
        s2 = reg_data["stage2_candidate_eviction"]
        s3 = reg_data["stage3_state_misclassification"]
        s4 = reg_data["stage4_relevance_rejection"]
        tp = reg_data["success_tp"]

        # Waterfall conservation law
        assert s1 + s2 + s3 + s4 + tp == total_rr, f"Conservation failed for {reg_key}: {s1}+{s2}+{s3}+{s4}+{tp} != {total_rr}"

        # Confusion matrix checks
        assert tp + reg_data["fn"] == total_rr
        assert reg_data["fp"] + reg_data["tn"] == reg_data["total_irrelevant_red"]
