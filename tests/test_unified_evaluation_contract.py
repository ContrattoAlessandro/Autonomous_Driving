"""Unit tests for E29 Unified Evaluation Contract & Cross-Ticket Normalization Standard."""

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

from scripts.unified_evaluation_contract import (
    compute_nll,
    compute_reliability_curve,
    optimize_safety_threshold,
)
from tlr_yolo_mtl.evaluation.contract import (
    EvaluationContractConfig,
    SafetyWaterfallBreakdown,
    deterministic_contract_split,
)


def test_contract_config_validation():
    # Valid contract default
    contract = EvaluationContractConfig()
    violations = contract.validate()
    assert len(violations) == 0, f"Default contract must be fully valid, got: {violations}"

    # Invalid primary checkpoint
    invalid_ckpt = EvaluationContractConfig(primary_checkpoint="wrong.pt")
    violations = invalid_ckpt.validate()
    assert any("Primary checkpoint must be" in v for v in violations)

    # Invalid IoU threshold
    invalid_iou = EvaluationContractConfig(iou_threshold=0.75)
    violations = invalid_iou.validate()
    assert any("Matching IoU threshold must be 0.50" in v for v in violations)

    # Invalid resolution
    invalid_res = EvaluationContractConfig(resolution=(640, 640))
    violations = invalid_res.validate()
    assert any("Canonical base resolution must be (800, 1600)" in v for v in violations)

    # Invalid token budget
    invalid_tokens = EvaluationContractConfig(k_tl=16, k_arrow=16)
    violations = invalid_tokens.validate()
    assert any("Candidate token pools must be K_TL=32, K_Arrow=32" in v for v in violations)


def test_deterministic_contract_split():
    # Strict determinism
    img_ids = [f"image_{i:04d}.png" for i in range(1000)]
    split_1 = [deterministic_contract_split(img_id) for img_id in img_ids]
    split_2 = [deterministic_contract_split(img_id) for img_id in img_ids]
    assert split_1 == split_2, "Split assignment must be 100% deterministic"

    # 50/50 balance check
    n_cal = sum(split_1)
    ratio = n_cal / len(img_ids)
    assert 0.45 <= ratio <= 0.55, f"Split ratio {ratio:.2f} must be close to 50%"

    # Salt sensitivity
    split_other = [deterministic_contract_split(img_id, salt="different_salt") for img_id in img_ids]
    assert split_1 != split_other, "Different salts must produce distinct partitions"


def test_safety_waterfall_breakdown():
    wf = SafetyWaterfallBreakdown(
        gt_relevant_red_total=100,
        perception_detected=90,
        perception_missed=10,
        candidate_selected=85,
        candidate_missed=5,
        state_classified_red=80,
        state_misclassified=5,
        relevance_accepted=75,
        relevance_rejected=5,
    )

    assert wf.end_to_end_recalled == 75
    assert math.isclose(wf.end_to_end_recall, 0.75)
    assert math.isclose(wf.perception_recall, 0.90)
    assert math.isclose(wf.candidate_selection_rate, 85 / 90)
    assert math.isclose(wf.state_classification_rate, 80 / 85)
    assert math.isclose(wf.relevance_acceptance_rate, 75 / 80)

    d = wf.to_dict()
    assert d["gt_relevant_red_total"] == 100
    assert d["end_to_end_recalled"] == 75


def test_compute_nll():
    targets = np.array([1, 1, 0, 0], dtype=np.int64)
    probs_good = np.array([0.9, 0.85, 0.1, 0.05])
    probs_bad = np.array([0.2, 0.3, 0.8, 0.7])

    nll_good = compute_nll(targets, probs_good)
    nll_bad = compute_nll(targets, probs_bad)

    assert nll_good < nll_bad
    assert nll_good > 0.0


def test_optimize_safety_threshold():
    # Construct 100 positives with scores ~0.8, 100 negatives with scores ~0.2
    rng = np.random.default_rng(42)
    pos_scores = rng.uniform(0.6, 0.95, size=100)
    neg_scores = rng.uniform(0.05, 0.4, size=100)
    scores = np.concatenate([pos_scores, neg_scores])
    targets = np.array([1] * 100 + [0] * 100)

    tau_90, prec_90, rec_90 = optimize_safety_threshold(targets, scores, target_recall=0.90)
    assert rec_90 >= 0.90
    assert prec_90 > 0.80

    tau_95, prec_95, rec_95 = optimize_safety_threshold(targets, scores, target_recall=0.95)
    assert rec_95 >= 0.95

    tau_975, prec_975, rec_975 = optimize_safety_threshold(targets, scores, target_recall=0.975)
    assert rec_975 >= 0.975

    assert tau_90 >= tau_95 >= tau_975, "Thresholds must be monotonic with increasing recall constraint"


def test_reliability_curve():
    targets = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.95, 0.6])

    curve = compute_reliability_curve(targets, probs, bins=5)
    assert len(curve["bin_centers"]) == 5
    assert len(curve["bin_accs"]) == 5
    assert sum(curve["bin_counts"]) == len(targets)
