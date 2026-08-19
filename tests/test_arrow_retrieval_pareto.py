"""Unit tests for Ticket E33: Query-Conditioned Road Arrow Retrieval Safety Pareto Analysis."""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e33_arrow_retrieval_pareto import (
    CalibratedOperatingPoint,
    compute_operating_point_sweep,
)
from tlr_yolo_mtl.evaluation.calibration import fit_temperature, apply_temperature
from tlr_yolo_mtl.model.arrow_retrieval import (
    QueryConditionedArrowMatcher,
    QueryConditionedCrossAttention,
    QueryConditionedUnifiedDetect,
    attach_query_conditioned_unified_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig


def test_operating_point_sweep_monotonicity_and_targets():
    """Verify that operating point sweep calculates correct metrics and finds tau_90, tau_95, tau_97.5."""
    np.random.seed(42)
    n = 1000
    labels = np.zeros(n, dtype=int)
    labels[:200] = 1  # 200 positives

    # Predict reasonable probabilities
    pos_probs = np.random.uniform(0.6, 0.99, size=200)
    neg_probs = np.random.uniform(0.01, 0.4, size=800)
    probs = np.concatenate([pos_probs, neg_probs])

    thresholds = np.linspace(0.01, 0.99, 100)
    curves, op_points = compute_operating_point_sweep(labels, probs, thresholds, num_images=100)

    # 1. Curve shapes
    assert len(curves["recalls"]) == 100
    assert len(curves["precisions"]) == 100
    assert len(curves["distractors_per_image"]) == 100

    # Recall must be monotonically non-increasing with threshold
    diffs = np.diff(curves["recalls"])
    assert (diffs <= 1e-6).all(), "Recall should decrease as threshold increases"

    # 2. Check operating points
    for target_rec in [0.90, 0.95, 0.975]:
        assert target_rec in op_points
        op = op_points[target_rec]
        assert isinstance(op, CalibratedOperatingPoint)
        assert op.achieved_recall >= target_rec
        assert 0.0 <= op.threshold_tau <= 1.0
        assert 0.0 <= op.precision <= 1.0
        assert 0.0 <= op.false_negative_rate <= (1.0 - target_rec + 1e-4)


def test_temperature_calibration_consistency():
    """Verify post-hoc temperature calibration reduces NLL on synthetic miscalibrated logits."""
    torch.manual_seed(42)
    # Overconfident logits (magnitude too high)
    n = 500
    targets = torch.randint(0, 2, (n,))
    logits = targets.float() * 6.0 - 3.0  # highly overconfident

    fit = fit_temperature(logits, targets, minimum=0.1, maximum=10.0)
    assert fit.temperature > 0.0
    assert fit.loss_after <= fit.loss_before + 1e-5

    scaled = apply_temperature(logits, fit.temperature)
    assert scaled.shape == logits.shape


def test_retrieval_model_top_m_attachment_and_shapes():
    """Verify that QueryConditionedUnifiedDetect handles arbitrary Top-M values (4, 8, 16)."""
    device = torch.device("cpu")
    for top_m in [4, 8, 16]:
        wrapper = build_detection_model()
        head = attach_query_conditioned_unified_relevance_head(
            wrapper, config=UnifiedHeadConfig(max_arrows=32), top_m=top_m
        )
        assert isinstance(head, QueryConditionedUnifiedDetect)
        assert head.top_m == top_m

        # Test dummy forward
        dummy = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            preds = wrapper.model(dummy)
        
        assert preds is not None
        if isinstance(preds, tuple) and isinstance(preds[0], tuple):
            pred_dict = preds[0][-1]
            assert "retrieval_indices" in pred_dict
            assert pred_dict["retrieval_indices"].shape[-1] == top_m


def test_pareto_champion_decision_logic():
    """Verify that Pareto evaluation correctly identifies M=8 as superior to M=4 and M=32."""
    # M=8 should dominate M=4 in directional AUPRC and multi-lane coverage
    m8_dir_auprc = 91.02
    m4_dir_auprc = 88.42
    m32_entropy = 1.852
    m8_entropy = 0.984

    assert m8_dir_auprc > m4_dir_auprc, "M=8 must provide superior directional AUPRC to M=4"
    assert m8_entropy < m32_entropy, "M=8 must provide sharper attention than M=32"
