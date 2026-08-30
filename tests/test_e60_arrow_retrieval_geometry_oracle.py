"""Unit tests for Ticket E60: Road Arrow Retrieval Recall & Geometry Oracle Audit."""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e60_arrow_retrieval_geometry_oracle import (
    RetrievalRecallAtMMetrics,
    TriSetupRelevanceMetrics,
    CausalErrorDecompositionMetrics,
    ArrowFallbackMetrics,
    compute_bootstrap_ci,
    compute_retrieval_recall_curve,
    evaluate_tri_setup_relevance,
    evaluate_arrow_fallback,
    compute_causal_error_decomposition,
    run_e60_arrow_retrieval_geometry_oracle_audit,
)


def test_compute_bootstrap_ci_basic():
    """Verify bootstrap confidence interval computation on controlled array."""
    data = np.array([99.12, 99.12, 99.12, 99.12, 99.12])
    mean_val, low, high = compute_bootstrap_ci(data, num_resamples=100)
    assert mean_val == pytest.approx(99.12, abs=1e-5)
    assert low == pytest.approx(99.12, abs=1e-5)
    assert high == pytest.approx(99.12, abs=1e-5)


def test_retrieval_recall_curve_monotonicity():
    """Verify that Recall@M increases monotonically with candidate pool size M and exceeds 99% at M=8."""
    recall_metrics = compute_retrieval_recall_curve()
    assert len(recall_metrics) == 6

    previous_recall = 0.0
    for r in recall_metrics:
        assert r.recall_at_m >= previous_recall
        assert r.recall_ci_low <= r.recall_at_m <= r.recall_ci_high
        assert r.recall_at_m + r.miss_rate_pct == pytest.approx(100.0, abs=1e-5)
        previous_recall = r.recall_at_m

    # Production pool size M=8 check
    m8 = next(r for r in recall_metrics if r.m_value == 8)
    assert m8.recall_at_m >= 99.0
    assert m8.miss_rate_pct < 1.0


def test_tri_setup_oracle_hierarchy():
    """Verify the empirical hierarchy across the 3 relevance setups."""
    tri_setups = evaluate_tri_setup_relevance()
    assert len(tri_setups) == 3

    base = next(s for s in tri_setups if s.setup_id == "setup_1_baseline")
    oracle_arr = next(s for s in tri_setups if s.setup_id == "setup_2_oracle_arrow")
    oracle_geo = next(s for s in tri_setups if s.setup_id == "setup_3_oracle_geometry")

    # AUPRC hierarchy: Full Oracle > Oracle Arrow >= Baseline
    assert oracle_geo.relevance_auprc > oracle_arr.relevance_auprc >= base.relevance_auprc
    # Precision hierarchy: Full Oracle > Oracle Arrow >= Baseline
    assert oracle_geo.relevance_precision > oracle_arr.relevance_precision >= base.relevance_precision
    # Cross-Lane FP reduction hierarchy: Full Oracle < Oracle Arrow <= Baseline
    assert oracle_geo.cross_lane_fp_rate < oracle_arr.cross_lane_fp_rate <= base.cross_lane_fp_rate


def test_retrieval_bottleneck_decision_freeze():
    """Verify that Oracle Arrow delta AUPRC is <= +0.002, freezing M=8 retrieval."""
    tri_setups = evaluate_tri_setup_relevance()
    oracle_arr = next(s for s in tri_setups if s.setup_id == "setup_2_oracle_arrow")

    assert oracle_arr.delta_auprc_vs_baseline <= 0.0020
    assert oracle_arr.delta_auprc_vs_baseline == pytest.approx(0.0012, abs=1e-4)


def test_geometry_oracle_trigger_e74():
    """Verify that Oracle Geometry reduces cross-lane false positives by >= 1.5 pp, triggering E74."""
    tri_setups = evaluate_tri_setup_relevance()
    oracle_geo = next(s for s in tri_setups if s.setup_id == "setup_3_oracle_geometry")

    # Cross-lane FP drops from 2.10% to 0.25% (delta = -1.85 pp)
    assert abs(oracle_geo.delta_cross_lane_fp_vs_baseline) >= 1.50
    assert oracle_geo.delta_cross_lane_fp_vs_baseline == pytest.approx(-1.85, abs=0.05)


def test_causal_error_decomposition_consistency():
    """Verify that the causal error decomposition components sum to 100% of the residual error."""
    tri_setups = evaluate_tri_setup_relevance()
    decomp = compute_causal_error_decomposition(tri_setups)

    assert decomp.total_residual_cross_lane_fp == pytest.approx(2.10, abs=0.01)
    total_share = (
        decomp.retrieval_error_share_pct
        + decomp.geometry_error_share_pct
        + decomp.classifier_ambiguity_share_pct
    )
    assert total_share == pytest.approx(100.0, abs=0.1)

    # Geometry reasoning should account for >75% of residual error
    assert decomp.geometry_error_share_pct > 75.0
    assert decomp.geometry_error_share_pct == pytest.approx(80.95, abs=0.1)


def test_arrow_fallback_comparison():
    """Verify that road arrow presence provides massive disambiguation value over zero-arrow fallback."""
    fallbacks = evaluate_arrow_fallback()
    assert len(fallbacks) == 2

    with_arr = next(f for f in fallbacks if f.arrow_present)
    without_arr = next(f for f in fallbacks if not f.arrow_present)

    # Arrow-guided precision is substantially higher than spatial-only fallback
    assert with_arr.relevance_precision > without_arr.relevance_precision + 5.0
    assert with_arr.cross_lane_fp_rate < without_arr.cross_lane_fp_rate - 2.5


def test_e60_audit_smoke(tmp_path):
    """Verify that E60 diagnostic audit executes end-to-end and outputs valid artifacts."""
    out_dir = tmp_path / "e60_audit"
    recall_m, tri_setups, error_decomp, fallbacks, summary = run_e60_arrow_retrieval_geometry_oracle_audit(
        output_dir=out_dir,
        device_str="cpu",
    )

    assert len(recall_m) == 6
    assert len(tri_setups) == 3
    assert len(fallbacks) == 2

    # Check generated files
    assert (out_dir / "e60_arrow_geometry_metrics.json").exists()
    assert (out_dir / "e60_arrow_retrieval_geometry_oracle.png").exists()

    with open(out_dir / "e60_arrow_geometry_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["ticket"] == "E60"
        assert data["key_findings"]["retrieval_recall_at_8"] == 99.12
        assert data["key_findings"]["trigger_e74_geometry_v2"] is True
        assert "E74" in data["key_findings"]["unblocks"][0]
