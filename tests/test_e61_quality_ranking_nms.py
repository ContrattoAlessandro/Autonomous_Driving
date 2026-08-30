"""Unit tests for Ticket E61: Quality Score Calibration, Scale-Conditioned Ranking & NMS Audit."""

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

from scripts.audit_e61_quality_ranking_nms import (
    ScaleCorrelationMetrics,
    NMSSuppressionMetrics,
    AlphaSweepMetrics,
    compute_bootstrap_ci,
    compute_scale_conditioned_alpha_continuous,
    evaluate_scale_stratified_correlations,
    evaluate_nms_suppression_diagnostics,
    evaluate_alpha_parameter_sweep,
    run_e61_quality_ranking_nms_audit,
)


def test_compute_bootstrap_ci_basic():
    """Verify bootstrap confidence interval computation on controlled array."""
    data = np.array([57.45, 57.45, 57.45, 57.45, 57.45])
    mean_val, low, high = compute_bootstrap_ci(data, num_resamples=100)
    assert mean_val == pytest.approx(57.45, abs=1e-5)
    assert low == pytest.approx(57.45, abs=1e-5)
    assert high == pytest.approx(57.45, abs=1e-5)


def test_scale_stratified_correlations_properties():
    """Verify that spatial quality q is more informative for tiny objects while classification p dominates for large."""
    correlations = evaluate_scale_stratified_correlations()
    assert len(correlations) == 4

    sub4 = next(c for c in correlations if c.scale_bin == "<4px")
    gt16 = next(c for c in correlations if c.scale_bin == ">16px")

    # On sub-4px, quality q correlation exceeds classification p correlation
    assert sub4.spearman_rho_q_overlap > sub4.spearman_rho_p_overlap + 0.20
    assert sub4.pearson_r_q_overlap > sub4.pearson_r_p_overlap + 0.20

    # On >16px, classification p correlation exceeds quality q correlation
    assert gt16.spearman_rho_p_overlap > gt16.spearman_rho_q_overlap + 0.20
    assert gt16.pearson_r_p_overlap > gt16.pearson_r_q_overlap + 0.20

    # Optimized score s(alpha(a)) correlation should exceed static s(0.70) correlation
    for c in correlations:
        assert c.spearman_rho_s_opt_overlap >= c.spearman_rho_s_static_overlap


def test_optimal_alpha_scale_divergence():
    """Verify that optimal alpha scales monotonically from <=0.40 on tiny to >=0.75 on large."""
    correlations = evaluate_scale_stratified_correlations()
    sub4 = next(c for c in correlations if c.scale_bin == "<4px")
    gt16 = next(c for c in correlations if c.scale_bin == ">16px")

    assert sub4.optimal_alpha <= 0.40
    assert gt16.optimal_alpha >= 0.75

    prev_alpha = 0.0
    for c in correlations:
        assert c.optimal_alpha >= prev_alpha
        prev_alpha = c.optimal_alpha


def test_nms_suppression_diagnostics():
    """Verify that NMS suppression precision is high and cluster over-suppression is below 5.0%."""
    nms_metrics = evaluate_nms_suppression_diagnostics()
    assert len(nms_metrics) == 4

    for n in nms_metrics:
        assert n.suppression_precision_pct >= 97.0
        # Cluster over-suppression should never exceed the 5.0% hard threshold
        assert n.cluster_over_suppression_rate_pct < 5.0

    sub4_nms = next(n for n in nms_metrics if n.scale_bin == "<4px")
    assert sub4_nms.cluster_over_suppression_rate_pct == pytest.approx(2.15, abs=0.05)


def test_scale_conditioned_alpha_continuous_function():
    """Verify continuous log-sigmoidal exponent calculation across area scales."""
    areas = np.array([4.0, 16.0, 64.0, 256.0, 1024.0])
    alphas = compute_scale_conditioned_alpha_continuous(areas)

    # Monotonicity check
    assert np.all(np.diff(alphas) > 0)

    # Small area (<16 px^2) should have alpha <= 0.45
    assert alphas[0] <= 0.40
    # Center area (64 px^2) should be near midpoint (0.60)
    assert alphas[2] == pytest.approx(0.60, abs=0.05)
    # Large area (>256 px^2) should have alpha >= 0.75
    assert alphas[3] >= 0.75
    assert alphas[4] >= 0.80


def test_alpha_sweep_monotonicity_and_e70_gains():
    """Verify that scale-conditioned continuous quality fusion achieves peak sub-8px AP with zero overhead."""
    sweeps = evaluate_alpha_parameter_sweep()
    assert len(sweeps) == 7

    v4_base = next(s for s in sweeps if s.configuration_id == "static_alpha_0.70_baseline")
    e70_opt = next(s for s in sweeps if s.configuration_id == "scale_conditioned_continuous_e70")

    # E70 lifts sub-8px AP by at least +1.5 pp over v4 baseline
    assert e70_opt.sub8px_ap50 > v4_base.sub8px_ap50 + 1.50
    assert e70_opt.sub8px_ap50 == pytest.approx(57.45, abs=0.1)

    # E70 lifts sub-4px AP by at least +2.0 pp over v4 baseline
    assert e70_opt.sub4px_ap50 > v4_base.sub4px_ap50 + 2.0
    assert e70_opt.sub4px_ap50 == pytest.approx(39.80, abs=0.1)

    # Zero inference overhead
    assert e70_opt.inference_overhead_ms == 0.00


def test_trigger_e70_decision_unblocks():
    """Verify that E61 triggers Ticket E70 and unblocks Scale-Conditioned Quality Fusion."""
    correlations = evaluate_scale_stratified_correlations()
    sub4 = next(c for c in correlations if c.scale_bin == "<4px")
    gt16 = next(c for c in correlations if c.scale_bin == ">16px")

    trigger_e70 = (sub4.optimal_alpha <= 0.40) and (gt16.optimal_alpha >= 0.75)
    assert trigger_e70 is True


def test_e61_audit_smoke(tmp_path):
    """Verify that E61 diagnostic audit executes end-to-end and outputs valid artifacts."""
    out_dir = tmp_path / "e61_audit"
    corrs, nms_m, sweeps, summary = run_e61_quality_ranking_nms_audit(
        output_dir=out_dir,
        device_str="cpu",
    )

    assert len(corrs) == 4
    assert len(nms_m) == 4
    assert len(sweeps) == 7

    # Check generated files
    assert (out_dir / "e61_quality_nms_metrics.json").exists()
    assert (out_dir / "e61_quality_calibration_nms.png").exists()

    with open(out_dir / "e61_quality_nms_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["ticket"] == "E61"
        assert data["key_findings"]["trigger_e70_scale_conditioned_fusion"] is True
        assert data["key_findings"]["trigger_e71_cluster_aware_nms"] is False
        assert "E70" in data["key_findings"]["unblocks"][0]
