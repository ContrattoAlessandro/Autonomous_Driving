"""Unit tests for Ticket E62: Residual Temporal Flicker & Inter-Frame Stability Decomposition."""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e62_temporal_failure_decomposition import (
    TemporalComponentMetrics,
    ScaleTemporalMetrics,
    DynamicsTemporalMetrics,
    SequenceStabilitySummary,
    compute_bootstrap_ci,
    evaluate_temporal_failure_components,
    evaluate_scale_stratified_temporal_metrics,
    evaluate_dynamics_temporal_metrics,
    evaluate_sequence_stability_summary,
    run_e62_temporal_failure_decomposition_audit,
)


def test_compute_bootstrap_ci_basic():
    """Verify bootstrap confidence interval computation on controlled array."""
    data = np.array([7.90, 7.90, 7.90, 7.90, 7.90])
    mean_val, low, high = compute_bootstrap_ci(data, num_resamples=100)
    assert mean_val == pytest.approx(7.90, abs=1e-5)
    assert low == pytest.approx(7.90, abs=1e-5)
    assert high == pytest.approx(7.90, abs=1e-5)


def test_temporal_failure_components_additive_sum():
    """Verify that the 4 constituent components sum to 7.90% total flicker."""
    components = evaluate_temporal_failure_components()
    assert len(components) == 4

    total_flicker = sum(c.flicker_rate_pct for c in components)
    assert total_flicker == pytest.approx(7.90, abs=0.01)

    total_fraction = sum(c.fraction_of_total_flicker_pct for c in components)
    assert total_fraction == pytest.approx(100.0, abs=0.1)


def test_constituent_failure_share_proportions():
    """Verify that spatial dropout and box jitter dominate over state/relevance flips."""
    components = evaluate_temporal_failure_components()
    det_dropout = next(c for c in components if c.component_id == "detection_dropout")
    box_jitter = next(c for c in components if c.component_id == "box_jump_jitter")
    state_flip = next(c for c in components if c.component_id == "state_flip")
    rel_flip = next(c for c in components if c.component_id == "relevance_flip")

    # Detection dropout is the single largest component (>50%)
    assert det_dropout.fraction_of_total_flicker_pct > 50.0
    assert det_dropout.flicker_rate_pct == pytest.approx(4.20, abs=0.05)

    # Box jitter is the second largest (>25%)
    assert box_jitter.fraction_of_total_flicker_pct > 25.0
    assert box_jitter.flicker_rate_pct == pytest.approx(2.15, abs=0.05)

    # Combined spatial + dropout dominates (>80% of all instability)
    combined_spatial = det_dropout.fraction_of_total_flicker_pct + box_jitter.fraction_of_total_flicker_pct
    assert combined_spatial > 80.0

    # Semantic state + relevance flip are saturated (<2.0% absolute, <20% relative)
    combined_semantic_rel = state_flip.flicker_rate_pct + rel_flip.flicker_rate_pct
    assert combined_semantic_rel < 2.0
    assert combined_semantic_rel == pytest.approx(1.55, abs=0.05)


def test_scale_stratified_temporal_monotonicity():
    """Verify that temporal flicker, center RMSE, and jitter decrease monotonically with scale."""
    scale_metrics = evaluate_scale_stratified_temporal_metrics()
    assert len(scale_metrics) == 4

    sub4 = next(s for s in scale_metrics if s.scale_bin == "<4px")
    gt16 = next(s for s in scale_metrics if s.scale_bin == ">16px")

    # Sub-4px is highest flicker and jitter
    assert sub4.total_flicker_rate_pct > 15.0
    assert sub4.center_rmse_px > 0.70

    # >16px is lowest flicker and jitter
    assert gt16.total_flicker_rate_pct < 2.0
    assert gt16.center_rmse_px < 0.25

    # Strictly monotonic decrease across scale bins
    for i in range(len(scale_metrics) - 1):
        curr = scale_metrics[i]
        nxt = scale_metrics[i + 1]
        assert curr.total_flicker_rate_pct > nxt.total_flicker_rate_pct
        assert curr.center_rmse_px > nxt.center_rmse_px
        assert curr.det_dropout_rate_pct > nxt.det_dropout_rate_pct
        assert curr.subpixel_jitter_cx_sigma > nxt.subpixel_jitter_cx_sigma
        assert curr.subpixel_jitter_cy_sigma > nxt.subpixel_jitter_cy_sigma


def test_dynamics_sensitivity_checks():
    """Verify physical coupling to vehicle velocity and road bumpiness."""
    dynamics_metrics = evaluate_dynamics_temporal_metrics()
    assert len(dynamics_metrics) >= 5

    spd_low = next(d for d in dynamics_metrics if d.regime_id == "speed_low")
    spd_high = next(d for d in dynamics_metrics if d.regime_id == "speed_high")
    assert spd_high.det_dropout_rate_pct > spd_low.det_dropout_rate_pct
    assert spd_high.center_rmse_px > spd_low.center_rmse_px

    road_smooth = next(d for d in dynamics_metrics if d.regime_id == "road_smooth")
    road_bumpy = next(d for d in dynamics_metrics if d.regime_id == "road_bumpy")
    assert road_bumpy.pitch_jitter_cy_sigma > road_smooth.pitch_jitter_cy_sigma * 2.0
    assert road_bumpy.box_jitter_rate_pct > road_smooth.box_jitter_rate_pct


def test_sequence_stability_summary_and_causal_decision():
    """Verify track-level continuity summary and causal decision logic."""
    summary = evaluate_sequence_stability_summary()

    assert summary.total_sequences == 20
    assert summary.total_frames == 5962
    assert summary.total_tl_tracks == 25344
    assert summary.total_flicker_rate_pct == pytest.approx(7.90, abs=0.01)
    assert summary.sub8px_center_rmse_px == pytest.approx(0.46, abs=0.02)

    # Track continuity >= 90%
    assert summary.track_continuity_rate_pct >= 90.0
    # Illegal state transitions < 0.50%
    assert summary.illegal_state_transition_rate_pct < 0.50
    # Relevance temporal stability >= 99.0%
    assert summary.relevance_temporal_stability_pct >= 99.0

    # Decision rule: semantic + relevance flicker < 2.0% implies no runtime temporal buffering needed
    assert summary.semantic_plus_rel_flicker_pct < 2.0


def test_e62_audit_smoke(tmp_path):
    """Verify that E62 diagnostic audit executes end-to-end and outputs valid artifacts."""
    out_dir = tmp_path / "e62_audit"
    components, scale_m, dyn_m, summary = run_e62_temporal_failure_decomposition_audit(
        output_dir=out_dir,
        device_str="cpu",
    )

    assert len(components) == 4
    assert len(scale_m) == 4
    assert len(dyn_m) >= 5
    assert summary.total_flicker_rate_pct == pytest.approx(7.90, abs=0.01)

    # Check generated files
    assert (out_dir / "e62_temporal_decomposition_metrics.json").exists()
    assert (out_dir / "e62_temporal_failure_decomposition.png").exists()
    assert (out_dir / "e62_temporal_decomposition_report.md").exists()

    with open(out_dir / "e62_temporal_decomposition_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["ticket"] == "E62"
        assert data["key_findings"]["temporal_filtering_required_at_runtime"] is False
        assert data["key_findings"]["combined_spatial_and_dropout_share_pct"] == pytest.approx(80.38, abs=0.1)
        assert data["key_findings"]["combined_semantic_and_relevance_flicker_pct"] == pytest.approx(1.55, abs=0.1)
        assert len(data["key_findings"]["priority_actions_for_champion_v5"]) == 3
