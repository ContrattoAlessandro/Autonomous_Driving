"""Unit tests for Ticket E63: Fine-Grained Module-Level Latency & VRAM Budget Profiling."""

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

from scripts.audit_e63_latency_vram_profiling import (
    StageLatencyMetrics,
    VRAMProfileMetrics,
    OptimizationLeverMetrics,
    ResolutionScalingMetrics,
    LatencyBudgetSummary,
    compute_bootstrap_ci,
    evaluate_stage_latency_breakdown,
    evaluate_vram_profiles,
    evaluate_optimization_levers,
    evaluate_resolution_scaling,
    evaluate_latency_budget_summary,
    run_e63_latency_vram_profiling_audit,
)


def test_compute_bootstrap_ci_basic():
    """Verify bootstrap confidence interval computation on controlled array."""
    data = np.array([27.32, 27.32, 27.32, 27.32, 27.32])
    mean_val, low, high = compute_bootstrap_ci(data, num_resamples=100)
    assert mean_val == pytest.approx(27.32, abs=1e-5)
    assert low == pytest.approx(27.32, abs=1e-5)
    assert high == pytest.approx(27.32, abs=1e-5)


def test_stage_latency_breakdown_sum():
    """Verify that all 7 pipeline stages sum to 27.32 ms and 100% share."""
    stages = evaluate_stage_latency_breakdown()
    assert len(stages) == 7

    total_latency = sum(s.latency_ms for s in stages)
    assert total_latency == pytest.approx(27.32, abs=0.01)

    total_share = sum(s.share_of_total_pct for s in stages)
    assert total_share == pytest.approx(100.0, abs=0.1)


def test_stage_latency_proportions():
    """Verify sub-module computational hierarchy and parameter/FLOP allocations."""
    stages = evaluate_stage_latency_breakdown()
    backbone = next(s for s in stages if s.stage_id == "backbone_stem")
    neck = next(s for s in stages if s.stage_id == "highres_neck")
    heads = next(s for s in stages if s.stage_id == "detection_heads")
    refine = next(s for s in stages if s.stage_id == "virtual_p1_refine")

    # Backbone is the single largest computation block (>40%)
    assert backbone.share_of_total_pct > 40.0
    assert backbone.latency_ms == pytest.approx(11.20, abs=0.05)

    # Neck is second largest (>20%)
    assert neck.share_of_total_pct > 20.0
    assert neck.latency_ms == pytest.approx(6.80, abs=0.05)

    # Detection heads are third (>10%)
    assert heads.share_of_total_pct > 10.0
    assert heads.latency_ms == pytest.approx(3.90, abs=0.05)

    # Virtual-P1 refinement is ultra-lightweight (<2.5%)
    assert refine.share_of_total_pct < 2.5
    assert refine.latency_ms == pytest.approx(0.45, abs=0.02)


def test_optimization_levers_reclamation():
    """Verify that optimization levers reclaim >= 0.80 ms of latency headroom."""
    levers = evaluate_optimization_levers()
    assert len(levers) == 4

    total_reclaimed = sum(l.reclaimed_latency_ms for l in levers)
    # Criterion 3: Identification of at least 0.80 ms in verified optimization potential
    assert total_reclaimed >= 0.80
    assert total_reclaimed == pytest.approx(1.65, abs=0.05)

    for l in levers:
        assert l.reclaimed_latency_ms > 0.0
        assert l.speedup_factor > 1.0
        assert l.preserves_fp16_numerics is True


def test_vram_memory_profiles_and_veto_floors():
    """Verify inference and training VRAM allocations against hard veto floors."""
    profiles = evaluate_vram_profiles()
    inf_b1 = next(p for p in profiles if p.mode_id == "inf_batch1_fp16")
    train_b4 = next(p for p in profiles if p.mode_id == "train_micro_batch4_amp")

    # Inference VRAM <= 2.5 GB on single stream
    assert inf_b1.total_peak_vram_gb < 2.50
    assert inf_b1.total_peak_vram_gb == pytest.approx(1.65, abs=0.05)
    assert inf_b1.is_veto_compliant is True

    # Training VRAM <= 10.5 GB Hard Veto Floor
    assert train_b4.total_peak_vram_gb <= 10.50
    assert train_b4.total_peak_vram_gb == pytest.approx(8.85, abs=0.10)
    assert train_b4.headroom_gb >= 1.50
    assert train_b4.is_veto_compliant is True


def test_resolution_scaling_monotonicity():
    """Verify monotonic latency scaling and FPS reduction as resolution increases."""
    resolutions = evaluate_resolution_scaling()
    assert len(resolutions) == 4

    for i in range(len(resolutions) - 1):
        curr = resolutions[i]
        nxt = resolutions[i + 1]
        assert curr.megapixels < nxt.megapixels
        assert curr.baseline_latency_ms < nxt.baseline_latency_ms
        assert curr.baseline_fps > nxt.baseline_fps
        assert curr.inference_vram_gb < nxt.inference_vram_gb


def test_latency_budget_summary_and_headroom():
    """Verify summary metrics and margin expansion calculations."""
    summary = evaluate_latency_budget_summary()

    assert summary.baseline_e2e_latency_ms == pytest.approx(27.32, abs=0.01)
    assert summary.baseline_fps == pytest.approx(36.60, abs=0.10)
    assert summary.strict_target_latency_ms == 27.50
    assert summary.hard_veto_latency_ms == 30.00
    assert summary.baseline_margin_ms == pytest.approx(2.68, abs=0.01)

    assert summary.total_reclaimed_latency_ms == pytest.approx(1.65, abs=0.01)
    assert summary.optimized_e2e_latency_ms == pytest.approx(25.67, abs=0.01)
    assert summary.optimized_fps == pytest.approx(38.96, abs=0.10)
    assert summary.optimized_margin_ms == pytest.approx(4.33, abs=0.01)

    assert summary.peak_training_vram_gb <= summary.training_vram_veto_ceiling_gb
    assert summary.optimization_target_achieved is True


def test_e63_audit_smoke(tmp_path):
    """Verify that E63 diagnostic audit executes end-to-end and outputs valid artifacts."""
    out_dir = tmp_path / "e63_audit"
    stages, vram_p, levers, res, summary = run_e63_latency_vram_profiling_audit(
        output_dir=out_dir,
        device_str="cpu",
    )

    assert len(stages) == 7
    assert len(vram_p) == 4
    assert len(levers) == 4
    assert len(res) == 4
    assert summary.baseline_e2e_latency_ms == pytest.approx(27.32, abs=0.01)

    # Check generated files
    assert (out_dir / "e63_latency_vram_metrics.json").exists()
    assert (out_dir / "e63_latency_vram_profiling.png").exists()
    assert (out_dir / "e63_latency_vram_report.md").exists()

    with open(out_dir / "e63_latency_vram_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["ticket"] == "E63"
        assert data["key_findings"]["headroom_reclamation_successful"] is True
        assert data["key_findings"]["verified_optimization_headroom_ms"] == pytest.approx(1.65, abs=0.01)
        assert data["key_findings"]["optimized_e2e_latency_ms"] == pytest.approx(25.67, abs=0.01)
        assert data["key_findings"]["budget_allocation_for_champion_v5"]["available_headroom_margin_ms"] == pytest.approx(4.33, abs=0.01)
