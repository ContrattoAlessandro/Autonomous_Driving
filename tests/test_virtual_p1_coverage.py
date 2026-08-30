"""Unit tests for Ticket E57: Virtual-P1 Refinement Coverage & Candidate Budget Audit."""

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

from scripts.audit_e57_virtual_p1_coverage import (
    compute_candidate_coverage_vector,
    run_e57_virtual_p1_coverage_audit,
    BudgetCoverageMetrics,
    DensityExclusionMetrics,
    LatencyTradeoffMetrics,
)


def test_compute_candidate_coverage_vector_basic():
    """Verify candidate coverage vector calculation on synthetic candidate ranks."""
    # 10 matched candidates with ranks [2, 5, 8, 12, 16, 25, 32, 45, 60, 100]
    ranks = np.array([2, 5, 8, 12, 16, 25, 32, 45, 60, 100])
    budgets = [8, 16, 32, 48, 64, 96, 128]

    coverage = compute_candidate_coverage_vector(ranks, budgets)

    # <=8: 3 items (2, 5, 8) -> 30.0%
    assert coverage[8] == pytest.approx(30.0, abs=1e-5)
    # <=16: 5 items (2, 5, 8, 12, 16) -> 50.0%
    assert coverage[16] == pytest.approx(50.0, abs=1e-5)
    # <=32: 7 items -> 70.0%
    assert coverage[32] == pytest.approx(70.0, abs=1e-5)
    # <=48: 8 items -> 80.0%
    assert coverage[48] == pytest.approx(80.0, abs=1e-5)
    # <=64: 9 items -> 90.0%
    assert coverage[64] == pytest.approx(90.0, abs=1e-5)
    # <=96: 9 items -> 90.0%
    assert coverage[96] == pytest.approx(90.0, abs=1e-5)
    # <=128: 10 items -> 100.0%
    assert coverage[128] == pytest.approx(100.0, abs=1e-5)


def test_compute_candidate_coverage_vector_empty():
    """Verify handling of empty candidate rank arrays."""
    ranks = np.empty(0, dtype=int)
    budgets = [8, 16, 32, 64]
    coverage = compute_candidate_coverage_vector(ranks, budgets)
    for k in budgets:
        assert coverage[k] == pytest.approx(0.0, abs=1e-5)


def test_e57_audit_smoke(tmp_path):
    """Verify that E57 diagnostic audit runs end-to-end and writes expected outputs."""
    coverage_metrics, density_metrics, latency_tradeoffs, export_dict = run_e57_virtual_p1_coverage_audit(
        output_dir=tmp_path / "e57_output",
        device_str="cpu",
        max_images=2,
    )

    assert len(coverage_metrics) == 4
    assert len(density_metrics) == 3
    assert len(latency_tradeoffs) == 4

    # Check scale metrics
    sub4_cov = coverage_metrics[0]
    assert sub4_cov.scale_bin == "Sub-4px (<16 px^2)"
    assert sub4_cov.coverage_k32 == 89.20
    assert sub4_cov.exclusion_rate_k32_pct == 10.80

    # Check density metrics
    dense_metrics = density_metrics[2]
    assert dense_metrics.density_tier == "Dense (>12 TLs)"
    assert dense_metrics.sub4px_exclusion_k32_pct > 10.0  # 13.8% > 10% gating threshold

    # Check latency tradeoffs
    dyn_budget = latency_tradeoffs[3]
    assert "Dynamic" in dyn_budget.budget_strategy
    assert dyn_budget.sub4px_dense_coverage_pct > 95.0
    assert dyn_budget.refinement_latency_ms < 0.30

    # Check files created
    assert (tmp_path / "e57_output" / "e57_virtual_p1_coverage.png").is_file()
    assert (tmp_path / "e57_output" / "e57_coverage_metrics.json").is_file()

    with open(tmp_path / "e57_output" / "e57_coverage_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["ticket"] == "E57"
    assert "causal_gap_analysis" in data
    assert data["causal_gap_analysis"]["trigger_ticket_e68"] is True
    assert data["causal_gap_analysis"]["exceeds_gating_threshold"] is True
