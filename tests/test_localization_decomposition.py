"""Unit tests for Ticket E56: Localization Error Decomposition & Oracle Bounding Box Audit."""

import json
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e56_localization_decomposition import (
    compute_spatial_error_vector,
    run_e56_localization_decomposition_audit,
    ScaleLocalizationMetrics,
    RefinementDeltaMetrics,
    DualOracleMetrics,
)


def test_compute_spatial_error_vector_basic():
    """Verify parametric spatial error vector calculation on known box pairs."""
    # Box 1: Exact match 10x20
    # Box 2: Shifted by 1px right, 2px down, +2px width, +4px height
    pred = np.array([
        [10.0, 20.0, 20.0, 40.0],
        [11.0, 22.0, 23.0, 46.0],
    ])
    gt = np.array([
        [10.0, 20.0, 20.0, 40.0],
        [10.0, 20.0, 20.0, 40.0],
    ])

    errors = compute_spatial_error_vector(pred, gt)

    # Box 1 (identical)
    assert errors["abs_cx"][0] == pytest.approx(0.0, abs=1e-5)
    assert errors["abs_cy"][0] == pytest.approx(0.0, abs=1e-5)
    assert errors["abs_w"][0] == pytest.approx(0.0, abs=1e-5)
    assert errors["abs_h"][0] == pytest.approx(0.0, abs=1e-5)
    assert errors["iou"][0] == pytest.approx(1.0, abs=1e-5)
    assert errors["nwd"][0] == pytest.approx(1.0, abs=1e-5)

    # Box 2 (shifted and scaled)
    # pred cx = (11+23)/2 = 17, gt cx = 15 -> abs_cx = 2.0
    # pred cy = (22+46)/2 = 34, gt cy = 30 -> abs_cy = 4.0
    # pred w = 12, gt w = 10 -> abs_w = 2.0
    # pred h = 24, gt h = 20 -> abs_h = 4.0
    assert errors["abs_cx"][1] == pytest.approx(2.0, abs=1e-4)
    assert errors["abs_cy"][1] == pytest.approx(4.0, abs=1e-4)
    assert errors["abs_w"][1] == pytest.approx(2.0, abs=1e-4)
    assert errors["abs_h"][1] == pytest.approx(4.0, abs=1e-4)
    assert 0.0 < errors["iou"][1] < 1.0
    assert 0.0 < errors["nwd"][1] < 1.0


def test_compute_spatial_error_vector_empty():
    """Verify handling of empty bounding box arrays."""
    pred = np.empty((0, 4))
    gt = np.empty((0, 4))
    errors = compute_spatial_error_vector(pred, gt)
    assert len(errors["abs_cx"]) == 0
    assert len(errors["iou"]) == 0


def test_e56_audit_smoke(tmp_path):
    """Verify that E56 diagnostic audit runs end-to-end and writes expected outputs."""
    loc_metrics, refinement_deltas, oracle_metrics, export_dict = run_e56_localization_decomposition_audit(
        output_dir=tmp_path / "e56_output",
        device_str="cpu",
        max_images=2,
    )

    assert len(loc_metrics) == 4
    assert len(refinement_deltas) == 4
    assert len(oracle_metrics) == 3

    # Check metrics
    sub4_loc = loc_metrics[0]
    assert sub4_loc.scale_bin == "<4px"
    assert sub4_loc.center_rmse_px > 0.0
    assert sub4_loc.scale_rmse_px > 0.0

    # Check Dual Oracle
    baseline = oracle_metrics[0]
    oracle_box = oracle_metrics[1]
    oracle_class = oracle_metrics[2]
    assert oracle_box.map50_95 > baseline.map50_95
    assert oracle_box.map50_95 - baseline.map50_95 >= 15.0  # +24.0 pp gain

    # Check files created
    assert (tmp_path / "e56_output" / "e56_localization_error_decomposition.png").is_file()
    assert (tmp_path / "e56_output" / "e56_localization_metrics.json").is_file()

    with open(tmp_path / "e56_output" / "e56_localization_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["ticket"] == "E56"
    assert "causal_gap_analysis" in data
    assert data["causal_gap_analysis"]["prioritize_ticket_e69"] is True
