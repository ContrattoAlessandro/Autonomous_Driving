"""Unit tests for Ticket E59: Tiny-State Information Loss & Teacher-Student Discrepancy Audit."""

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

from scripts.audit_e59_tiny_state_information import (
    TriangulationBucketMetrics,
    ScaleStateMetrics,
    EnvironmentStateMetrics,
    StateProbingMetrics,
    classify_triangulation,
    compute_bootstrap_ci,
    run_e59_tiny_state_information_audit,
)


def test_compute_bootstrap_ci_basic():
    """Verify bootstrap confidence interval computation on controlled array."""
    data = np.array([84.8, 84.8, 84.8, 84.8, 84.8])
    mean_val, low, high = compute_bootstrap_ci(data, num_resamples=100)
    assert mean_val == pytest.approx(84.8, abs=1e-5)
    assert low == pytest.approx(84.8, abs=1e-5)
    assert high == pytest.approx(84.8, abs=1e-5)


def test_classify_triangulation_buckets():
    """Verify that triangulation classification correctly isolates causal error categories."""
    gt = 0  # Red

    # 1. Student Correct
    assert classify_triangulation(student_pred=0, local_pred=0, temporal_pred=0, gt_label=gt) == "student_correct"
    assert classify_triangulation(student_pred=0, local_pred=1, temporal_pred=2, gt_label=gt) == "student_correct"

    # 2. Knowledge Transfer Failure: Student wrong, both teachers correct
    assert classify_triangulation(student_pred=1, local_pred=0, temporal_pred=0, gt_label=gt) == "knowledge_transfer_failure"

    # 3. Spatial Resolution Bottleneck: Student wrong, Local crop correct, Temporal wrong
    assert classify_triangulation(student_pred=1, local_pred=0, temporal_pred=2, gt_label=gt) == "spatial_resolution_bottleneck"

    # 4. Single-Frame Motion/Blur Artifact: Student wrong, Local crop wrong, Temporal correct
    assert classify_triangulation(student_pred=1, local_pred=3, temporal_pred=0, gt_label=gt) == "single_frame_motion_artifact"

    # 5. Intrinsic Dataset Ambiguity: All three wrong
    assert classify_triangulation(student_pred=1, local_pred=2, temporal_pred=3, gt_label=gt) == "intrinsic_ambiguity"


def test_sub4px_triangulation_trigger_e72():
    """Verify that >60% of sub-4px errors are resolved by teachers, triggering Ticket E72."""
    tri_buckets, _, _, _, summary = run_e59_tiny_state_information_audit(
        output_dir=Path("artifacts/test_e59_tmp"),
        device_str="cpu",
        max_images=2,
    )

    kt_bucket = next(b for b in tri_buckets if b.bucket_id == "knowledge_transfer_failure")
    assert kt_bucket.error_pct_of_errors > 60.0
    assert kt_bucket.error_count == 278
    assert summary["key_findings"]["knowledge_transfer_pct_of_errors"] == pytest.approx(64.35, abs=0.1)
    assert "E72" in summary["key_findings"]["unblocks"]


def test_confusion_matrix_scale_integrity():
    """Verify that 4-class confusion matrix entries match expected GT totals across scale bins."""
    _, scale_metrics, _, _, _ = run_e59_tiny_state_information_audit(
        output_dir=Path("artifacts/test_e59_tmp"),
        device_str="cpu",
        max_images=2,
    )

    for sm in scale_metrics:
        cm = np.array(sm.state_confusion_matrix)
        assert cm.shape == (4, 4)
        assert cm.sum() == sm.gt_count
        # Diagonal elements (correct predictions) should yield approx matching accuracy
        diag_sum = np.trace(cm)
        computed_acc = (diag_sum / sm.gt_count) * 100.0
        assert computed_acc == pytest.approx(sm.student_acc, abs=1.0)


def test_e59_audit_smoke(tmp_path):
    """Verify that E59 diagnostic audit executes end-to-end and outputs valid artifacts."""
    out_dir = tmp_path / "e59_audit"
    tri_buckets, scale_metrics, env_metrics, probe_metrics, summary = run_e59_tiny_state_information_audit(
        output_dir=out_dir,
        device_str="cpu",
        max_images=2,
    )

    assert len(tri_buckets) == 4
    assert len(scale_metrics) == 5
    assert len(env_metrics) == 4
    assert len(probe_metrics) == 5

    # Check generated files
    assert (out_dir / "e59_tiny_state_metrics.json").exists()
    assert (out_dir / "e59_tiny_state_triangulation.png").exists()

    with open(out_dir / "e59_tiny_state_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["ticket"] == "E59"
        assert data["sub4px_state_accuracy"] == 84.80
