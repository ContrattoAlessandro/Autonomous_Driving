"""Unit tests for Ticket E58: Scale-Adaptive NWD-TAL Supervision & Anchor Assignment Audit."""

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

from scripts.audit_e58_nwd_tal_assignment import (
    AssignerAllocationMetrics,
    AssignerComparisonMetrics,
    compute_bootstrap_ci,
    run_e58_nwd_tal_assignment_audit,
)
from tlr_yolo_mtl.training.tal import build_task_aligned_assigner, compute_nwd_similarity
from ultralytics.utils.tal import make_anchors


def test_compute_bootstrap_ci_basic():
    """Verify bootstrap confidence interval computation on controlled array."""
    data = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    mean_val, low, high = compute_bootstrap_ci(data, num_resamples=100)
    assert mean_val == pytest.approx(10.0, abs=1e-5)
    assert low == pytest.approx(10.0, abs=1e-5)
    assert high == pytest.approx(10.0, abs=1e-5)


def test_nwd_tal_prevents_tiny_anchor_starvation():
    """Verify that NWD-Aware TAL allocates positive anchors when IoU is strictly zero."""
    strides = [4.0, 8.0, 16.0, 32.0]
    feats = [
        torch.zeros(1, 64, 240, 480),
        torch.zeros(1, 128, 120, 240),
        torch.zeros(1, 256, 60, 120),
        torch.zeros(1, 512, 30, 60),
    ]
    anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)

    std_assigner = build_task_aligned_assigner(
        assigner_type="standard",
        topk=10,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=strides,
    )
    nwd_assigner = build_task_aligned_assigner(
        assigner_type="nwd",
        topk=10,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=strides,
        nwd_weight=0.5,
        nwd_constant=12.0,
        area_threshold=64.0,
        mode="scale_adaptive",
    )

    # Sub-4px GT Box (3x3 pixels)
    gt_bboxes = torch.tensor([[[100.0, 100.0, 103.0, 103.0]]])
    gt_labels = torch.zeros((1, 1, 1))
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)

    # Predictions with 4px offset (zero IoU with 3x3 GT)
    pd_bboxes_scaled = anchor_points.unsqueeze(0) * stride_tensor
    pd_bboxes_xyxy = torch.cat([pd_bboxes_scaled + 2.0, pd_bboxes_scaled + 5.0], dim=-1)
    pd_scores = torch.full((1, anchor_points.shape[0], 2), 0.5)

    _, _, _, fg_std, tgt_gt_std = std_assigner(
        pd_scores, pd_bboxes_xyxy, anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
    )
    _, _, _, fg_nwd, tgt_gt_nwd = nwd_assigner(
        pd_scores, pd_bboxes_xyxy, anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
    )

    n_pos_std = int((fg_std & (tgt_gt_std == 0)).sum().item())
    n_pos_nwd = int((fg_nwd & (tgt_gt_nwd == 0)).sum().item())

    # Standard TAL starves (<=1 anchor), whereas NWD-TAL recovers multiple positive anchors
    assert n_pos_std <= 1
    assert n_pos_nwd >= 4
    assert n_pos_nwd > n_pos_std


def test_e58_audit_smoke(tmp_path):
    """Verify that E58 diagnostic audit runs end-to-end and writes expected outputs."""
    alloc_metrics, comp_metrics, export_dict = run_e58_nwd_tal_assignment_audit(
        output_dir=tmp_path / "e58_output",
        device_str="cpu",
        max_images=2,
    )

    assert len(alloc_metrics) == 4
    assert len(comp_metrics) == 2

    # Check scale allocation metrics
    sub4_alloc = alloc_metrics[0]
    assert sub4_alloc.scale_bin == "Sub-4px (<16 px^2)"
    assert sub4_alloc.mean_n_pos >= 4.0
    assert sub4_alloc.p2_allocation_pct > 95.0  # >98% on P2
    assert sub4_alloc.starvation_rate_pct < 5.0

    # Check head-to-head comparison
    nwd_comp = comp_metrics[1]
    assert nwd_comp.assigner_type == "NWD-Aware TAL (Champion v4)"
    assert nwd_comp.sub4px_starvation_rate_pct < 5.0
    assert nwd_comp.sub4px_mean_n_pos > 5.0

    # Check files created
    assert (tmp_path / "e58_output" / "e58_nwd_tal_assignment.png").is_file()
    assert (tmp_path / "e58_output" / "e58_assignment_metrics.json").is_file()

    with open(tmp_path / "e58_output" / "e58_assignment_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["ticket"] == "E58"
    assert "causal_gap_analysis" in data
    assert data["causal_gap_analysis"]["trigger_ticket_e67"] is False  # Supervision is adequate
    assert data["causal_gap_analysis"]["sub4px_starvation_rate_pct"] < 15.0
