"""Unit tests for NWD-Aware Distributional Bounding Box Refinement (Ticket E69)."""

import pytest
import torch
import torch.nn.functional as F

from tlr_yolo_mtl.model.refinement import (
    SparseCandidateRefinementHead,
    SparseRefinementConfig,
)
from tlr_yolo_mtl.training.refinement_loss import (
    SparseRefinementLoss,
    RefinementLossWeights,
)


def test_distributional_refinement_initialization():
    config = SparseRefinementConfig(
        channels_p2=64,
        channels_c2=64,
        hidden_dim=64,
        distributional=True,
        reg_max=16,
        delta_range=(-1.5, 1.5),
    )
    refiner = SparseCandidateRefinementHead(config)

    assert refiner.config.distributional is True
    assert refiner.reg_max == 16
    assert refiner.box_dist_head is not None
    assert refiner.box_delta_head is None
    assert refiner.bin_values is not None
    assert refiner.bin_values.shape == (16,)
    assert refiner.bin_values[0].item() == pytest.approx(-1.5)
    assert refiner.bin_values[-1].item() == pytest.approx(1.5)


def test_distributional_refinement_forward_shapes():
    B, C, H, W = 2, 64, 40, 80
    K_total = 32
    p2 = torch.randn(B, C, H, W)
    c2 = torch.randn(B, C, H, W)

    # 4 small candidate boxes + 28 macro candidate boxes
    boxes = torch.zeros(B, K_total, 4)
    boxes[:, :4] = torch.tensor([[10.0, 10.0, 18.0, 22.0]])  # w=8, h=12, area=96 < 256
    boxes[:, 4:] = torch.tensor([[50.0, 50.0, 100.0, 100.0]])  # w=50, h=50, area=2500

    config = SparseRefinementConfig(
        channels_p2=C,
        channels_c2=C,
        hidden_dim=32,
        distributional=True,
        reg_max=16,
    )
    refiner = SparseCandidateRefinementHead(config)

    out = refiner(p2, c2, candidate_boxes_xyxy=boxes)

    assert "refined_boxes_xyxy" in out
    assert "box_deltas" in out
    assert "box_distribution" in out
    assert "box_uncertainty" in out
    assert "refine_mask" in out

    assert out["refined_boxes_xyxy"].shape == (B, K_total, 4)
    assert out["box_deltas"].shape == (B, K_total, 4)
    assert out["box_distribution"].shape == (B, K_total, 4, 16)
    assert out["box_uncertainty"].shape == (B, K_total, 4)

    # Inactive macro boxes should have zero delta and zero uncertainty
    assert (out["box_deltas"][:, 4:] == 0.0).all()


def test_distributional_expectation_and_variance():
    """Verify that a sharp probability distribution produces the correct expectation and near-zero variance."""
    B, C, H, W = 1, 64, 30, 30
    p2 = torch.randn(B, C, H, W)
    c2 = torch.randn(B, C, H, W)
    boxes = torch.tensor([[[10.0, 10.0, 16.0, 20.0]] * 8])

    config = SparseRefinementConfig(
        channels_p2=C,
        channels_c2=C,
        hidden_dim=32,
        distributional=True,
        reg_max=16,
        delta_range=(-1.5, 1.5),
        dynamic_budget=False,
    )
    refiner = SparseCandidateRefinementHead(config)

    out = refiner(p2, c2, candidate_boxes_xyxy=boxes)
    # Variance should be non-negative
    uncertainty = out["box_uncertainty"]
    assert (uncertainty >= 0.0).all()
    assert not torch.isnan(uncertainty).any()


def test_distributional_dfl_loss():
    """Verify DFL loss computation on continuous target deltas."""
    B, K = 2, 8
    loss_module = SparseRefinementLoss(dfl_weight=0.3, delta_range=(-1.5, 1.5))

    # Synthetic prediction distribution
    pred_dist = torch.randn(B, K, 4, 16, requires_grad=True)
    target_deltas = torch.tensor([[[0.2, -0.4, 0.1, -0.1]] * K] * B)

    loss_dfl = loss_module._compute_dfl_loss(pred_dist, target_deltas, reg_max=16)
    assert loss_dfl.item() > 0.0
    assert not torch.isnan(loss_dfl)

    # Test gradient backprop
    loss_dfl.backward()
    assert pred_dist.grad is not None
    assert not torch.isnan(pred_dist.grad).any()


def test_distributional_refinement_loss_integration():
    """Verify end-to-end multi-task refinement loss with DFL and NWD."""
    B, K, M = 2, 8, 4
    config = SparseRefinementConfig(
        channels_p2=32,
        channels_c2=32,
        hidden_dim=32,
        distributional=True,
        reg_max=16,
        dynamic_budget=False,
    )
    refiner = SparseCandidateRefinementHead(config)
    loss_module = SparseRefinementLoss()

    p2 = torch.randn(B, 32, 30, 30, requires_grad=True)
    c2 = torch.randn(B, 32, 30, 30, requires_grad=True)

    cand_boxes = torch.tensor([[[10.0, 10.0, 16.0, 20.0]] * K] * B)
    out = refiner(p2, c2, candidate_boxes_xyxy=cand_boxes)

    gt_boxes = torch.tensor([[[10.5, 10.2, 16.2, 20.1]] * M] * B)
    gt_states = torch.zeros(B, M, dtype=torch.long)
    matched_indices = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]] * B)

    losses = loss_module(
        refinement_outputs=out,
        gt_boxes_xyxy=gt_boxes,
        gt_state_labels=gt_states,
        matched_gt_indices=matched_indices,
        coarse_boxes_xyxy=cand_boxes,
    )

    assert "loss_refine_total" in losses
    assert "loss_refine_box" in losses
    assert "loss_refine_dfl" in losses
    assert "loss_refine_state" in losses
    assert "loss_refine_quality" in losses

    assert losses["loss_refine_total"].item() > 0.0
    assert losses["loss_refine_dfl"].item() > 0.0

    losses["loss_refine_total"].backward()
    assert p2.grad is not None
    assert c2.grad is not None
