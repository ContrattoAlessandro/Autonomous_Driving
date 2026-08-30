"""Unit tests for Dynamic Scene-Adaptive Sparse Refinement Budget (Ticket E68)."""

import pytest
import torch

from tlr_yolo_mtl.model.refinement import (
    SparseCandidateRefinementHead,
    SparseRefinementConfig,
    select_dynamic_refinement_budget,
)


def test_select_dynamic_budget_tiers():
    tiers = (8, 16, 32, 48, 64)
    assert select_dynamic_refinement_budget(2, tiers=tiers) == 8
    assert select_dynamic_refinement_budget(8, tiers=tiers) == 8
    assert select_dynamic_refinement_budget(9, tiers=tiers) == 16
    assert select_dynamic_refinement_budget(16, tiers=tiers) == 16
    assert select_dynamic_refinement_budget(25, tiers=tiers) == 32
    assert select_dynamic_refinement_budget(35, tiers=tiers) == 48
    assert select_dynamic_refinement_budget(55, tiers=tiers) == 64
    assert select_dynamic_refinement_budget(100, tiers=tiers) == 64


def test_dynamic_refinement_forward_shapes():
    B, C, H, W = 2, 64, 60, 120
    K_total = 64
    p2 = torch.randn(B, C, H, W)
    c2 = torch.randn(B, C, H, W)

    # Create dummy candidate boxes: 4 small boxes (<256 px^2) + 60 macro boxes
    boxes = torch.zeros(B, K_total, 4)
    # Small boxes
    boxes[:, :4] = torch.tensor([[10.0, 10.0, 18.0, 22.0]])  # w=8, h=12, area=96
    # Macro boxes
    boxes[:, 4:] = torch.tensor([[50.0, 50.0, 100.0, 100.0]]) # w=50, h=50, area=2500

    config = SparseRefinementConfig(
        channels_p2=C,
        channels_c2=C,
        hidden_dim=64,
        area_threshold=256.0,
        dynamic_budget=True,
        budget_tiers=(8, 16, 32, 48, 64),
        min_budget=8,
        max_budget=64,
    )
    refiner = SparseCandidateRefinementHead(config)

    out = refiner(p2, c2, candidate_boxes_xyxy=boxes)

    assert "refined_boxes_xyxy" in out
    assert "refined_state_logits" in out
    assert "box_deltas" in out
    assert "refine_mask" in out

    assert out["refined_boxes_xyxy"].shape == (B, K_total, 4)
    assert out["refined_state_logits"].shape == (B, K_total, 4)
    assert out["box_deltas"].shape == (B, K_total, 4)

    # Active mask should be True only for the 4 small boxes
    assert out["refine_mask"][:, :4].all()
    assert not out["refine_mask"][:, 4:].any()

    # Inactive macro boxes should have zero delta
    assert (out["box_deltas"][:, 4:] == 0.0).all()


def test_dynamic_refinement_backward():
    B, C, H, W = 1, 64, 30, 30
    p2 = torch.randn(B, C, H, W, requires_grad=True)
    c2 = torch.randn(B, C, H, W, requires_grad=True)
    boxes = torch.tensor([[[5.0, 5.0, 10.0, 15.0]] * 16])  # [1, 16, 4]

    refiner = SparseCandidateRefinementHead(
        channels_p2=C,
        channels_c2=C,
        hidden_dim=32,
        dynamic_budget=True,
        budget_tiers=(8, 16, 32),
    )
    out = refiner(p2, c2, candidate_boxes_xyxy=boxes)
    loss = out["refined_boxes_xyxy"].sum() + out["refined_state_logits"].sum()
    loss.backward()

    assert p2.grad is not None
    assert c2.grad is not None
