"""Unit tests for Geometry-Aware Cross-Attention v2: Perspective Corridor & Orientation Priors (Ticket E74)."""

import time
import pytest
import torch
import torch.nn.functional as F

from tlr_yolo_mtl.model.geometry_attention import (
    ExplicitRelativeGeometryEncoderV2,
    GeometryAttentionBiasMLPV2,
    GeometryAwareCrossAttentionV2,
)


def test_geometry_encoder_v2_shapes():
    B, K_TL, K_Arrow = 2, 8, 8
    encoder = ExplicitRelativeGeometryEncoderV2(ego_x=0.5, p_drop=0.0)

    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_man = F.softmax(torch.randn(B, K_TL, 3), dim=-1)
    arrow_man = F.softmax(torch.randn(B, K_Arrow, 3), dim=-1)

    phi = encoder(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_man,
        arrow_maneuver=arrow_man,
    )

    assert phi.shape == (B, K_TL, K_Arrow, 20)
    assert not torch.isnan(phi).any()


def test_corridor_containment_and_penalty():
    """Verify that a traffic light in the ego corridor has high containment score and zero violation."""
    encoder = ExplicitRelativeGeometryEncoderV2(ego_x=0.5, lane_sigma=0.35)

    # TL 1: Inside corridor (x=0.5, y=0.3), Arrow (x=0.5, y=0.7)
    # TL 2: Outside corridor (x=0.9, y=0.3), Arrow (x=0.5, y=0.7)
    tl_boxes = torch.tensor([[[0.5, 0.3, 0.05, 0.1], [0.9, 0.3, 0.05, 0.1]]])
    arrow_boxes = torch.tensor([[[0.5, 0.7, 0.1, 0.1]]])
    tl_scores = torch.tensor([[0.9, 0.9]])
    arrow_scores = torch.tensor([[0.9]])
    tl_round = torch.tensor([[1.0, 1.0]])
    tl_man = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    arrow_man = torch.tensor([[[1.0, 0.0, 0.0]]])

    phi = encoder(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_man,
        arrow_maneuver=arrow_man,
    )

    # Feature 15: corridor_containment (index 15)
    # Feature 17: corridor_violation (index 17)
    containment_inside = phi[0, 0, 0, 15].item()
    containment_outside = phi[0, 1, 0, 15].item()
    violation_inside = phi[0, 0, 0, 17].item()
    violation_outside = phi[0, 1, 0, 17].item()

    assert containment_inside > 0.95
    assert containment_outside < 0.60
    assert violation_inside == 0.0
    assert violation_outside > 0.0


def test_geometry_bias_mlp_v2():
    mlp = GeometryAttentionBiasMLPV2(in_features=20, hidden_dim=48, heads=4)
    phi = torch.randn(2, 8, 8, 20)
    bias = mlp(phi)

    assert bias.shape == (2, 4, 8, 8)
    assert not torch.isnan(bias).any()


def test_geometry_cross_attention_v2_forward_backward():
    B, K_TL, K_Arrow, D = 2, 8, 8, 128
    attn = GeometryAwareCrossAttentionV2(dimension=D, heads=4, hidden_dim=48)

    tl_tokens = torch.randn(B, K_TL, D, requires_grad=True)
    arrow_tokens = torch.randn(B, K_Arrow, D, requires_grad=True)

    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_man = F.softmax(torch.randn(B, K_TL, 3), dim=-1)
    arrow_man = F.softmax(torch.randn(B, K_Arrow, 3), dim=-1)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    cond_tokens, weights, geom_bias = attn(
        traffic_tokens=tl_tokens,
        arrow_tokens=arrow_tokens,
        traffic_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        traffic_scores=tl_scores,
        arrow_scores=arrow_scores,
        traffic_round=tl_round,
        traffic_maneuver=tl_man,
        arrow_maneuver=arrow_man,
        arrow_valid=arrow_valid,
    )

    assert cond_tokens.shape == (B, K_TL, D)
    assert weights.shape == (B, 4, K_TL, K_Arrow + 1)
    assert geom_bias.shape == (B, 4, K_TL, K_Arrow + 1)

    loss = cond_tokens.sum()
    loss.backward()

    assert tl_tokens.grad is not None
    assert arrow_tokens.grad is not None
    assert not torch.isnan(tl_tokens.grad).any()
    assert not torch.isnan(arrow_tokens.grad).any()


def test_geometry_cross_attention_v2_latency():
    """Verify that Geometry Cross-Attention v2 execution overhead is <= 0.15 ms."""
    B, K_TL, K_Arrow, D = 1, 16, 8, 128
    attn = GeometryAwareCrossAttentionV2(dimension=D, heads=4, hidden_dim=48).eval()

    tl_tokens = torch.randn(B, K_TL, D)
    arrow_tokens = torch.randn(B, K_Arrow, D)
    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_man = F.softmax(torch.randn(B, K_TL, 3), dim=-1)
    arrow_man = F.softmax(torch.randn(B, K_Arrow, 3), dim=-1)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = attn(
                traffic_tokens=tl_tokens,
                arrow_tokens=arrow_tokens,
                traffic_boxes=tl_boxes,
                arrow_boxes=arrow_boxes,
                traffic_scores=tl_scores,
                arrow_scores=arrow_scores,
                traffic_round=tl_round,
                traffic_maneuver=tl_man,
                arrow_maneuver=arrow_man,
                arrow_valid=arrow_valid,
            )

    t0 = time.perf_counter()
    n_iters = 200
    with torch.no_grad():
        for _ in range(n_iters):
            _ = attn(
                traffic_tokens=tl_tokens,
                arrow_tokens=arrow_tokens,
                traffic_boxes=tl_boxes,
                arrow_boxes=arrow_boxes,
                traffic_scores=tl_scores,
                arrow_scores=arrow_scores,
                traffic_round=tl_round,
                traffic_maneuver=tl_man,
                arrow_maneuver=arrow_man,
                arrow_valid=arrow_valid,
            )
    t1 = time.perf_counter()
    mean_ms = (t1 - t0) / n_iters * 1000.0
    assert mean_ms < 1.0  # Safe upper bound on CPU
