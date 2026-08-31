"""Unit and integration tests for Ticket 05: Vanishing Point 18D Geometry Descriptor."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.geometry_attention import (
    ExplicitRelativeGeometryEncoder,
    GeometryAttentionBiasMLP,
    GeometryAwareCrossAttention,
    GeometryAwareUnifiedDetect,
    attach_geometry_aware_unified_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig


def test_vanishing_point_18d_feature_dimensions_and_properties() -> None:
    """Verify feature dimensions, properties, and tensor shapes for 18D and 14D modes."""
    B, K_TL, K_Arrow = 3, 10, 14

    # Default: 18D Vanishing Point descriptor
    encoder_18d = ExplicitRelativeGeometryEncoder(ego_x=0.5, vp_x=0.5, vp_y=0.5, include_vanishing_point=True)
    assert encoder_18d.feature_dim == 18

    # Legacy: 14D descriptor
    encoder_14d = ExplicitRelativeGeometryEncoder(ego_x=0.5, include_vanishing_point=False)
    assert encoder_14d.feature_dim == 14

    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_maneuver = torch.softmax(torch.randn(B, K_TL, 3), dim=-1)
    arrow_maneuver = torch.softmax(torch.randn(B, K_Arrow, 3), dim=-1)

    phi_18d = encoder_18d(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_maneuver,
        arrow_maneuver=arrow_maneuver,
    )
    assert phi_18d.shape == (B, K_TL, K_Arrow, 18)
    assert not torch.isnan(phi_18d).any()
    assert not torch.isinf(phi_18d).any()

    phi_14d = encoder_14d(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_maneuver,
        arrow_maneuver=arrow_maneuver,
    )
    assert phi_14d.shape == (B, K_TL, K_Arrow, 14)


def test_vanishing_point_exact_geometric_values() -> None:
    """Verify exact analytical values of the 4 vanishing point descriptor dimensions."""
    vp_x, vp_y = 0.50, 0.50
    encoder = ExplicitRelativeGeometryEncoder(ego_x=0.50, vp_x=vp_x, vp_y=vp_y, include_vanishing_point=True)

    # 1 batch, 2 TLs, 1 Arrow
    # TL 0: Overhead ego lane center (cx=0.50, cy=0.25, w=0.03, h=0.08)
    # TL 1: Right adjacent lane gantry (cx=0.80, cy=0.35, w=0.03, h=0.08)
    # Arrow 0: Ego lane road arrow (cx=0.50, cy=0.85, w=0.10, h=0.15)
    tl_boxes = torch.tensor([[[0.50, 0.25, 0.03, 0.08], [0.80, 0.35, 0.03, 0.08]]])
    arrow_boxes = torch.tensor([[[0.50, 0.85, 0.10, 0.15]]])

    tl_scores = torch.tensor([[0.95, 0.90]])
    arrow_scores = torch.tensor([[0.88]])
    tl_round = torch.tensor([[1.0, 1.0]])
    tl_man = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    ar_man = torch.tensor([[[0.0, 1.0, 0.0]]])

    phi = encoder(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=tl_scores,
        arrow_scores=arrow_scores,
        tl_round=tl_round,
        tl_maneuver=tl_man,
        arrow_maneuver=ar_man,
    )

    # Dimensions 14, 15, 16, 17 are [tl_dx_vp, tl_dy_vp, dist_horizon, theta_road]
    # TL 0 (Overhead center):
    # dx_vp = 0.50 - 0.50 = 0.00
    # dy_vp = 0.25 - 0.50 = -0.25
    # dist_horizon = |0.50 - 0.25| = 0.25
    # theta_road = atan2(-0.25, 0.00) / pi = -0.50 (-pi/2)
    expected_tl0_vp = torch.tensor([0.00, -0.25, 0.25, -0.50])
    actual_tl0_vp = phi[0, 0, 0, 14:]
    assert torch.allclose(actual_tl0_vp, expected_tl0_vp, atol=1e-5)

    # TL 1 (Right adjacent lane):
    # dx_vp = 0.80 - 0.50 = +0.30
    # dy_vp = 0.35 - 0.50 = -0.15
    # dist_horizon = |0.50 - 0.35| = 0.15
    # theta_road = atan2(-0.15, +0.30) / pi = -0.14758...
    expected_theta1 = math.atan2(-0.15, 0.30) / math.pi
    expected_tl1_vp = torch.tensor([0.30, -0.15, 0.15, expected_theta1])
    actual_tl1_vp = phi[0, 1, 0, 14:]
    assert torch.allclose(actual_tl1_vp, expected_tl1_vp, atol=1e-5)


def test_vanishing_point_perspective_symmetry() -> None:
    """Verify lateral perspective symmetry around the corridor vanishing point axis."""
    vp_x, vp_y = 0.50, 0.50
    encoder = ExplicitRelativeGeometryEncoder(ego_x=0.50, vp_x=vp_x, vp_y=vp_y, include_vanishing_point=True)

    delta = 0.25
    # TL Left: cx = 0.50 - 0.25 = 0.25, cy = 0.30
    # TL Right: cx = 0.50 + 0.25 = 0.75, cy = 0.30
    tl_boxes = torch.tensor([[[0.50 - delta, 0.30, 0.02, 0.06], [0.50 + delta, 0.30, 0.02, 0.06]]])
    arrow_boxes = torch.tensor([[[0.50, 0.80, 0.08, 0.12]]])

    phi = encoder(
        tl_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        tl_scores=torch.tensor([[0.9, 0.9]]),
        arrow_scores=torch.tensor([[0.9]]),
        tl_round=torch.tensor([[1.0, 1.0]]),
        tl_maneuver=torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]),
        arrow_maneuver=torch.tensor([[[0.0, 1.0, 0.0]]]),
    )

    tl_left_vp = phi[0, 0, 0, 14:]   # [dx_vp, dy_vp, dist_h, theta]
    tl_right_vp = phi[0, 1, 0, 14:]

    # dx_vp should be opposite in sign
    assert torch.allclose(tl_left_vp[0], -tl_right_vp[0], atol=1e-6)
    # dy_vp and dist_horizon should be strictly identical
    assert torch.allclose(tl_left_vp[1], tl_right_vp[1], atol=1e-6)
    assert torch.allclose(tl_left_vp[2], tl_right_vp[2], atol=1e-6)


def test_vanishing_point_18d_gradient_flow() -> None:
    """Verify backward gradient flow through all 18 input channels of the geometry MLP."""
    B, K_TL, K_Arrow, D, H = 2, 6, 8, 64, 4
    attn = GeometryAwareCrossAttention(
        dimension=D, heads=H, hidden_dim=32, include_vanishing_point=True, use_confidence_gating=True
    )
    # Set non-zero weights on the final projection layer to test full chain-rule backpropagation
    nn.init.normal_(attn.geometry_mlp.network[-1].weight, std=0.1)

    tl_tokens = torch.randn(B, K_TL, D, requires_grad=True)
    arrow_tokens = torch.randn(B, K_Arrow, D, requires_grad=True)
    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_scores = torch.rand(B, K_TL)
    arrow_scores = torch.rand(B, K_Arrow)
    tl_round = torch.rand(B, K_TL)
    tl_man = torch.rand(B, K_TL, 3)
    ar_man = torch.rand(B, K_Arrow, 3)
    ar_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    conditioned, weights, bias = attn(
        traffic_tokens=tl_tokens,
        arrow_tokens=arrow_tokens,
        traffic_boxes=tl_boxes,
        arrow_boxes=arrow_boxes,
        traffic_scores=tl_scores,
        arrow_scores=arrow_scores,
        traffic_round=tl_round,
        traffic_maneuver=tl_man,
        arrow_maneuver=ar_man,
        arrow_valid=ar_valid,
    )

    loss = conditioned.sum() + bias.sum()
    loss.backward()

    assert tl_tokens.grad is not None
    assert arrow_tokens.grad is not None
    # Check that the first Linear layer in MLP receives gradient for all 18 dimensions
    mlp_weight_grad = attn.geometry_mlp.network[0].weight.grad
    assert mlp_weight_grad is not None
    assert mlp_weight_grad.shape == (32, 18)
    assert not torch.isnan(mlp_weight_grad).any()
    assert (mlp_weight_grad.abs().sum(dim=0) > 0).all(), "All 18 input channels must receive non-zero gradients"


def test_vanishing_point_18d_fp16_cuda() -> None:
    """Verify FP16 inference execution on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for FP16 test")

    device = torch.device("cuda")
    B, K_TL, K_Arrow, D, H = 2, 8, 12, 64, 4
    attn = GeometryAwareCrossAttention(
        dimension=D, heads=H, hidden_dim=32, include_vanishing_point=True, use_confidence_gating=True
    ).to(device).half()

    tl_tokens = torch.randn(B, K_TL, D, device=device, dtype=torch.float16)
    arrow_tokens = torch.randn(B, K_Arrow, D, device=device, dtype=torch.float16)
    tl_boxes = torch.rand(B, K_TL, 4, device=device, dtype=torch.float16)
    arrow_boxes = torch.rand(B, K_Arrow, 4, device=device, dtype=torch.float16)
    tl_scores = torch.rand(B, K_TL, device=device, dtype=torch.float16)
    arrow_scores = torch.rand(B, K_Arrow, device=device, dtype=torch.float16)
    tl_round = torch.rand(B, K_TL, device=device, dtype=torch.float16)
    tl_man = torch.rand(B, K_TL, 3, device=device, dtype=torch.float16)
    ar_man = torch.rand(B, K_Arrow, 3, device=device, dtype=torch.float16)
    ar_valid = torch.ones(B, K_Arrow, device=device, dtype=torch.bool)

    with torch.inference_mode():
        conditioned, weights, bias = attn(
            traffic_tokens=tl_tokens,
            arrow_tokens=arrow_tokens,
            traffic_boxes=tl_boxes,
            arrow_boxes=arrow_boxes,
            traffic_scores=tl_scores,
            arrow_scores=arrow_scores,
            traffic_round=tl_round,
            traffic_maneuver=tl_man,
            arrow_maneuver=ar_man,
            arrow_valid=ar_valid,
        )

    assert conditioned.dtype == torch.float16
    assert weights.dtype == torch.float16
    assert bias.dtype == torch.float16
    assert not torch.isnan(conditioned).any()


def test_full_unified_detector_with_18d_descriptor() -> None:
    """Verify integration of 18D geometry encoder with full detection model."""
    cfg_path = PROJECT_ROOT / "configs/model/tlr_yolo11n_p2.yaml"
    if not cfg_path.is_file():
        pytest.skip("Model config not found")

    wrapper = build_detection_model(cfg_path)
    head_config = UnifiedHeadConfig(
        token_dim=64,
        attention_heads=4,
        max_traffic_lights=16,
        max_arrows=16,
    )
    unified = attach_geometry_aware_unified_relevance_head(
        wrapper,
        config=head_config,
        hidden_dim=32,
        include_vanishing_point=True,
        vp_x=0.5,
        vp_y=0.5,
        use_confidence_gating=True,
    )

    assert isinstance(unified, GeometryAwareUnifiedDetect)
    assert isinstance(unified.cross_attention, GeometryAwareCrossAttention)
    assert unified.cross_attention.geometry_encoder.include_vanishing_point is True
    assert unified.cross_attention.geometry_encoder.feature_dim == 18

    # Forward pass on synthetic input tensor
    x = torch.randn(1, 3, 384, 640)
    out = wrapper.model(x)

    assert isinstance(out, dict)
    assert "relevance_logits" in out
    assert "attention_weights" in out
    assert "attention_geometry_bias" in out
    assert out["relevance_logits"].shape == (1, 1, 16)
