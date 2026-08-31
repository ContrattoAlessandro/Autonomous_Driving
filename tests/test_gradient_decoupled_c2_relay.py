"""Unit and integration tests for Ticket 01: Gradient-Decoupled C2 Texture Relay."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.neck import (
    GradientDecoupledC2Relay,
    GradientDecoupledC2RelayConfig,
    ScaleAwareFeatureRelay,
    ScaleAwareFeatureRelayV2,
    ScaleAwareRelayConfig,
    ScaleAwareRelayV2Config,
)


def test_gradient_isolation_c2_stop_gradient_true():
    """Verify that c2_stop_gradient=True strictly zeroes out backward gradient flow to C2."""
    B, C_c2, C_p2, H, W = 2, 64, 64, 24, 48
    c2 = torch.randn(B, C_c2, H, W, requires_grad=True)
    p2 = torch.randn(B, C_p2, H, W, requires_grad=True)

    relay = GradientDecoupledC2Relay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        gating_type="spatial_channel",
        c2_stop_gradient=True,
    )

    p2_refined = relay([c2, p2])
    loss = p2_refined.sum()
    loss.backward()

    # C2 must receive NO gradients due to stop-gradient barrier
    assert c2.grad is None or c2.grad.abs().sum().item() == 0.0

    # P2 and relay parameters MUST receive non-zero gradients
    assert p2.grad is not None
    assert p2.grad.abs().sum().item() > 0.0
    assert relay.c2_proj[0].weight.grad is not None
    assert relay.c2_proj[0].weight.grad.abs().sum().item() > 0.0
    assert relay.gate[0].weight.grad is not None
    assert relay.gate[0].weight.grad.abs().sum().item() > 0.0


def test_gradient_leakage_c2_stop_gradient_false():
    """Verify that c2_stop_gradient=False allows gradients to backpropagate into C2."""
    B, C_c2, C_p2, H, W = 2, 64, 64, 24, 48
    c2 = torch.randn(B, C_c2, H, W, requires_grad=True)
    p2 = torch.randn(B, C_p2, H, W, requires_grad=True)

    relay = ScaleAwareFeatureRelay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        gating_type="spatial_channel",
        c2_stop_gradient=False,
    )

    p2_refined = relay([c2, p2])
    loss = p2_refined.sum()
    loss.backward()

    # C2 receives backpropagated gradients when stop-gradient is disabled
    assert c2.grad is not None
    assert c2.grad.abs().sum().item() > 0.0


def test_backbone_gradient_isolation_end_to_end():
    """Verify that shallow backbone convolutional weights remain unperturbed when stop_gradient=True."""
    B, In_C, C_c2, C_p2, H, W = 2, 3, 64, 64, 24, 48
    x = torch.randn(B, In_C, H, W)

    # 1. Test isolated model (c2_stop_gradient=True)
    backbone_conv_isolated = nn.Conv2d(In_C, C_c2, kernel_size=3, padding=1, bias=False)
    relay_isolated = GradientDecoupledC2Relay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        gating_type="spatial_channel",
        c2_stop_gradient=True,
    )

    c2_feat = backbone_conv_isolated(x)
    dummy_p2 = torch.zeros(B, C_p2, H, W, requires_grad=False)
    p2_out = relay_isolated([c2_feat, dummy_p2])
    loss_isolated = p2_out.sum()
    loss_isolated.backward()

    # Shallow backbone conv weight gradient must be None (zero gradient flow from relay)
    assert backbone_conv_isolated.weight.grad is None

    # 2. Test leaky model (c2_stop_gradient=False)
    backbone_conv_leaky = nn.Conv2d(In_C, C_c2, kernel_size=3, padding=1, bias=False)
    relay_leaky = ScaleAwareFeatureRelay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        gating_type="spatial_channel",
        c2_stop_gradient=False,
    )

    c2_feat_leaky = backbone_conv_leaky(x)
    p2_out_leaky = relay_leaky([c2_feat_leaky, dummy_p2])
    loss_leaky = p2_out_leaky.sum()
    loss_leaky.backward()

    # Leaky backbone conv weight gradient is non-zero
    assert backbone_conv_leaky.weight.grad is not None
    assert backbone_conv_leaky.weight.grad.abs().sum().item() > 0.0


def test_v2_dual_gate_gradient_isolation():
    """Verify gradient isolation for ScaleAwareFeatureRelayV2 with dual-branch tiny saliency gate."""
    B, C_c2, C_p2, H, W = 2, 64, 64, 24, 48
    c2 = torch.randn(B, C_c2, H, W, requires_grad=True)
    p2 = torch.randn(B, C_p2, H, W, requires_grad=True)

    relay_v2 = ScaleAwareFeatureRelayV2(
        c2_channels=C_c2,
        p2_channels=C_p2,
        gating_type="dual_gate",
        c2_stop_gradient=True,
    )

    p2_refined = relay_v2([c2, p2])
    loss = p2_refined.sum()
    loss.backward()

    assert c2.grad is None or c2.grad.abs().sum().item() == 0.0
    assert p2.grad is not None
    assert relay_v2.gate_tiny[0].weight.grad is not None
    assert relay_v2.gate_normal[0].weight.grad is not None


@pytest.mark.parametrize("gating_type", ["spatial_channel", "spatial_only", "channel_only", "direct_sum"])
def test_all_gating_modes_forward_backward(gating_type: str):
    """Test all gating modes for GradientDecoupledC2Relay."""
    B, C_c2, C_p2, H, W = 2, 64, 64, 16, 32
    c2 = torch.randn(B, C_c2, H, W, requires_grad=True)
    p2 = torch.randn(B, C_p2, H, W, requires_grad=True)

    relay = GradientDecoupledC2Relay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        gating_type=gating_type,
        c2_stop_gradient=True,
    )

    out = relay([c2, p2])
    assert out.shape == (B, C_p2, H, W)
    loss = out.sum()
    loss.backward()
    assert c2.grad is None or c2.grad.abs().sum().item() == 0.0
    assert p2.grad is not None


def test_spatial_size_mismatch_interpolation():
    """Verify that spatial dimension mismatches between C2 and P2 are smoothly interpolated."""
    B, C_c2, C_p2 = 2, 64, 64
    c2 = torch.randn(B, C_c2, 48, 96)  # Larger spatial resolution
    p2 = torch.randn(B, C_p2, 24, 48)  # Target P2 resolution

    relay = GradientDecoupledC2Relay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        c2_stop_gradient=True,
    )

    out = relay([c2, p2])
    assert out.shape == (B, C_p2, 24, 48)


def test_fp16_and_eval_mode():
    """Verify FP16 precision and eval mode inference."""
    B, C_c2, C_p2, H, W = 1, 64, 64, 24, 48
    c2 = torch.randn(B, C_c2, H, W, dtype=torch.float16)
    p2 = torch.randn(B, C_p2, H, W, dtype=torch.float16)

    relay = GradientDecoupledC2Relay(
        c2_channels=C_c2,
        p2_channels=C_p2,
        c2_stop_gradient=True,
    ).half().eval()

    with torch.no_grad():
        out = relay([c2, p2])

    assert out.dtype == torch.float16
    assert out.shape == (B, C_p2, H, W)


def test_dataclass_configs():
    """Verify configuration dataclasses."""
    cfg = GradientDecoupledC2RelayConfig(
        enabled=True,
        gating_type="spatial_channel",
        c2_channels=64,
        p2_channels=64,
        c2_stop_gradient=True,
    )
    assert cfg.enabled is True
    assert cfg.c2_stop_gradient is True

    cfg_v2 = ScaleAwareRelayV2Config(
        enabled=True,
        c2_stop_gradient=True,
    )
    assert cfg_v2.c2_stop_gradient is True
