"""Unit tests for Scale-Conditioned C2 -> P2 Feature Relay v2 (Ticket E66)."""

import pytest
import torch
from torch import nn

from tlr_yolo_mtl.model.neck import (
    ScaleAwareFeatureRelayV2,
    ScaleAwareRelayV2Config,
)


def test_relay_v2_initialization():
    relay = ScaleAwareFeatureRelayV2(c2_channels=64, p2_channels=64, gating_type="dual_gate")
    assert relay.c2_channels == 64
    assert relay.p2_channels == 64
    assert relay.gating_type == "dual_gate"
    assert relay.gate_normal is not None
    assert relay.gate_tiny is not None


def test_relay_v2_forward_shape():
    B, C, H, W = 2, 64, 120, 240
    c2 = torch.randn(B, C, H, W)
    p2 = torch.randn(B, C, H, W)

    relay = ScaleAwareFeatureRelayV2(c2_channels=C, p2_channels=C, gating_type="dual_gate")
    out = relay([c2, p2])

    assert out.shape == (B, C, H, W)
    assert not torch.isnan(out).any()


def test_relay_v2_backward_gradients():
    B, C, H, W = 2, 64, 32, 64
    c2 = torch.randn(B, C, H, W, requires_grad=True)
    p2 = torch.randn(B, C, H, W, requires_grad=True)

    relay = ScaleAwareFeatureRelayV2(c2_channels=C, p2_channels=C, gating_type="dual_gate")
    out = relay([c2, p2])
    loss = out.sum()
    loss.backward()

    assert c2.grad is not None
    assert p2.grad is not None
    assert not torch.isnan(c2.grad).any()
    assert not torch.isnan(p2.grad).any()


def test_relay_v2_tiny_impulse_preservation():
    """Verify that a sharp point impulse in C2 activates the tiny saliency gate."""
    B, C, H, W = 1, 64, 32, 32
    c2 = torch.zeros(B, C, H, W)
    p2 = torch.zeros(B, C, H, W)

    # Place a sharp point impulse at center (representing sub-4px traffic light)
    c2[:, :, 16, 16] = 5.0

    relay = ScaleAwareFeatureRelayV2(c2_channels=C, p2_channels=C, gating_type="dual_gate")
    
    # Check that gamma_tiny is receptive to point signal
    with torch.no_grad():
        c2_proj = relay.c2_proj(c2)
        gamma = relay.gate_tiny(c2_proj)
        assert gamma.shape == (B, 1, H, W)
        assert gamma[:, :, 16, 16] > 0.0
