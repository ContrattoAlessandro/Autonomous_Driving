"""Unit and integration tests for Ticket E51: Scale-Aware C2 -> P2 Feature Relay."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.neck import (
    ScaleAwareFeatureRelay,
    ScaleAwareRelayConfig,
)


def test_scale_aware_feature_relay_forward_backward():
    """Verify forward shape, gating modes, and gradient backpropagation."""
    B, H, W = 2, 30, 60
    C_c2, C_p2 = 64, 64

    gating_types = ["spatial_channel", "spatial_only", "channel_only", "direct_sum"]

    for gating in gating_types:
        relay = ScaleAwareFeatureRelay(
            c2_channels=C_c2,
            p2_channels=C_p2,
            gating_type=gating,
            hidden_ratio=0.5,
            residual_scale=1.0,
        )

        c2 = torch.randn(B, C_c2, H, W, requires_grad=True)
        p2 = torch.randn(B, C_p2, H, W, requires_grad=True)

        out = relay([c2, p2])
        assert out.shape == (B, C_p2, H, W), f"Failed shape for gating {gating}"

        loss = out.sum()
        loss.backward()

        assert c2.grad is not None, f"c2.grad is None for {gating}"
        assert p2.grad is not None, f"p2.grad is None for {gating}"
        assert not torch.isnan(c2.grad).any(), f"NaN in c2.grad for {gating}"
        assert not torch.isnan(p2.grad).any(), f"NaN in p2.grad for {gating}"


def test_scale_aware_feature_relay_flexible_inputs():
    """Verify flexible input formats (tuple, list, positional args, spatial interpolation)."""
    B, H, W = 2, 32, 32
    C_c2, C_p2 = 64, 64

    relay = ScaleAwareFeatureRelay(c2_channels=C_c2, p2_channels=C_p2, gating_type="spatial_channel")

    c2 = torch.randn(B, C_c2, H, W)
    p2 = torch.randn(B, C_p2, H, W)

    # Calling as list [c2, p2]
    out1 = relay([c2, p2])
    assert out1.shape == (B, C_p2, H, W)

    # Calling with positional args relay(c2, p2)
    out2 = relay(c2, p2)
    assert out2.shape == (B, C_p2, H, W)

    # Calling with different spatial dimensions (c2 is 16x16, p2 is 32x32)
    c2_small = torch.randn(B, C_c2, 16, 16)
    out3 = relay([c2_small, p2])
    assert out3.shape == (B, C_p2, H, W)


def test_parameter_footprint_constraint():
    """Verify parameter count remains strictly <= 0.08M (Criterion 3: <= 0.10M)."""
    # YOLO11s configuration: C2=64, P2=64
    relay_s = ScaleAwareFeatureRelay(
        c2_channels=64,
        p2_channels=64,
        gating_type="spatial_channel",
        hidden_ratio=0.5,
    )
    total_params_s = sum(p.numel() for p in relay_s.parameters())
    assert total_params_s <= 80_000, f"YOLO11s relay has {total_params_s} params (> 80k)"

    # YOLO11l configuration: C2=128, P2=128
    relay_l = ScaleAwareFeatureRelay(
        c2_channels=128,
        p2_channels=128,
        gating_type="spatial_channel",
        hidden_ratio=0.5,
    )
    total_params_l = sum(p.numel() for p in relay_l.parameters())
    assert total_params_l <= 100_000, f"YOLO11l relay has {total_params_l} params (> 100k)"


def test_relay_model_building_and_forward():
    """Verify model instantiation from YAML and forward smoke test."""
    cfg_path = PROJECT_ROOT / "configs" / "model" / "tlr_yolo11s_p2_relay.yaml"
    assert cfg_path.exists(), f"Model config {cfg_path} not found"

    wrapper = build_detection_model(cfg_path)
    wrapper.model.eval()

    # Forward smoke pass with small resolution
    x_small = torch.randn(1, 3, 192, 320)
    with torch.no_grad():
        out_small = wrapper.model(x_small)
    assert out_small is not None

    # Check that ScaleAwareFeatureRelay is present in the architecture
    has_relay = any(isinstance(m, ScaleAwareFeatureRelay) for m in wrapper.model.modules())
    assert has_relay, "ScaleAwareFeatureRelay not found in instantiated model modules"


def test_amp_numerical_stability():
    """Verify mixed-precision numerical stability."""
    relay = ScaleAwareFeatureRelay(c2_channels=64, p2_channels=64, gating_type="spatial_channel")
    c2 = torch.randn(2, 64, 24, 48, dtype=torch.float32)
    p2 = torch.randn(2, 64, 24, 48, dtype=torch.float32)

    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        out = relay([c2, p2])

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
