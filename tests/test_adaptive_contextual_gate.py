"""Unit tests for Ticket E23: Per-Query Adaptive Contextual Gate (g_i Dynamic Residual Gating)."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.adaptive_gate import (
    AdaptiveContextualGate,
    AdaptiveGatedUnifiedDetect,
    attach_adaptive_gated_unified_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig


def test_adaptive_contextual_gate_shapes_and_bounds():
    """Verify that AdaptiveContextualGate computes valid [0, 1] bounded gates and valid telemetry."""
    gate_module = AdaptiveContextualGate(token_dim=128, hidden_dim=64)

    B, K_TL, K_Arrow, D, H = 2, 32, 16, 128, 4
    traffic_tokens = torch.randn(B, K_TL, D)
    traffic_round = torch.rand(B, K_TL)
    attention_weights = torch.softmax(torch.randn(B, H, K_TL, K_Arrow + 1), dim=-1)
    arrow_scores = torch.rand(B, K_Arrow)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)
    local_context_delta = torch.randn(B, 1, K_TL)
    conditioned_context_delta = torch.randn(B, 1, K_TL)

    gates, telemetry = gate_module(
        traffic_tokens=traffic_tokens,
        traffic_round=traffic_round,
        attention_weights=attention_weights,
        arrow_scores=arrow_scores,
        arrow_valid=arrow_valid,
        local_context_delta=local_context_delta,
        conditioned_context_delta=conditioned_context_delta,
        enforce_round_fallback=True,
    )

    assert gates.shape == (B, 1, K_TL)
    assert (gates >= 0.0).all() and (gates <= 1.0).all()
    assert not torch.isnan(gates).any()
    assert "mean_gate_overall" in telemetry
    assert "mean_gate_directional" in telemetry
    assert "mean_gate_round" in telemetry
    assert "mean_entropy" in telemetry
    assert "mean_null_mass" in telemetry


def test_round_fallback_guarantee():
    """Verify that pure round signals (P(round) = 1.0) strictly collapse contextual gating to 0.0."""
    gate_module = AdaptiveContextualGate(token_dim=128, hidden_dim=64)

    B, K_TL, K_Arrow, D, H = 1, 4, 8, 128, 4
    traffic_tokens = torch.randn(B, K_TL, D)
    # Traffic lights: 2 pure round (1.0), 2 pure directional (0.0)
    traffic_round = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    attention_weights = torch.softmax(torch.randn(B, H, K_TL, K_Arrow + 1), dim=-1)
    arrow_scores = torch.rand(B, K_Arrow)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)
    local_context_delta = torch.randn(B, 1, K_TL)
    conditioned_context_delta = torch.randn(B, 1, K_TL)

    gates, _ = gate_module(
        traffic_tokens=traffic_tokens,
        traffic_round=traffic_round,
        attention_weights=attention_weights,
        arrow_scores=arrow_scores,
        arrow_valid=arrow_valid,
        local_context_delta=local_context_delta,
        conditioned_context_delta=conditioned_context_delta,
        enforce_round_fallback=True,
    )

    # Pure round lights must have gate exactly 0.0
    assert torch.allclose(gates[0, 0, :2], torch.zeros(2), atol=1e-6)
    # Directional lights must have positive gates in (0, 1]
    assert (gates[0, 0, 2:] > 0.0).all()


def test_adaptive_gated_unified_detector_forward():
    """Verify forward pass of AdaptiveGatedUnifiedDetect within the detection wrapper."""
    wrapper = build_detection_model()
    attach_adaptive_gated_unified_relevance_head(
        wrapper, config=UnifiedHeadConfig(), enforce_round_fallback=True
    )

    dummy = torch.randn(1, 3, 384, 768)
    with torch.inference_mode():
        out = wrapper.model(dummy)
        assert out is not None
        if isinstance(out, tuple) and isinstance(out[0], tuple):
            pred_dict = out[0][-1]
            assert "relevance_logits" in pred_dict
            assert "adaptive_gates" in pred_dict
            gates = pred_dict["adaptive_gates"]
            assert (gates >= 0.0).all() and (gates <= 1.0).all()
