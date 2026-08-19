"""Unit tests for E16: Capacity-Matched Local+ Baseline and Causal Decomposition."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import pytest
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect

from tlr_yolo_mtl.model.local_plus import (
    LocalPlusRelevanceBranch,
    LocalPlusResidualBlock,
    LocalPlusTrafficControlDetect,
    attach_local_plus_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
)



def test_local_plus_parameter_parity():
    """Verify Local+ MLP branch parameter count strictly matches cross-attention context branch."""
    # 1. Build unified model with cross-attention
    dummy_detect = Detect(nc=2, ch=(64, 128, 256))
    dummy_detect.stride = torch.tensor([8.0, 16.0, 32.0])
    cfg = UnifiedHeadConfig(token_dim=128, token_feature_dim=64, attention_heads=4)
    unified_head = UnifiedTrafficControlDetect(dummy_detect, config=cfg)
    ctx_params = sum(p.numel() for p in unified_head.context_parameters())

    # 2. Build Local+ branch
    local_plus_branch = LocalPlusRelevanceBranch(
        token_feature_dim=64,
        position_dim=32,
        hidden_dim=128,
        head_hidden_dim=96,
        num_blocks=3,
    )
    lp_params = local_plus_branch.count_parameters()

    # Parameter counts:
    # Cross-Attention: 127,655 params
    # Local+ MLP:     127,618 params
    assert ctx_params == 127655, f"Unexpected cross-attention context params: {ctx_params}"
    assert lp_params["total"] == 127618, f"Unexpected Local+ params: {lp_params['total']}"

    rel_diff = abs(lp_params["total"] - ctx_params) / ctx_params
    assert rel_diff < 0.001, f"Parameter difference exceeds 0.1%: {rel_diff:.4%}"


def test_residual_block_forward_and_gradient():
    """Test LocalPlusResidualBlock forward, residual addition and gradient backward."""
    block = LocalPlusResidualBlock(dim=128)
    x = torch.randn(4, 32, 128, requires_grad=True)
    out = block(x)
    assert out.shape == (4, 32, 128)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_local_plus_branch_forward_shapes():
    """Test LocalPlusRelevanceBranch forward output shape and zero-gate behavior."""
    branch = LocalPlusRelevanceBranch(
        token_feature_dim=64,
        position_dim=32,
        hidden_dim=128,
        head_hidden_dim=96,
        num_blocks=3,
    )

    b, k_tl = 2, 32
    feats = torch.randn(b, k_tl, 64)
    boxes = torch.rand(b, k_tl, 4)
    rounds = torch.rand(b, k_tl)
    mans = torch.rand(b, k_tl, 3)
    scores = torch.rand(b, k_tl)

    # Initial zero gate
    delta_zero = branch(
        traffic_features=feats,
        traffic_boxes=boxes,
        traffic_round=rounds,
        traffic_maneuver=mans,
        traffic_scores=scores,
        use_gate=True,
    )
    assert delta_zero.shape == (b, 1, k_tl)
    assert torch.allclose(delta_zero, torch.zeros_like(delta_zero)), "Zero gate should return exact zero delta"

    # Non-zero gate or use_gate=False
    delta_raw = branch(
        traffic_features=feats,
        traffic_boxes=boxes,
        traffic_round=rounds,
        traffic_maneuver=mans,
        traffic_scores=scores,
        use_gate=False,
    )
    assert delta_raw.shape == (b, 1, k_tl)
    assert not torch.allclose(delta_raw, torch.zeros_like(delta_raw)), "Raw delta should be non-zero"


def test_local_plus_detector_forward_eval():
    """Test LocalPlusTrafficControlDetect forward pass in eval mode."""
    dummy_detect = Detect(nc=2, ch=(32, 64, 128))
    dummy_detect.stride = torch.tensor([8.0, 16.0, 32.0])
    cfg = UnifiedHeadConfig(token_dim=128, token_feature_dim=64, attention_heads=4)
    head = LocalPlusTrafficControlDetect(dummy_detect, config=cfg).eval()

    features = [
        torch.zeros((1, 32, 40, 80)),
        torch.zeros((1, 64, 20, 40)),
        torch.zeros((1, 128, 10, 20)),
    ]
    with torch.no_grad():
        decoded, raw = head(features)

    assert "relevance_logits" in raw
    assert "local_relevance_logits" in raw
    assert "local_plus_delta" in raw
    assert raw["relevance_logits"].shape == (1, 1, cfg.max_traffic_lights)
    assert raw["local_relevance_logits"].shape == (1, 1, cfg.max_traffic_lights)


def test_attach_local_plus_relevance_head():
    """Test attaching LocalPlusTrafficControlDetect to a model wrapper."""
    wrapper = build_detection_model("configs/model/tlr_yolo11n.yaml")
    attach_local_plus_relevance_head(wrapper)
    head = wrapper.model.model[-1]
    assert isinstance(head, LocalPlusTrafficControlDetect)
    assert hasattr(wrapper.model, "stride")
    assert len(head.local_plus_parameters()) > 0



def test_local_plus_gradient_flow():
    """Test gradient propagation through Local+ Residual MLP branch."""
    dummy_detect = Detect(nc=2, ch=(32, 64, 128))
    dummy_detect.stride = torch.tensor([8.0, 16.0, 32.0])
    head = LocalPlusTrafficControlDetect(dummy_detect).train()
    head.local_plus_branch.gate.data.fill_(1.0)

    features = [
        torch.randn((2, 32, 20, 20), requires_grad=True),
        torch.randn((2, 64, 10, 10), requires_grad=True),
        torch.randn((2, 128, 5, 5), requires_grad=True),
    ]
    output = head(features)
    assert isinstance(output, dict)
    rel_logits = output["relevance_logits"]
    loss = rel_logits.sum()
    loss.backward()

    # Check Local+ branch gradients
    for name, param in head.local_plus_branch.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
        assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} has non-finite gradients"
