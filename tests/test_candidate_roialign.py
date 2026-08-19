"""Unit tests for E28 Candidate-Centered Multi-Scale ROIAlign Attribute module."""

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

from tlr_yolo_mtl.model.roialign_attributes import (
    CandidateAttributeTower,
    CandidateMultiScaleROIAlign,
)


def test_candidate_multiscale_roialign_forward() -> None:
    B, K = 2, 32
    C_p2, C_p3 = 64, 128
    H_p2, W_p2 = 200, 400  # stride 4 for 800x1600
    H_p3, W_p3 = 100, 200  # stride 8 for 800x1600

    p2_feat = torch.randn(B, C_p2, H_p2, W_p2, requires_grad=True)
    p3_feat = torch.randn(B, C_p3, H_p3, W_p3, requires_grad=True)

    # Random candidate boxes in pixel coordinates [x1, y1, x2, y2]
    boxes = torch.zeros(B, K, 4)
    boxes[:, :, 0] = torch.rand(B, K) * 1500.0
    boxes[:, :, 1] = torch.rand(B, K) * 700.0
    boxes[:, :, 2] = boxes[:, :, 0] + torch.rand(B, K) * 40.0 + 4.0
    boxes[:, :, 3] = boxes[:, :, 1] + torch.rand(B, K) * 60.0 + 8.0

    extractor = CandidateMultiScaleROIAlign(
        channels_p2=C_p2,
        channels_p3=C_p3,
        roi_size=(3, 3),
        embed_dim=128,
    )

    tokens = extractor(p2_feat, p3_feat, boxes)
    assert tokens.shape == (B, K, 128)

    # Verify backprop to feature maps
    loss = tokens.sum()
    loss.backward()
    assert p2_feat.grad is not None
    assert p3_feat.grad is not None
    assert p2_feat.grad.abs().sum() > 0
    assert p3_feat.grad.abs().sum() > 0


def test_candidate_attribute_tower_forward() -> None:
    B, K, D = 2, 32, 128
    tokens = torch.randn(B, K, D, requires_grad=True)
    tower = CandidateAttributeTower(embed_dim=D)

    out = tower(tokens)
    assert out["state_logits"].shape == (B, K, 4)
    assert out["round_logits"].shape == (B, K)
    assert out["maneuver_logits"].shape == (B, K, 3)

    assert out["state_probs"].shape == (B, K, 4)
    assert out["round_probs"].shape == (B, K)
    assert out["maneuver_probs"].shape == (B, K, 3)

    # Check sum of probabilities for state
    state_sum = out["state_probs"].sum(dim=-1)
    assert torch.allclose(state_sum, torch.ones_like(state_sum), atol=1e-5)
