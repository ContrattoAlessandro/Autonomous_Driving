"""Unit tests for Ticket E25: Normalized Relative Geometry Encoding & Relation MLP."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.relation_geometry import (
    NormalizedRelativeGeometryEncoder,
    RelationMLP,
    RelationGeometryCrossAttention,
    RelationGeometryUnifiedDetect,
    compute_normalized_scene_ranks,
    attach_relation_geometry_unified_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig


def test_scene_ranks_computation():
    """Verify ordinal scene rank computation in [0, 1]."""
    values = torch.tensor([[10.0, 30.0, 20.0, 50.0]])
    ranks = compute_normalized_scene_ranks(values)
    # Order should be 0 (10), 2 (30), 1 (20), 3 (50) -> normalized by 3: [0.0, 2/3, 1/3, 1.0]
    expected = torch.tensor([[0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0]])
    assert torch.allclose(ranks, expected, atol=1e-5)


def test_relation_geometry_encoder_and_dropout():
    """Verify NormalizedRelativeGeometryEncoder produces expected 10-d feature vectors and respects dropout."""
    encoder = NormalizedRelativeGeometryEncoder(ego_x=0.5, p_drop=0.3)

    B, K_TL, K_Arrow = 2, 8, 16
    tl_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    tl_valid = torch.ones(B, K_TL, dtype=torch.bool)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    # Eval mode: no dropout
    encoder.eval()
    geom_eval = encoder(tl_boxes, arrow_boxes, tl_valid, arrow_valid)
    assert geom_eval.shape == (B, K_TL, K_Arrow, 10)
    assert not torch.isnan(geom_eval).any()

    # Train mode: dropout active
    encoder.train()
    geom_train = encoder(tl_boxes, arrow_boxes, tl_valid, arrow_valid)
    assert geom_train.shape == (B, K_TL, K_Arrow, 10)


def test_relation_mlp_and_cross_attention():
    """Verify RelationMLP and RelationGeometryCrossAttention forward execution and weights."""
    mlp = RelationMLP(in_features=10, hidden_dim=32, heads=4)
    feats = torch.randn(2, 8, 16, 10)
    bias = mlp(feats)
    assert bias.shape == (2, 4, 8, 16)

    attn = RelationGeometryCrossAttention(dimension=128, heads=4, p_drop=0.2)
    B, K_TL, K_Arrow, D, H = 2, 8, 16, 128, 4
    traffic_tokens = torch.randn(B, K_TL, D)
    arrow_tokens = torch.randn(B, K_Arrow, D)
    traffic_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    traffic_round = torch.rand(B, K_TL)
    traffic_maneuver = torch.rand(B, K_TL, 3)
    arrow_maneuver = torch.rand(B, K_Arrow, 3)
    arrow_ego_lane = torch.rand(B, K_Arrow)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    out, weights, full_bias = attn(
        traffic_tokens,
        arrow_tokens,
        traffic_boxes=traffic_boxes,
        arrow_boxes=arrow_boxes,
        traffic_round=traffic_round,
        traffic_maneuver=traffic_maneuver,
        arrow_maneuver=arrow_maneuver,
        arrow_ego_lane=arrow_ego_lane,
        arrow_valid=arrow_valid,
    )

    assert out.shape == (B, K_TL, D)
    assert weights.shape == (B, H, K_TL, K_Arrow + 1)
    assert full_bias.shape == (B, H, K_TL, K_Arrow + 1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


def test_relation_geometry_unified_detector_forward():
    """Verify forward pass of RelationGeometryUnifiedDetect in wrapper."""
    wrapper = build_detection_model()
    attach_relation_geometry_unified_relevance_head(
        wrapper, config=UnifiedHeadConfig(max_arrows=32), p_drop=0.2
    )

    dummy = torch.randn(1, 3, 384, 768)
    with torch.inference_mode():
        out = wrapper.model(dummy)
        assert out is not None
        if isinstance(out, tuple) and isinstance(out[0], tuple):
            pred_dict = out[0][-1]
            assert "relevance_logits" in pred_dict
            assert "attention_weights" in pred_dict
            assert "attention_geometry_bias" in pred_dict
