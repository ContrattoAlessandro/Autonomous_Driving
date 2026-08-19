"""Unit tests for Ticket E24: Query-Conditioned Road Arrow Selection (Top-M per TL Query)."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.arrow_retrieval import (
    QueryConditionedArrowMatcher,
    QueryConditionedCrossAttention,
    QueryConditionedUnifiedDetect,
    attach_query_conditioned_unified_relevance_head,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig


def test_arrow_matcher_shapes_and_masking():
    """Verify that QueryConditionedArrowMatcher scores pairs and masks invalid arrows."""
    matcher = QueryConditionedArrowMatcher(token_dim=128, hidden_dim=64)

    B, K_TL, K_Arrow, D = 2, 16, 32, 128
    traffic_tokens = torch.randn(B, K_TL, D)
    arrow_tokens = torch.randn(B, K_Arrow, D)
    traffic_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    arrow_scores = torch.rand(B, K_Arrow)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)
    arrow_valid[:, 20:] = False  # Mark arrows 20..31 invalid

    scores = matcher(
        traffic_tokens=traffic_tokens,
        arrow_tokens=arrow_tokens,
        traffic_boxes=traffic_boxes,
        arrow_boxes=arrow_boxes,
        arrow_scores=arrow_scores,
        arrow_valid=arrow_valid,
    )

    assert scores.shape == (B, K_TL, K_Arrow)
    assert not torch.isnan(scores).any()
    # Invalid arrows must have large negative scores
    assert (scores[:, :, 20:] <= -1e4).all()
    # Valid arrows must have finite scores
    assert (scores[:, :, :20] > -1e4).all()


def test_query_conditioned_cross_attention_shapes_and_null_fallback():
    """Verify cross-attention over Top-M retrieved arrows and exact null fallback."""
    top_m = 8
    attn = QueryConditionedCrossAttention(dimension=128, heads=4, top_m=top_m)

    B, K_TL, K_Arrow, D, H = 2, 16, 32, 128, 4
    traffic_tokens = torch.randn(B, K_TL, D)
    arrow_tokens = torch.randn(B, K_Arrow, D)
    traffic_boxes = torch.rand(B, K_TL, 4)
    arrow_boxes = torch.rand(B, K_Arrow, 4)
    traffic_round = torch.rand(B, K_TL)
    traffic_maneuver = torch.rand(B, K_TL, 3)
    arrow_maneuver = torch.rand(B, K_Arrow, 3)
    arrow_ego_lane = torch.rand(B, K_Arrow)
    arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)
    retrieval_indices = torch.randint(0, K_Arrow, (B, K_TL, top_m))

    out, weights, bias = attn(
        traffic_tokens,
        arrow_tokens,
        traffic_boxes=traffic_boxes,
        arrow_boxes=arrow_boxes,
        traffic_round=traffic_round,
        traffic_maneuver=traffic_maneuver,
        arrow_maneuver=arrow_maneuver,
        arrow_ego_lane=arrow_ego_lane,
        arrow_valid=arrow_valid,
        retrieval_indices=retrieval_indices,
    )

    assert out.shape == (B, K_TL, D)
    assert weights.shape == (B, H, K_TL, top_m + 1)
    assert bias.shape == (B, H, K_TL, top_m + 1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)

    # Test complete arrow-less fallback (all arrows invalid)
    arrow_valid_zero = torch.zeros(B, K_Arrow, dtype=torch.bool)
    out_zero, weights_zero, _ = attn(
        traffic_tokens,
        arrow_tokens,
        traffic_boxes=traffic_boxes,
        arrow_boxes=arrow_boxes,
        traffic_round=traffic_round,
        traffic_maneuver=traffic_maneuver,
        arrow_maneuver=arrow_maneuver,
        arrow_ego_lane=arrow_ego_lane,
        arrow_valid=arrow_valid_zero,
        retrieval_indices=retrieval_indices,
    )
    # When all arrows are invalid, attention must collapse 100% to NullToken (last index)
    assert torch.allclose(weights_zero[:, :, :, -1], torch.ones_like(weights_zero[:, :, :, -1]), atol=1e-4)


def test_query_conditioned_unified_detector_forward():
    """Verify forward pass of QueryConditionedUnifiedDetect within the detection model wrapper."""
    wrapper = build_detection_model()
    attach_query_conditioned_unified_relevance_head(
        wrapper, config=UnifiedHeadConfig(max_arrows=32), top_m=8
    )

    dummy = torch.randn(1, 3, 384, 768)
    with torch.inference_mode():
        out = wrapper.model(dummy)
        assert out is not None
        if isinstance(out, tuple) and isinstance(out[0], tuple):
            pred_dict = out[0][-1]
            assert "relevance_logits" in pred_dict
            assert "retrieval_indices" in pred_dict
            assert "arrow_matching_scores" in pred_dict
            ret_idx = pred_dict["retrieval_indices"]
            assert ret_idx.shape == (1, 32, 8)
            assert "attention_weights" in pred_dict
            attn_w = pred_dict["attention_weights"]
            assert attn_w.shape == (1, 4, 32, 9)  # 8 arrows + 1 null token
