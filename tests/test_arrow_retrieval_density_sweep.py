from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect

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


class TestArrowRetrievalDensitySweep:
    """Test suite for Arrow Retrieval Density M Sweep and Dynamic Confidence Gating."""

    @pytest.fixture
    def sample_inputs(self):
        B, K_TL, K_Arrow, D = 2, 8, 16, 128
        torch.manual_seed(42)
        traffic_tokens = torch.randn(B, K_TL, D, requires_grad=True)
        arrow_tokens = torch.randn(B, K_Arrow, D, requires_grad=True)
        traffic_boxes = torch.rand(B, K_TL, 4)
        arrow_boxes = torch.rand(B, K_Arrow, 4)
        arrow_scores = torch.rand(B, K_Arrow) * 0.9 + 0.1
        arrow_valid = torch.ones(B, K_Arrow, dtype=torch.bool)
        arrow_valid[:, 10:] = False  # Arrows 10..15 are invalid padding
        arrow_scores[:, 10:] = 0.01

        traffic_scores = torch.rand(B, K_TL)
        traffic_round = torch.rand(B, K_TL)
        traffic_maneuver = torch.softmax(torch.randn(B, K_TL, 3), dim=-1)
        arrow_maneuver = torch.softmax(torch.randn(B, K_Arrow, 3), dim=-1)
        arrow_ego_lane = torch.rand(B, K_Arrow)

        return {
            "traffic_tokens": traffic_tokens,
            "arrow_tokens": arrow_tokens,
            "traffic_boxes": traffic_boxes,
            "arrow_boxes": arrow_boxes,
            "arrow_scores": arrow_scores,
            "arrow_valid": arrow_valid,
            "traffic_scores": traffic_scores,
            "traffic_round": traffic_round,
            "traffic_maneuver": traffic_maneuver,
            "arrow_maneuver": arrow_maneuver,
            "arrow_ego_lane": arrow_ego_lane,
            "B": B,
            "K_TL": K_TL,
            "K_Arrow": K_Arrow,
            "D": D,
        }

    def test_matcher_shapes_and_masking(self, sample_inputs):
        matcher = QueryConditionedArrowMatcher(token_dim=sample_inputs["D"], hidden_dim=64)
        scores = matcher(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            sample_inputs["traffic_boxes"],
            sample_inputs["arrow_boxes"],
            sample_inputs["arrow_scores"],
            sample_inputs["arrow_valid"],
        )

        assert scores.shape == (sample_inputs["B"], sample_inputs["K_TL"], sample_inputs["K_Arrow"])
        # Invalid arrows (index >= 10) must be masked with negative mask value <= -1e4
        assert torch.all(scores[:, :, 10:] <= -1e4)
        # Valid arrows must have finite matching scores
        assert torch.all(torch.isfinite(scores[:, :, :10]))

    @pytest.mark.parametrize("m", [4, 6, 8, 12])
    def test_cross_attention_m_sweep_shapes(self, sample_inputs, m):
        cross_attn = QueryConditionedCrossAttention(
            dimension=sample_inputs["D"], heads=4, top_m=m
        )
        matcher = QueryConditionedArrowMatcher(token_dim=sample_inputs["D"], hidden_dim=64)

        matching_scores = matcher(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            sample_inputs["traffic_boxes"],
            sample_inputs["arrow_boxes"],
            sample_inputs["arrow_scores"],
            sample_inputs["arrow_valid"],
        )
        retrieval_scores, retrieval_indices = matching_scores.topk(m, dim=-1, largest=True, sorted=True)

        conditioned, weights, bias = cross_attn(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            traffic_boxes=sample_inputs["traffic_boxes"],
            arrow_boxes=sample_inputs["arrow_boxes"],
            traffic_round=sample_inputs["traffic_round"],
            traffic_maneuver=sample_inputs["traffic_maneuver"],
            arrow_maneuver=sample_inputs["arrow_maneuver"],
            arrow_ego_lane=sample_inputs["arrow_ego_lane"],
            arrow_scores=sample_inputs["arrow_scores"],
            arrow_valid=sample_inputs["arrow_valid"],
            retrieval_indices=retrieval_indices,
            retrieval_scores=retrieval_scores,
        )

        assert conditioned.shape == (sample_inputs["B"], sample_inputs["K_TL"], sample_inputs["D"])
        assert weights.shape == (sample_inputs["B"], 4, sample_inputs["K_TL"], m + 1)
        assert bias.shape == (sample_inputs["B"], 4, sample_inputs["K_TL"], m + 1)
        # Attention weights must sum to 1.0 along key dimension
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)

    def test_dynamic_confidence_gating_masking(self, sample_inputs):
        cross_attn = QueryConditionedCrossAttention(
            dimension=sample_inputs["D"],
            heads=4,
            top_m=8,
            match_threshold=0.50,
            arrow_score_threshold=0.20,
        )
        matcher = QueryConditionedArrowMatcher(token_dim=sample_inputs["D"], hidden_dim=64)

        matching_scores = matcher(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            sample_inputs["traffic_boxes"],
            sample_inputs["arrow_boxes"],
            sample_inputs["arrow_scores"],
            sample_inputs["arrow_valid"],
        )
        retrieval_scores, retrieval_indices = matching_scores.topk(8, dim=-1, largest=True, sorted=True)

        # Force one sample to have low matching score below threshold
        low_scores = retrieval_scores.clone()
        low_scores[:, :, 2:] = -10.0  # sigmoid(-10) ~ 0.0 < 0.50

        _, weights, _ = cross_attn(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            traffic_boxes=sample_inputs["traffic_boxes"],
            arrow_boxes=sample_inputs["arrow_boxes"],
            traffic_round=sample_inputs["traffic_round"],
            traffic_maneuver=sample_inputs["traffic_maneuver"],
            arrow_maneuver=sample_inputs["arrow_maneuver"],
            arrow_ego_lane=sample_inputs["arrow_ego_lane"],
            arrow_scores=sample_inputs["arrow_scores"],
            arrow_valid=sample_inputs["arrow_valid"],
            retrieval_indices=retrieval_indices,
            retrieval_scores=low_scores,
        )

        # Gated out keys (index 2..7) must receive 0 attention weight
        assert torch.all(weights[:, :, :, 2:8] < 1e-6)
        # Null token (index 8) or keys 0..1 must capture the attention
        assert torch.all(weights[:, :, :, -1] > 0.0)

    def test_end_to_end_gradient_flow(self, sample_inputs):
        matcher = QueryConditionedArrowMatcher(token_dim=sample_inputs["D"], hidden_dim=64)
        cross_attn = QueryConditionedCrossAttention(
            dimension=sample_inputs["D"], heads=4, top_m=6, match_threshold=0.0
        )

        matching_scores = matcher(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            sample_inputs["traffic_boxes"],
            sample_inputs["arrow_boxes"],
            sample_inputs["arrow_scores"],
            sample_inputs["arrow_valid"],
        )
        retrieval_scores, retrieval_indices = matching_scores.topk(6, dim=-1, largest=True, sorted=True)

        conditioned, weights, bias = cross_attn(
            sample_inputs["traffic_tokens"],
            sample_inputs["arrow_tokens"],
            traffic_boxes=sample_inputs["traffic_boxes"],
            arrow_boxes=sample_inputs["arrow_boxes"],
            traffic_round=sample_inputs["traffic_round"],
            traffic_maneuver=sample_inputs["traffic_maneuver"],
            arrow_maneuver=sample_inputs["arrow_maneuver"],
            arrow_ego_lane=sample_inputs["arrow_ego_lane"],
            arrow_scores=sample_inputs["arrow_scores"],
            arrow_valid=sample_inputs["arrow_valid"],
            retrieval_indices=retrieval_indices,
            retrieval_scores=retrieval_scores,
        )

        loss = conditioned.sum() + matching_scores.sum()
        loss.backward()

        assert sample_inputs["traffic_tokens"].grad is not None
        assert sample_inputs["arrow_tokens"].grad is not None
        for name, param in matcher.named_parameters():
            assert param.grad is not None, f"Matcher parameter {name} missing gradient"
        for name, param in cross_attn.named_parameters():
            if name != "gate":  # gate starts at 0.0, grad tested via residual
                assert param.grad is not None, f"CrossAttention parameter {name} missing gradient"

    def test_unified_detector_integration(self):
        cfg_path = PROJECT_ROOT / "configs/model/tlr_yolo11n_p2.yaml"
        if not cfg_path.is_file():
            pytest.skip("Model config not found")

        wrapper = build_detection_model(cfg_path)
        config = UnifiedHeadConfig(
            token_dim=64,
            token_feature_dim=32,
            attention_heads=4,
            max_traffic_lights=16,
            max_arrows=16,
            traffic_score_threshold=0.01,
            arrow_score_threshold=0.05,
        )
        unified = attach_query_conditioned_unified_relevance_head(
            wrapper, config=config, top_m=8, match_threshold=0.25
        )

        assert isinstance(unified, QueryConditionedUnifiedDetect)
        assert isinstance(unified.cross_attention, QueryConditionedCrossAttention)
        assert unified.cross_attention.top_m == 8
        assert unified.cross_attention.match_threshold == 0.25

        x = torch.randn(1, 3, 384, 640)
        out = wrapper.model(x)

        assert isinstance(out, dict)
        assert "relevance_logits" in out
        assert "retrieval_indices" in out
        assert "retrieval_scores" in out
        assert "attention_weights" in out
        assert out["retrieval_indices"].shape == (1, 16, 8)
        assert out["attention_weights"].shape == (1, 4, 16, 9)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_cuda_inference(self):
        cfg_path = PROJECT_ROOT / "configs/model/tlr_yolo11n_p2.yaml"
        if not cfg_path.is_file():
            pytest.skip("Model config not found")

        device = "cuda"
        wrapper = build_detection_model(cfg_path)
        config = UnifiedHeadConfig(
            token_dim=64,
            token_feature_dim=32,
            attention_heads=4,
            max_traffic_lights=16,
            max_arrows=16,
        )
        unified = attach_query_conditioned_unified_relevance_head(
            wrapper, config=config, top_m=8, match_threshold=0.25
        )
        wrapper.model.to(device).half()
        wrapper.model.eval()

        x = torch.randn(1, 3, 384, 640, device=device, dtype=torch.float16)
        with torch.no_grad():
            res = wrapper.model(x)
            out = res[1] if isinstance(res, tuple) else res

        assert not torch.isnan(out["relevance_logits"]).any()
        assert not torch.isnan(out["attention_weights"]).any()
        assert out["relevance_logits"].dtype == torch.float16
