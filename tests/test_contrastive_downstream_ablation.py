"""Unit tests for Ticket E35: TL <-> Road Arrow Contrastive Downstream Relevance Ablation."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.model.arrow_retrieval import (
    attach_query_conditioned_unified_relevance_head,
)
from tlr_yolo_mtl.training.losses import (
    MultiTaskLossWeights,
    TLRMultiTaskCriterion,
    TLRMultiTaskLossResult,
)
from tlr_yolo_mtl.training.contrastive_loss import (
    TLArrowContrastiveLoss,
    TLArrowContrastiveProjector,
)


def test_multitask_loss_weights_contrastive_field():
    """Verify MultiTaskLossWeights includes contrastive field and default value."""
    weights = MultiTaskLossWeights()
    assert hasattr(weights, "contrastive")
    assert weights.contrastive == 0.0

    custom_weights = MultiTaskLossWeights(contrastive=0.10)
    assert custom_weights.contrastive == 0.10


def test_unified_detect_exposes_candidate_tokens_and_maneuvers():
    """Verify UnifiedDetect._build_tokens returns candidate tokens and maneuvers."""
    wrapper = build_detection_model()
    attach_unified_relevance_head(
        wrapper,
        config=UnifiedHeadConfig(max_traffic_lights=8, max_arrows=16),
    )
    model = wrapper.model.eval()

    dummy_img = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        decoded, raw = model(dummy_img)

    assert "traffic_tokens" in raw
    assert "arrow_tokens" in raw
    assert "traffic_candidate_maneuver" in raw
    assert "arrow_candidate_maneuver" in raw
    assert "traffic_candidate_round" in raw

    assert raw["traffic_tokens"].shape == (1, 8, 128)
    assert raw["arrow_tokens"].shape == (1, 16, 128)


def test_query_conditioned_detect_exposes_candidate_tokens():
    """Verify QueryConditionedUnifiedDetect._build_tokens returns candidate tokens."""
    wrapper = build_detection_model()
    attach_query_conditioned_unified_relevance_head(
        wrapper,
        config=UnifiedHeadConfig(max_traffic_lights=8, max_arrows=16),
        top_m=8,
    )
    model = wrapper.model.eval()

    dummy_img = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        decoded, raw = model(dummy_img)

    assert "traffic_tokens" in raw
    assert "arrow_tokens" in raw
    assert "traffic_candidate_maneuver" in raw
    assert "arrow_candidate_maneuver" in raw
    assert raw["traffic_tokens"].shape == (1, 8, 128)


def test_tlr_multitask_criterion_contrastive_loss_integration():
    """Verify TLRMultiTaskCriterion computes and logs contrastive loss when weight > 0."""
    wrapper = build_detection_model()
    attach_unified_relevance_head(
        wrapper,
        config=UnifiedHeadConfig(max_traffic_lights=4, max_arrows=8),
    )
    model = wrapper.model.train()

    criterion = TLRMultiTaskCriterion(
        model,
        weights=MultiTaskLossWeights(contrastive=0.10),
    )

    B = 2
    dummy_img = torch.randn(B, 3, 256, 256)
    raw = model(dummy_img)

    # Create dummy batch with valid fields
    dummy_batch = {
        "unified_detection_valid": torch.ones(B, dtype=torch.bool),
        "traffic_relevance_valid": torch.ones(B, dtype=torch.bool),
        "object_batch_idx": torch.tensor([0, 0, 1, 1]),
        "object_cls": torch.tensor([0, 1, 0, 1]),
        "object_bboxes": torch.tensor([
            [10.0, 10.0, 30.0, 50.0],
            [40.0, 40.0, 80.0, 90.0],
            [15.0, 15.0, 35.0, 55.0],
            [50.0, 50.0, 90.0, 100.0],
        ]),
        "object_state": torch.tensor([0, 0, 1, 0]),
        "object_round": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "object_maneuver": torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        "object_relevance": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "object_ego_lane": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }

    result = criterion(raw, dummy_batch)

    assert isinstance(result, TLRMultiTaskLossResult)
    assert result.total > 0.0
    assert "contrastive_loss" in result.metrics
    assert "contrastive_margin" in result.metrics
    assert result.contrastive is not None


def test_contrastive_loss_zero_weight_passthrough():
    """Verify that lambda_contrastive = 0.0 adds zero contrastive loss and avoids overhead."""
    wrapper = build_detection_model()
    attach_unified_relevance_head(
        wrapper,
        config=UnifiedHeadConfig(max_traffic_lights=4, max_arrows=8),
    )
    model = wrapper.model.train()

    criterion_zero = TLRMultiTaskCriterion(
        model,
        weights=MultiTaskLossWeights(contrastive=0.0),
    )

    B = 2
    dummy_img = torch.randn(B, 3, 256, 256)
    raw = model(dummy_img)

    dummy_batch = {
        "unified_detection_valid": torch.ones(B, dtype=torch.bool),
        "traffic_relevance_valid": torch.ones(B, dtype=torch.bool),
        "object_batch_idx": torch.tensor([0, 1]),
        "object_cls": torch.tensor([0, 0]),
        "object_bboxes": torch.tensor([
            [10.0, 10.0, 30.0, 50.0],
            [15.0, 15.0, 35.0, 55.0],
        ]),
        "object_state": torch.tensor([0, 1]),
        "object_round": torch.tensor([1.0, 0.0]),
        "object_maneuver": torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        "object_relevance": torch.tensor([1.0, 1.0]),
        "object_ego_lane": torch.tensor([1.0, 1.0]),
    }

    result = criterion_zero(raw, dummy_batch)
    assert float(result.contrastive.item()) == 0.0
    assert float(result.metrics["contrastive_loss"].item()) == 0.0
