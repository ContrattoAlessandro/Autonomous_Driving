"""Unit tests for Counterfactual Hard-Negative Mining and Balanced Sampling (Ticket E43)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.data.counterfactual_sampling import (
    DEFAULT_COUNTERFACTUAL_CONFIG,
    CounterfactualMiningConfig,
    CounterfactualPairType,
    CounterfactualRelevancePair,
    CounterfactualRelevanceSampler,
    encode_counterfactual_relevance_targets,
    mine_scene_counterfactual_pairs,
)
from tlr_yolo_mtl.data.schema import (
    BBox,
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.model.relevance import assigned_relevance_focal_bce
from tlr_yolo_mtl.training.data import (
    CanonicalMultiTaskDataset,
    canonical_multitask_collate,
    prepare_training_sample,
)


def _create_synthetic_intersection_record(
    num_pos_tls: int = 1,
    num_distractor_tls: int = 2,
    num_arrows: int = 2,
    mast_arm_neighbors: bool = True,
) -> ImageRecord:
    """Create a synthetic multi-object intersection record with known ground-truth geometry."""
    traffic_lights: list[TrafficLightAnnotation] = []

    # Positive TL (ego lane, straight, y=200, x=800)
    for i in range(num_pos_tls):
        traffic_lights.append(
            TrafficLightAnnotation(
                bbox_xyxy=(780.0 + i * 20.0, 180.0, 810.0 + i * 20.0, 240.0),
                state="green",
                pictogram="circle",
                relevance=1,
                valid_relevance=True,
            )
        )

    # Distractor TLs
    for j in range(num_distractor_tls):
        if mast_arm_neighbors and j == 0 and num_pos_tls > 0:
            # Mounted on the same mast-arm (dx = 60px, dy = 5px)
            box = (850.0, 185.0, 880.0, 245.0)
            pict = "left"
        else:
            # Farther away or distinct lane
            box = (400.0 + j * 150.0, 250.0, 430.0 + j * 150.0, 310.0)
            pict = "right" if j % 2 == 0 else "circle"

        traffic_lights.append(
            TrafficLightAnnotation(
                bbox_xyxy=box,
                state="red",
                pictogram=pict,
                relevance=0,
                valid_relevance=True,
            )
        )

    # Road Arrows
    road_arrows: list[RoadArrowAnnotation] = []
    directions_multihot = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]  # straight, left, right
    for k in range(num_arrows):
        road_arrows.append(
            RoadArrowAnnotation(
                bbox_xyxy=(700.0 + k * 80.0, 600.0, 760.0 + k * 80.0, 720.0),
                direction_multihot=directions_multihot[k % len(directions_multihot)],
            )
        )


    return ImageRecord(
        image_id="synthetic_e43_test",
        sequence_id="seq_01",
        source_dataset="DTLD",
        split="train",
        image_path="synthetic.jpg",
        original_height=800,
        original_width=1600,
        traffic_lights=traffic_lights,
        road_arrows=road_arrows,
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_relevance=True,
            arrow_detection=True,
        ),
    )



def test_counterfactual_mining_config_validation():
    """Test configuration validation and ratio constraints."""
    cfg = CounterfactualMiningConfig()
    cfg.validate()
    assert cfg.target_pos_ratio == 0.40
    assert cfg.target_easy_neg_ratio == 0.30
    assert cfg.target_cross_lane_hard_ratio == 0.15
    assert cfg.target_spatial_neighbor_hard_ratio == 0.15

    # Invalid sum of ratios
    with pytest.raises(ValueError, match="target ratios must sum to 1.0"):
        CounterfactualMiningConfig(target_pos_ratio=0.5, target_easy_neg_ratio=0.9).validate()

    # Invalid spatial threshold
    with pytest.raises(ValueError, match="spatial thresholds must be positive"):
        CounterfactualMiningConfig(spatial_dx_threshold_px=-10.0).validate()


def test_taxonomy_pair_mining():
    """Verify that all four taxonomy categories are correctly identified."""
    record = _create_synthetic_intersection_record(
        num_pos_tls=1, num_distractor_tls=2, num_arrows=2, mast_arm_neighbors=True
    )
    pairs = mine_scene_counterfactual_pairs(record, DEFAULT_COUNTERFACTUAL_CONFIG)

    pair_types = {p.pair_type for p in pairs}
    assert CounterfactualPairType.POSITIVE in pair_types
    assert CounterfactualPairType.EASY_NEGATIVE in pair_types
    assert (
        CounterfactualPairType.CROSS_LANE_CONFUSER in pair_types
        or CounterfactualPairType.OPPOSING_MANEUVER_CONFUSER in pair_types
    )
    assert CounterfactualPairType.SPATIAL_NEIGHBOR_CONFUSER in pair_types

    # Check that spatial neighbor confuser identifies anchor positive TL
    spatial_pairs = [p for p in pairs if p.pair_type == CounterfactualPairType.SPATIAL_NEIGHBOR_CONFUSER]
    assert len(spatial_pairs) >= 1
    assert spatial_pairs[0].relevance_label == 0
    assert spatial_pairs[0].weight >= 1.2
    assert spatial_pairs[0].metadata["same_mast_arm"] is True


def test_balanced_sampler_distribution_and_fallbacks():
    """Verify that CounterfactualRelevanceSampler enforces quotas and handles edge cases gracefully."""
    sampler = CounterfactualRelevanceSampler(DEFAULT_COUNTERFACTUAL_CONFIG)

    # 1. Normal scene with all categories
    record = _create_synthetic_intersection_record(num_pos_tls=2, num_distractor_tls=3, num_arrows=3)
    sampled = sampler.sample_pairs(record, max_pairs=16)
    assert len(sampled) <= 16
    pos_count = sum(1 for p in sampled if p.pair_type == CounterfactualPairType.POSITIVE)
    neg_count = sum(1 for p in sampled if p.relevance_label == 0)
    assert pos_count > 0
    assert neg_count > 0

    # 2. Scene without arrows (fallback to spatial neighbors + easy negatives)
    no_arrow_record = _create_synthetic_intersection_record(num_pos_tls=1, num_distractor_tls=2, num_arrows=0)
    sampled_no_arrows = sampler.sample_pairs(no_arrow_record, max_pairs=8)
    assert len(sampled_no_arrows) > 0
    assert any(p.pair_type == CounterfactualPairType.POSITIVE for p in sampled_no_arrows)

    # 3. Scene with only distractors (no positive lights)
    only_distractor_record = _create_synthetic_intersection_record(num_pos_tls=0, num_distractor_tls=3, num_arrows=2)
    sampled_distractors = sampler.sample_pairs(only_distractor_record, max_pairs=8)
    assert len(sampled_distractors) > 0
    assert all(p.relevance_label == 0 for p in sampled_distractors)


def test_encode_counterfactual_relevance_targets():
    """Test tensor target encoding for multi-task loss consumption."""
    record = _create_synthetic_intersection_record(num_pos_tls=1, num_distractor_tls=2, num_arrows=2)
    targets = encode_counterfactual_relevance_targets(record, DEFAULT_COUNTERFACTUAL_CONFIG)

    assert "counterfactual_weights" in targets
    assert "counterfactual_confuser_mask" in targets
    assert "is_hard_negative" in targets

    weights = targets["counterfactual_weights"]
    mask = targets["counterfactual_confuser_mask"]
    assert weights.shape == (3,)  # 1 pos + 2 distractors
    assert mask.shape == (3,)
    assert weights.dtype == torch.float32
    assert mask.dtype == torch.bool
    # At least one distractor was mined as a confuser
    assert mask.any()


def test_assigned_relevance_focal_bce_with_weights():
    """Test that focal BCE seamlessly supports instance weighting without shape errors."""
    batch_size = 2
    anchors = 10
    num_gt = 4

    logits = torch.randn(batch_size, 1, anchors, requires_grad=True)
    padded_targets = torch.randint(0, 2, (batch_size, num_gt)).long()
    foreground_mask = torch.ones((batch_size, anchors), dtype=torch.bool)
    target_gt_indices = torch.randint(0, num_gt, (batch_size, anchors)).long()
    instance_weights = torch.tensor([[1.0, 1.5, 1.2, 0.8], [1.0, 1.0, 1.5, 1.2]], dtype=torch.float32)

    loss_unweighted, count1 = assigned_relevance_focal_bce(
        logits, padded_targets, foreground_mask, target_gt_indices
    )
    loss_weighted, count2 = assigned_relevance_focal_bce(
        logits, padded_targets, foreground_mask, target_gt_indices, instance_weights=instance_weights
    )

    assert count1 == count2
    assert loss_weighted.ndim == 0
    assert loss_weighted.item() > 0.0

    # Gradient flow check
    loss_weighted.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_prepare_training_sample_and_collate_integration():
    """Verify integration of counterfactual fields in prepare_training_sample and collator."""
    record = _create_synthetic_intersection_record(num_pos_tls=1, num_distractor_tls=2, num_arrows=2)
    fake_img = (torch.rand((800, 1600, 3)).numpy() * 255).astype("uint8")

    sample = prepare_training_sample(
        fake_img,
        record,
        training=True,
        counterfactual_mining_enabled=True,
        counterfactual_config=DEFAULT_COUNTERFACTUAL_CONFIG,
    )

    assert "counterfactual_weights" in sample
    assert "counterfactual_confuser_mask" in sample
    assert "is_hard_negative" in sample
    assert isinstance(sample["counterfactual_weights"], torch.Tensor)

    # Test collate
    batch = canonical_multitask_collate([sample, sample])
    assert "counterfactual_weights" in batch
    assert "counterfactual_confuser_mask" in batch
    assert "is_hard_negative" in batch
    assert batch["counterfactual_weights"].shape[0] == 6  # 3 TLs * 2 samples
