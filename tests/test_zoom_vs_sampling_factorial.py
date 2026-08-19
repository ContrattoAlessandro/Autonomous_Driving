"""Unit tests for E32 Context-Preserving Zoom vs Hard-Example Sampling 2x2 Factorial Ablation."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_e32_zoom_vs_sampling_factorial import (
    FactorialConditionMetrics,
    FactorialDecomposition,
    compute_factorial_decomposition,
)
from tlr_yolo_mtl.data.schema import (
    IgnoreRegion,
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.data.zoom_augmentation import (
    DifficultyBucketedSampler,
    compute_context_envelope,
    context_preserving_zoom,
    zoom_crop_record,
)
from tlr_yolo_mtl.training.data import (
    CanonicalMultiTaskDataset,
    prepare_training_sample,
)


def _make_sample_record(image_id: str = "img_001", has_tiny: bool = True, has_arrows: bool = True) -> ImageRecord:
    tls = [
        TrafficLightAnnotation(
            bbox_xyxy=(500.0, 300.0, 503.5, 308.0) if has_tiny else (500.0, 300.0, 540.0, 420.0),
            state="red",
            pictogram="straight",
            relevance=1,
            valid_state=True,
            valid_pictogram=True,
            valid_relevance=True,
        ),
        TrafficLightAnnotation(
            bbox_xyxy=(600.0, 310.0, 615.0, 345.0),
            state="green",
            pictogram="round",
            relevance=0,
            valid_state=True,
            valid_pictogram=True,
            valid_relevance=True,
        ),
    ]
    arrows = (
        [
            RoadArrowAnnotation(
                bbox_xyxy=(520.0, 600.0, 580.0, 720.0),
                direction_multihot=(0, 1, 0),
                segmentation_xy=((520.0, 600.0), (580.0, 600.0), (550.0, 720.0)),
            )
        ]
        if has_arrows
        else []
    )
    ignores = [
        IgnoreRegion(
            bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
            reason="ambiguous",
        )
    ]
    return ImageRecord(
        image_id=image_id,
        image_path="dummy.png",
        original_width=1600,
        original_height=800,
        source_dataset="DTLD",
        split="train",
        sequence_id="seq_01",
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_state=True,
            traffic_light_pictogram=True,
            traffic_light_relevance=True,
            arrow_detection=True,
        ),
        traffic_lights=tls,
        road_arrows=arrows,
        ignore_regions=ignores,
    )


def test_factorial_decomposition_mathematics() -> None:
    # Test strictly additive case: A=40, B=45 (zoom +5), C=43 (sampler +3), D=48 (combined +8)
    d_add = compute_factorial_decomposition("Additive Test", 40.0, 45.0, 43.0, 48.0)
    assert d_add.delta_zoom_isolated == 5.0
    assert d_add.delta_sampler_isolated == 3.0
    assert d_add.delta_combined_total == 8.0
    assert math.isclose(d_add.main_effect_zoom, 5.0, abs_tol=1e-4)
    assert math.isclose(d_add.main_effect_sampler, 3.0, abs_tol=1e-4)
    assert math.isclose(d_add.interaction_term, 0.0, abs_tol=1e-4)
    assert math.isclose(d_add.additivity_efficiency_pct, 100.0, abs_tol=1e-4)
    assert d_add.interaction_type == "strictly additive"

    # Test sub-additive saturation case (E32 empirical values on Sub-4px recall)
    # A=43.96, B=48.74 (+4.78), C=46.12 (+2.16), D=50.12 (+6.16)
    d_sub = compute_factorial_decomposition("Sub-4px Recall", 43.96, 48.74, 46.12, 50.12)
    assert math.isclose(d_sub.delta_zoom_isolated, 4.78, abs_tol=1e-2)
    assert math.isclose(d_sub.delta_sampler_isolated, 2.16, abs_tol=1e-2)
    assert math.isclose(d_sub.delta_combined_total, 6.16, abs_tol=1e-2)
    assert math.isclose(d_sub.interaction_term, -0.78, abs_tol=1e-2)
    assert d_sub.additivity_efficiency_pct > 85.0
    assert d_sub.zoom_attribution_share_pct > 65.0
    assert d_sub.sampler_attribution_share_pct > 25.0
    assert "sub-additive" in d_sub.interaction_type


def test_prepare_training_sample_context_zoom() -> None:
    rec = _make_sample_record("test_zoom", has_tiny=True, has_arrows=True)
    dummy_img = np.zeros((800, 1600, 3), dtype=np.uint8)

    # 1. With context_zoom=False
    sample_no_zoom = prepare_training_sample(
        dummy_img,
        rec,
        target_size=(800, 1600),
        training=True,
        horizontal_flip=False,
        context_zoom=False,
    )
    assert isinstance(sample_no_zoom["image"], torch.Tensor)
    assert sample_no_zoom["image"].shape == (3, 800, 1600)
    assert sample_no_zoom["bboxes"].shape[0] == 2

    # 2. With context_zoom=True, zoom_prob=1.0
    sample_zoom = prepare_training_sample(
        dummy_img,
        rec,
        target_size=(800, 1600),
        training=True,
        horizontal_flip=False,
        context_zoom=True,
        zoom_prob=1.0,
        rng=random.Random(42),
    )
    assert isinstance(sample_zoom["image"], torch.Tensor)
    assert sample_zoom["image"].shape == (3, 800, 1600)
    assert sample_zoom["bboxes"].shape[0] == 2
    # Check normalized bounding boxes are within [0, 1]
    assert (sample_zoom["bboxes"] >= 0.0).all() and (sample_zoom["bboxes"] <= 1.0).all()


def test_difficulty_bucketed_sampler_coverage() -> None:
    # Create records with mix of tiny, directional, and standard
    records = [
        _make_sample_record(f"tiny_{i}", has_tiny=True, has_arrows=True)
        for i in range(10)
    ] + [
        _make_sample_record(f"std_{i}", has_tiny=False, has_arrows=False)
        for i in range(10)
    ]
    sampler = DifficultyBucketedSampler(
        records,
        micro_batch_size=4,
        effective_batch_size=8,
        hard_weights=(0.50, 0.30, 0.20),
        seed=42,
    )

    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 4
        assert all(0 <= idx < len(records) for idx in batch)
