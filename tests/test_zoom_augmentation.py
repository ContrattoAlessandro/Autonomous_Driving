"""Unit tests for E27 Context-Preserving Zoom Augmentation and Bucketed Sampler."""

from __future__ import annotations

import random
import numpy as np
import pytest

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    TrafficLightAnnotation,
    RoadArrowAnnotation,
    IgnoreRegion,
    TaskValidity,
)
from tlr_yolo_mtl.data.zoom_augmentation import (
    compute_context_envelope,
    zoom_crop_record,
    context_preserving_zoom,
    DifficultyBucketedSampler,
)


def _make_dummy_record() -> ImageRecord:
    tls = [
        TrafficLightAnnotation(
            bbox_xyxy=(500.0, 300.0, 520.0, 350.0),
            state="red",
            pictogram="straight",
            relevance=1,
            valid_state=True,
            valid_pictogram=True,
            valid_relevance=True,
        ),
        TrafficLightAnnotation(
            bbox_xyxy=(600.0, 305.0, 615.0, 345.0),
            state="green",
            pictogram="round",
            relevance=0,
            valid_state=True,
            valid_pictogram=True,
            valid_relevance=True,
        ),
    ]
    arrows = [
        RoadArrowAnnotation(
            bbox_xyxy=(520.0, 600.0, 580.0, 720.0),
            direction_multihot=(0, 1, 0),
            segmentation_xy=((520.0, 600.0), (580.0, 600.0), (550.0, 720.0)),
        )
    ]
    ignores = [
        IgnoreRegion(
            bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
            reason="ambiguous",
        )
    ]
    return ImageRecord(
        image_id="test_001",
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


def test_compute_context_envelope() -> None:
    rec = _make_dummy_record()
    crop_box = compute_context_envelope(rec, margin_factor=1.5)
    x1, y1, x2, y2 = crop_box

    assert x1 >= 0 and y1 >= 0
    assert x2 <= 1600 and y2 <= 800
    assert x2 > x1 and y2 > y1

    # Ensure all TLs and arrows are contained in the crop box
    for tl in rec.traffic_lights:
        assert x1 <= tl.bbox_xyxy[0] and y1 <= tl.bbox_xyxy[1]
        assert x2 >= tl.bbox_xyxy[2] and y2 >= tl.bbox_xyxy[3]

    for ar in rec.road_arrows:
        assert x1 <= ar.bbox_xyxy[0] and y1 <= ar.bbox_xyxy[1]
        assert x2 >= ar.bbox_xyxy[2] and y2 >= ar.bbox_xyxy[3]


def test_zoom_crop_record() -> None:
    rec = _make_dummy_record()
    crop_box = (400, 200, 800, 800)  # w=400, h=600
    cropped = zoom_crop_record(rec, crop_box)

    assert cropped.original_width == 400
    assert cropped.original_height == 600
    assert len(cropped.traffic_lights) == 2
    assert len(cropped.road_arrows) == 1

    # Check that coordinates shifted by (-x1, -y1)
    tl0 = cropped.traffic_lights[0]
    assert tl0.bbox_xyxy == (100.0, 100.0, 120.0, 150.0)

    # Check segmentation shifted
    seg0 = cropped.road_arrows[0].segmentation_xy
    assert seg0[0] == (120.0, 400.0)


def test_context_preserving_zoom_full() -> None:
    rec = _make_dummy_record()
    dummy_img = np.zeros((800, 1600, 3), dtype=np.uint8)
    rng = random.Random(42)

    # Trigger zoom with probability 1.0
    cropped_img, cropped_rec = context_preserving_zoom(dummy_img, rec, zoom_prob=1.0, rng=rng)

    assert cropped_img.shape[0] == cropped_rec.original_height
    assert cropped_img.shape[1] == cropped_rec.original_width
    assert cropped_rec.original_width < 1600
    assert len(cropped_rec.traffic_lights) == 2


def test_difficulty_bucketed_sampler() -> None:
    records = [_make_dummy_record() for _ in range(20)]
    sampler = DifficultyBucketedSampler(records, micro_batch_size=4, effective_batch_size=8, seed=42)

    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 4
