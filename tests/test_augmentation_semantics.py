"""Unit tests verifying semantic invariance and label consistency under data augmentations (W4)."""

from __future__ import annotations

import random
import numpy as np
from tlr_yolo_mtl.data.geometry import horizontal_flip_box
from tlr_yolo_mtl.data.schema import (
    RoadArrowAnnotation,
    BBox,
    IgnoreRegion,
    ImageRecord,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.data.taxonomy import (
    flip_direction_multihot,
    flip_pictogram,
)
from tlr_yolo_mtl.data.transforms import horizontal_flip_record
from tlr_yolo_mtl.training.data import _photometric_augment
from tlr_yolo_mtl.training.engine import load_training_config


def test_horizontal_flip_box_coordinates() -> None:
    """Verify box coordinate transformation under horizontal flip."""
    box: BBox = (100.0, 50.0, 250.0, 300.0)
    image_width = 1600
    flipped = horizontal_flip_box(box, image_width)
    assert flipped == (1350.0, 50.0, 1500.0, 300.0)
    # Double flip returns identity
    re_flipped = horizontal_flip_box(flipped, image_width)
    assert all(abs(a - b) < 1e-5 for a, b in zip(re_flipped, box))


def test_flip_direction_multihot_inversion() -> None:
    """Verify [left, straight, right] -> [right, straight, left]."""
    assert flip_direction_multihot((1, 0, 0)) == (0, 0, 1)  # Left -> Right
    assert flip_direction_multihot((0, 0, 1)) == (1, 0, 0)  # Right -> Left
    assert flip_direction_multihot((0, 1, 0)) == (0, 1, 0)  # Straight -> Straight
    assert flip_direction_multihot((1, 1, 0)) == (0, 1, 1)  # Straight-Left -> Straight-Right
    assert flip_direction_multihot((0, 1, 1)) == (1, 1, 0)  # Straight-Right -> Straight-Left
    assert flip_direction_multihot((1, 0, 1)) == (1, 0, 1)  # Left-Right -> Left-Right


def test_flip_pictogram_inversion() -> None:
    """Verify pictogram string inversion for compound and atomic arrows."""
    assert flip_pictogram("left") == "right"
    assert flip_pictogram("right") == "left"
    assert flip_pictogram("straight_left") == "straight_right"
    assert flip_pictogram("straight_right") == "straight_left"
    assert flip_pictogram("round") == "round"
    assert flip_pictogram("straight") == "straight"
    assert flip_pictogram(None) is None


def test_horizontal_flip_record_full_invariance() -> None:
    """Verify atomic flip on ImageRecord with TL, Arrow, and Ignore regions."""
    record = ImageRecord(
        image_id="test_001",
        image_path="test.png",
        source_dataset="DTLD",
        original_width=1600,
        original_height=800,
        split="train",
        sequence_id="seq_01",
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_state=True,
            traffic_light_pictogram=True,
            traffic_light_relevance=True,
            arrow_detection=True,
        ),
        traffic_lights=[
            TrafficLightAnnotation(
                bbox_xyxy=(100.0, 100.0, 150.0, 200.0),
                state="red",
                valid_state=True,
                round_target=0,
                valid_round=True,
                pictogram="left",
                valid_pictogram=True,
                maneuver_multihot=(1, 0, 0),
                valid_maneuver=True,
                relevance=1,
                valid_relevance=True,
            ),
        ],
        road_arrows=[
            RoadArrowAnnotation(
                bbox_xyxy=(500.0, 600.0, 700.0, 750.0),
                direction_multihot=(1, 1, 0),
                segmentation_xy=((500.0, 600.0), (700.0, 750.0)),
            ),
        ],
        ignore_regions=[
            IgnoreRegion(bbox_xyxy=(0.0, 0.0, 50.0, 50.0), reason="reflection"),
        ],
    )

    flipped = horizontal_flip_record(record)

    # 1. TL checks
    tl = flipped.traffic_lights[0]
    assert tl.bbox_xyxy == (1450.0, 100.0, 1500.0, 200.0)
    assert tl.state == "red"  # State invariant
    assert tl.round_target == 0
    assert tl.pictogram == "right"
    assert tl.maneuver_multihot == (0, 0, 1)  # Left -> Right
    assert tl.relevance == 1

    # 2. Arrow checks
    arr = flipped.road_arrows[0]
    assert arr.bbox_xyxy == (900.0, 600.0, 1100.0, 750.0)
    assert arr.direction_multihot == (0, 1, 1)  # Straight-Left -> Straight-Right
    assert arr.segmentation_xy == ((1100.0, 600.0), (900.0, 750.0))

    # 3. Ignore region checks
    ign = flipped.ignore_regions[0]
    assert ign.bbox_xyxy == (1550.0, 0.0, 1600.0, 50.0)


def test_photometric_augmentation_color_preservation() -> None:
    """Verify conservative photometric augmentations preserve color polarity."""
    rng = random.Random(42)
    # Synthetic red, yellow, green pixel swatches
    img_red = np.zeros((64, 64, 3), dtype=np.uint8)
    img_red[..., 0] = 240  # High R, low G, low B

    img_green = np.zeros((64, 64, 3), dtype=np.uint8)
    img_green[..., 1] = 240  # High G, low R, low B

    for seed in range(50):
        rng = random.Random(seed)
        aug_red = _photometric_augment(img_red, rng)
        # Red channel should dominate over green and blue
        assert aug_red[..., 0].mean() > aug_red[..., 1].mean() + 50
        assert aug_red[..., 0].mean() > aug_red[..., 2].mean() + 50

        rng = random.Random(seed)
        aug_green = _photometric_augment(img_green, rng)
        # Green channel should dominate over red and blue
        assert aug_green[..., 1].mean() > aug_green[..., 0].mean() + 50
        assert aug_green[..., 1].mean() > aug_green[..., 2].mean() + 50


def test_contextual_augmentations_disabled() -> None:
    """Confirm active config has horizontal flip, mosaic, mixup isolation."""
    config = load_training_config("configs/tlr_yolov8s_train.yaml")
    # Mosaic/Mixup should not be present or must be disabled
    assert config.get("mosaic", 0.0) == 0.0
    assert config.get("mixup", 0.0) == 0.0
