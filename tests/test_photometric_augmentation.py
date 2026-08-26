"""Unit and integration tests for Ticket E39: Physics-Grounded Photometric Traffic Light Augmentation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.data.photometric_augmentation import (
    DEFAULT_PHOTOMETRIC_CONFIG,
    EMISSIVE_SPECTRA,
    PhotometricAugmentationConfig,
    apply_exposure_and_gamma,
    apply_physics_photometric_augmentation,
    apply_sensor_noise_and_defocus,
    apply_wet_lens_glare,
    estimate_lamp_center,
    synthesize_lamp_bloom,
)
from tlr_yolo_mtl.data.schema import (
    BBox,
    IgnoreRegion,
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.training.data import prepare_training_sample


def _create_sample_record(
    image_id: str = "rec_e39_001",
    img_w: int = 1600,
    img_h: int = 800,
    tl_boxes: list[BBox] | None = None,
    states: list[str] | None = None,
) -> ImageRecord:
    if tl_boxes is None:
        tl_boxes = [
            (500.0, 200.0, 510.0, 230.0),  # Vertical 10x30, red
            (700.0, 200.0, 710.0, 230.0),  # Vertical 10x30, yellow
            (900.0, 200.0, 910.0, 230.0),  # Vertical 10x30, green
            (1100.0, 200.0, 1110.0, 230.0),  # Vertical 10x30, off
        ]
    if states is None:
        states = ["red", "yellow", "green", "off"]

    tls = [
        TrafficLightAnnotation(
            bbox_xyxy=box,
            state=states[i % len(states)],
            pictogram="straight",
            relevance=1 if i == 0 else 0,
            valid_state=True,
            valid_pictogram=True,
            valid_relevance=True,
            round_target=0,
            maneuver_multihot=(0, 1, 0),
            valid_round=True,
            valid_maneuver=True,
        )
        for i, box in enumerate(tl_boxes)
    ]

    return ImageRecord(
        image_id=image_id,
        image_path=f"dummy_{image_id}.png",
        original_width=img_w,
        original_height=img_h,
        source_dataset="DTLD",
        split="train",
        sequence_id="seq_01",
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_state=True,
            traffic_light_pictogram=True,
            traffic_light_relevance=True,
            arrow_detection=True,
            traffic_light_round=True,
            traffic_light_maneuver=True,
            arrow_ego_lane=True,
        ),
        traffic_lights=tls,
        road_arrows=[],
        ignore_regions=[],
    )


def test_estimate_lamp_center_vertical():
    box: BBox = (100.0, 200.0, 110.0, 230.0)  # w=10, h=30
    cx_red, cy_red = estimate_lamp_center(box, "red")
    cx_yellow, cy_yellow = estimate_lamp_center(box, "yellow")
    cx_green, cy_green = estimate_lamp_center(box, "green")

    assert cx_red == pytest.approx(105.0)
    assert cx_yellow == pytest.approx(105.0)
    assert cx_green == pytest.approx(105.0)

    # Red is in top aspect (~20% from top -> 200 + 6 = 206)
    assert cy_red == pytest.approx(206.0)
    # Yellow is in middle aspect (~50% from top -> 200 + 15 = 215)
    assert cy_yellow == pytest.approx(215.0)
    # Green is in bottom aspect (~80% from top -> 200 + 24 = 224)
    assert cy_green == pytest.approx(224.0)


def test_estimate_lamp_center_square_or_horizontal():
    box: BBox = (100.0, 200.0, 130.0, 210.0)  # horizontal w=30, h=10
    cx, cy = estimate_lamp_center(box, "red")
    assert cx == pytest.approx(115.0)
    assert cy == pytest.approx(205.0)


def test_synthesize_lamp_bloom_emissive_vs_off():
    img = np.full((100, 100, 3), 40, dtype=np.uint8)
    box: BBox = (40.0, 30.0, 50.0, 60.0)

    # Red bloom
    bloomed_red = synthesize_lamp_bloom(img, box, "red", intensity=0.8, radius_scale=1.0)
    assert not np.array_equal(bloomed_red, img)
    # Red channel around lamp center (45, 36) should increase significantly
    assert bloomed_red[36, 45, 0] > img[36, 45, 0] + 50
    # Green channel should stay much lower than red channel at bloom center
    assert bloomed_red[36, 45, 0] > bloomed_red[36, 45, 1]

    # Off state should produce NO bloom
    bloomed_off = synthesize_lamp_bloom(img, box, "off", intensity=0.8)
    assert np.array_equal(bloomed_off, img)


def test_synthesize_lamp_bloom_boundary_safety():
    img = np.full((50, 50, 3), 50, dtype=np.uint8)
    # Box partially outside top-left boundary
    box: BBox = (-10.0, -10.0, 10.0, 20.0)
    bloomed = synthesize_lamp_bloom(img, box, "green", intensity=0.5)
    assert bloomed.shape == (50, 50, 3)
    assert bloomed.dtype == np.uint8


def test_apply_exposure_and_gamma():
    img = np.full((50, 50, 3), 100, dtype=np.uint8)

    # Gamma < 1.0 increases brightness
    bright = apply_exposure_and_gamma(img, gamma=0.8, exposure_scale=1.0)
    assert np.mean(bright) > np.mean(img)

    # Gamma > 1.0 decreases brightness
    dark = apply_exposure_and_gamma(img, gamma=1.3, exposure_scale=1.0)
    assert np.mean(dark) < np.mean(img)

    # Highlight clipping on near-white regions
    img_bright = np.full((50, 50, 3), 240, dtype=np.uint8)
    clipped = apply_exposure_and_gamma(img_bright, gamma=1.0, exposure_scale=1.0, clip_highlights=True)
    assert np.mean(clipped) >= 240


def test_apply_sensor_noise_and_defocus():
    img = np.full((60, 60, 3), 128, dtype=np.uint8)
    rng = random.Random(42)

    noisy = apply_sensor_noise_and_defocus(
        img, apply_noise=True, noise_sigma=5.0, apply_defocus=False, rng=rng
    )
    assert noisy.shape == img.shape
    assert noisy.dtype == np.uint8
    assert np.std(noisy.astype(np.float32)) > 0.5

    blurred = apply_sensor_noise_and_defocus(
        noisy, apply_noise=False, apply_defocus=True, kernel_size=3, rng=rng
    )
    assert blurred.shape == img.shape
    # Blurring should reduce high-frequency standard deviation
    assert np.std(blurred.astype(np.float32)) <= np.std(noisy.astype(np.float32))


def test_apply_wet_lens_glare():
    img = np.full((100, 100, 3), 30, dtype=np.uint8)
    tls = [
        TrafficLightAnnotation(
            bbox_xyxy=(45.0, 30.0, 55.0, 60.0),
            state="red",
            valid_state=True,
        )
    ]
    rng = random.Random(123)
    glare_img = apply_wet_lens_glare(img, tls, glare_intensity=0.5, rng=rng)
    assert not np.array_equal(glare_img, img)
    assert np.max(glare_img) > 30


def test_strict_hue_preservation_constraint():
    """Verify that photometric augmentation never distorts canonical chromatic hue."""
    record = _create_sample_record()
    img = np.full((800, 1600, 3), 100, dtype=np.uint8)

    cfg = PhotometricAugmentationConfig(
        photometric_prob=1.0,
        enable_lamp_bloom=True,
        lamp_bloom_prob=1.0,
        max_hue_jitter=0.004,  # strict hue constraint
    )

    rng = random.Random(42)
    aug_img = apply_physics_photometric_augmentation(img, record, config=cfg, rng=rng)

    # For each traffic light, check that hue at the bloom center belongs strictly to correct color
    for tl in record.traffic_lights:
        state = tl.state
        if state not in EMISSIVE_SPECTRA:
            continue
        cx, cy = estimate_lamp_center(tl.bbox_xyxy, state)
        pixel_bgr = aug_img[int(cy), int(cx)]
        hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_RGB2HSV)[0, 0]
        hue = float(hsv[0])  # in [0, 180]

        if state == "red":
            # Red hue is in [0, 10] or [170, 180]
            assert (hue <= 15.0 or hue >= 165.0), f"Red lamp hue corrupted: {hue}"
        elif state == "yellow":
            # Yellow hue is in [15, 38]
            assert (15.0 <= hue <= 45.0), f"Yellow lamp hue corrupted: {hue}"
        elif state == "green":
            # Green hue is in [45, 95]
            assert (45.0 <= hue <= 95.0), f"Green lamp hue corrupted: {hue}"


def test_prepare_training_sample_integration():
    record = _create_sample_record()
    img = np.full((800, 1600, 3), 120, dtype=np.uint8)
    rng = random.Random(42)

    sample = prepare_training_sample(
        img,
        record,
        target_size=(800, 1600),
        training=True,
        photometric_suite_enabled=True,
        rng=rng,
    )

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 800, 1600)
    assert sample["bboxes"].shape[0] == len(record.traffic_lights)
    assert sample["tl_state"].shape[0] == len(record.traffic_lights)


def test_eval_mode_zero_photometric_overhead():
    """In evaluation / inference mode (training=False), output tensor is deterministic and unaugmented."""
    record = _create_sample_record()
    img = np.full((800, 1600, 3), 120, dtype=np.uint8)

    sample1 = prepare_training_sample(
        img, record, target_size=(800, 1600), training=False, photometric_suite_enabled=True
    )
    sample2 = prepare_training_sample(
        img, record, target_size=(800, 1600), training=False, photometric_suite_enabled=True
    )

    assert torch.equal(sample1["image"], sample2["image"])
