"""Unit and integration tests for Ticket E38: Scale-Matched Augmentation & Paired Copy-Paste."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    TrafficLightAnnotation,
    RoadArrowAnnotation,
    IgnoreRegion,
    TaskValidity,
)
from tlr_yolo_mtl.data.scale_matched_augmentation import (
    BIN_SUB_8PX,
    BIN_8_TO_16PX,
    BIN_GT_16PX,
    classify_box_scale_bin,
    get_record_scale_stats,
    compute_scale_matched_envelope,
    scale_matched_zoom,
    paired_copy_paste,
)
from tlr_yolo_mtl.training.data import prepare_training_sample


def _create_sample_record(
    image_id: str = "rec_001",
    img_w: int = 1600,
    img_h: int = 800,
    tl_boxes: list[tuple[float, float, float, float]] | None = None,
    paired_arrow: bool = True,
) -> ImageRecord:
    if tl_boxes is None:
        tl_boxes = [
            (500.0, 250.0, 506.0, 265.0),   # 6x15 -> sub-8px
            (700.0, 240.0, 712.0, 270.0),   # 12x30 -> 8-16px
            (900.0, 200.0, 925.0, 260.0),   # 25x60 -> gt-16px
        ]

    tls = [
        TrafficLightAnnotation(
            bbox_xyxy=box,
            state="red" if i == 0 else "green",
            pictogram="straight" if i == 0 else "round",
            relevance=1 if i == 0 else 0,
            valid_state=True,
            valid_pictogram=True,
            valid_relevance=True,
            round_target=0 if i == 0 else 1,
            maneuver_multihot=(0, 1, 0) if i == 0 else None,
            valid_round=True,
            valid_maneuver=True if i == 0 else False,
        )
        for i, box in enumerate(tl_boxes)
    ]

    arrows = []
    if paired_arrow:
        arrows.append(
            RoadArrowAnnotation(
                bbox_xyxy=(490.0, 550.0, 530.0, 680.0),
                direction_multihot=(0, 1, 0),
                segmentation_xy=((490.0, 550.0), (530.0, 550.0), (510.0, 680.0)),
                is_ego_lane=1,
                valid_ego_lane=True,
            )
        )

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
        road_arrows=arrows,
        ignore_regions=[],
    )


def test_classify_box_scale_bin() -> None:
    # Sub-8px side or small area
    assert classify_box_scale_bin((100.0, 100.0, 106.0, 114.0)) == BIN_SUB_8PX  # w=6, h=14 -> min_side=6
    assert classify_box_scale_bin((100.0, 100.0, 107.0, 108.0)) == BIN_SUB_8PX  # area=56 < 64

    # 8-16px side
    assert classify_box_scale_bin((100.0, 100.0, 110.0, 125.0)) == BIN_8_TO_16PX  # w=10, h=25 -> min_side=10

    # >16px side
    assert classify_box_scale_bin((100.0, 100.0, 120.0, 150.0)) == BIN_GT_16PX  # w=20, h=50 -> min_side=20


def test_get_record_scale_stats() -> None:
    rec = _create_sample_record()
    stats = get_record_scale_stats(rec)

    assert stats["num_tls"] == 3
    assert stats["num_sub_8px"] == 1
    assert stats["num_8_to_16px"] == 1
    assert stats["num_gt_16px"] == 1
    assert stats["min_side"] == pytest.approx(6.0)


def test_compute_scale_matched_envelope() -> None:
    rec = _create_sample_record()
    rng = random.Random(42)

    for target_bin in [BIN_SUB_8PX, BIN_8_TO_16PX, BIN_GT_16PX]:
        x1, y1, x2, y2 = compute_scale_matched_envelope(rec, target_bin=target_bin, rng=rng)
        assert x1 >= 0 and y1 >= 0
        assert x2 <= rec.original_width and y2 <= rec.original_height
        assert x2 > x1 and y2 > y1
        # Check aspect ratio roughly 2.0
        w = x2 - x1
        h = y2 - y1
        aspect = w / h
        assert 1.7 <= aspect <= 2.3


def test_scale_matched_zoom_execution() -> None:
    rec = _create_sample_record()
    dummy_img = np.full((rec.original_height, rec.original_width, 3), 128, dtype=np.uint8)
    rng = random.Random(123)

    cropped_img, cropped_rec = scale_matched_zoom(
        dummy_img,
        rec,
        zoom_prob=1.0,
        scale_quotas=(0.50, 0.30, 0.20),
        rng=rng,
    )

    assert cropped_img.shape[0] == cropped_rec.original_height
    assert cropped_img.shape[1] == cropped_rec.original_width
    assert cropped_rec.original_width <= rec.original_width
    assert len(cropped_rec.traffic_lights) > 0
    # Canonical validation check
    cropped_rec.validate()


def test_paired_copy_paste_execution() -> None:
    dest_rec = _create_sample_record("dest_001", tl_boxes=[(300.0, 200.0, 315.0, 235.0)], paired_arrow=False)
    donor_rec = _create_sample_record("donor_001", tl_boxes=[(500.0, 250.0, 508.0, 270.0)], paired_arrow=True)

    dest_img = np.full((dest_rec.original_height, dest_rec.original_width, 3), 100, dtype=np.uint8)
    donor_img = np.full((donor_rec.original_height, donor_rec.original_width, 3), 150, dtype=np.uint8)

    rng = random.Random(42)

    pasted_img, pasted_rec = paired_copy_paste(
        dest_img,
        dest_rec,
        donor_img,
        donor_rec,
        copy_paste_prob=1.0,
        blend_strength=0.95,
        rng=rng,
    )

    assert pasted_img.shape == dest_img.shape
    assert len(pasted_rec.traffic_lights) == len(dest_rec.traffic_lights) + 1
    # Check that arrow was also pasted
    assert len(pasted_rec.road_arrows) == 1
    # Check metadata tracking
    assert pasted_rec.metadata.get("paired_copy_paste") is True
    assert pasted_rec.metadata.get("donor_image_id") == "donor_001"
    # Canonical validation
    pasted_rec.validate()


def test_prepare_training_sample_integration() -> None:
    rec = _create_sample_record("main_rec")
    donor_rec = _create_sample_record("donor_rec")
    img = np.full((rec.original_height, rec.original_width, 3), 120, dtype=np.uint8)
    donor_img = np.full((donor_rec.original_height, donor_rec.original_width, 3), 140, dtype=np.uint8)

    rng = random.Random(999)

    sample = prepare_training_sample(
        img,
        rec,
        target_size=(800, 1600),
        training=True,
        scale_matched_zoom_enabled=True,
        scale_quotas=(0.40, 0.35, 0.25),
        paired_copy_paste_enabled=True,
        donor_image_rgb=donor_img,
        donor_record=donor_rec,
        copy_paste_prob=1.0,
        rng=rng,
    )

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 800, 1600)
    assert "bboxes" in sample
    assert "object_bboxes" in sample
    assert "object_state" in sample
    assert "object_round" in sample
    assert "object_maneuver" in sample
    assert "object_relevance" in sample
    assert sample["bboxes"].shape[1] == 4


def test_compute_kl_divergence_and_report() -> None:
    from scripts.audit_e38_scale_matched_paired_augmentation import (
        ScaleMatchedConditionMetrics,
        compute_kl_divergence,
        format_e38_markdown_report,
    )

    p = [0.4, 0.35, 0.25]
    q = [0.4, 0.35, 0.25]
    assert compute_kl_divergence(p, q) == pytest.approx(0.0, abs=1e-5)

    p2 = [0.5, 0.3, 0.2]
    assert compute_kl_divergence(p2, q) > 0.0

    dummy_cond = ScaleMatchedConditionMetrics(
        condition_id="test",
        condition_name="test_cond",
        has_scale_matched_zoom=True,
        has_paired_copy_paste=True,
        pct_sub_8px=40.0,
        pct_8_to_16px=35.0,
        pct_gt_16px=25.0,
        kl_divergence_to_target=0.01,
        ap_tl_sub8px=33.0,
        ap_tl_8_16px=68.0,
        ap_tl_16_32px=87.0,
        ap_tl_gt32px=94.5,
        ap_tl_50=72.8,
        ap_arrow_50=96.1,
        map50=84.5,
        map50_95=60.6,
        recall_tl_sub8px=54.8,
        recall_tl_8_16px=81.6,
        recall_tl_16_32px=92.6,
        recall_tl_gt32px=98.1,
        relevance_auprc=0.918,
        relevance_f1=0.864,
        relevant_red_recall_tau50=87.8,
        relevant_red_recall_tau95=96.8,
        state_accuracy=94.3,
        state_macro_f1=0.844,
        round_f1=0.892,
        latency_ms=26.81,
        fps=37.3,
    )

    report = format_e38_markdown_report(dummy_cond, dummy_cond, dummy_cond, {})
    assert "E38 Diagnostic Audit" in report
    assert "Criterion 1" in report
    assert "PASSED" in report

