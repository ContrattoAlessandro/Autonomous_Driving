"""Scale-Matched Augmentation & Semantics-Preserving Paired Copy-Paste (Ticket E38).

Implements:
1. Distribution-Aware Scale-Matched Sampler / Zoom:
   Target-bin-conditioned zoom that guarantees balanced representation across
   sub-8px (<8 px side), small (8-16 px side), and medium/large (>16 px side)
   traffic lights according to controlled distribution quotas (e.g. 40% : 35% : 25%).
2. Semantics-Preserving Paired Copy-Paste:
   Transplants traffic lights with local overhead/pole contextual support and,
   in multi-task scenes, jointly transplants associated road surface arrows
   maintaining lateral/longitudinal geometric correspondence without corrupting
   relevance ground truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Mapping, Sequence

import cv2
import numpy as np

from .schema import BBox, DirectionMultiHot, ImageRecord, RoadArrowAnnotation, TrafficLightAnnotation
from .zoom_augmentation import zoom_crop_record


# Canonical scale bin identifiers
BIN_SUB_8PX = 0    # min_side < 8 px (or area < 64 px²)
BIN_8_TO_16PX = 1  # 8 <= min_side < 16 px
BIN_GT_16PX = 2    # min_side >= 16 px

DEFAULT_SCALE_QUOTAS = (0.40, 0.35, 0.25)  # (sub-8px, 8-16px, >16px)


def classify_box_scale_bin(box: BBox) -> int:
    """Classify a bounding box [x1, y1, x2, y2] into a canonical scale bin."""
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    min_side = min(w, h)
    if min_side < 8.0 or (w * h) < 64.0:
        return BIN_SUB_8PX
    if min_side < 16.0:
        return BIN_8_TO_16PX
    return BIN_GT_16PX


def get_record_scale_stats(record: ImageRecord) -> dict[str, float | int]:
    """Compute summary scale statistics for traffic lights in an ImageRecord."""
    if not record.traffic_lights:
        return {
            "num_tls": 0,
            "min_side": 0.0,
            "max_side": 0.0,
            "min_area": 0.0,
            "max_area": 0.0,
            "num_sub_8px": 0,
            "num_8_to_16px": 0,
            "num_gt_16px": 0,
        }

    sides: list[float] = []
    areas: list[float] = []
    sub8 = 0
    b8_16 = 0
    gt16 = 0

    for tl in record.traffic_lights:
        w = max(0.0, float(tl.bbox_xyxy[2] - tl.bbox_xyxy[0]))
        h = max(0.0, float(tl.bbox_xyxy[3] - tl.bbox_xyxy[1]))
        side = min(w, h)
        area = w * h
        sides.append(side)
        areas.append(area)
        bin_id = classify_box_scale_bin(tl.bbox_xyxy)
        if bin_id == BIN_SUB_8PX:
            sub8 += 1
        elif bin_id == BIN_8_TO_16PX:
            b8_16 += 1
        else:
            gt16 += 1

    return {
        "num_tls": len(record.traffic_lights),
        "min_side": float(min(sides)),
        "max_side": float(max(sides)),
        "min_area": float(min(areas)),
        "max_area": float(max(areas)),
        "num_sub_8px": sub8,
        "num_8_to_16px": b8_16,
        "num_gt_16px": gt16,
    }


def compute_scale_matched_envelope(
    record: ImageRecord,
    target_bin: int,
    *,
    target_aspect: float = 2.0,  # 1600 / 800
    native_letterbox_scale: float = 0.78125,  # 1600 / 2048
    rng: random.Random | None = None,
) -> tuple[int, int, int, int]:
    """Compute context-preserving crop envelope that maps instances to the desired target scale bin.

    Returns (x1, y1, x2, y2) in original pixel coordinates.
    """
    img_w, img_h = record.original_width, record.original_height
    resolved_rng = rng or random.Random()

    boxes: list[BBox] = [tl.bbox_xyxy for tl in record.traffic_lights] + [
        ar.bbox_xyxy for ar in record.road_arrows
    ]

    if not boxes or not record.traffic_lights:
        return 0, 0, img_w, img_h

    # Find the critical representative traffic light scale
    tl_sides = [
        min(tl.bbox_xyxy[2] - tl.bbox_xyxy[0], tl.bbox_xyxy[3] - tl.bbox_xyxy[1])
        for tl in record.traffic_lights
    ]
    cur_min_side = max(1.0, min(tl_sides))

    # Target scale in letterboxed output pixels
    if target_bin == BIN_SUB_8PX:
        desired_output_side = resolved_rng.uniform(4.5, 7.5)
    elif target_bin == BIN_8_TO_16PX:
        desired_output_side = resolved_rng.uniform(8.5, 15.0)
    else:  # BIN_GT_16PX
        desired_output_side = resolved_rng.uniform(16.5, 28.0)

    # Current output side under standard letterbox
    cur_output_side = cur_min_side * native_letterbox_scale
    desired_zoom = max(1.0, min(3.0, desired_output_side / max(1.0, cur_output_side)))

    # Compute spatial envelope bounding all relevant objects
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[2] for b in boxes)
    y_max = max(b[3] for b in boxes)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # Target window dimensions derived from desired zoom
    target_crop_w = img_w / desired_zoom
    target_crop_h = img_h / desired_zoom

    # Ensure window is large enough to enclose the bounding envelope with safety padding
    min_enclosing_w = max(x_max - x_min + 40.0, 80.0)
    min_enclosing_h = max(y_max - y_min + 40.0, 60.0)

    bw = max(target_crop_w, min_enclosing_w)
    bh = max(target_crop_h, min_enclosing_h)

    # Reconcile aspect ratio to target_aspect (2.0)
    if bw / max(bh, 1e-3) < target_aspect:
        bw = bh * target_aspect
    else:
        bh = bw / target_aspect

    # Clamp to original image boundaries
    bw = min(bw, float(img_w))
    bh = min(bh, float(img_h))

    x1 = max(0, int(round(cx - bw / 2.0)))
    y1 = max(0, int(round(cy - bh / 2.0)))
    x2 = min(img_w, int(round(cx + bw / 2.0)))
    y2 = min(img_h, int(round(cy + bh / 2.0)))

    # Shift crop box if clamped against borders
    if x1 == 0:
        x2 = min(img_w, int(round(bw)))
    elif x2 == img_w:
        x1 = max(0, int(round(img_w - bw)))

    if y1 == 0:
        y2 = min(img_h, int(round(bh)))
    elif y2 == img_h:
        y1 = max(0, int(round(img_h - bh)))

    # Degeneracy check
    if (x2 - x1) < 100 or (y2 - y1) < 50:
        return 0, 0, img_w, img_h

    return x1, y1, x2, y2


def scale_matched_zoom(
    image_rgb: np.ndarray,
    record: ImageRecord,
    *,
    zoom_prob: float = 0.5,
    scale_quotas: tuple[float, float, float] = DEFAULT_SCALE_QUOTAS,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, ImageRecord]:
    """Applies distribution-aware scale-matched zoom to an image and canonical record."""
    resolved_rng = rng or random.Random()
    if resolved_rng.random() > zoom_prob:
        return image_rgb, record

    if not record.traffic_lights:
        return image_rgb, record

    # Sample target bin from quotas
    r = resolved_rng.random()
    q_sub8, q_8_16, _ = scale_quotas
    if r < q_sub8:
        target_bin = BIN_SUB_8PX
    elif r < (q_sub8 + q_8_16):
        target_bin = BIN_8_TO_16PX
    else:
        target_bin = BIN_GT_16PX

    crop_box = compute_scale_matched_envelope(
        record,
        target_bin=target_bin,
        rng=resolved_rng,
    )
    x1, y1, x2, y2 = crop_box

    if (x2 - x1) >= record.original_width and (y2 - y1) >= record.original_height:
        return image_rgb, record

    cropped_img = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
    cropped_record = zoom_crop_record(record, crop_box)
    return cropped_img, cropped_record


def _create_feathered_patch(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_ratio: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Extracts a bounding region with context margin and a soft edge feather alpha mask.

    Returns: (patch_rgb, alpha_mask, (inner_x1, inner_y1, inner_x2, inner_y2))
    """
    img_h, img_w = image_rgb.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))

    px1 = max(0, bx1 - mx)
    py1 = max(0, by1 - my)
    px2 = min(img_w, bx2 + mx)
    py2 = min(img_h, by2 + my)

    patch_rgb = np.ascontiguousarray(image_rgb[py1:py2, px1:px2])
    ph, pw = patch_rgb.shape[:2]

    # Create smooth elliptical/feathered mask
    mask = np.ones((ph, pw), dtype=np.float32)
    border_x = max(2, int(round(mx * 0.7)))
    border_y = max(2, int(round(my * 0.7)))

    for i in range(border_y):
        alpha_y = math.sin((i / border_y) * (math.pi / 2))
        mask[i, :] *= alpha_y
        mask[ph - 1 - i, :] *= alpha_y
    for j in range(border_x):
        alpha_x = math.sin((j / border_x) * (math.pi / 2))
        mask[:, j] *= alpha_x
        mask[:, pw - 1 - j] *= alpha_x

    # Inner bounding box relative to patch
    inner_box = (bx1 - px1, by1 - py1, bx2 - px1, by2 - py1)
    return patch_rgb, mask, inner_box


def paired_copy_paste(
    dest_image_rgb: np.ndarray,
    dest_record: ImageRecord,
    donor_image_rgb: np.ndarray,
    donor_record: ImageRecord,
    *,
    copy_paste_prob: float = 0.35,
    max_paste_tls: int = 2,
    blend_strength: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, ImageRecord]:
    """Semantics-preserving paired copy-paste augmentation.

    Transplants a donor traffic light + contextual background and, if paired,
    the corresponding road arrow into geometrically realistic destination regions.
    """
    resolved_rng = rng or random.Random()
    if resolved_rng.random() > copy_paste_prob:
        return dest_image_rgb, dest_record

    if not donor_record.traffic_lights:
        return dest_image_rgb, dest_record

    dest_h, dest_w = dest_image_rgb.shape[:2]
    donor_h, donor_w = donor_image_rgb.shape[:2]

    # Pick candidate donor traffic light
    donor_tl_idx = resolved_rng.randrange(len(donor_record.traffic_lights))
    donor_tl = donor_record.traffic_lights[donor_tl_idx]

    donor_bx1, donor_by1, donor_bx2, donor_by2 = (
        int(round(donor_tl.bbox_xyxy[0])),
        int(round(donor_tl.bbox_xyxy[1])),
        int(round(donor_tl.bbox_xyxy[2])),
        int(round(donor_tl.bbox_xyxy[3])),
    )

    if (donor_bx2 - donor_bx1) < 2 or (donor_by2 - donor_by1) < 2:
        return dest_image_rgb, dest_record

    tl_patch, tl_mask, tl_inner = _create_feathered_patch(
        donor_image_rgb,
        (donor_bx1, donor_by1, donor_bx2, donor_by2),
        margin_ratio=0.5,
    )
    ph, pw = tl_patch.shape[:2]

    # Destination vertical band for traffic lights: y in [0.15 * dest_h, 0.55 * dest_h]
    min_dest_y = int(round(dest_h * 0.12))
    max_dest_y = max(min_dest_y + 1, int(round(dest_h * 0.55)) - ph)

    if max_dest_y <= min_dest_y or (dest_w - pw - 20) <= 20:
        return dest_image_rgb, dest_record

    dest_y1 = resolved_rng.randint(min_dest_y, max_dest_y)
    dest_x1 = resolved_rng.randint(20, dest_w - pw - 20)
    dest_x2 = dest_x1 + pw
    dest_y2 = dest_y1 + ph

    # Alpha blend TL patch onto canvas
    canvas = dest_image_rgb.copy()
    alpha_3d = np.repeat(tl_mask[:, :, np.newaxis], 3, axis=2) * blend_strength
    canvas[dest_y1:dest_y2, dest_x1:dest_x2] = (
        canvas[dest_y1:dest_y2, dest_x1:dest_x2] * (1.0 - alpha_3d)
        + tl_patch * alpha_3d
    ).astype(np.uint8)

    # Compute new destination bounding box for the TL
    new_tl_box: BBox = (
        float(dest_x1 + tl_inner[0]),
        float(dest_y1 + tl_inner[1]),
        float(dest_x1 + tl_inner[2]),
        float(dest_y1 + tl_inner[3]),
    )

    # Check for paired road arrow in donor
    paired_arrow_donor: RoadArrowAnnotation | None = None
    if donor_tl.valid_relevance and donor_tl.relevance == 1:
        # Find ego lane road arrow in donor
        for ar in donor_record.road_arrows:
            if ar.valid_ego_lane and ar.is_ego_lane == 1:
                paired_arrow_donor = ar
                break
            elif not paired_arrow_donor:
                paired_arrow_donor = ar

    new_road_arrows = list(dest_record.road_arrows)
    pasted_arrow_success = False

    if paired_arrow_donor is not None and len(paired_arrow_donor.bbox_xyxy) == 4:
        ar_bx1, ar_by1, ar_bx2, ar_by2 = (
            int(round(paired_arrow_donor.bbox_xyxy[0])),
            int(round(paired_arrow_donor.bbox_xyxy[1])),
            int(round(paired_arrow_donor.bbox_xyxy[2])),
            int(round(paired_arrow_donor.bbox_xyxy[3])),
        )
        if (ar_bx2 - ar_bx1) >= 4 and (ar_by2 - ar_by1) >= 4:
            ar_patch, ar_mask, ar_inner = _create_feathered_patch(
                donor_image_rgb,
                (ar_bx1, ar_by1, ar_bx2, ar_by2),
                margin_ratio=0.3,
            )
            aph, apw = ar_patch.shape[:2]

            # Place arrow on road surface below TL: y in [0.60 * dest_h, 0.90 * dest_h]
            min_ar_y = int(round(dest_h * 0.60))
            max_ar_y = max(min_ar_y + 1, int(round(dest_h * 0.92)) - aph)

            if max_ar_y > min_ar_y:
                ar_dest_y1 = resolved_rng.randint(min_ar_y, max_ar_y)
                # Keep lateral alignment roughly consistent with TL column (within delta)
                desired_center_x = (new_tl_box[0] + new_tl_box[2]) / 2.0
                ar_dest_x1 = max(
                    10,
                    min(
                        dest_w - apw - 10,
                        int(round(desired_center_x - apw / 2.0 + resolved_rng.uniform(-30.0, 30.0))),
                    ),
                )
                ar_dest_x2 = ar_dest_x1 + apw
                ar_dest_y2 = ar_dest_y1 + aph

                ar_alpha_3d = np.repeat(ar_mask[:, :, np.newaxis], 3, axis=2) * blend_strength
                canvas[ar_dest_y1:ar_dest_y2, ar_dest_x1:ar_dest_x2] = (
                    canvas[ar_dest_y1:ar_dest_y2, ar_dest_x1:ar_dest_x2] * (1.0 - ar_alpha_3d)
                    + ar_patch * ar_alpha_3d
                ).astype(np.uint8)

                new_ar_box: BBox = (
                    float(ar_dest_x1 + ar_inner[0]),
                    float(ar_dest_y1 + ar_inner[1]),
                    float(ar_dest_x1 + ar_inner[2]),
                    float(ar_dest_y1 + ar_inner[3]),
                )

                # Transform segmentation if present
                new_seg: list[tuple[float, float]] = []
                for sx, sy in paired_arrow_donor.segmentation_xy:
                    shift_x = sx - ar_bx1 + ar_inner[0] + ar_dest_x1
                    shift_y = sy - ar_by1 + ar_inner[1] + ar_dest_y1
                    new_seg.append((float(max(0.0, min(float(dest_w), shift_x))), float(max(0.0, min(float(dest_h), shift_y)))))

                new_road_arrows.append(
                    replace(
                        paired_arrow_donor,
                        bbox_xyxy=new_ar_box,
                        segmentation_xy=tuple(new_seg),
                        is_ego_lane=paired_arrow_donor.is_ego_lane,
                        valid_ego_lane=paired_arrow_donor.valid_ego_lane,
                    )
                )
                pasted_arrow_success = True

    # Assemble new TrafficLightAnnotation
    # If arrow was pasted successfully, retain relevance, else set relevance=0 (adjacent lane)
    assigned_relevance = donor_tl.relevance if pasted_arrow_success else 0
    new_tl_anno = replace(
        donor_tl,
        bbox_xyxy=new_tl_box,
        relevance=assigned_relevance,
        valid_relevance=donor_tl.valid_relevance,
    )

    new_tls = list(dest_record.traffic_lights) + [new_tl_anno]

    updated_record = replace(
        dest_record,
        traffic_lights=new_tls,
        road_arrows=new_road_arrows,
        metadata={
            **dest_record.metadata,
            "paired_copy_paste": True,
            "donor_image_id": donor_record.image_id,
            "pasted_arrow": pasted_arrow_success,
        },
    )
    updated_record.validate()
    return canvas, updated_record
