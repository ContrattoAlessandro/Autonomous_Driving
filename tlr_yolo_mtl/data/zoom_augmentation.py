"""Context-Preserving Whole-Scene Zoom Augmentation & Difficulty-Bucketed Hard Sampling.

Provides:
1. Context-preserving whole-scene zoom augmentation: scales up tiny traffic lights
   by 1.5x - 2.5x without breaking lane-level spatial topology or TL <-> Arrow
   relevance semantics.
2. Difficulty-bucketed hard sampler: boosts sampling rates for tiny traffic lights,
   directional signals, and arrow-present intersections.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Iterator, Mapping, Sequence

import cv2
import numpy as np
from torch.utils.data import Sampler

from .schema import BBox, ImageRecord


def compute_context_envelope(
    record: ImageRecord,
    *,
    margin_factor: float = 1.4,
    target_aspect: float = 2.0,  # 1600 / 800
) -> tuple[int, int, int, int]:
    """Compute an intersection-centric bounding window enclosing all relevant objects.

    Returns (x1, y1, x2, y2) in original pixel coordinates.
    """
    img_w, img_h = record.original_width, record.original_height

    boxes: list[BBox] = []
    # Include all traffic lights and road arrows
    for tl in record.traffic_lights:
        boxes.append(tl.bbox_xyxy)
    for ar in record.road_arrows:
        boxes.append(ar.bbox_xyxy)

    if not boxes:
        return 0, 0, img_w, img_h

    # Compute bounding envelope
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[2] for b in boxes)
    y_max = max(b[3] for b in boxes)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = max(x_max - x_min, 50.0) * margin_factor
    bh = max(y_max - y_min, 30.0) * margin_factor

    # Adjust aspect ratio to roughly match target aspect
    if bw / max(bh, 1e-3) < target_aspect:
        bw = bh * target_aspect
    else:
        bh = bw / target_aspect

    # Ensure minimum zoom (at least 1.2x zoom vs full image)
    max_w = img_w * 0.85
    max_h = img_h * 0.85
    bw = min(bw, max_w)
    bh = min(bh, max_h)

    x1 = max(0, int(round(cx - bw / 2.0)))
    y1 = max(0, int(round(cy - bh / 2.0)))
    x2 = min(img_w, int(round(cx + bw / 2.0)))
    y2 = min(img_h, int(round(cy + bh / 2.0)))

    # Re-clamp to ensure valid non-degenerate box
    if x2 - x1 < 100 or y2 - y1 < 50:
        return 0, 0, img_w, img_h

    return x1, y1, x2, y2


def zoom_crop_record(
    record: ImageRecord,
    crop_box: tuple[int, int, int, int],
) -> ImageRecord:
    """Transform bounding boxes and segmentations relative to the cropped sub-window."""
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop box: {crop_box}")

    def transform_box(box: BBox) -> BBox | None:
        bx1 = max(0.0, min(float(crop_w), box[0] - x1))
        by1 = max(0.0, min(float(crop_h), box[1] - y1))
        bx2 = max(0.0, min(float(crop_w), box[2] - x1))
        by2 = max(0.0, min(float(crop_h), box[3] - y1))
        if (bx2 - bx1) >= 1.0 and (by2 - by1) >= 1.0:
            return (bx1, by1, bx2, by2)
        return None

    new_traffic_lights = []
    for item in record.traffic_lights:
        t_box = transform_box(item.bbox_xyxy)
        if t_box is not None:
            new_traffic_lights.append(replace(item, bbox_xyxy=t_box))

    new_road_arrows = []
    for item in record.road_arrows:
        t_box = transform_box(item.bbox_xyxy)
        if t_box is not None:
            new_seg = tuple(
                (float(max(0.0, min(float(crop_w), px - x1))), float(max(0.0, min(float(crop_h), py - y1))))
                for px, py in item.segmentation_xy
            )
            new_road_arrows.append(replace(item, bbox_xyxy=t_box, segmentation_xy=new_seg))

    new_ignore_regions = []
    for item in record.ignore_regions:
        t_box = transform_box(item.bbox_xyxy)
        if t_box is not None:
            new_ignore_regions.append(replace(item, bbox_xyxy=t_box))

    cropped = replace(
        record,
        original_width=crop_w,
        original_height=crop_h,
        traffic_lights=new_traffic_lights,
        road_arrows=new_road_arrows,
        ignore_regions=new_ignore_regions,
        metadata={**record.metadata, "context_zoom": True, "crop_box": crop_box},
    )
    cropped.validate()
    return cropped


def context_preserving_zoom(
    image_rgb: np.ndarray,
    record: ImageRecord,
    *,
    zoom_prob: float = 0.5,
    margin_range: tuple[float, float] = (1.2, 1.8),
    rng: random.Random | None = None,
) -> tuple[np.ndarray, ImageRecord]:
    """Applies context-preserving zoom augmentation to an image and its canonical record."""
    resolved_rng = rng or random.Random()
    if resolved_rng.random() > zoom_prob:
        return image_rgb, record

    if not record.traffic_lights and not record.road_arrows:
        return image_rgb, record

    margin = resolved_rng.uniform(margin_range[0], margin_range[1])
    crop_box = compute_context_envelope(record, margin_factor=margin)
    x1, y1, x2, y2 = crop_box

    # If full frame was returned, do nothing
    if (x2 - x1) == record.original_width and (y2 - y1) == record.original_height:
        return image_rgb, record

    cropped_img = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
    cropped_record = zoom_crop_record(record, crop_box)
    return cropped_img, cropped_record


class DifficultyBucketedSampler(Sampler[list[int]]):
    """Stratified sampler weighting tiny traffic lights and directional scenes higher."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        micro_batch_size: int = 8,
        effective_batch_size: int = 32,
        hard_weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
        seed: int = 42,
    ) -> None:
        self.micro_batch_size = int(micro_batch_size)
        self.effective_batch_size = int(effective_batch_size)
        self.seed = int(seed)
        self.epoch = 0

        # Partition indices into 3 buckets:
        # 1. Tiny TLs (area < 64 or side < 6)
        # 2. Directional / Arrow scenes
        # 3. Standard / Round-only scenes
        self.tiny_indices: list[int] = []
        self.directional_indices: list[int] = []
        self.standard_indices: list[int] = []

        for idx, rec in enumerate(records):
            has_tiny = False
            has_directional = False
            has_arrows = len(rec.road_arrows) > 0

            for tl in rec.traffic_lights:
                bw = tl.bbox_xyxy[2] - tl.bbox_xyxy[0]
                bh = tl.bbox_xyxy[3] - tl.bbox_xyxy[1]
                area = bw * bh
                if area < 64.0 or min(bw, bh) < 6.0:
                    has_tiny = True
                if tl.valid_pictogram and tl.pictogram in {"left", "straight", "right", "straight_left", "straight_right"}:
                    has_directional = True

            if has_tiny:
                self.tiny_indices.append(idx)
            elif has_directional or has_arrows:
                self.directional_indices.append(idx)
            else:
                self.standard_indices.append(idx)

        # Fallbacks for empty buckets
        if not self.tiny_indices:
            self.tiny_indices = list(range(len(records)))
        if not self.directional_indices:
            self.directional_indices = list(range(len(records)))
        if not self.standard_indices:
            self.standard_indices = list(range(len(records)))

        # Quotas per effective batch (sum = effective_batch_size)
        q_tiny = max(1, int(round(effective_batch_size * hard_weights[0])))
        q_dir = max(1, int(round(effective_batch_size * hard_weights[1])))
        q_std = max(1, effective_batch_size - q_tiny - q_dir)
        self.quotas = {
            "tiny": q_tiny,
            "directional": q_dir,
            "standard": q_std,
        }
        self.total_samples = len(records)
        self.windows_per_epoch = max(1, math.ceil(self.total_samples / self.effective_batch_size))

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.micro_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.windows_per_epoch * self.accumulation_steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = {
            "tiny": list(self.tiny_indices),
            "directional": list(self.directional_indices),
            "standard": list(self.standard_indices),
        }
        positions = {k: len(v) for k, v in pools.items()}

        def take(group: str, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                if positions[group] >= len(pools[group]):
                    rng.shuffle(pools[group])
                    positions[group] = 0
                avail = min(count - len(selected), len(pools[group]) - positions[group])
                selected.extend(pools[group][positions[group] : positions[group] + avail])
                positions[group] += avail
            return selected

        for _ in range(self.windows_per_epoch):
            window: list[int] = []
            for group, count in self.quotas.items():
                window.extend(take(group, count))
            rng.shuffle(window)
            for offset in range(0, len(window), self.micro_batch_size):
                yield window[offset : offset + self.micro_batch_size]
