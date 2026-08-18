"""Annotation transforms whose semantics must match image augmentation."""

from __future__ import annotations

from dataclasses import replace

from .geometry import horizontal_flip_box
from .schema import ImageRecord
from .taxonomy import flip_direction_multihot, flip_pictogram


def horizontal_flip_record(record: ImageRecord) -> ImageRecord:
    """Flip boxes and every direction-bearing target as one atomic operation."""

    width = record.original_width
    traffic_lights = [
        replace(
            item,
            bbox_xyxy=horizontal_flip_box(item.bbox_xyxy, width),
            pictogram=flip_pictogram(item.pictogram),
            maneuver_multihot=(
                None
                if item.maneuver_multihot is None
                else flip_direction_multihot(item.maneuver_multihot)
            ),
        )
        for item in record.traffic_lights
    ]
    road_arrows = [
        replace(
            item,
            bbox_xyxy=horizontal_flip_box(item.bbox_xyxy, width),
            direction_multihot=flip_direction_multihot(item.direction_multihot),
            segmentation_xy=tuple(
                (float(width) - x, y) for x, y in item.segmentation_xy
            ),
        )
        for item in record.road_arrows
    ]
    ignore_regions = [
        replace(item, bbox_xyxy=horizontal_flip_box(item.bbox_xyxy, width))
        for item in record.ignore_regions
    ]
    flipped = replace(
        record,
        traffic_lights=traffic_lights,
        road_arrows=road_arrows,
        ignore_regions=ignore_regions,
        metadata={**record.metadata, "horizontal_flip": True},
    )
    flipped.validate()
    return flipped
