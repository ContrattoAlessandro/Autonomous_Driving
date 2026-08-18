"""DriveU Traffic Light Dataset v2.0 converter.

Unlike the legacy YOLO conversion, this adapter never drops a vehicle-light
box merely because one attribute is unknown.  Attribute masks and ignore
regions preserve exactly which gradients are valid.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from ..geometry import clip_box, xywh_to_xyxy
from ..schema import (
    IgnoreRegion,
    ImageRecord,
    TaskValidity,
    TrafficLightAnnotation,
)
from ..taxonomy import (
    factor_pictogram,
    map_binary_relevance,
    map_pictogram,
    map_state,
    normalize_label,
    normalize_occlusion,
)
from .common import ConversionResult, dimensions, image_index

DTLD_SIZE = (2048, 1024)


def _assert_unannotated_images(images_dir: Path) -> None:
    """Reject the known local DTLD preview tree with burned-in annotations."""

    if any(part.casefold() == "dtld_jpg" for part in images_dir.resolve().parts):
        raise ValueError(
            "DTLD_jpg contains preview images with boxes/text burned into the pixels; "
            "use the unannotated DTLD_jpg_plain tree"
        )


def _source_parts(source_path: str) -> tuple[str | None, str | None, str | None]:
    clean = source_path.replace("\\", "/").removeprefix("./")
    parts = PurePosixPath(clean).parts
    parent = "/".join(parts[:-1]) or None
    city = parts[0] if len(parts) > 1 else None
    route = "/".join(parts[:2]) if len(parts) > 2 else city
    return city, route, parent


def _ignore_reason(attributes: dict[str, Any], pictogram_ignore: bool) -> str | None:
    if normalize_occlusion(attributes.get("occlusion")) == "fully_occluded":
        return "fully_occluded"
    if normalize_label(attributes.get("direction")) in {"back", "rear", "backward"}:
        return "back_facing_traffic_light"
    if pictogram_ignore:
        return "non_vehicle_pictogram"
    return None


def convert_dtld_file(
    labels_json: str | Path,
    images_dir: str | Path,
    split: str,
    *,
    limit: int | None = None,
    verify_dimensions: bool = False,
    strict_images: bool = True,
) -> ConversionResult:
    labels_path = Path(labels_json)
    image_root = Path(images_dir)
    _assert_unannotated_images(image_root)
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    rows = payload.get("images", payload if isinstance(payload, list) else ())
    index = image_index(image_root, recursive=True)
    records: list[ImageRecord] = []
    stats: Counter[str] = Counter()

    for source_image in rows:
        if limit is not None and len(records) >= limit:
            break
        source_path = str(
            source_image.get("image_path")
            or source_image.get("path")
            or source_image.get("filename")
            or ""
        )
        stem = Path(source_path).stem
        image_path = index.get(stem)
        if image_path is None:
            stats["missing_images"] += 1
            if strict_images:
                raise FileNotFoundError(f"DTLD image not found for {source_path!r} in {image_root}")
            continue
        width, height = dimensions(
            image_path, DTLD_SIZE, verify=verify_dimensions
        )
        city, route, sequence = _source_parts(source_path)
        traffic_lights: list[TrafficLightAnnotation] = []
        ignore_regions: list[IgnoreRegion] = []

        for source_label in source_image.get("labels", source_image.get("objects", ())):
            try:
                raw_box = xywh_to_xyxy(
                    float(source_label["x"]),
                    float(source_label["y"]),
                    float(source_label["w"]),
                    float(source_label["h"]),
                )
            except (KeyError, TypeError, ValueError):
                stats["invalid_box_rows"] += 1
                continue
            box = clip_box(raw_box, width, height)
            if box is None:
                stats["degenerate_boxes"] += 1
                continue

            attributes = dict(source_label.get("attributes", {}))
            state = map_state(attributes.get("state"))
            pictogram = map_pictogram(attributes.get("pictogram"))
            factorized = factor_pictogram(attributes.get("pictogram"))
            ignore_reason = _ignore_reason(
                attributes, pictogram.ignore_object or factorized.ignore_object
            )
            if ignore_reason is not None:
                ignore_regions.append(
                    IgnoreRegion(
                        bbox_xyxy=box,
                        reason=ignore_reason,
                        source_label=normalize_label(attributes.get("pictogram")),
                    )
                )
                stats[f"ignore/{ignore_reason}"] += 1
                continue

            relevance, valid_relevance = map_binary_relevance(attributes.get("relevance"))
            occlusion = normalize_occlusion(attributes.get("occlusion"))
            traffic_lights.append(
                TrafficLightAnnotation(
                    bbox_xyxy=box,
                    state=state.target,
                    pictogram=pictogram.target,
                    relevance=relevance,
                    occlusion=occlusion,
                    valid_state=state.valid,
                    valid_pictogram=pictogram.valid,
                    valid_relevance=valid_relevance,
                    round_target=factorized.round,
                    maneuver_multihot=factorized.maneuver,
                    valid_round=factorized.valid_round,
                    valid_maneuver=factorized.valid_maneuver,
                    source_attributes=attributes,
                )
            )
            stats["traffic_lights"] += 1
            stats[f"state/{state.target if state.valid else 'masked'}"] += 1
            stats[f"pictogram/{pictogram.target if pictogram.valid else 'masked'}"] += 1
            stats[
                f"round/{factorized.round if factorized.valid_round else 'masked'}"
            ] += 1
            if factorized.valid_maneuver:
                stats[f"maneuver/{factorized.maneuver}"] += 1
            stats[f"occlusion/{occlusion}"] += 1
            if valid_relevance:
                stats[f"relevance/{relevance}"] += 1

        record = ImageRecord(
            image_id=f"DTLD/{split}/{stem}",
            image_path=str(image_path),
            source_dataset="DTLD",
            original_width=width,
            original_height=height,
            split=split,
            sequence_id=f"DTLD/{sequence}" if sequence else f"DTLD/{stem}",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_state=True,
                traffic_light_pictogram=True,
                traffic_light_relevance=True,
                traffic_light_round=True,
                traffic_light_maneuver=True,
            ),
            traffic_lights=traffic_lights,
            ignore_regions=ignore_regions,
            metadata={
                "official_split": split,
                "image_variant": "plain_unannotated",
                "city": city,
                "route": route,
                "source_image_path": source_path,
            },
        )
        record.validate()
        records.append(record)
        stats["images"] += 1
        stats["image_variant/plain_unannotated"] += 1

    return ConversionResult(records=records, stats=stats)


def convert_dtld_root(
    labels_root: str | Path,
    images_root: str | Path,
    *,
    limit_per_split: int | None = None,
    verify_dimensions: bool = False,
    strict_images: bool = True,
) -> ConversionResult:
    labels = Path(labels_root)
    images = Path(images_root)
    records: list[ImageRecord] = []
    stats: Counter[str] = Counter()
    for split in ("train", "test"):
        result = convert_dtld_file(
            labels / f"DTLD_{split}.json",
            images / split,
            split,
            limit=limit_per_split,
            verify_dimensions=verify_dimensions,
            strict_images=strict_images,
        )
        records.extend(result.records)
        stats.update(result.stats)
    return ConversionResult(records=records, stats=stats)
