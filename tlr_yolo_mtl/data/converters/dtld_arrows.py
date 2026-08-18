"""Fuse exhaustive human road-arrow labels into canonical DTLD records.

The annotation project mirrors every DTLD official-train image using a
``dtld_`` filename prefix and YOLO detection labels.  Empty label files are
intentional, reviewed negative images.  Official DTLD test records are left
arrow-masked because they are not part of the reviewed annotation project.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ..geometry import clip_box, image_size, yolo_to_xyxy
from ..schema import ImageRecord, RoadArrowAnnotation
from .common import ConversionResult, IMAGE_SUFFIXES

DTLD_ARROW_DIRECTIONS: dict[int, tuple[str, tuple[int, int, int]]] = {
    0: ("straight", (0, 1, 0)),
    1: ("left", (1, 0, 0)),
    2: ("right", (0, 0, 1)),
    3: ("straight_left", (1, 1, 0)),
    4: ("straight_right", (0, 1, 1)),
}
CLIP_REPORT_TOLERANCE_PX = 0.01


def _annotation_stem(path: Path) -> str:
    stem = path.stem
    if not stem.casefold().startswith("dtld_"):
        raise ValueError(f"annotated DTLD filename must start with 'dtld_': {path}")
    canonical = stem[5:]
    if not canonical:
        raise ValueError(f"annotated DTLD filename has an empty canonical stem: {path}")
    return canonical.casefold()


def _index_files(paths: Iterable[Path], kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(paths):
        key = _annotation_stem(path)
        previous = result.get(key)
        if previous is not None:
            raise ValueError(
                f"duplicate {kind} for canonical DTLD stem {key!r}: "
                f"{previous} and {path}"
            )
        result[key] = path.resolve()
    return result


def _coverage_error(
    label: str,
    expected: set[str],
    actual: set[str],
) -> ValueError:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return ValueError(
        f"{label}: missing={len(missing)} {missing[:5]}, "
        f"extra={len(extra)} {extra[:5]}"
    )


def _parse_label_file(
    path: Path,
    *,
    width: int,
    height: int,
    stats: Counter[str],
) -> list[RoadArrowAnnotation]:
    arrows: list[RoadArrowAnnotation] = []
    seen: set[tuple[tuple[float, float, float, float], tuple[int, int, int]]] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        location = f"{path}:{line_number}"
        if len(fields) != 5:
            raise ValueError(
                f"{location}: expected exactly 5 YOLO fields, got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            cx, cy, box_width, box_height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{location}: non-numeric YOLO row {line!r}") from exc
        if class_id not in DTLD_ARROW_DIRECTIONS:
            raise ValueError(
                f"{location}: unsupported arrow class {class_id}; "
                f"expected {sorted(DTLD_ARROW_DIRECTIONS)}"
            )
        values = (cx, cy, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{location}: YOLO coordinates must be finite")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError(f"{location}: YOLO centre must be in [0, 1]")
        if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
            raise ValueError(f"{location}: YOLO width/height must be in (0, 1]")

        raw_box = yolo_to_xyxy(cx, cy, box_width, box_height, width, height)
        box = clip_box(raw_box, width, height)
        if box is None:
            raise ValueError(f"{location}: clipping produced a zero-area box")
        clamp_delta = max(abs(before - after) for before, after in zip(raw_box, box))
        clipped = clamp_delta > CLIP_REPORT_TOLERANCE_PX
        class_name, direction = DTLD_ARROW_DIRECTIONS[class_id]
        duplicate_key = (box, direction)
        if duplicate_key in seen:
            raise ValueError(f"{location}: duplicate arrow annotation {line!r}")
        seen.add(duplicate_key)
        arrows.append(
            RoadArrowAnnotation(
                bbox_xyxy=box,
                direction_multihot=direction,
                source_attributes={
                    "annotation_source": "dataset_ALL_USER_ANNOTATED",
                    "annotation_status": "human_verified_exhaustive",
                    "source_class_id": class_id,
                    "source_class_name": class_name,
                    "box_clipped_to_image": clipped,
                    "max_clip_delta_px": clamp_delta if clipped else 0.0,
                    "source_label_file": str(path),
                    "source_label_line": line_number,
                },
            )
        )
        stats["arrows"] += 1
        stats[f"class/{class_id}/{class_name}"] += 1
        stats["boundary_clamped_boxes"] += int(clamp_delta > 0.0)
        stats["clipped_boxes"] += int(clipped)
    return arrows


def fuse_dtld_arrow_annotations(
    records: Iterable[ImageRecord],
    annotations_root: str | Path,
    *,
    require_exact_coverage: bool = True,
) -> ConversionResult:
    """Attach exhaustive arrows to every DTLD official-train record.

    The canonical DTLD image remains the training image.  The duplicate image
    in the annotation project is used to prove filename coverage and dimensions.
    Coverage, rows, classes, coordinates, and accidental overwrites are all
    validated before a record is changed.
    """

    materialized = list(records)
    root = Path(annotations_root).resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"DTLD arrow annotations require images/ and labels/ under {root}"
        )

    images = _index_files(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        "annotation image",
    )
    labels = _index_files(
        (path for path in labels_dir.iterdir() if path.is_file() and path.suffix.casefold() == ".txt"),
        "annotation label",
    )
    if set(images) != set(labels):
        raise _coverage_error("image/label coverage mismatch", set(images), set(labels))

    targets: dict[str, ImageRecord] = {}
    for record in materialized:
        official_split = str(record.metadata.get("official_split", record.split))
        if record.source_dataset != "DTLD" or official_split != "train":
            continue
        key = Path(record.image_path).stem.casefold()
        if key in targets:
            raise ValueError(f"duplicate canonical DTLD official-train stem: {key}")
        if record.task_valid.arrow_detection or record.road_arrows:
            raise ValueError(
                f"refusing to overwrite existing arrow supervision: {record.image_id}"
            )
        targets[key] = record

    if not targets:
        raise ValueError("no DTLD official-train records are available for arrow fusion")
    if require_exact_coverage and set(targets) != set(labels):
        raise _coverage_error(
            "annotation coverage mismatch", set(targets), set(labels)
        )
    missing = set(targets) - set(labels)
    if missing:
        raise _coverage_error("annotation coverage mismatch", set(targets), set(labels))

    stats: Counter[str] = Counter()
    fused_by_id: dict[str, ImageRecord] = {}
    for key, record in targets.items():
        annotation_image = images[key]
        actual_size = image_size(annotation_image)
        expected_size = (record.original_width, record.original_height)
        if actual_size != expected_size:
            raise ValueError(
                f"dimension mismatch for {annotation_image}: "
                f"annotation={actual_size}, canonical={expected_size}"
            )
        arrows = _parse_label_file(
            labels[key],
            width=record.original_width,
            height=record.original_height,
            stats=stats,
        )
        task_valid = replace(record.task_valid, arrow_detection=True)
        metadata = {
            **record.metadata,
            "arrow_annotation_source": "dataset_ALL_USER_ANNOTATED",
            "arrow_annotation_root": str(root),
            "arrow_annotation_status": "human_verified_exhaustive",
            "arrow_label_path": str(labels[key]),
            "arrow_annotation_image_path": str(annotation_image),
            "paired_relevance_arrow": bool(task_valid.traffic_light_relevance),
        }
        fused = replace(
            record,
            task_valid=task_valid,
            road_arrows=arrows,
            metadata=metadata,
        )
        fused.validate()
        fused_by_id[record.image_id] = fused
        stats["annotated_images"] += 1
        stats["positive_images" if arrows else "negative_images"] += 1
        stats["paired_relevance_arrow_images"] += int(
            task_valid.traffic_light_relevance
        )

    result = [fused_by_id.get(record.image_id, record) for record in materialized]
    return ConversionResult(records=result, stats=stats)
