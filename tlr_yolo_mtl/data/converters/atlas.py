"""ATLAS composite-label converter with factorized attribute masks."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..geometry import clip_box, yolo_to_xyxy
from ..schema import ImageRecord, TaskValidity, TrafficLightAnnotation
from ..taxonomy import split_atlas_class
from .common import ConversionResult, dimensions, parse_yolo_row

ATLAS_SIZE = (1920, 1200)
ATLAS_CAMERAS = ("front_medium", "front_tele", "front_wide")
TIMESTAMP_RE = re.compile(r"(?:front_(?:medium|tele|wide)_)?(\d+)-(\d+)$")


def read_atlas_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(\d+)\s*:\s*['\"]?([^'\"#]+)", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    if not names:
        raise ValueError(f"no ATLAS classes found in {path}")
    return names


def _atlas_sequence(stem: str) -> tuple[str, int | None]:
    match = TIMESTAMP_RE.search(stem)
    if not match:
        return f"ATLAS/{stem}", None
    seconds, nanoseconds = int(match.group(1)), int(match.group(2))
    timestamp_ns = seconds * 1_000_000_000 + nanoseconds
    block = timestamp_ns // 30_000_000_000
    return f"ATLAS/30s/{block}", timestamp_ns


def convert_atlas(
    root: str | Path,
    *,
    limit_per_partition: int | None = None,
    verify_dimensions: bool = False,
) -> ConversionResult:
    atlas_root = Path(root)
    names = read_atlas_names(atlas_root / "ATLAS_classes.yaml")
    records: list[ImageRecord] = []
    stats: Counter[str] = Counter()

    for split in ("train", "test"):
        for camera in ATLAS_CAMERAS:
            image_dir = atlas_root / split / camera / "images"
            label_dir = atlas_root / split / camera / "labels"
            if not image_dir.exists():
                continue
            partition_count = 0
            for image_path in sorted(
                path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ):
                if limit_per_partition is not None and partition_count >= limit_per_partition:
                    break
                width, height = dimensions(
                    image_path,
                    ATLAS_SIZE,
                    verify=verify_dimensions,
                    # ATLAS labels are normalized but the downloaded JPEGs do
                    # not all share the nominal 1920x1200 canvas.
                    prefer_actual=True,
                )
                if (width, height) != ATLAS_SIZE:
                    stats["non_nominal_image_dimensions"] += 1
                traffic_lights: list[TrafficLightAnnotation] = []
                label_path = label_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        row = parse_yolo_row(line)
                        if row is None:
                            stats["invalid_label_rows"] += 1
                            continue
                        class_id, cx, cy, box_width, box_height = row
                        class_name = names.get(class_id)
                        if class_name is None:
                            stats["unknown_classes"] += 1
                            continue
                        box = clip_box(
                            yolo_to_xyxy(
                                cx, cy, box_width, box_height, width, height
                            ),
                            width,
                            height,
                        )
                        if box is None:
                            stats["degenerate_boxes"] += 1
                            continue
                        state, pictogram = split_atlas_class(class_name)
                        traffic_lights.append(
                            TrafficLightAnnotation(
                                bbox_xyxy=box,
                                state=state.target,
                                pictogram=pictogram.target,
                                relevance=None,
                                occlusion="unknown",
                                valid_state=state.valid,
                                valid_pictogram=pictogram.valid,
                                valid_relevance=False,
                                source_attributes={
                                    "class_id": class_id,
                                    "class_name": class_name,
                                },
                            )
                        )
                        stats["traffic_lights"] += 1
                        stats[f"state/{state.target if state.valid else 'masked'}"] += 1
                        stats[
                            f"pictogram/{pictogram.target if pictogram.valid else 'masked'}"
                        ] += 1

                sequence_id, timestamp_ns = _atlas_sequence(image_path.stem)
                record = ImageRecord(
                    image_id=f"ATLAS/{split}/{camera}/{image_path.stem}",
                    image_path=str(image_path.resolve()),
                    source_dataset="ATLAS",
                    original_width=width,
                    original_height=height,
                    split=split,
                    sequence_id=sequence_id,
                    task_valid=TaskValidity(
                        traffic_light_detection=True,
                        traffic_light_state=True,
                        traffic_light_pictogram=True,
                    ),
                    traffic_lights=traffic_lights,
                    metadata={
                        "official_split": split,
                        "camera": camera,
                        "timestamp_ns": timestamp_ns,
                    },
                )
                record.validate()
                records.append(record)
                partition_count += 1
                stats["images"] += 1
    return ConversionResult(records=records, stats=stats)
