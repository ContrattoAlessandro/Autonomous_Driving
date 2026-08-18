"""LISA Traffic Light Dataset converter.

The canonical BOX CSVs annotate complete video sequences.  Every frame is
therefore emitted: frames absent from a CSV are valid detection negatives,
not images with a missing task label.  Sample folders bundled in the archive
are excluded because they duplicate parts of the training clips.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..geometry import clip_box
from ..schema import ImageRecord, TaskValidity, TrafficLightAnnotation
from ..taxonomy import map_pictogram, map_state, normalize_label
from .common import ConversionResult, dimensions

LISA_SIZE = (1280, 960)

LISA_TAGS: dict[str, tuple[str, str]] = {
    "go": ("green", "circle"),
    "go_left": ("green", "arrow_left"),
    "go_forward": ("green", "arrow_straight"),
    "stop": ("red", "circle"),
    "stop_left": ("red", "arrow_left"),
    "stop_forward": ("red", "arrow_straight"),
    "warning": ("yellow", "circle"),
    "warning_left": ("yellow", "arrow_left"),
    "warning_forward": ("yellow", "arrow_straight"),
}


def _normalize_tag(value: str) -> str:
    normalized = normalize_label(value)
    # LISA uses camelCase names such as goLeft and warningLeft.
    compact = normalized.replace("_", "")
    aliases = {
        "goleft": "go_left",
        "goforward": "go_forward",
        "stopleft": "stop_left",
        "stopforward": "stop_forward",
        "warningleft": "warning_left",
        "warningforward": "warning_forward",
    }
    return aliases.get(compact, normalized)


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    name: str
    split: str
    domain: str
    annotation_file: Path
    frames_dir: Path


def _annotation_root(root: Path) -> Path:
    candidates = (root / "Annotations" / "Annotations", root / "Annotations", root)
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("frameAnnotationsBOX.csv")):
            return candidate
    raise FileNotFoundError(f"LISA BOX annotations not found under {root}")


def _sequence_specs(root: Path) -> list[SequenceSpec]:
    annotations = _annotation_root(root)
    specs: list[SequenceSpec] = []
    for annotation_file in sorted(annotations.rglob("frameAnnotationsBOX.csv")):
        relative_parent = annotation_file.parent.relative_to(annotations)
        parts = relative_parent.parts
        if not parts:
            continue
        collection = parts[0]
        if collection in {"dayTrain", "nightTrain"}:
            if len(parts) != 2:
                continue
            clip = parts[1]
            frames_dir = root / collection / collection / clip / "frames"
            split = "train"
            sequence_name = f"{collection}/{clip}"
        elif collection in {
            "daySequence1",
            "daySequence2",
            "nightSequence1",
            "nightSequence2",
        }:
            frames_dir = root / collection / collection / "frames"
            split = "test"
            sequence_name = collection
        else:
            continue
        domain = "night" if collection.lower().startswith("night") else "day"
        if not frames_dir.exists():
            raise FileNotFoundError(
                f"LISA frames for {sequence_name} not found: {frames_dir}"
            )
        specs.append(
            SequenceSpec(
                name=sequence_name,
                split=split,
                domain=domain,
                annotation_file=annotation_file,
                frames_dir=frames_dir,
            )
        )
    if not specs:
        raise ValueError(f"no canonical LISA sequences found under {root}")
    return specs


def _column_map(fieldnames: list[str] | None) -> dict[str, str]:
    return {name.strip().lower(): name for name in (fieldnames or [])}


def _read_annotations(
    path: Path, stats: Counter[str]
) -> dict[str, list[tuple[str, float, float, float, float]]]:
    by_image: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    seen: set[tuple[str, str, float, float, float, float]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        columns = _column_map(reader.fieldnames)
        required = {
            "filename",
            "annotation tag",
            "upper left corner x",
            "upper left corner y",
            "lower right corner x",
            "lower right corner y",
        }
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        for row in reader:
            filename = Path(row[columns["filename"]].replace("\\", "/")).name
            tag = row[columns["annotation tag"]].strip()
            try:
                coords = (
                    float(row[columns["upper left corner x"]]),
                    float(row[columns["upper left corner y"]]),
                    float(row[columns["lower right corner x"]]),
                    float(row[columns["lower right corner y"]]),
                )
            except ValueError:
                stats["invalid_box_rows"] += 1
                continue
            key = (filename.lower(), tag, *coords)
            if key in seen:
                stats["duplicate_box_rows"] += 1
                continue
            seen.add(key)
            by_image[filename.lower()].append((tag, *coords))
    return by_image


def convert_lisa(
    root: str | Path,
    *,
    limit_per_split: int | None = None,
    verify_dimensions: bool = False,
) -> ConversionResult:
    lisa_root = Path(root)
    records: list[ImageRecord] = []
    stats: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for spec in _sequence_specs(lisa_root):
        if limit_per_split is not None and split_counts[spec.split] >= limit_per_split:
            continue
        annotations = _read_annotations(spec.annotation_file, stats)
        image_paths = sorted(
            path
            for path in spec.frames_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_path in image_paths:
            if limit_per_split is not None and split_counts[spec.split] >= limit_per_split:
                break
            width, height = dimensions(
                image_path, LISA_SIZE, verify=verify_dimensions
            )
            traffic_lights: list[TrafficLightAnnotation] = []
            rows_by_box: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
            for tag, x1, y1, x2, y2 in annotations.get(image_path.name.lower(), ()):
                rows_by_box[(x1, y1, x2, y2)].append(tag)
            for (x1, y1, x2, y2), tags in rows_by_box.items():
                box = clip_box((x1, y1, x2, y2), width, height)
                if box is None:
                    stats["degenerate_boxes"] += 1
                    continue
                mapped = [
                    LISA_TAGS.get(_normalize_tag(tag), ("unknown", "unknown"))
                    for tag in tags
                ]
                states = [map_state(state_name) for state_name, _ in mapped]
                pictograms = [
                    map_pictogram(pictogram_name) for _, pictogram_name in mapped
                ]
                state_targets = {item.target for item in states if item.valid}
                pictogram_targets = {item.target for item in pictograms if item.valid}
                valid_state = all(item.valid for item in states) and len(state_targets) == 1
                valid_pictogram = (
                    all(item.valid for item in pictograms)
                    and len(pictogram_targets) == 1
                )
                state_target = next(iter(state_targets)) if valid_state else None
                pictogram_target = (
                    next(iter(pictogram_targets)) if valid_pictogram else None
                )
                if len(tags) > 1:
                    stats["merged_same_box_rows"] += len(tags) - 1
                if len(state_targets) > 1:
                    stats["conflicting_state_boxes"] += 1
                if len(pictogram_targets) > 1:
                    stats["conflicting_pictogram_boxes"] += 1
                traffic_lights.append(
                    TrafficLightAnnotation(
                        bbox_xyxy=box,
                        state=state_target,
                        pictogram=pictogram_target,
                        relevance=None,
                        occlusion="unknown",
                        valid_state=valid_state,
                        valid_pictogram=valid_pictogram,
                        valid_relevance=False,
                        source_attributes={"tags": tags},
                    )
                )
                stats["traffic_lights"] += 1
                stats[f"state/{state_target if valid_state else 'masked'}"] += 1
                stats[
                    f"pictogram/{pictogram_target if valid_pictogram else 'masked'}"
                ] += 1
                for tag, state, pictogram in zip(tags, states, pictograms):
                    if not state.valid or not pictogram.valid:
                        stats[f"unknown_tag/{tag}"] += 1

            if not traffic_lights:
                stats["negative_images"] += 1
            record = ImageRecord(
                image_id=f"LISA/{spec.split}/{spec.name}/{image_path.stem}",
                image_path=str(image_path.resolve()),
                source_dataset="LISA",
                original_width=width,
                original_height=height,
                split=spec.split,
                sequence_id=f"LISA/{spec.name}",
                task_valid=TaskValidity(
                    traffic_light_detection=True,
                    traffic_light_state=True,
                    traffic_light_pictogram=True,
                ),
                traffic_lights=traffic_lights,
                metadata={
                    "official_split": spec.split,
                    "domain": spec.domain,
                    "annotation_file": str(spec.annotation_file.resolve()),
                },
            )
            record.validate()
            records.append(record)
            split_counts[spec.split] += 1
            stats["images"] += 1
            stats[f"images/{spec.split}"] += 1
            stats[f"domain/{spec.domain}"] += 1

    return ConversionResult(records=records, stats=stats)
