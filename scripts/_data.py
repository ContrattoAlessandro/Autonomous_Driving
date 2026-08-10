"""Shared dataset-path and YOLO-label validation helpers.

The training entry point still delegates all augmentation, batching and loss
handling to Ultralytics.  This module only makes the dataset contract
explicit: paths are resolved once, split files are checked, and labels are
validated without rewriting them.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
YOLO_EDGE_TOLERANCE = 1e-4


def resolve_project_path(value: str | Path, *, base: Path = ROOT) -> Path:
    """Resolve a path from the current directory or from the project root."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    candidates = (Path.cwd() / path, base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Keep the project-root interpretation for useful error messages.
    return (base / path).resolve()


def _resolve_dataset_root(data_yaml: Path, raw_path: Any) -> Path:
    if raw_path in (None, ""):
        return data_yaml.parent.resolve()

    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()

    candidates = (
        Path.cwd() / path,
        data_yaml.parent / path,
        ROOT / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def _split_list_path(dataset_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        return None
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def read_image_list(list_file: Path, dataset_root: Path) -> list[Path]:
    """Read an Ultralytics image-list file and return absolute image paths."""
    if not list_file.exists():
        raise FileNotFoundError(f"Split list does not exist: {list_file}")

    images: list[Path] = []
    missing: list[Path] = []
    for raw_line in list_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        image = Path(line)
        if not image.is_absolute():
            image = dataset_root / image
        image = image.resolve()
        if image.exists():
            images.append(image)
        else:
            missing.append(image)

    if missing:
        preview = ", ".join(str(p) for p in missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} image path(s) from {list_file} do not exist; "
            f"examples: {preview}"
        )
    return images


def label_for_image(image: Path) -> Path:
    """Return the canonical YOLO label path for an image path."""
    parts = list(image.parts)
    image_dir_indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if image_dir_indices:
        i = image_dir_indices[-1]
        return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
    return image.with_suffix(".txt")


def _validate_label_file(label: Path, nc: int | None) -> int:
    errors: list[str] = []
    n_boxes = 0
    try:
        lines = label.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label.read_text().splitlines()

    for line_no, raw_line in enumerate(lines, 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"line {line_no}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_value = float(fields[0])
            values = [float(v) for v in fields[1:]]
        except ValueError:
            errors.append(f"line {line_no}: non-numeric value")
            continue

        if not math.isfinite(class_value) or class_value != int(class_value):
            errors.append(f"line {line_no}: class id is not an integer")
        elif nc is not None and not 0 <= int(class_value) < nc:
            errors.append(f"line {line_no}: class id {int(class_value)} outside [0,{nc - 1}]")

        if any(not math.isfinite(v) for v in values):
            errors.append(f"line {line_no}: non-finite box value")
            continue
        cx, cy, width, height = values
        if width <= 0 or height <= 0:
            errors.append(f"line {line_no}: width/height must be positive")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"line {line_no}: normalized values outside YOLO range")
        if not (
            -YOLO_EDGE_TOLERANCE <= cx - width / 2
            <= cx + width / 2 <= 1 + YOLO_EDGE_TOLERANCE
        ):
            errors.append(f"line {line_no}: x bounds outside image")
        if not (
            -YOLO_EDGE_TOLERANCE <= cy - height / 2
            <= cy + height / 2 <= 1 + YOLO_EDGE_TOLERANCE
        ):
            errors.append(f"line {line_no}: y bounds outside image")
        n_boxes += 1

    if errors:
        raise ValueError(f"Invalid YOLO label {label}: " + "; ".join(errors[:4]))
    return n_boxes


def validate_data_config(data_yaml: Path, *, validate_labels: bool = True) -> tuple[Path, dict[str, Any]]:
    """Validate a dataset YAML and return an absolute, trainer-ready YAML."""
    import yaml

    source = data_yaml.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source}")

    nc = data.get("nc")
    nc_int = int(nc) if nc is not None else None
    names = data.get("names")
    if nc_int is not None and names is not None:
        if isinstance(names, dict):
            keys = sorted(int(k) for k in names)
            if keys != list(range(nc_int)):
                raise ValueError(f"{source}: names keys must be exactly 0..{nc_int - 1}")
            if any(not isinstance(value, str) for value in names.values()):
                raise ValueError(
                    f"{source}: every class name must be quoted text; YAML may parse names such as 'off' as booleans"
                )
        elif len(names) != nc_int:
            raise ValueError(f"{source}: nc={nc_int} but names has {len(names)} entries")
        elif any(not isinstance(value, str) for value in names):
            raise ValueError(f"{source}: every class name must be quoted text")

    dataset_root = _resolve_dataset_root(source, data.get("path"))
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root} (from {source})")
    data["path"] = str(dataset_root)

    split_images: dict[str, list[Path]] = {}
    for split in SPLITS:
        list_file = _split_list_path(dataset_root, data.get(split))
        if list_file is None:
            continue
        if not list_file.exists():
            raise FileNotFoundError(f"{split} list does not exist: {list_file}")
        data[split] = str(list_file)
        split_images[split] = read_image_list(list_file, dataset_root)

    if not split_images.get("train"):
        raise ValueError(f"Training split is empty in {source}")
    if not split_images.get("val"):
        raise ValueError(f"Validation split is empty in {source}")

    train_set = set(split_images.get("train", []))
    val_set = set(split_images.get("val", []))
    test_set = set(split_images.get("test", []))
    for first_name, first_set, second_name, second_set in (
        ("train", train_set, "val", val_set),
        ("train", train_set, "test", test_set),
        ("val", val_set, "test", test_set),
    ):
        overlap = first_set & second_set
        if overlap:
            sample = next(iter(overlap))
            raise ValueError(
                f"Dataset leakage: {first_name}/{second_name} share {len(overlap)} image(s); "
                f"example: {sample}"
            )

    if validate_labels:
        for split, images in split_images.items():
            for image in images:
                label = label_for_image(image)
                if label.exists():
                    _validate_label_file(label, nc_int)

    resolved_dir = ROOT / "runs" / "_resolved_data"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolved_dir / source.name
    resolved.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolved, data
