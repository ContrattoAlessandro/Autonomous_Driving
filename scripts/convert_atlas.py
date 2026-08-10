"""Prepare the native ATLAS YOLO split files.

ATLAS already ships YOLO annotations with the official 25 class IDs.  This
script only validates those annotations and writes image-list files; it never
rewrites, merges or remaps labels.
"""
from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_ATLAS_CLASSES = 25
YOLO_EDGE_TOLERANCE = 1e-4


def list_images(raw: Path, split: str, focals: list[str]) -> list[Path]:
    images: list[Path] = []
    for focal in focals:
        image_dir = raw / split / focal / "images"
        images.extend(sorted(image_dir.glob("*.jpg")))
        images.extend(sorted(image_dir.glob("*.jpeg")))
        images.extend(sorted(image_dir.glob("*.png")))
    return images


def label_for_image(image: Path) -> Path:
    # Native ATLAS layout: <focal>/images/foo.jpg and <focal>/labels/foo.txt.
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def read_class_names(raw: Path) -> dict[int, str]:
    """Read the official ATLAS_classes.yaml without changing its IDs."""
    class_file = raw / "ATLAS_classes.yaml"
    if not class_file.exists():
        raise SystemExit(f"Missing official class map: {class_file}")

    names: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)\s*:\s*(\S+)\s*$")
    for line in class_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            names[int(match.group(1))] = match.group(2)
    if sorted(names) != list(range(N_ATLAS_CLASSES)):
        raise SystemExit(
            f"{class_file} must define exactly class IDs 0..{N_ATLAS_CLASSES - 1}; "
            f"found {sorted(names)}"
        )
    return names


def validate_labels(images: list[Path], *, n_classes: int) -> tuple[int, int]:
    """Validate native labels and return (images_without_boxes, box_count)."""
    missing: list[Path] = []
    errors: list[str] = []
    empty = 0
    boxes = 0

    for image in images:
        label = label_for_image(image)
        if not label.exists():
            missing.append(label)
            continue
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty += 1
        for line_no, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"{label}:{line_no}: expected 5 fields")
                continue
            try:
                cls_value = float(fields[0])
                cx, cy, width, height = (float(v) for v in fields[1:])
            except ValueError:
                errors.append(f"{label}:{line_no}: non-numeric value")
                continue
            valid_cls = math.isfinite(cls_value) and cls_value == int(cls_value) and 0 <= int(cls_value) < n_classes
            valid_box = all(math.isfinite(v) for v in (cx, cy, width, height))
            valid_box = valid_box and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1
            valid_box = valid_box and (
                -YOLO_EDGE_TOLERANCE <= cx - width / 2
                <= cx + width / 2 <= 1 + YOLO_EDGE_TOLERANCE
            )
            valid_box = valid_box and (
                -YOLO_EDGE_TOLERANCE <= cy - height / 2
                <= cy + height / 2 <= 1 + YOLO_EDGE_TOLERANCE
            )
            if not valid_cls or not valid_box:
                errors.append(f"{label}:{line_no}: class/box outside native YOLO range")
                continue
            boxes += 1

    if missing:
        preview = ", ".join(str(p) for p in missing[:3])
        errors.append(f"missing labels for {len(missing)} image(s), e.g. {preview}")
    if errors:
        raise SystemExit("ATLAS validation failed without modifying labels:\n  " + "\n  ".join(errors[:12]))
    return empty, boxes


def write_list(out_path: Path, images: list[Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for image in images:
            # Absolute paths make the list independent of the launch directory.
            handle.write(str(image.resolve()).replace("\\", "/") + "\n")
    print(f"  wrote {out_path} ({len(images)} images)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT.parent / "dataset_ATLAS" / "ATLAS",
        help="ATLAS root containing ATLAS_classes.yaml, train/ and test/",
    )
    parser.add_argument(
        "--focals",
        type=str,
        default="front_medium,front_tele,front_wide",
        help="comma-separated camera folders",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="fraction carved from native train for validation (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "yolo" / "atlas")
    parser.add_argument("--max-train", type=int, default=None, help="optional cap for a smoke split")
    parser.add_argument("--max-val", type=int, default=None, help="optional cap for a smoke split")
    parser.add_argument("--max-test", type=int, default=None, help="optional cap for a smoke split")
    parser.add_argument(
        "--allow-test-as-val",
        action="store_true",
        help="explicitly allow the old/leaky behavior when --val-frac 0 is requested",
    )
    parser.add_argument("--skip-label-validation", action="store_true")
    args = parser.parse_args()

    raw = args.raw.resolve()
    if not raw.exists():
        raise SystemExit(f"ATLAS not found at {raw}")
    names = read_class_names(raw)
    focals = [f.strip() for f in args.focals.split(",") if f.strip()]
    if not focals:
        raise SystemExit("At least one focal folder is required")

    train_images = list_images(raw, "train", focals)
    test_images = list_images(raw, "test", focals)
    if not train_images or not test_images:
        raise SystemExit(f"No images found: train={len(train_images)} test={len(test_images)}")
    print(f"ATLAS native classes={len(names)} train={len(train_images)} test={len(test_images)} focals={focals}")
    print(f"class map: {names}")

    if not args.skip_label_validation:
        empty_train, boxes_train = validate_labels(train_images, n_classes=N_ATLAS_CLASSES)
        empty_test, boxes_test = validate_labels(test_images, n_classes=N_ATLAS_CLASSES)
        print(f"validated labels: train_boxes={boxes_train} test_boxes={boxes_test} empty={empty_train + empty_test}")

    if not 0 < args.val_frac < 1:
        if not (args.val_frac == 0 and args.allow_test_as_val):
            raise SystemExit(
                "Use 0 < --val-frac < 1 to keep validation and test separate. "
                "Pass --allow-test-as-val only for an explicit legacy run."
            )

    if args.val_frac == 0:
        val_images = list(test_images)
        train_split = list(train_images)
    else:
        rng = random.Random(args.seed)
        train_pool = list(train_images)
        rng.shuffle(train_pool)
        n_val = max(1, int(round(len(train_pool) * args.val_frac)))
        if n_val >= len(train_pool):
            raise SystemExit("Validation fraction leaves no training images")
        val_images = train_pool[:n_val]
        train_split = train_pool[n_val:]

    sets = {"train": set(train_split), "val": set(val_images), "test": set(test_images)}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[first] & sets[second]
        if overlap:
            raise SystemExit(f"Split leakage between {first}/{second}: {len(overlap)} shared images")

    if args.max_train is not None:
        train_split = train_split[: max(1, args.max_train)]
    if args.max_val is not None:
        val_images = val_images[: max(1, args.max_val)]
    if args.max_test is not None:
        test_images = test_images[: max(1, args.max_test)]

    write_list(args.out / "train.txt", train_split)
    write_list(args.out / "val.txt", val_images)
    write_list(args.out / "test.txt", test_images)
    print(
        f"Done: train={len(train_split)} val={len(val_images)} test={len(test_images)}. "
        "ATLAS labels were not rewritten."
    )


if __name__ == "__main__":
    main()
