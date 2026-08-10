"""Prepare the native ATLAS YOLO split files.

ATLAS already ships YOLO annotations with the official 25 class IDs.  This
script only validates those annotations and writes image-list files; it never
rewrites, merges or remaps labels.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_ATLAS_CLASSES = 25
YOLO_EDGE_TOLERANCE = 1e-4
ATLAS_FILENAME_RE = re.compile(
    r"^(front_(?:medium|tele|wide))_(\d+)-(\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
DEFAULT_TEMPORAL_BLOCK_SECONDS = 30.0
DEFAULT_TEMPORAL_GUARD_BLOCKS = 1


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


def image_timestamp_ns(image: Path) -> int:
    """Return the timestamp encoded in an ATLAS filename.

    ATLAS filenames contain a seconds/subseconds pair.  Some files have a
    subsecond field larger than 1e9; adding the fields (rather than parsing a
    decimal string) keeps ordering correct for both forms.
    """
    match = ATLAS_FILENAME_RE.match(image.name)
    if not match:
        raise SystemExit(
            f"Cannot derive a timestamp from ATLAS filename {image.name!r}; "
            "temporal splitting requires the native timestamp naming scheme."
        )
    return int(match.group(2)) * 1_000_000_000 + int(match.group(3))


def content_hash_groups(images: list[Path]) -> dict[str, list[Path]]:
    """Group images by SHA-256 so split checks also cover renamed copies."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        digest = hashlib.sha256()
        with image.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        groups[digest.hexdigest()].append(image)
    return {key: sorted(value) for key, value in groups.items()}


def deduplicate_train_images(
    groups: dict[str, list[Path]],
) -> tuple[list[Path], int, int]:
    """Remove duplicate content from the native train pool.

    If the same bytes have conflicting or missing labels, all copies are
    dropped rather than arbitrarily selecting one annotation.  This keeps the
    native test split untouched while preventing a duplicate from crossing the
    generated train/validation split.
    """
    kept: list[Path] = []
    dropped = 0
    ambiguous = 0
    for members in groups.values():
        if len(members) == 1:
            kept.append(members[0])
            continue

        label_contents: list[bytes] = []
        labels_complete = True
        for image in members:
            label = label_for_image(image)
            if not label.exists():
                labels_complete = False
                break
            label_contents.append(label.read_bytes())

        if not labels_complete or len(set(label_contents)) != 1:
            ambiguous += len(members)
            dropped += len(members)
            print(
                "  dropped duplicate-content group with non-identical/missing labels: "
                + ", ".join(image.name for image in members)
            )
            continue

        kept.append(members[0])
        dropped += len(members) - 1

    return sorted(kept), dropped, ambiguous


def temporal_train_val_split(
    images: list[Path],
    *,
    val_frac: float,
    seed: int,
    block_seconds: float,
    guard_blocks: int,
) -> tuple[list[Path], list[Path], int, int]:
    """Split native ATLAS training images into isolated temporal blocks.

    The ATLAS paper states that images are captured synchronously but sampled
    sparsely to avoid near-identical annotations.  We preserve that intent for
    the trainer's validation split: all cameras sharing a time bucket stay in
    the same split, and one neighbouring bucket on each side is excluded from
    training as an embargo.  The official native test split is never touched.
    """
    block_ns = int(round(block_seconds * 1_000_000_000))
    if block_ns <= 0:
        raise ValueError("block_seconds must be positive")
    if guard_blocks < 0:
        raise ValueError("guard_blocks must be non-negative")

    blocks: dict[int, list[Path]] = defaultdict(list)
    for image in images:
        bucket = image_timestamp_ns(image) // block_ns
        blocks[bucket].append(image)

    keys = list(blocks)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target = max(1, int(round(len(images) * val_frac)))
    selected: set[int] = set()
    unavailable: set[int] = set()
    val_count = 0
    for bucket in keys:
        if bucket in unavailable:
            continue
        selected.add(bucket)
        val_count += len(blocks[bucket])
        unavailable.update(
            range(bucket - guard_blocks, bucket + guard_blocks + 1)
        )
        if val_count >= target:
            break

    if not selected:
        raise SystemExit("Temporal split could not select a validation block")

    val = sorted(image for bucket in selected for image in blocks[bucket])
    train = sorted(
        image
        for bucket, bucket_images in blocks.items()
        if bucket not in unavailable
        for image in bucket_images
    )
    excluded = len(images) - len(train) - len(val)
    if not train or not val:
        raise SystemExit(
            "Temporal validation split leaves no training or validation images; "
            "reduce the guard or validation fraction."
        )
    return train, val, excluded, len(selected)


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
        help="fraction of the native train pool targeted for validation (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temporal-block-seconds",
        type=float,
        default=DEFAULT_TEMPORAL_BLOCK_SECONDS,
        help=(
            "length of temporal blocks used for train/val splitting; all ATLAS "
            "cameras in a block stay together (default: 30)"
        ),
    )
    parser.add_argument(
        "--temporal-guard-blocks",
        type=int,
        default=DEFAULT_TEMPORAL_GUARD_BLOCKS,
        help=(
            "neighbouring temporal blocks excluded around validation blocks "
            "as an embargo (default: 1)"
        ),
    )
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

    if args.temporal_block_seconds <= 0:
        raise SystemExit("--temporal-block-seconds must be positive")
    if args.temporal_guard_blocks < 0:
        raise SystemExit("--temporal-guard-blocks cannot be negative")

    print("checking native train/test content hashes ...")
    train_hash_groups = content_hash_groups(train_images)
    test_hash_groups = content_hash_groups(test_images)
    shared_content = set(train_hash_groups) & set(test_hash_groups)
    if shared_content:
        raise SystemExit(
            "Native ATLAS train/test content leakage: "
            f"{len(shared_content)} identical image hash(es) are shared."
        )

    original_train_count = len(train_images)
    train_images, duplicate_count, ambiguous_duplicate_count = deduplicate_train_images(
        train_hash_groups
    )
    if duplicate_count:
        print(
            f"deduplicated native train: removed={duplicate_count} "
            f"ambiguous={ambiguous_duplicate_count} "
            f"remaining={len(train_images)}/{original_train_count}"
        )

    if args.val_frac == 0:
        val_images = list(test_images)
        train_split = list(train_images)
        excluded_guard = 0
        selected_blocks = 0
    else:
        train_split, val_images, excluded_guard, selected_blocks = temporal_train_val_split(
            train_images,
            val_frac=args.val_frac,
            seed=args.seed,
            block_seconds=args.temporal_block_seconds,
            guard_blocks=args.temporal_guard_blocks,
        )
        print(
            f"temporal split: blocks={selected_blocks} "
            f"block_seconds={args.temporal_block_seconds:g} "
            f"guard_blocks={args.temporal_guard_blocks} "
            f"guard_excluded={excluded_guard}"
        )

    sets = {"train": set(train_split), "val": set(val_images), "test": set(test_images)}
    split_pairs = (("train", "val"), ("train", "test"))
    if args.val_frac != 0:
        split_pairs += (("val", "test"),)
    for first, second in split_pairs:
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
        f"guard_excluded={excluded_guard}. ATLAS labels were not rewritten."
    )


if __name__ == "__main__":
    main()
