"""convert_bosch.py — Bosch Small Traffic Lights Dataset (BSTLD) -> YOLO format.

Verified schema (from bosch-ros-pkg/bstld read_label_file.py + README):
  top-level : list of dicts, each = one image with 'path' and 'boxes'.
  box       : 'x_min','x_max','y_min','y_max' (pixel ints, top-left/bottom-right)
              + 'label' field with state string.
  states    : 'Green','Yellow','Red','Off'  (exact casing from the README results table).

Split files live in label_files/ (e.g. test.yaml, train.yaml). We map each
.yaml to a YOLO split of the same stem.

Usage:
    python scripts/convert_bosch.py --raw /path/to/bstld
"""
from __future__ import annotations
import argparse
from pathlib import Path

from _common import (
    STATE2ID, ConvertStats, bbox_to_yolo, corners_to_xywh, pick,
    write_yolo_label, link_or_copy_image, yolo_root, raw_root, split_summary_append,
    add_common_args,
)

BOSCH_STATE_ALIASES = {
    "green": "green", "yellow": "yellow", "red": "red", "off": "off",
}


def norm_state(s: str | None) -> str:
    if not s:
        return "unknown"
    s = str(s).strip().lower()
    return BOSCH_STATE_ALIASES.get(s, "unknown")


def label_files(raw: Path) -> list[Path]:
    out = list((raw / "label_files").glob("*.yaml")) + list((raw / "label_files").glob("*.yml"))
    if not out:
        out = list(raw.glob("**/*.yaml"))
    return sorted(out)


def split_of(lf: Path) -> str:
    """Map a label-file stem to a split name. Bosch convention:
       'test.yaml' -> test, anything else -> train."""
    name = lf.stem.lower()
    if name.startswith("test"):
        return "test"
    if name.startswith("val"):
        return "val"
    return "train"


def image_dims(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def convert(raw: Path, dry: bool = False) -> ConvertStats:
    stats = ConvertStats(source="bosch/ALL")
    seen = set()

    for lf in label_files(raw):
        split = split_of(lf)
        import yaml
        images = yaml.safe_load(lf.read_text()) or []
        for img in images:
            stats.add()
            rel = pick(img, "path", "filename") or ""
            img_path = raw / rel
            stem = Path(rel).stem
            if not img_path.exists() or stem in seen:
                stats.skipped_images += (stem in seen)
                if stem in seen:
                    continue
            seen.add(stem)

            try:
                w, h = image_dims(img_path)
            except Exception:
                stats.skipped_images += 1
                continue

            rows_a, rows_b, rows_c = [], [], []
            for box in pick(img, "boxes", "objects", default=[]) or []:
                try:
                    x, y, ww, hh = corners_to_xywh(
                        float(pick(box, "x_min", "left")),
                        float(pick(box, "y_min", "top")),
                        float(pick(box, "x_max", "right")),
                        float(pick(box, "y_max", "bottom")),
                    )
                    cx, cy, bw, bh = bbox_to_yolo(x, y, ww, hh, w, h)
                except Exception:
                    stats.skipped_boxes += 1
                    continue

                rows_a.append((0, cx, cy, bw, bh))
                stats.by_class_tierA[0] += 1

                state = norm_state(pick(box, "label", "state", "class"))
                bid = STATE2ID[state]
                rows_b.append((bid, cx, cy, bw, bh))
                rows_c.append((bid, cx, cy, bw, bh))   # no pictograms in Bosch
                stats.by_class_tierB[bid] += 1

            stats.n_boxes += len(rows_a)
            if not rows_a:
                stats.n_empty_label += 1
            if dry:
                continue

            for tier, rows in (("tierA", rows_a), ("tierB", rows_b), ("tierC", rows_c)):
                lp = yolo_root() / "bosch" / tier / "labels" / split / f"{stem}.txt"
                write_yolo_label(lp, rows)
            link_or_copy_image(img_path, yolo_root() / "bosch" / "images" / split / img_path.name)

    print(stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Bosch STL -> YOLO (tier A/B/C)")
    add_common_args(p)
    args = p.parse_args()
    raw = args.raw or (raw_root() / "bosch")
    if not raw.exists():
        raise SystemExit(f"Bosch raw dir not found: {raw}. See https://hci.iwr.uni-heidelberg.de/node/6132")
    stats = convert(raw, dry=args.dry_run)
    split_summary_append(stats)


if __name__ == "__main__":
    main()
