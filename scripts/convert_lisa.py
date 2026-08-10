"""convert_lisa.py — LISA Traffic Light Dataset -> YOLO format.

Verified schema (from the LISA docs + community YOLO converters):
  Each clip has a CSV: 'frameAnnotationsBOX.csv' with header
    Filename, Annotation Tag, Upper left corner X, Upper right corner Y,
    Lower right corner X, Lower right corner Y, Origin frame number, Origin frame,
    Origin track, Origin track frame number
  (header names vary in casing/spelling across releases -> looked up defensively.)
  'Annotation tag' values (6 used in practice):
    go, goLeft, goForward, stop, stopLeft, warning, warningLeft
  Day/night test split: the dayTrain/ nightTrain/ dayTest/ nightTest/ folders.
    dayTest/nightTest images -> 'test'; dayTrain/nightTrain -> 'train'.

Usage:
    python scripts/convert_lisa.py --raw /path/to/LISA
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from _common import (
    STATE2ID, ConvertStats, bbox_to_yolo, corners_to_xywh,
    write_yolo_label, link_or_copy_image, yolo_root, raw_root, split_summary_append,
    add_common_args,
)

LISA_TAG2STATE = {
    "go": "green", "goleft": "green", "goforward": "green",
    "stop": "red", "stopleft": "red",
    "warning": "yellow", "warningleft": "yellow",
}


def norm_tag(tag: str) -> str:
    return LISA_TAG2STATE.get(tag.strip().lower().replace(" ", ""), "unknown")


def find_csvs(raw: Path) -> list[Path]:
    return sorted(raw.glob("**/frameAnnotationsBOX.csv")) + sorted(raw.glob("**/frameAnnotations*.csv"))


def split_for(csv_path: Path) -> str:
    p = str(csv_path).lower().replace("\\", "/")
    if "daytest/" in p or "nighttest/" in p or "/test/" in p:
        return "test"
    return "train"


def image_dims(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def convert(raw: Path, dry: bool = False) -> ConvertStats:
    stats = ConvertStats(source="lisa/ALL")
    # image -> list of (tag, xmin, ymin, xmax, ymax), keyed by the image filename
    by_image: dict[str, list[tuple]] = defaultdict(list)
    img_to_split: dict[str, str] = {}

    for csvf in find_csvs(raw):
        split = split_for(csvf)
        with csvf.open(newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            for row in reader:
                fname = (row.get(cols.get("filename", "Filename")) or "").strip()
                tag = (row.get(cols.get("annotation tag", "Annotation tag")) or "").strip()
                if not fname or not tag:
                    continue
                try:
                    xmin = float(row[cols["upper left corner x"]])
                    ymin = float(row[cols["upper left corner y"]])
                    xmax = float(row[cols["lower right corner x"]])
                    ymax = float(row[cols["lower right corner y"]])
                except (KeyError, ValueError):
                    stats.skipped_boxes += 1
                    continue
                # LISA filenames sometimes repeat with a leading path; normalize.
                fname = fname.replace("\\", "/").split("/")[-1]
                by_image[fname].append((tag, xmin, ymin, xmax, ymax))
                img_to_split[fname] = split

    # locate image files on disk
    img_index = {}
    for ext in (".png", ".jpg", ".jpeg"):
        for p in raw.glob(f"**/*{ext}"):
            img_index[p.name.lower()] = p

    for fname, boxes in by_image.items():
        stats.add()
        img_path = img_index.get(fname.lower())
        if not img_path:
            stats.skipped_images += 1
            continue
        try:
            w, h = image_dims(img_path)
        except Exception:
            stats.skipped_images += 1
            continue
        stem = img_path.stem
        split = img_to_split[fname]

        rows_a, rows_b, rows_c = [], [], []
        for tag, xmin, ymin, xmax, ymax in boxes:
            try:
                x, y, ww, hh = corners_to_xywh(xmin, ymin, xmax, ymax)
                cx, cy, bw, bh = bbox_to_yolo(x, y, ww, hh, w, h)
            except Exception:
                stats.skipped_boxes += 1
                continue
            rows_a.append((0, cx, cy, bw, bh))
            stats.by_class_tierA[0] += 1
            state = norm_tag(tag)
            bid = STATE2ID[state]
            rows_b.append((bid, cx, cy, bw, bh))
            rows_c.append((bid, cx, cy, bw, bh))
            stats.by_class_tierB[bid] += 1

        stats.n_boxes += len(rows_a)
        if not rows_a:
            stats.n_empty_label += 1
        if dry:
            continue

        for tier, rows in (("tierA", rows_a), ("tierB", rows_b), ("tierC", rows_c)):
            lp = yolo_root() / "lisa" / tier / "labels" / split / f"{stem}.txt"
            write_yolo_label(lp, rows)
        link_or_copy_image(img_path, yolo_root() / "lisa" / "images" / split / img_path.name)

    print(stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Convert LISA TL -> YOLO (tier A/B/C)")
    add_common_args(p)
    args = p.parse_args()
    raw = args.raw or (raw_root() / "lisa")
    if not raw.exists():
        raise SystemExit(f"LISA raw dir not found: {raw}. See http://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html")
    stats = convert(raw, dry=args.dry_run)
    split_summary_append(stats)


if __name__ == "__main__":
    main()
