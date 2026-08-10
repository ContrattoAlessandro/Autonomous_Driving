"""_common.py — shared utilities for dataset converters.

Every converter turns a public traffic-light dataset into the canonical YOLO
layout used by this project:

    datasets/yolo/<source>/
        images/{train,val,test}/<id>.<ext>
        labels/{train,val,test}/<id>.txt      # one row per box: "cls cx cy w h"

Key design choices:
  * Labels are written as a DICT per image and finalized with write_yolo_label(),
    so converters can be tolerant of missing/odd native fields.
  * A small normalization layer (bbox_to_yolo, pick) lets each converter be
    defensive about exact field names, which vary across DTLD v1/v2,
    Bosch README vs shipped YAML, and LISA clip variants.
  * The convert_*() entry points are CLI-callable and idempotent (re-running
    overwrites outputs).
"""
from __future__ import annotations
import argparse
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Tier-B unified state class ids (must match configs/data_tierB.yaml).
STATE2ID: dict[str, int] = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "red_yellow": 3,
    "off": 4,
    "unknown": 5,
}
ID2STATE: dict[int, str] = {v: k for k, v in STATE2ID.items()}

# Tier-A single class.
DETECT_ID = 0  # "traffic_light"


@dataclass
class ConvertStats:
    """Counter container, printed + dumped to results at the end of each run."""
    source: str
    n_images: int = 0
    n_boxes: int = 0
    n_empty_label: int = 0          # images with no boxes (YOLO needs an empty .txt)
    by_class_tierA: dict = field(default_factory=lambda: {0: 0})
    by_class_tierB: dict = field(default_factory=lambda: {c: 0 for c in STATE2ID.values()})
    skipped_boxes: int = 0
    skipped_images: int = 0

    def add(self) -> None:
        self.n_images += 1

    def __str__(self) -> str:
        return (
            f"[{self.source}] images={self.n_images} boxes={self.n_boxes} "
            f"empty={self.n_empty_label} skipped_img={self.skipped_images} "
            f"skipped_box={self.skipped_boxes} tierA={self.by_class_tierA} "
            f"tierB={self.by_class_tierB}"
        )


def pick(d: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present value among `keys` in dict `d` (case-insensitive
    on the key name). Handles schema drift between DTLD v1/v2, Bosch variants."""
    lower = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return default


def bbox_to_yolo(x: float, y: float, w: float, h: float, img_w: float, img_h: float) -> tuple[float, float, float, float]:
    """Convert absolute (x, y, w, h) [top-left, pixels] -> normalized YOLO (cx, cy, w, h)."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("image dimensions must be positive")
    # Clip both corners, not only width/height.  Clipping just w/h leaves a
    # box whose center is in range but whose edge is still outside the image.
    x1 = max(0.0, min(float(x), float(img_w)))
    y1 = max(0.0, min(float(y), float(img_h)))
    x2 = max(0.0, min(float(x + w), float(img_w)))
    y2 = max(0.0, min(float(y + h), float(img_h)))
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("degenerate box")
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    return cx, cy, w / img_w, h / img_h


def corners_to_xywh(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max) -> (x, y, w, h) top-left + size."""
    x, y = min(x_min, x_max), min(y_min, y_max)
    return x, y, abs(x_max - x_min), abs(y_max - y_min)


def write_yolo_label(label_path: Path, rows: Sequence[Sequence[float]]) -> bool:
    """Write a YOLO label file. rows = list of (cls, cx, cy, w, h).
    An image with no boxes still gets an (empty) .txt file — required by Ultralytics.
    Returns True if the file has >= 1 box, False if empty."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for r in rows:
            if len(r) != 5:
                raise ValueError(f"YOLO row must have 5 values, got {r}")
            cls, cx, cy, w, h = (float(v) for v in r)
            if not math.isfinite(cls) or cls != int(cls):
                raise ValueError(f"class id must be a finite integer, got {r[0]}")
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError(f"non-finite YOLO row: {r}")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"normalized YOLO row outside range: {r}")
            if not (0 <= cx - w / 2 <= cx + w / 2 <= 1 and 0 <= cy - h / 2 <= cy + h / 2 <= 1):
                raise ValueError(f"YOLO box crosses image boundary: {r}")
            f.write(" ".join(f"{v:.6g}" for v in r) + "\n")
    return len(rows) > 0


def link_or_copy_image(src: Path, dst: Path) -> None:
    """Hardlink if possible (cheap on same volume), else symlink, else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)        # py3.10+
        return
    except Exception:
        pass
    try:
        os_symlink(src.resolve(), dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def os_symlink(target: Path, link: Path) -> None:
    import os
    os.symlink(target, link)


def yolo_root() -> Path:
    """datasets/yolo relative to this file."""
    return Path(__file__).resolve().parent.parent / "datasets" / "yolo"


def raw_root() -> Path:
    """datasets/raw relative to this file."""
    return Path(__file__).resolve().parent.parent / "datasets" / "raw"


def split_summary_append(stats: ConvertStats) -> None:
    """Append one converter's stats to results/splits_summary.csv (create header if new)."""
    import csv
    out = Path(__file__).resolve().parent.parent / "results" / "splits_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source", "n_images", "n_boxes", "n_empty", "skipped_img", "skipped_box", "tierA", "tierB"])
        w.writerow([stats.source, stats.n_images, stats.n_boxes, stats.n_empty_label,
                    stats.skipped_images, stats.skipped_boxes, stats.by_class_tierA, stats.by_class_tierB])


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw", type=Path, default=None, help="raw dataset root (default: datasets/raw/<source>)")
    p.add_argument("--dry-run", action="store_true", help="scan only, write nothing")
