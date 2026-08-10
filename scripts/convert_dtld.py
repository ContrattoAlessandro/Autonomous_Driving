"""convert_dtld.py — DriveU Traffic Light Dataset (DTLD) -> YOLO format.

DTLD schema (v1 YAML and v2 JSON both supported via defensive field lookup):
  top-level : list of images, each dict has 'path', 'width', 'height', 'objects'.
  object    : bounding box fields + 'attributes' dict with keys
              'direction','relevance','occlusion','orientation','state','pictogram'.
  Box coords: 'x','y','w','h' (top-left + size, pixels) per the UTD spec.
              We also accept x_min/x_max/y_min/y_max as a fallback.

State  -> Tier-B class via STATE2ID (red/yellow/green/red_yellow/off/unknown).
        Tier-C pictogram classes are derived from 'pictogram' attribute.
All    -> Tier-A single class (0 = traffic_light).

Usage:
    python scripts/convert_dtld.py --raw /path/to/DTLD
    python scripts/convert_dtld.py --raw /path/to/DTLD --splits yaml/udt2.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from _common import (
    STATE2ID, ID2STATE, ConvertStats, bbox_to_yolo, corners_to_xywh, pick,
    write_yolo_label, link_or_copy_image, yolo_root, raw_root, split_summary_append,
    add_common_args,
)

# --- Tier-C pictogram class ids (offset above Tier-B's 0..5) ---
# Final nc/names for data_tierC.yaml are confirmed by harmonize_labels.py.
PICTOGRAM2ID: dict[str, int] = {
    "arrow_left": 6,
    "arrow_straight": 7,
    "arrow_right": 8,
    "pedestrian": 9,
    "bicycle": 10,
    "tram": 11,
    "arrow_back_left": 12,
    "arrow_back_right": 13,
}

DTLD_STATE_ALIASES = {
    "red": "red", "yellow": "yellow", "green": "green",
    "red_yellow": "red_yellow", "yellow_red": "red_yellow",
    "off": "off", "unknown": "unknown", "none": "off",
}


def norm_state(s: str | None) -> str:
    if not s:
        return "unknown"
    s = str(s).strip().lower()
    return DTLD_STATE_ALIASES.get(s, "unknown")


def find_label_files(raw: Path) -> list[Path]:
    """DTLD ships one YAML/JSON label file per sequence (each lists many images).
    Locate them regardless of nesting."""
    files = []
    for pat in ("**/*.yaml", "**/*.yml", "**/*.json"):
        files.extend(p for p in raw.glob(pat) if p.name.lower() not in ("readme",))
    # dedupe by suffix priority: prefer json (v2)
    seen_stems, out = set(), []
    for p in sorted(files, key=lambda x: (x.suffix != ".json", str(x))):
        if p.stem in seen_stems:
            continue
        seen_stems.add(p.stem)
        out.append(p)
    return out


def parse_label_file(p: Path) -> list[dict]:
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    import yaml
    data = yaml.safe_load(p.read_text())
    # Some DTLD releases wrap in a top-level dict; unwrap to the image list.
    if isinstance(data, dict):
        for k in ("images", "frames", "data"):
            if isinstance(data.get(k), list):
                return data[k]
    return data or []


def image_base(img: dict) -> str:
    """Filename without extension — used as the YOLO id."""
    path = pick(img, "path", "filename", "file", "image_path") or ""
    return Path(str(path)).stem


def image_dims(img: dict) -> tuple[int, int]:
    w = pick(img, "width", "img_width", "cols")
    h = pick(img, "height", "img_height", "rows")
    if w and h:
        return int(w), int(h)
    return 0, 0  # signal: must read from image later


def object_box(obj: dict, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) in absolute pixels, or None if unusable."""
    x = pick(obj, "x", "x_min", "left")
    y = pick(obj, "y", "y_min", "top")
    w = pick(obj, "w", "width")
    h = pick(obj, "h", "height")
    if w is not None and h is not None and x is not None and y is not None:
        try:
            return bbox_to_yolo(float(x), float(y), float(w), float(h), img_w, img_h)
        except Exception:
            return None
    x_max = pick(obj, "x_max", "right")
    y_max = pick(obj, "y_max", "bottom")
    if None not in (x, y, x_max, y_max):
        try:
            xx, yy, ww, hh = corners_to_xywh(float(x), float(y), float(x_max), float(y_max))
            return bbox_to_yolo(xx, yy, ww, hh, img_w, img_h)
        except Exception:
            return None
    return None


def convert(raw: Path, split_dir: str = "train", dry: bool = False) -> ConvertStats:
    stats = ConvertStats(source=f"dtld/{split_dir}")
    out_img = yolo_root() / "dtld" / "images" / split_dir
    out_lbl = yolo_root() / "dtld" / "labels" / split_dir

    for lf in find_label_files(raw):
        for img in parse_label_file(lf):
            stats.add()
            stem = image_base(img)
            if not stem:
                stats.skipped_images += 1
                continue

            img_path = raw / (pick(img, "path", "filename", "file", "image_path") or "")
            if not img_path.exists():
                # try siblings
                cand = [lf.parent / (stem + s) for s in (".png", ".jpg", ".jpeg", ".bmp")]
                img_path = next((c for c in cand if c.exists()), img_path)

            w, h = image_dims(img)
            if (not w or not h) and img_path.exists():
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = im.size
            if not w or not h:
                stats.skipped_images += 1
                continue

            rows_a, rows_b, rows_c = [], [], []
            for obj in pick(img, "objects", "boxes", "lights", "annotations", default=[]) or []:
                attrs = pick(obj, "attributes", "attr", default={}) or {}
                box = object_box(obj, w, h)
                if box is None:
                    stats.skipped_boxes += 1
                    continue
                cx, cy, bw, bh = box

                # --- Tier A: single class ---
                rows_a.append((0, cx, cy, bw, bh))
                stats.by_class_tierA[0] += 1

                # --- Tier B: state ---
                state = norm_state(pick(attrs, "state"))
                bid = STATE2ID[state]
                rows_b.append((bid, cx, cy, bw, bh))
                stats.by_class_tierB[bid] += 1

                # --- Tier C: state, but arrows override to pictogram classes ---
                picto = str(pick(attrs, "pictogram") or "").strip().lower()
                cid = PICTOGRAM2ID.get(picto, bid)
                rows_c.append((cid, cx, cy, bw, bh))

            stats.n_boxes += len(rows_a)
            if dry:
                continue

            # write three parallel label sets (tierA/B/C) so harmonize step is trivial
            for tier, rows in (("tierA", rows_a), ("tierB", rows_b), ("tierC", rows_c)):
                lp = yolo_root() / "dtld" / tier / "labels" / split_dir / f"{stem}.txt"
                has = write_yolo_label(lp, rows)
                if not has:
                    stats.n_empty_label += 0  # tracked per-image below

            if not rows_a:
                stats.n_empty_label += 1
            if img_path.exists():
                link_or_copy_image(img_path, out_img / img_path.name)

    print(stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Convert DTLD -> YOLO (tier A/B/C)")
    add_common_args(p)
    p.add_argument("--splits", type=Path, default=None,
                   help="optional DTLD split YAML (map split->list of label files)")
    args = p.parse_args()
    raw = args.raw or (raw_root() / "dtld")
    if not raw.exists():
        raise SystemExit(f"DTLD raw dir not found: {raw}. Download from https://www.traffic-light-data.com/")

    # DTLD has no official train/val/test split. We dump everything to 'train'
    # here; harmonize_labels.py performs the stratified re-split afterward.
    stats = convert(raw, split_dir="train", dry=args.dry_run)
    split_summary_append(stats)


if __name__ == "__main__":
    main()
