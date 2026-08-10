"""convert_oi.py — Open Images V7 'Traffic light' class -> YOLO format (Tier A only).

Open Images has NO traffic-light STATE labels, so this script writes ONLY the
Tier-A (single class 'traffic_light') label set. It contributes to the detection
baseline / hard-negative corpus; it is excluded from Tier B and Tier C.

Uses `fiftyone` to download + filter, which is the simplest robust path (handles
the Open Images CSV metadata + GCS image download).

Usage:
    python scripts/convert_oi.py --max-images 8000        # cap for disk/time
    python scripts/convert_oi.py --splits train,validation,test
"""
from __future__ import annotations
import argparse
from pathlib import Path

from _common import (
    ConvertStats, bbox_to_yolo, write_yolo_label, link_or_copy_image,
    yolo_root, split_summary_append, add_common_args,
)


def split_map(oi_split: str) -> str:
    return {"train": "train", "validation": "val", "test": "test"}.get(oi_split, "train")


def convert(max_images: int | None, splits: list[str], dry: bool) -> ConvertStats:
    import fiftyone as fo
    stats = ConvertStats(source="openimages/ALL")

    for oi_split in splits:
        print(f"[oi] loading split '{oi_split}' (this hits the network)...")
        ds = fo.zoo.load_zoo_dataset(
            "open-images-v7",
            split=oi_split,
            label_types=["detections"],
            classes=["Traffic light"],
            max_samples=max_images,
            dataset_name=f"oi_tl_{oi_split}",
        )
        ysplit = split_map(oi_split)
        out_img = yolo_root() / "openimages" / "images" / ysplit

        for sample in ds:
            stats.add()
            det = sample.ground_truth
            if det is None or len(det.detections) == 0:
                stats.n_empty_label += 1
                continue
            w, h = sample.metadata.width, sample.metadata.height
            stem = Path(sample.filepath).stem
            rows_a = []
            for d in det.detections:
                bx, by, bw, bh = d.bounding_box  # fiftyone: relative [x,y,w,h] already
                # ensure clamped
                try:
                    # already normalized — re-clamp via bbox_to_yolo using px form
                    cx, cy, nw, nh = bbox_to_yolo(bx * w, by * h, bw * w, bh * h, w, h)
                except Exception:
                    stats.skipped_boxes += 1
                    continue
                rows_a.append((0, cx, cy, nw, nh))
                stats.by_class_tierA[0] += 1
            stats.n_boxes += len(rows_a)
            if dry:
                continue
            lp = yolo_root() / "openimages" / "tierA" / "labels" / ysplit / f"{stem}.txt"
            write_yolo_label(lp, rows_a)
            link_or_copy_image(Path(sample.filepath), out_img / Path(sample.filepath).name)

    print(stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Open Images V7 'Traffic light' -> YOLO (Tier A only)")
    add_common_args(p)
    p.add_argument("--max-images", type=int, default=6000,
                   help="cap per Open Images split (default 6000)")
    p.add_argument("--splits", type=str, default="train,validation,test",
                   help="comma-separated Open Images splits")
    args = p.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    stats = convert(args.max_images, splits, args.dry_run)
    split_summary_append(stats)


if __name__ == "__main__":
    main()
