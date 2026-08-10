"""eda.py — exploratory analysis of the unified YOLO dataset.

Produces (into results/eda/):
  * class_distribution_<tier>.png   bar chart of class counts
  * bbox_size_hist_<tier>.png       histogram of box areas (sqrt(px*px)) on log scale
  * size_bins_<tier>.csv            COCO-style small/medium/large object counts
  * bbox_scatter_<tier>.png         w vs h scatter (log-log) — visualizes the P2 head target
  * results/eda/summary_<tier>.csv  per-class counts + % small objects

The headline number reported in the README / thesis is the **% of objects with
sqrt(area) < 32 px** (COCO "small"): a high fraction (>40%) justifies the P2 head
and imgsz=1280.

Usage:
    python scripts/eda.py --tier B
    python scripts/eda.py --tier all
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from _common import yolo_root

TIERS = ("tierA", "tierB", "tierC")
SIZE_BINS = [(0, 16), (16, 32), (32, 96), (96, 1 << 16)]   # sqrt(area) px edges
BIN_NAMES = ["8-16", "16-32", "32-96", ">96"]              # midpoints for display
ID2STATE = {0: "red", 1: "yellow", 2: "green", 3: "red_yellow",
            4: "off", 5: "unknown",
            6: "arrow_left", 7: "arrow_straight", 8: "arrow_right", 9: "pedestrian",
            10: "bicycle", 11: "tram", 12: "arrow_back_left", 13: "arrow_back_right"}


def image_size_from_path(img: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(img) as im:
        return im.size


def collect(tier: str) -> tuple[Counter, list[float], Counter, int, int]:
    """Return (class_counts, sqrt_areas, size_bin_counts, n_imgs, n_imgs_no_label)."""
    classes = Counter()
    sqrt_areas: list[float] = []
    bins = Counter()
    n_imgs = n_empty = 0
    seen_imgs = set()

    lbl_root = yolo_root() / tier / "labels"
    for split in ("train", "val", "test"):
        for lbl in (lbl_root / split).glob("*.txt"):
            # find the image
            img = None
            for cand in (yolo_root() / tier / "images" / split).glob(f"{lbl.stem}.*"):
                img = cand
                break
            if img is None:
                for src in ("dtld", "bosch", "lisa", "openimages"):
                    cand_dir = yolo_root() / src / "images" / split
                    for ext in (".png", ".jpg", ".jpeg"):
                        c = cand_dir / f"{lbl.stem}{ext}"
                        if c.exists():
                            img = c
                            break
                    if img:
                        break
            if img is None or img in seen_imgs:
                continue
            seen_imgs.add(img)
            n_imgs += 1
            try:
                W, H = image_size_from_path(img)
            except Exception:
                continue
            rows = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
            if not rows:
                n_empty += 1
            for r in rows:
                cls = int(float(r[0]))
                _, _, w, h = (float(x) for x in r[1:5])
                area_sqrt = (w * W) ** 0.5 * (h * H) ** 0.5  # geometric mean side
                classes[cls] += 1
                sqrt_areas.append(area_sqrt)
                for i, (lo, hi) in enumerate(SIZE_BINS):
                    if lo <= area_sqrt < hi:
                        bins[BIN_NAMES[i]] += 1
                        break
    return classes, sqrt_areas, bins, n_imgs, n_empty


def run(tier: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = yolo_root().parent / "results" / "eda"
    out.mkdir(parents=True, exist_ok=True)

    classes, sqrt_areas, bins, n_imgs, n_empty = collect(tier)
    if not sqrt_areas:
        print(f"[{tier}] no labels found under {yolo_root() / tier} — run converters first.")
        return

    n_total = sum(classes.values())
    n_small = bins["8-16"] + bins["16-32"]
    pct_small = 100.0 * n_small / max(1, n_total)
    print(f"[{tier}] images={n_imgs} (empty={n_empty}) objects={n_total} "
          f"small(<32px)={n_small} ({pct_small:.1f}%)")

    # --- class distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = sorted(classes)
    ax.bar([ID2STATE.get(k, str(k)) for k in keys], [classes[k] for k in keys])
    ax.set_ylabel("count")
    ax.set_title(f"{tier} — class distribution  (N={n_total})")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / f"class_distribution_{tier}.png", dpi=130)
    plt.close(fig)

    # --- bbox size histogram (log x) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sqrt_areas, bins=np.logspace(0.7, 2.6, 60))
    ax.set_xscale("log")
    ax.axvline(32, color="r", ls="--", label="COCO small boundary (32 px)")
    ax.axvline(16, color="orange", ls="--", label="xsmall (16 px)")
    ax.set_xlabel("geometric-mean box side  √(w·h)  [px]")
    ax.set_ylabel("# boxes")
    ax.set_title(f"{tier} — object size distribution  ({pct_small:.0f}% < 32 px)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"bbox_size_hist_{tier}.png", dpi=130)
    plt.close(fig)

    # --- size bins CSV + small-object summary ---
    with (out / f"size_bins_{tier}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size_bin", "count", "fraction"])
        for name in BIN_NAMES:
            c = bins[name]
            w.writerow([name, c, f"{c / max(1, n_total):.4f}"])
        w.writerow([])
        w.writerow(["total_objects", n_total])
        w.writerow(["pct_small_lt_32", f"{pct_small:.2f}"])

    # --- per-class CSV ---
    with (out / f"summary_{tier}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "count", "fraction"])
        for k in keys:
            w.writerow([k, ID2STATE.get(k, str(k)), classes[k], f"{classes[k] / max(1, n_total):.4f}"])

    print(f"[{tier}] wrote plots/csv to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA over the unified YOLO dataset")
    ap.add_argument("--tier", choices=TIERS + ["all"], default="all")
    args = ap.parse_args()
    tiers = TIERS if args.tier == "all" else (args.tier,)
    for t in tiers:
        run(t)


if __name__ == "__main__":
    main()
