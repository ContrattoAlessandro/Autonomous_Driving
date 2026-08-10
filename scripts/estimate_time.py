"""estimate_time.py — wall-clock training-time estimates for YOLO26x-P2.

Combines:
  - MEASURED throughput (ms/image) on the user's RTX 5070 12GB, from _bench_clean
  - realistic dataset sizes
  - epoch counts from the training recipe
to estimate per-run wall-clock time.

All throughput numbers are MEASURED (not theoretical) on this exact GPU, at the
batch size that actually fits 12GB without VRAM thrashing. A +25% fudge factor
covers the mosaic/mixup/hsv augmentation turned on in real training (the bench
ran with aug=0 to isolate compute).
"""
from __future__ import annotations

# ---- MEASURED throughput on RTX 5070 12GB (img/sec, batch that fits VRAM) ----
# Source: scripts/_bench_clean.py + isolated follow-up runs (see results/_bench_clean.json).
# These already EXCLUDE the CPU-spilling regime: batches were chosen so VRAM peak
# stays under ~7.7 GB (well within 12 GB).
THROUGHPUT_IMG_PER_SEC = {
    # variant:    (imgsz, img/sec measured,  batch_used,  notes)
    "yolo26n-p2":  (1280, 40.7,  8, "fits comfortably"),
    "yolo26s-p2":  (1280, 23.7,  4, "fits comfortably"),
    "yolo26m-p2":  (1280, 10.8,  2, "batch 4 thrashes; use 2"),
    "yolo26l-p2":  (1280,  9.8,  2, "fits at batch 2"),
    "yolo26x-p2":  (1280,  5.6,  1, "batch 2 thrashes; use 1"),
    "yolo26x-p2@640": (640, 22.0, 4, "smaller imgsz, 4x bigger batch"),
}

# ---- Dataset sizes (image counts). Tier B = DTLD+Bosch+LISA ----
# Conservative midpoints from docs/datasets.md (to be confirmed by eda.py).
DATASET_IMAGES = {
    "DTLD":       23_500,
    "Bosch STL":  13_400,
    "LISA":       43_000,
    "Open Images (Tier A only)": 6_000,   # capped via convert_oi --max-images
}
TIER_B_IMAGES = DATASET_IMAGES["DTLD"] + DATASET_IMAGES["Bosch STL"] + DATASET_IMAGES["LISA"]  # ~80k
TIER_A_IMAGES = TIER_B_IMAGES + DATASET_IMAGES["Open Images (Tier A only)"]                   # ~86k

# ---- Epoch counts (from configs/hyp_base.yaml and the search recipe) ----
EPOCHS_FULL = 300
EPOCHS_SEARCH_TRIAL = 40

AUG_FUDGE = 1.25   # +25% for mosaic/mixup/hsv/copy-paste turned on in real training
HOURS_PER_DAY = 24


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h*60:.0f} min"
    if h < 48:
        return f"{h:.1f} h ({h/HOURS_PER_DAY:.1f} days)"
    return f"{h:.0f} h ({h/HOURS_PER_DAY:.1f} days)"


def epoch_time(n_images: int, img_per_sec: float) -> float:
    """Seconds per epoch = images / (img/sec), with aug fudge + ~5% validation."""
    train_s = (n_images / img_per_sec) * AUG_FUDGE
    val_s = train_s * 0.05
    return train_s + val_s


def main() -> None:
    print("=" * 78)
    print("YOLO26x-P2 training-time estimates  —  RTX 5070 12 GB")
    print("Throughput MEASURED on this GPU (not theoretical). +25% for real augmentation.")
    print("=" * 78)

    print(f"\nDataset sizes (Tier B = DTLD+Bosch+LISA = {TIER_B_IMAGES:,} imgs)")
    for k, v in DATASET_IMAGES.items():
        print(f"  {k:30s} {v:>8,} imgs")

    print(f"\n--- FULL TRAINING RUN: Tier B, {EPOCHS_FULL} epochs, {TIER_B_IMAGES:,} images ---")
    print(f"{'variant':16s} {'imgsz':>6s} {'img/s':>7s} {'s/epoch':>9s} {'total':>14s}")
    for v, (imgsz, ips, batch, note) in THROUGHPUT_IMG_PER_SEC.items():
        sep = epoch_time(TIER_B_IMAGES, ips)
        total_h = sep * EPOCHS_FULL / 3600.0
        print(f"{v:16s} {imgsz:>6d} {ips:>7.1f} {sep:>9.0f} {fmt_hours(total_h):>14s}   [{note}]")

    print(f"\n--- TIER A (detection baseline, {TIER_A_IMAGES:,} imgs incl. Open Images) ---")
    for v in ("yolo26n-p2", "yolo26s-p2", "yolo26m-p2", "yolo26x-p2"):
        imgsz, ips, batch, _ = THROUGHPUT_IMG_PER_SEC[v]
        sep = epoch_time(TIER_A_IMAGES, ips)
        total_h = sep * EPOCHS_FULL / 3600.0
        print(f"{v:16s} {fmt_hours(total_h):>14s}")

    print(f"\n--- HYPERPARAMETER SEARCH (Tier B, {EPOCHS_SEARCH_TRIAL} ep/trial) ---")
    n_trials = 30
    # use s-p2 or m-p2 for the search (x is too slow for 30 trials)
    print(f"  {n_trials} trials x {EPOCHS_SEARCH_TRIAL} ep/trial:")
    for v in ("yolo26s-p2", "yolo26m-p2"):
        imgsz, ips, batch, _ = THROUGHPUT_IMG_PER_SEC[v]
        sep = epoch_time(TIER_B_IMAGES, ips)
        total_h = sep * EPOCHS_SEARCH_TRIAL * n_trials / 3600.0
        print(f"  {v:16s} {fmt_hours(total_h):>14s}")

    print(f"\n--- RECOMMENDED CONFIG (realistic on 12 GB) ---")
    print("  Primary:   yolo26x-p2 @ imgsz=1280, batch=1, 300 ep, Tier B")
    _, ips_x, _, _ = THROUGHPUT_IMG_PER_SEC["yolo26x-p2"]
    sep = epoch_time(TIER_B_IMAGES, ips_x)
    print(f"             -> ~{fmt_hours(sep*EPOCHS_FULL/3600)} wall-clock")
    print("  Faster alt: yolo26x-p2 @ imgsz=640, batch=4, 300 ep (trades small-object recall)")
    _, ips_640, _, _ = THROUGHPUT_IMG_PER_SEC["yolo26x-p2@640"]
    sep640 = epoch_time(TIER_B_IMAGES, ips_640)
    print(f"             -> ~{fmt_hours(sep640*EPOCHS_FULL/3600)} wall-clock")
    print("  Sanity run: yolo26m-p2 @ imgsz=1280, batch=2 (decent accuracy, 5x faster than x)")
    _, ips_m, _, _ = THROUGHPUT_IMG_PER_SEC["yolo26m-p2"]
    sep_m = epoch_time(TIER_B_IMAGES, ips_m)
    print(f"             -> ~{fmt_hours(sep_m*EPOCHS_FULL/3600)} wall-clock")

    print("\nNotes:")
    print(" - 'batch that fits': the auto-batcher's polyfit misfires for the heavy P2")
    print("   variants on 12 GB (suggests 16 -> silent VRAM thrashing, 10-50x slower).")
    print("   PIN the batch explicitly (batch=1 for x@1280, batch=2 for m/l@1280).")
    print(" - Validation every epoch adds ~5%; the search includes ~5% val overhead/trial.")


if __name__ == "__main__":
    main()
