"""_bench_clean.py — reliable per-iteration timing via Ultralytics callbacks.

Instead of reconstructing Ultralytics' internal loss machinery, we drive the real
trainer (model.train) and capture wall-clock timestamps in an on_train_batch_end
callback. The first few iterations are discarded as warmup (CUDA lazy init,
worker spin-up). The median of the rest is the per-iteration time, which we
convert to ms/image.

This measures the SAME code path real training uses (dataloading + aug + fwd +
bwd + step), so estimates derived from it are faithful.
"""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# (variant, batch) — batches chosen to fit 12 GB at imgsz=1280 (see _bench auto-batch
# results: the polyfit misfires for heavy variants, so we pin explicitly).
JOBS = [
    ("yolo26n-p2", 8),
    ("yolo26s-p2", 4),
    ("yolo26m-p2", 4),
    ("yolo26l-p2", 2),
    ("yolo26x-p2", 2),
]
IMGSZ = 1280
N_TRAIN = 80   # synthetic frames -> at batch 8 that's 10 iters/epoch, enough to time


def make_dataset(n: int) -> None:
    import numpy as np
    from PIL import Image
    root = ROOT / "_bench"
    for split, k in (("train", n), ("val", 8)):
        d = root / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(k):
            Image.fromarray((np.random.rand(1080, 1920, 3) * 255).astype("uint8")).save(d / f"fr{i:04d}.jpg")
            (root / "labels" / split / f"fr{i:04d}.txt").write_text("0 0.5 0.2 0.01 0.02\n")


def time_variant(variant: str, batch: int) -> dict:
    from ultralytics import YOLO

    iter_times: list[float] = []
    last_t = {"v": None}

    def on_batch_end(trainer):
        now = time.perf_counter()
        if last_t["v"] is not None:
            iter_times.append(now - last_t["v"])
        last_t["v"] = now

    model = YOLO(f"{variant}.yaml")
    model.add_callback("on_train_batch_end", on_batch_end)
    run_dir = ROOT / "runs" / "_bench" / f"{variant}_b{batch}"
    model.train(
        data=str(ROOT / "_bench_data.yaml"),
        epochs=1, imgsz=IMGSZ, batch=batch, workers=4, device=0,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True, verbose=False,
        # keep aug cheap so we measure the compute-dominated regime; real training
        # adds ~10-30% from mosaic/mixup/hsv, applied as a fudge factor below.
        mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        copy_paste=0.0, erasing=0.0, fliplr=0.0, scale=0.0, translate=0.0,
        optimizer="SGD", lr0=0.01, close_mosaic=0,
        seed=42, deterministic=False, val=False, plots=False, save=False,
    )
    # discard first 2 iters (lazy init / worker spin-up)
    clean = iter_times[2:] if len(iter_times) > 2 else iter_times
    ms_iter = statistics.median(clean) * 1000.0 if clean else None
    params_M = sum(p.numel() for p in model.model.parameters()) / 1e6
    return {
        "variant": variant, "batch": batch, "imgsz": IMGSZ,
        "ms_per_iter": round(ms_iter, 1) if ms_iter else None,
        "ms_per_image": round(ms_iter / batch, 2) if ms_iter else None,
        "img_per_sec": round(1000.0 * batch / ms_iter, 1) if ms_iter else None,
        "params_M": round(params_M, 1),
        "n_iters_timed": len(clean),
    }


def main() -> None:
    make_dataset(N_TRAIN)
    results = []
    for v, b in JOBS:
        try:
            r = time_variant(v, b)
            results.append(r)
            print(f"[done] {v:14s} batch={b}  params={r['params_M']:6.1f}M  "
                  f"{r['ms_per_iter']:7.1f} ms/iter  ({r['ms_per_image']:6.2f} ms/img, "
                  f"{r['img_per_sec']:6.1f} img/s)  n={r['n_iters_timed']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            results.append({"variant": v, "batch": b, "error": "OOM"})
            print(f"[OOM]  {v:14s} batch={b}", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            results.append({"variant": v, "batch": b, "error": str(e)})
            print(f"[FAIL] {v:14s} batch={b}: {type(e).__name__}: {e}", flush=True)
    out = ROOT / "results" / "_bench_clean.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
