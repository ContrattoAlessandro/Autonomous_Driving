"""_bench_heavy.py — timing for m/l/x-p2 with EXPLICIT batch sizes.

The Ultralytics auto-batcher's polyfit misfires for the heavy P2 variants on a
12 GB card (suggests batch 16 -> instant OOM). We pin a conservative batch that
fits and time one epoch. The per-image throughput is what we need for estimates;
batch size only affects it mildly (larger batch = slightly better GPU util).

Pinned batches (chosen to fit 12 GB at imgsz=1280; tune if OOM):
  yolo26m-p2: batch 4
  yolo26l-p2: batch 2
  yolo26x-p2: batch 2
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = [
    ("yolo26m-p2", 4),
    ("yolo26l-p2", 2),
    ("yolo26x-p2", 2),
]


def run_variant(v: str, batch: int) -> dict:
    from ultralytics import YOLO
    model = YOLO(f"{v}.yaml")
    run_dir = ROOT / "runs" / "_bench" / v
    model.train(
        data=str(ROOT / "_bench_data.yaml"),
        epochs=1, imgsz=1280, batch=batch, workers=4, device=0,
        project=str(run_dir.parent), name=v, exist_ok=True, verbose=False,
        mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        copy_paste=0.0, erasing=0.0, fliplr=0.0, scale=0.0, translate=0.0,
        optimizer="SGD", lr0=0.01, close_mosaic=0,
        seed=42, deterministic=False,
    )
    import csv
    rcsv = run_dir / "results.csv"
    epoch_time = None
    with rcsv.open() as f:
        rows = list(csv.DictReader(f))
    if rows and "time" in rows[-1] and rows[-1]["time"]:
        epoch_time = float(rows[-1]["time"])
    return {
        "variant": v, "batch": batch,
        "epoch_time_min": epoch_time,   # Ultralytics 'time' column = total minutes
        "sec_per_img": (epoch_time * 60.0 / 50.0) if epoch_time else None,
        "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
    }


def main() -> None:
    results = []
    for v, b in JOBS:
        try:
            r = run_variant(v, b)
            results.append(r)
            print(f"[done] {v}: batch={r['batch']}  epoch_time={r['epoch_time_min']}min  "
                  f"sec/img={r['sec_per_img']}  params={r['params_M']:.1f}M", flush=True)
        except Exception as e:
            print(f"[FAIL] {v}: {type(e).__name__}: {e}", flush=True)
            results.append({"variant": v, "batch": b, "error": str(e)})
    out = ROOT / "results" / "_bench_p2_heavy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
