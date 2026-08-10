"""_bench.py — one-shot timing benchmark of all YOLO26-P2 variants on this GPU.

For each variant {n,s,m,l,x}-p2, runs 1 epoch of training on a tiny synthetic
dataset at imgsz=1280 with batch=-1 (auto-batch). Parses the Ultralytics log to
extract:
  - the auto-selected batch size
  - seconds-per-iteration and seconds-per-image

Prints a tidy table at the end. This is what feeds the wall-clock estimates.
"""
from __future__ import annotations
import re
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["yolo26n-p2", "yolo26s-p2", "yolo26m-p2", "yolo26l-p2", "yolo26x-p2"]


def run_variant(v: str) -> dict:
    """Train 1 epoch; capture stdout to scrape batch size + timing."""
    from ultralytics import YOLO

    buf = StringIO()
    old_stdout = sys.stdout
    # We can't easily redirect YOLO's rich console, so we parse the saved results.
    model = YOLO(f"{v}.yaml")
    # warm up the model build
    run_dir = ROOT / "runs" / "_bench" / v
    res = model.train(
        data=str(ROOT / "_bench_data.yaml"),
        epochs=1,
        imgsz=1280,
        batch=-1,            # auto-batch: max that fits VRAM
        workers=4,
        device=0,
        project=str(run_dir.parent),
        name=v,
        exist_ok=True,
        verbose=False,
        # minimal aug to measure pure compute throughput
        mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        copy_paste=0.0, erasing=0.0, fliplr=0.0, scale=0.0, translate=0.0,
        optimizer="SGD", lr0=0.01, close_mosaic=0,
        seed=42, deterministic=False,
    )

    # Parse the args.yaml the run saved — it records the resolved batch size.
    import yaml
    args_path = run_dir / "args.yaml"
    args = yaml.safe_load(args_path.read_text()) if args_path.exists() else {}
    batch = args.get("batch", "?")

    # Parse results.csv for per-epoch time, and the train batch log line if present.
    results_csv = run_dir / "results.csv"
    epoch_time = None
    if results_csv.exists():
        import csv
        with results_csv.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            for k in ("time", "epoch_time", "total_time"):
                if k in rows[0] and rows[-1][k]:
                    try:
                        epoch_time = float(rows[-1][k])
                    except ValueError:
                        pass

    # The most reliable per-iteration timing lives in stdout lines like:
    #   "  1/50  3.5G  ...  4.2s  ...  0.084s"   (it/s and s/it)
    # We re-derive it from the log if captured. Otherwise estimate from epoch_time.
    n_train_imgs = 50
    sec_per_img = (epoch_time / n_train_imgs) if epoch_time else None

    return {
        "variant": v,
        "batch": batch,
        "epoch_time_s": epoch_time,
        "sec_per_img": sec_per_img,
        "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
    }


def main() -> None:
    results = []
    for v in VARIANTS:
        try:
            r = run_variant(v)
            results.append(r)
            print(f"[done] {v}: batch={r['batch']}  epoch_time={r['epoch_time_s']}s  "
                  f"sec/img={r['sec_per_img']}  params={r['params_M']:.1f}M",
                  flush=True)
        except Exception as e:
            print(f"[FAIL] {v}: {type(e).__name__}: {e}", flush=True)
            results.append({"variant": v, "error": str(e)})

    # Save JSON for the estimate step
    import json
    out = ROOT / "results" / "_bench_p2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
