"""Benchmark official YOLOv8 rectangular batching.

This is intentionally a thin benchmark around ``model.train(rect=True)``.
There is no custom trainer, tensor reshape, or Ultralytics monkey-patch here.
``rect=True`` keeps the official aspect-ratio-aware batching behavior; the
``H W`` argument is only a reporting hint because stock Ultralytics receives an
integer ``imgsz``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOBS = [("yolov8n", [16, 8]), ("yolov8s", [8, 4])]


def make_dataset() -> Path:
    import numpy as np
    from PIL import Image

    image_root = ROOT / "_bench" / "images"
    label_root = ROOT / "_bench" / "labels"
    for split, count in (("train", 80), ("val", 10)):
        image_dir = image_root / split
        label_dir = label_root / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Image.fromarray((np.random.rand(1080, 1920, 3) * 255).astype("uint8"))
            image.save(image_dir / f"fr{index:04d}.jpg")
            (label_dir / f"fr{index:04d}.txt").write_text("0 0.5 0.2 0.01 0.02\n")

    data_yaml = ROOT / "_bench_data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {ROOT.as_posix()}/_bench",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: traffic_light",
                "",
            ]
        )
    )
    return data_yaml


def model_source(variant: str) -> str:
    local = ROOT / "configs" / "model" / f"{variant}.yaml"
    return str(local) if local.exists() else variant + ".yaml"


def measure(variant: str, imgsz_hw: list[int], batch: int, warmup: int = 2) -> dict:
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times: list[float] = []
    last_time: float | None = None

    def on_batch_end(_trainer) -> None:
        nonlocal last_time
        now = time.perf_counter()
        if last_time is not None:
            times.append(now - last_time)
        last_time = now

    model = YOLO(model_source(variant))
    trainer = model.train(
        data=str(ROOT / "_bench_data.yaml"),
        epochs=1,
        imgsz=max(imgsz_hw),
        batch=batch,
        workers=0,
        device=0,
        rect=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        scale=0.0,
        translate=0.0,
        optimizer="SGD",
        lr0=0.01,
        close_mosaic=0,
        seed=42,
        deterministic=False,
        amp=True,
        val=False,
        plots=False,
        save=False,
        project=str(ROOT / "runs" / "_bench"),
        name=f"rect_official_{variant}_b{batch}",
        exist_ok=True,
    )
    del trainer
    torch.cuda.synchronize()

    clean = times[warmup:]
    if not clean:
        return {"variant": variant, "batch": batch, "status": "no_data"}
    median_ms = statistics.median(clean) * 1000.0
    image_ms = median_ms / batch
    return {
        "variant": variant,
        "batch": batch,
        "imgsz_hint_hw": list(imgsz_hw),
        "status": "ok",
        "ms_per_iter": round(median_ms, 1),
        "ms_per_image": round(image_ms, 2),
        "img_per_sec": round(1000.0 * batch / median_ms, 1),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="n,s", help="YOLOv8 scales, e.g. n,s")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[736, 1280], metavar=("H", "W"))
    args = parser.parse_args()

    make_dataset()
    selected = {name: batches for name, batches in DEFAULT_JOBS}
    variants = [f"yolov8{scale.strip()}" for scale in args.variants.split(",")]
    results = []
    print(f"official rect=True, imgsz={args.imgsz} hint, GPU={torch.cuda.get_device_name(0)}")
    for variant in variants:
        for batch in selected.get(variant, [8, 4, 2, 1]):
            try:
                result = measure(variant, args.imgsz, batch)
            except torch.cuda.OutOfMemoryError:
                result = {"variant": variant, "batch": batch, "status": "oom"}
                torch.cuda.empty_cache()
            except Exception as exc:
                result = {"variant": variant, "batch": batch, "status": f"err:{type(exc).__name__}"}
                print(f"[error] {variant} batch={batch}: {exc}")
            results.append(result)
            print(result)
            if result.get("status") == "ok":
                break

    out = ROOT / "results" / "_bench_rect_real.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
