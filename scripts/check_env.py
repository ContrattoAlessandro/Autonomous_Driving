"""Verify the YOLOv8/YOLO26 no-P2 training environment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALID_SCALES = ("n", "s", "m", "l", "x")
VALID_FAMILIES = ("yolov8", "yolo26")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify environment for YOLOv8/YOLO26 training")
    parser.add_argument("--family", choices=VALID_FAMILIES, default="yolov8")
    parser.add_argument("--scale", choices=VALID_SCALES, default="n")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    ok = True

    section("ultralytics / torch")
    try:
        import torch
        import ultralytics

        print(f"ultralytics : {ultralytics.__version__}")
        print(f"torch       : {torch.__version__}")
    except Exception as exc:
        print(f"FAIL: cannot import ultralytics/torch: {exc}")
        return 1

    section("GPU")
    print(f"CUDA built : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device     : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM       : {props.total_memory / 1024**3:.1f} GB")
    else:
        print("WARN: CUDA is not visible; training will be CPU-only.")

    model_stem = f"{args.family}{args.scale}"
    section(f"model config: {model_stem}.yaml")
    local_cfg = root / "configs" / "model" / f"{model_stem}.yaml"
    cfg_src = str(local_cfg) if local_cfg.exists() else f"{model_stem}.yaml"
    try:
        from ultralytics import YOLO

        model = YOLO(cfg_src)
        model.info()
        print(f"model built OK: {cfg_src}")
    except Exception as exc:
        print(f"FAIL: cannot build {cfg_src}: {exc}")
        ok = False

    section(f"COCO weights: {model_stem}.pt")
    try:
        weights = YOLO(f"{model_stem}.pt")
        print(f"weights OK: nc={getattr(weights.model, 'nc', '?')}")
    except Exception as exc:
        print(f"FAIL: cannot load COCO weights: {exc}")
        ok = False

    section("ATLAS config")
    atlas_yaml = root / "configs" / "data_atlas.yaml"
    text = atlas_yaml.read_text(encoding="utf-8")
    if "nc: 25" in text and "circle_green" in text and "arrow_straight_right_green" in text:
        print("ATLAS config OK: native 25-class taxonomy")
    else:
        print("FAIL: ATLAS config does not declare the native 25 classes")
        ok = False

    section("summary")
    print("ENV OK" if ok else "ENV HAS PROBLEMS (see above)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
