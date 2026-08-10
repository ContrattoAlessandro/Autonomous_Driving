"""Official Ultralytics hyperparameter tuning for the YOLOv8 baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _data import ROOT, resolve_project_path, validate_data_config
except ImportError:  # supports ``python -m scripts.tune`` from tl_detection/
    from scripts._data import ROOT, resolve_project_path, validate_data_config

MODEL_CFG = ROOT / "configs" / "model" / "yolov8n.yaml"
DEFAULT_HYP = ROOT / "configs" / "hyp_base.yaml"
DATA_PRESETS = {
    "atlas": ROOT / "configs" / "data_atlas.yaml",
    "A": ROOT / "configs" / "data_tierA.yaml",
    "B": ROOT / "configs" / "data_tierB.yaml",
    "C": ROOT / "configs" / "data_tierC.yaml",
}

# Keep the search space limited to parameters understood by YOLOv8's official
# trainer.  In particular, do not tune a custom optimizer or a P2-only option.
SEARCH_SPACE = {
    "lr0": (1e-4, 1e-2),
    "lrf": (0.005, 0.05),
    "momentum": (0.90, 0.98),
    "weight_decay": (1e-5, 1e-3),
    "box": (5.0, 9.0),
    "cls": (0.3, 1.0),
    "mosaic": (0.25, 0.75),
    "mixup": (0.0, 0.05),
    "close_mosaic": (10, 20),
    "hsv_h": (0.0, 0.02),
    "hsv_s": (0.25, 0.6),
    "hsv_v": (0.2, 0.5),
    "scale": (0.3, 0.7),
    "translate": (0.05, 0.15),
}


def resolve_data(value: str) -> Path:
    if value.lower() == "atlas":
        return DATA_PRESETS["atlas"]
    if value in DATA_PRESETS:
        return DATA_PRESETS[value]
    return resolve_project_path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune official YOLOv8 hyperparameters")
    parser.add_argument("--data", default="atlas", help="atlas, A, B, C, or a dataset YAML path")
    parser.add_argument("--model", type=Path, default=MODEL_CFG)
    parser.add_argument("--init", choices=["coco", "scratch"], default="coco")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=-1)
    args = parser.parse_args()

    model_path = resolve_project_path(args.model)
    if not model_path.exists():
        candidate = ROOT / "configs" / "model" / Path(args.model)
        if candidate.exists():
            model_path = candidate.resolve()
    if "p2" in model_path.stem.lower():
        raise SystemExit("P2 models are disabled; tune a YOLOv8 model without P2.")
    data_yaml, _ = validate_data_config(resolve_data(args.data), validate_labels=False)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if args.init == "coco":
        scale = model_path.stem.lower().replace("yolov8", "", 1)[0]
        model = model.load(f"yolov8{scale}.pt")

    tune_kwargs = dict(
        data=str(data_yaml),
        space=SEARCH_SPACE,
        iterations=args.iters,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="auto",
        val=True,
        plots=True,
        save=True,
        project=str(ROOT / "runs" / "tune"),
        name=f"yolov8_{data_yaml.stem}_{args.init}",
    )
    print(f"[tune] YOLOv8 no-P2, {args.iters} trials, {args.epochs} epochs, imgsz={args.imgsz}")
    model.tune(**tune_kwargs)
    print("[tune] done; inspect runs/tune/ for the best hyperparameters")


if __name__ == "__main__":
    main()
