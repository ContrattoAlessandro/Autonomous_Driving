"""Evaluate a YOLOv8 checkpoint with the official Ultralytics metrics.

The default dataset is native ATLAS (25 classes).  The script evaluates the
held-out ``test`` split and provides optional size-stratified recall and speed
measurements without changing the model's prediction or NMS code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _data import ROOT, label_for_image, read_image_list, resolve_project_path, validate_data_config
except ImportError:  # supports ``python -m scripts.eval`` from tl_detection/
    from scripts._data import ROOT, label_for_image, read_image_list, resolve_project_path, validate_data_config

SIZE_BINS = [(0, 16), (16, 32), (32, 96), (96, 1 << 16)]
BIN_NAMES = ["0-16", "16-32", "32-96", ">=96"]
DATA_PRESETS = {
    "atlas": ROOT / "configs" / "data_atlas.yaml",
    "A": ROOT / "configs" / "data_tierA.yaml",
    "B": ROOT / "configs" / "data_tierB.yaml",
    "C": ROOT / "configs" / "data_tierC.yaml",
}


def resolve_data(arg: str | None, tier: str) -> Path:
    if arg is None:
        return DATA_PRESETS[tier]
    if arg.lower() == "atlas":
        return DATA_PRESETS["atlas"]
    if arg in DATA_PRESETS:
        return DATA_PRESETS[arg]
    return resolve_project_path(arg)


def eval_standard(weights: Path, data_yaml: Path, imgsz: int, device: str | None = None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = dict(
        data=str(data_yaml),
        imgsz=imgsz,
        split="test",
        project=str(weights.parent.parent / "eval"),
        name="test",
    )
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    return _metrics_to_dict(metrics)


def _metrics_to_dict(metrics) -> dict:
    out: dict = {}
    for attr in ("fitness", "box.map50", "box.map", "box.mp", "box.mr"):
        try:
            value = metrics
            for part in attr.split("."):
                value = getattr(value, part)
            out[attr] = float(value)
        except Exception:
            pass
    try:
        names = metrics.names
        # Ultralytics stores per-class arrays only for classes present in the
        # evaluated split; map array positions through ap_class_index.
        class_indices = [int(i) for i in metrics.box.ap_class_index]
        out["per_class"] = {}
        for position, class_id in enumerate(class_indices):
            class_name = names[class_id] if isinstance(names, dict) else names[class_id]
            out["per_class"][class_name] = {
                "ap50": float(metrics.box.ap50[position]),
                "ap": float(metrics.box.ap[position]),
                "p": float(metrics.box.p[position]),
                "r": float(metrics.box.r[position]),
            }
    except Exception:
        pass
    return out


def eval_size_stratified(weights: Path, data_yaml: Path, imgsz: int, device: str | None = None) -> dict:
    """Report recall by GT box size using Ultralytics predictions at IoU=.5."""
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    test_list = Path(data["test"])
    image_paths = read_image_list(test_list, dataset_root)
    model = YOLO(str(weights))

    n_per_bin = {name: 0 for name in BIN_NAMES}
    tp_per_bin = {name: 0 for name in BIN_NAMES}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = label_for_image(image_path)
        if not label_path.exists():
            continue

        gts = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            cls, cx, cy, box_w, box_h = (float(value) for value in fields)
            side = ((box_w * width) * (box_h * height)) ** 0.25
            gts.append((int(cls), cx * width, cy * height, box_w * width, box_h * height, side))

        predict_kwargs = dict(imgsz=imgsz, verbose=False, conf=0.25)
        if device is not None:
            predict_kwargs["device"] = device
        results = model.predict(str(image_path), **predict_kwargs)
        predictions = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append(
                    (int(box.cls[0]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                )

        for cls, gx, gy, gw, gh, side in gts:
            bin_index = next(i for i, (low, high) in enumerate(SIZE_BINS) if low <= side < high)
            bin_name = BIN_NAMES[bin_index]
            n_per_bin[bin_name] += 1
            if any(
                pred_cls == cls and _iou(gx, gy, gw, gh, px, py, pw, ph) > 0.5
                for pred_cls, px, py, pw, ph in predictions
            ):
                tp_per_bin[bin_name] += 1

    return {
        name: {
            "n": n_per_bin[name],
            "recall": tp_per_bin[name] / n_per_bin[name] if n_per_bin[name] else None,
        }
        for name in BIN_NAMES
    }


def _iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2) -> float:
    ax1, ay1, ax2, ay2 = cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2
    bx1, by1, bx2, by2 = cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def eval_speed(weights: Path, imgsz_list=(640, 1280), onnx: bool = False, device: str = "0") -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for size in imgsz_list:
        out[f"imgsz{size}"] = model.benchmark(imgsz=size, device=device)
    if onnx:
        out["onnx_export"] = str(model.export(format="onnx", imgsz=1280, dynamic=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 checkpoint")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--tier", choices=["atlas", "A", "B", "C"], default="atlas")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--size", action="store_true", help="run size-stratified recall")
    parser.add_argument("--speed", action="store_true", help="run latency benchmark")
    parser.add_argument("--onnx", action="store_true", help="export ONNX during speed test")
    args = parser.parse_args()

    weights = resolve_project_path(args.weights)
    source_data = resolve_data(args.data, args.tier)
    data_yaml, _ = validate_data_config(source_data, validate_labels=False)
    output_dir = ROOT / "results" / "eval" / weights.parent.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[eval] official Ultralytics test mAP/P/R ...")
    metrics = eval_standard(weights, data_yaml, args.imgsz, args.device)
    (output_dir / "ultralytics_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.size:
        print("[eval] size-stratified recall ...")
        size_metrics = eval_size_stratified(weights, data_yaml, args.imgsz, args.device)
        (output_dir / "size_stratified.json").write_text(json.dumps(size_metrics, indent=2), encoding="utf-8")
        print(json.dumps(size_metrics, indent=2))

    if args.speed:
        print("[eval] latency benchmark ...")
        speed = eval_speed(weights, onnx=args.onnx, device=args.device or "0")
        (output_dir / "speed.json").write_text(json.dumps(speed, indent=2, default=str), encoding="utf-8")
        print(json.dumps(speed, indent=2, default=str))


if __name__ == "__main__":
    main()
