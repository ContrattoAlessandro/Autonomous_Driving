"""Train an official Ultralytics YOLO detector on the project datasets.

The mainline experiments use YOLOv8 or YOLO26 without a P2 head.  The script
deliberately keeps the Ultralytics trainer, loss, DataLoader and augmentation
pipeline unchanged.  It only adds project-root path resolution and a fail-fast
dataset check before calling ``model.train``.

Examples (run from this directory or from the repository root)::

    python scripts/train.py --data atlas --init coco --epochs 300 --batch 8
    python scripts/train.py --data atlas --model yolov8s.yaml --imgsz 1280 --epochs 300 --batch 2 --rect
    python scripts/train.py --data atlas --model yolo26n.yaml --init coco --epochs 300 --batch 8
    python scripts/train.py --resume runs/yolov8n_data_atlas_coco/weights/last.pt

ATLAS is used with its native 25 pictogram-state classes.  No class remapping
or label rewriting is performed here.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from _data import ROOT, resolve_project_path, validate_data_config
except ImportError:  # supports ``python -m scripts.train`` from tl_detection/
    from scripts._data import ROOT, resolve_project_path, validate_data_config

MODEL_CFG = ROOT / "configs" / "model" / "yolov8n.yaml"
DEFAULT_HYP = ROOT / "configs" / "hyp_base.yaml"
VALID_SCALES = ("n", "s", "m", "l", "x")
SUPPORTED_FAMILIES = ("yolov8", "yolo26")

DATA_PRESETS = {
    "atlas": ROOT / "configs" / "data_atlas.yaml",
    "A": ROOT / "configs" / "data_tierA.yaml",
    "B": ROOT / "configs" / "data_tierB.yaml",
    "C": ROOT / "configs" / "data_tierC.yaml",
}


def extract_model_spec(model_arg: str | Path, *, allow_p2: bool = False) -> tuple[str, str]:
    """Return ``(family, scale)`` for a supported model identifier."""
    stem = Path(str(model_arg)).stem.lower()
    if "p2" in stem and not allow_p2:
        raise SystemExit(
            "P2 models require the explicit --allow-p2 ablation flag. "
            "The default mainline uses YOLOv8/YOLO26 models without '-p2'."
        )
    family_pattern = "|".join(re.escape(family) for family in SUPPORTED_FAMILIES)
    match = re.search(rf"(?:^|[^a-z])({family_pattern})([nsmlx])(?:$|[-_.])", stem)
    if not match:
        raise SystemExit(
            f"Cannot infer a supported YOLO family/scale from '{model_arg}'. "
            "Expected yolov8n/s/m/l/x or yolo26n/s/m/l/x (.yaml or .pt)."
        )
    return match.group(1), match.group(2)


def extract_v8_scale(model_arg: str | Path, *, allow_p2: bool = False) -> str:
    """Backward-compatible helper returning the scale for a YOLOv8 model."""
    family, scale = extract_model_spec(model_arg, allow_p2=allow_p2)
    if family != "yolov8":
        raise SystemExit(f"Expected a YOLOv8 model, got '{model_arg}'.")
    return scale


def resolve_model_arg(model_arg: str | Path) -> str:
    """Resolve project model configs while retaining Ultralytics hub names."""
    raw = Path(str(model_arg))
    if raw.is_absolute() and raw.exists():
        return str(raw.resolve())
    candidates = (Path.cwd() / raw, ROOT / raw, ROOT / "configs" / "model" / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(model_arg)


def resolve_init_weights(
    model_arg: str | Path,
    init: str,
    *,
    allow_p2: bool = False,
) -> str | None:
    """Resolve official COCO weights matching the selected family and scale."""
    if init == "scratch":
        return None
    if init != "coco":
        raise SystemExit(f"Unsupported --init '{init}'. Use 'coco' or 'scratch'.")
    family, scale = extract_model_spec(model_arg, allow_p2=allow_p2)
    return f"{family}{scale}.pt"


def resolve_data(arg: str | None, tier: str) -> Path:
    if arg is None:
        return DATA_PRESETS[tier]
    key = arg.lower()
    if key == "atlas":
        return DATA_PRESETS["atlas"]
    if arg in DATA_PRESETS:
        return DATA_PRESETS[arg]
    return resolve_project_path(arg)


def assert_resumable_checkpoint(checkpoint: Path) -> None:
    """Prevent Ultralytics from silently falling back to a fresh default run."""
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"Checkpoint is not a resumable Ultralytics dictionary: {checkpoint}")
    epoch = payload.get("epoch", -1)
    optimizer = payload.get("optimizer")
    if not isinstance(epoch, int) or epoch < 0 or optimizer is None:
        raise SystemExit(
            f"Checkpoint is not resumable: {checkpoint}. "
            "Use a last.pt captured during an interrupted run (with epoch and optimizer state); "
            "a completed/stripped checkpoint can only be loaded for inference or fine-tuning."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8/YOLO26 on traffic-light data")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="atlas, A, B, C, or a path to a dataset YAML (default: atlas)",
    )
    parser.add_argument(
        "--tier",
        choices=["atlas", "A", "B", "C"],
        default="atlas",
        help="preset used when --data is omitted",
    )
    parser.add_argument(
        "--init",
        choices=["coco", "scratch"],
        default="coco",
        help="official YOLOv8/YOLO26 COCO initialization or random initialization",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_CFG),
        help="YOLOv8/YOLO26 model YAML/weight; P2 requires --allow-p2 (default: yolov8n.yaml)",
    )
    parser.add_argument(
        "--allow-p2",
        action="store_true",
        help="run an explicit P2 ablation; disabled in the mainline by default",
    )
    parser.add_argument("--hyp", type=Path, default=DEFAULT_HYP, help="training YAML overrides")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="number of training epochs (overrides hyp YAML; default: value in hyp YAML)",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="square input size, e.g. 1280")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="required for new training: positive integer, or -1 for Ultralytics auto-batch",
    )
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--rect", action="store_true", help="use Ultralytics' native aspect-ratio batching")
    parser.add_argument("--device", type=str, default=None, help="CUDA device, e.g. 0; default is Ultralytics auto")
    parser.add_argument("--name", type=str, default=None, help="run name")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="LAST_PT",
        help="resume from an explicit runs/.../weights/last.pt checkpoint",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="skip label validation only; split/path checks are still performed",
    )
    args = parser.parse_args()

    import yaml
    from ultralytics import YOLO

    data_yaml = resolve_data(args.data, args.tier)
    resolved_data, data_cfg = validate_data_config(
        data_yaml,
        validate_labels=not args.skip_data_check,
    )
    if data_yaml.stem == "data_atlas":
        if int(data_cfg.get("nc", -1)) != 25:
            raise SystemExit("ATLAS must be trained with its native nc=25 taxonomy.")
        print("[train] dataset=ATLAS native labels: 25 pictogram-state classes")

    if args.resume is not None:
        checkpoint = resolve_project_path(args.resume)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        assert_resumable_checkpoint(checkpoint)
        print(f"[train] resuming official Ultralytics checkpoint: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return

    if args.batch is None:
        raise SystemExit(
            "Specify the batch explicitly for a new run, e.g. --batch 8. "
            "Use --batch -1 to delegate batch selection to Ultralytics auto-batch."
        )
    if args.batch == 0 or args.batch < -1:
        raise SystemExit("--batch must be a positive integer or -1 for auto-batch.")
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs must be a positive integer.")

    model_arg = resolve_model_arg(args.model)
    model_family, scale = extract_model_spec(model_arg, allow_p2=args.allow_p2)
    init_weights = resolve_init_weights(model_arg, args.init, allow_p2=args.allow_p2)

    hyp_path = resolve_project_path(args.hyp)
    hyp = yaml.safe_load(hyp_path.read_text(encoding="utf-8")) or {}
    if not isinstance(hyp, dict):
        raise ValueError(f"Training YAML must contain a mapping: {hyp_path}")
    if isinstance(hyp.get("imgsz"), (list, tuple)):
        raise SystemExit(
            "This YOLOv8 mainline uses the official integer imgsz API. "
            "Use --imgsz 1280; do not pass [H,W]."
        )
    if args.epochs is not None:
        hyp["epochs"] = args.epochs
    if args.imgsz is not None:
        hyp["imgsz"] = args.imgsz
    if args.batch is not None:
        hyp["batch"] = args.batch
    if args.workers is not None:
        hyp["workers"] = args.workers
    if args.rect:
        hyp["rect"] = True
    if args.device is not None:
        hyp["device"] = args.device
    if args.init == "scratch":
        hyp["pretrained"] = False

    model = YOLO(model_arg)
    if init_weights:
        model = model.load(init_weights)

    data_name = data_yaml.stem.removeprefix("data_")
    model_variant = (
        f"{model_family}{scale}-p2"
        if "p2" in Path(model_arg).stem.lower()
        else f"{model_family}{scale}"
    )
    run_name = args.name or f"{model_variant}_{data_name}_{args.init}"
    mode = "rect=True" if hyp.get("rect", False) else "rect=False"
    print(
        f"[train] model={Path(model_arg).name} init={args.init} "
        f"data={data_yaml.name} imgsz={hyp.get('imgsz')} batch={hyp.get('batch')} "
        f"workers={hyp.get('workers')} {mode}"
    )

    train_kwargs = {k: v for k, v in hyp.items() if k not in {"model", "data"}}
    model.train(
        data=str(resolved_data),
        project=str(ROOT / "runs"),
        name=run_name,
        **train_kwargs,
    )


if __name__ == "__main__":
    main()
