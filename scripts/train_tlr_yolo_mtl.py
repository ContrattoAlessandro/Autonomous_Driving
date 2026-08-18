"""Launch the resource-aware four-phase TLR-YOLO-MTL training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.training.engine import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tlr_yolo_mtl_train.yaml")
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="COCO warm-start override (config default: yolo11n.pt); ignored on resume",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run directory; with --resume defaults to the checkpoint run",
    )
    parser.add_argument(
        "--batch",
        "--micro-batch-size",
        dest="micro_batch_size",
        type=int,
        default=None,
        help="physical batch per forward; effective batch remains 32",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers override (validated config default: 2)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="balanced effective batches per epoch (config default: 100)",
    )
    parser.add_argument("--device", default=None, help="device override, e.g. cuda")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume a full run from an epoch-complete weights/last.pt",
    )
    parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help="run only a named phase; may be repeated",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="optional hard cap for a bounded trial",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing output directory if it already contains training artifacts",
    )
    parser.add_argument(
        "--val-steps",
        type=int,
        default=None,
        help="max validation batches to evaluate per epoch (defaults to config or full val)",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="skip validation evaluation at epoch end",
    )
    args = parser.parse_args()
    if args.max_optimizer_steps is not None and args.max_optimizer_steps < 1:
        raise SystemExit("--max-optimizer-steps must be positive")
    if args.micro_batch_size is not None and args.micro_batch_size < 1:
        raise SystemExit("--batch must be positive")
    if args.workers is not None and args.workers < 0:
        raise SystemExit("--workers must be non-negative")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise SystemExit("--steps-per-epoch must be positive")
    if args.val_steps is not None and args.val_steps < 1:
        raise SystemExit("--val-steps must be positive")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.resume.resolve().parent.parent
            if args.resume is not None
            else Path("runs/tlr_yolo_mtl")
        )
    result = train_from_config(
        args.config,
        weights_path=args.weights,
        output_dir=output_dir,
        only_phases=args.phases,
        max_optimizer_steps=args.max_optimizer_steps,
        micro_batch_size=args.micro_batch_size,
        optimizer_steps_per_epoch=args.steps_per_epoch,
        workers=args.workers,
        device=args.device,
        resume_checkpoint=args.resume,
        overwrite=args.overwrite,
        val_steps_per_epoch=args.val_steps,
        skip_validation=args.no_val,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
