"""Probe full-resolution AMP train memory for a chosen physical batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.milestone2 import write_report
from tlr_yolo_mtl.training.diagnostics import run_full_resolution_memory_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tlr_yolo_mtl_train.yaml")
    )
    parser.add_argument(
        "--weights", type=Path, default=None, help="config default: yolo11n.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch",
        "--micro-batch-size",
        dest="micro_batch_size",
        type=int,
        default=1,
        help="physical batch to test (must divide effective batch 32)",
    )
    parser.add_argument(
        "--amp-initial-scale",
        type=float,
        default=None,
        help="diagnostic override for GradScaler initial scale",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/milestone7_memory_probe.json"),
    )
    args = parser.parse_args()
    if args.micro_batch_size < 1:
        raise SystemExit("--batch must be positive")
    if args.amp_initial_scale is not None and args.amp_initial_scale <= 0:
        raise SystemExit("--amp-initial-scale must be positive")
    report = run_full_resolution_memory_probe(
        args.config,
        weights_path=args.weights,
        device=args.device,
        micro_batch_size=args.micro_batch_size,
        amp_initial_scale=args.amp_initial_scale,
    )
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
