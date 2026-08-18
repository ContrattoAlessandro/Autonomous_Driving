"""Run one bounded DTLD-only unified-model forward/backward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.milestone2 import write_report
from tlr_yolo_mtl.training.diagnostics import run_multitask_training_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tlr_yolo_mtl_train.yaml")
    )
    parser.add_argument(
        "--weights", type=Path, default=None, help="config default: yolo11n.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/milestone7_training_smoke.json"),
    )
    args = parser.parse_args()
    report = run_multitask_training_smoke(
        args.config,
        weights_path=args.weights,
        device=args.device,
        image_size=args.image_size,
    )
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
