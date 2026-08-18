"""Build and smoke-test the active YOLO11 P3-P5 Milestone 2 detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.milestone2 import (
    DEFAULT_CONFIG,
    build_detection_model,
    export_detection_onnx,
    load_coco_warmstart,
    run_forward_smoke,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--export-onnx",
        type=Path,
        default=None,
        metavar="OUTPUT.onnx",
        help="also export and validate a fixed-shape detection ONNX model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/milestone2_smoke.json"),
    )
    args = parser.parse_args()

    model = build_detection_model(args.config)
    report: dict[str, object] = {
        "warmstart": (
            load_coco_warmstart(model, args.weights) if args.weights else None
        ),
        "forward": run_forward_smoke(
            model,
            input_size=(args.height, args.width),
            device=args.device,
            half=not args.fp32,
        ),
    }
    if args.export_onnx is not None:
        report["onnx"] = export_detection_onnx(
            model,
            args.export_onnx,
            input_size=(args.height, args.width),
            device="0" if args.device == "cuda" else args.device,
            fp16=not args.fp32,
        )
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
