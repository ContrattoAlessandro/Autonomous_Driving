"""Export and validate the complete fixed-shape deployment graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.deployment.export import (
    build_full_model,
    export_full_onnx,
    profile_pytorch_fp16,
    run_onnxruntime_parity,
)
from tlr_yolo_mtl.model.milestone2 import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("results/tlr_yolo_mtl/tlr-yolo-mtl-p3-p5-fp16.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/milestone9_deployment.json"),
    )
    args = parser.parse_args()
    wrapper, initialization = build_full_model(
        weights_path=args.weights, checkpoint=args.checkpoint
    )
    report = {
        "initialization": initialization,
        "onnx": export_full_onnx(
            wrapper,
            args.onnx,
            input_size=(args.height, args.width),
            device=args.device,
            half=True,
        ),
        "onnxruntime_parity": (
            None if args.skip_parity else run_onnxruntime_parity(wrapper)
        ),
        "pytorch_profile": (
            None
            if args.skip_profile
            else profile_pytorch_fp16(
                wrapper, input_size=(args.height, args.width)
            )
        ),
        "tensorrt": {
            "available": False,
            "validated": False,
            "reason": "TensorRT runtime is not installed in the current environment",
        },
    }
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
