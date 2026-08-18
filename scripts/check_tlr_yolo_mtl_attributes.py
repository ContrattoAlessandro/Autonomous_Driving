"""Validate the Milestone 3 state and pictogram heads on active YOLO11 P3-P5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.attributes import (
    attach_attribute_heads,
    run_assignment_gradient_smoke,
    run_attribute_forward_smoke,
    run_masking_gradient_smoke,
)
from tlr_yolo_mtl.model.milestone2 import (
    build_detection_model,
    load_coco_warmstart,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/milestone3_attributes.json"),
    )
    args = parser.parse_args()

    model = build_detection_model()
    warmstart = load_coco_warmstart(model, args.weights)
    attach_attribute_heads(model)
    report = {
        "warmstart": warmstart,
        "forward": run_attribute_forward_smoke(
            model,
            input_size=(args.height, args.width),
            device=args.device,
            half=not args.fp32,
        ),
        "masking": run_masking_gradient_smoke(args.device),
        "task_aligned_assignment": run_assignment_gradient_smoke(
            model, device=args.device
        ),
    }
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
