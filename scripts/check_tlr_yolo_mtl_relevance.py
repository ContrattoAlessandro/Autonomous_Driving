"""Validate the Milestone 5 P3-P5 local relevance head and masked loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.milestone2 import (
    build_detection_model,
    load_coco_warmstart,
    write_report,
)
from tlr_yolo_mtl.model.relevance import (
    attach_local_relevance_head,
    run_relevance_assignment_smoke,
    run_relevance_forward_smoke,
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
        default=Path("results/tlr_yolo_mtl/milestone5_relevance.json"),
    )
    args = parser.parse_args()

    model = build_detection_model()
    warmstart = load_coco_warmstart(model, args.weights)
    attach_local_relevance_head(model)
    report = {
        "warmstart": warmstart,
        "forward": run_relevance_forward_smoke(
            model,
            input_size=(args.height, args.width),
            device=args.device,
            half=not args.fp32,
        ),
        "task_aligned_assignment": run_relevance_assignment_smoke(
            model, device=args.device
        ),
    }
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
