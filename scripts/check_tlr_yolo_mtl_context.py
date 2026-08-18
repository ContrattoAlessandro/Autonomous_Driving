"""Validate Milestone 6 dense arrow context and FiLM relevance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.context import (
    attach_arrow_context_relevance,
    run_context_forward_smoke,
    run_context_gradient_smoke,
    summarize_context_pairing,
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
        "--records",
        type=Path,
        default=Path("datasets/tlr_mtl_dtld_paired/records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/milestone6_context.json"),
    )
    args = parser.parse_args()

    model = build_detection_model()
    warmstart = load_coco_warmstart(model, args.weights)
    attach_arrow_context_relevance(model)
    report = {
        "warmstart": warmstart,
        "dataset_pairing": summarize_context_pairing(args.records),
        "forward": run_context_forward_smoke(
            model,
            input_size=(args.height, args.width),
            device=args.device,
            half=not args.fp32,
        ),
        "controlled_gradient": run_context_gradient_smoke(
            model, device=args.device
        ),
    }
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
