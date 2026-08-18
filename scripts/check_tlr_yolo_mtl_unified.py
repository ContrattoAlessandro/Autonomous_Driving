"""Validate the active unified detector and gated TL-to-arrow attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlr_yolo_mtl.model.milestone2 import (
    build_detection_model,
    load_coco_warmstart,
    write_report,
)
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    attach_unified_relevance_head,
    run_unified_forward_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tlr_yolo_mtl/unified_attention_smoke.json"),
    )
    args = parser.parse_args()

    wrapper = build_detection_model(args.config) if args.config else build_detection_model()
    warmstart = load_coco_warmstart(wrapper, args.weights)
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig())
    report = {
        "warmstart": warmstart,
        "forward": run_unified_forward_smoke(
            wrapper,
            input_size=(args.height, args.width),
            device=args.device,
            half=not args.fp32,
        ),
    }
    output = write_report(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
