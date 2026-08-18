"""Command-line interface for TLR-YOLO-MTL dataset Milestone 1."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .data.converters import (
    convert_atlas,
    convert_dtld_root,
    convert_lisa,
    fuse_dtld_arrow_annotations,
)
from .data.io import read_records, write_json, write_records
from .data.overlays import generate_overlays
from .data.qa import build_qa_report
from .data.schema import ImageRecord, SCHEMA_VERSION, validate_records
from .data.splits import (
    assert_no_split_leakage,
    assign_default_splits,
    audit_split_leakage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "tlr_mtl_dtld_paired"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _print_stats(source: str, stats: Counter[str]) -> None:
    summary = ", ".join(f"{key}={value}" for key, value in sorted(stats.items()))
    print(f"[{source}] {summary}")


def _write_dataset(
    records: list[ImageRecord],
    output_dir: Path,
    *,
    conversion_stats: dict[str, Counter[str]] | None,
    hash_images: bool,
    overlays: bool,
    overlay_fraction: float,
    seed: int,
) -> None:
    validate_records(records)
    assert_no_split_leakage(records, hash_images=hash_images)
    qa = build_qa_report(records, hash_images=hash_images)
    qa["conversion_stats"] = {
        source: dict(sorted(stats.items()))
        for source, stats in sorted((conversion_stats or {}).items())
    }
    anomalies = qa["annotation_anomalies"]
    if any(int(value) for value in anomalies.values()):
        raise ValueError(f"normalized annotation anomalies remain: {anomalies}")
    output_dir.mkdir(parents=True, exist_ok=True)
    count = write_records(output_dir / "records.jsonl", records)
    split_ids = {
        split: [record.image_id for record in records if record.split == split]
        for split in ("train", "val", "test")
    }
    write_json(output_dir / "splits.json", split_ids)
    write_json(output_dir / "qa_report.json", qa)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_records": count,
        "sources": sorted({record.source_dataset for record in records}),
        "records": "records.jsonl",
        "splits": "splits.json",
        "qa_report": "qa_report.json",
        "images_are_referenced": True,
        "conversion_stats_in_qa": True,
    }
    if overlays:
        paths = generate_overlays(
            records,
            output_dir / "overlays",
            fraction=overlay_fraction,
            seed=seed,
        )
        manifest["n_overlays"] = len(paths)
        manifest["overlays"] = "overlays"
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"[done] {count} records -> {output_dir / 'records.jsonl'}; "
        f"QA leakage_ok={qa['split_leakage']['ok']}"
    )


def _prepare(args: argparse.Namespace) -> None:
    dtld = convert_dtld_root(
        _path(args.dtld_labels),
        _path(args.dtld_images),
        limit_per_split=args.limit_per_split,
        verify_dimensions=args.verify_dimensions,
    )
    paired_dtld = fuse_dtld_arrow_annotations(
        dtld.records,
        _path(args.dtld_arrows),
        require_exact_coverage=args.limit_per_split is None,
    )
    atlas = convert_atlas(
        _path(args.atlas),
        limit_per_partition=args.limit_per_split,
        verify_dimensions=args.verify_dimensions,
    )
    lisa = convert_lisa(
        _path(args.lisa),
        limit_per_split=args.limit_per_split,
        verify_dimensions=args.verify_dimensions,
    )
    _print_stats("DTLD", dtld.stats)
    _print_stats("DTLD_USER_ARROWS", paired_dtld.stats)
    _print_stats("ATLAS", atlas.stats)
    _print_stats("LISA", lisa.stats)

    records = paired_dtld.records + atlas.records + lisa.records
    records = assign_default_splits(records, seed=args.seed)
    _write_dataset(
        records,
        _path(args.output),
        conversion_stats={
            "DTLD": dtld.stats,
            "DTLD_USER_ARROWS": paired_dtld.stats,
            "ATLAS": atlas.stats,
            "LISA": lisa.stats,
        },
        hash_images=args.hash_images,
        overlays=not args.skip_overlays,
        overlay_fraction=args.overlay_fraction,
        seed=args.seed,
    )


def _convert(args: argparse.Namespace) -> None:
    if args.command == "convert-dtld":
        result = convert_dtld_root(
            _path(args.labels),
            _path(args.images),
            limit_per_split=args.limit_per_split,
            verify_dimensions=args.verify_dimensions,
        )
    elif args.command == "convert-atlas":
        result = convert_atlas(
            _path(args.root),
            limit_per_partition=args.limit_per_split,
            verify_dimensions=args.verify_dimensions,
        )
    else:
        result = convert_lisa(
            _path(args.root),
            limit_per_split=args.limit_per_split,
            verify_dimensions=args.verify_dimensions,
        )
    _print_stats(args.command, result.stats)
    count = write_records(_path(args.output), result.records)
    print(f"wrote {count} records to {_path(args.output)}")


def _split(args: argparse.Namespace) -> None:
    records = assign_default_splits(read_records(_path(args.input)), seed=args.seed)
    assert_no_split_leakage(records, hash_images=args.hash_images)
    write_records(_path(args.output), records)
    print(audit_split_leakage(records, hash_images=False)["split_counts"])


def _qa(args: argparse.Namespace) -> None:
    records = list(read_records(_path(args.input)))
    validate_records(records)
    report = build_qa_report(records, hash_images=args.hash_images)
    write_json(_path(args.output), report)
    if args.overlays:
        generate_overlays(
            records,
            _path(args.overlays),
            fraction=args.overlay_fraction,
            seed=args.seed,
        )
    if not report["split_leakage"]["ok"]:
        raise SystemExit("QA failed: split leakage detected")
    print(f"QA OK -> {_path(args.output)}")


def _add_limit_and_dimensions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--verify-dimensions", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tlr_yolo_mtl",
        description="TLR-YOLO-MTL unified-dataset tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="build paired DTLD plus auxiliary traffic-light sources",
    )
    prepare.add_argument("--dtld-labels", type=Path, default=WORKSPACE_ROOT / "DTLD" / "v2.0")
    prepare.add_argument(
        "--dtld-images", type=Path, default=WORKSPACE_ROOT / "DTLD_jpg_plain"
    )
    prepare.add_argument(
        "--dtld-arrows",
        type=Path,
        default=WORKSPACE_ROOT / "dataset_ALL_USER_ANNOTATED",
    )
    prepare.add_argument("--atlas", type=Path, default=WORKSPACE_ROOT / "dataset_ATLAS" / "ATLAS")
    prepare.add_argument("--lisa", type=Path, default=WORKSPACE_ROOT / "dataset_LISA" / "LISA")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--hash-images", action="store_true")
    prepare.add_argument("--skip-overlays", action="store_true")
    prepare.add_argument("--overlay-fraction", type=float, default=0.01)
    _add_limit_and_dimensions(prepare)
    prepare.set_defaults(handler=_prepare)

    dtld = subparsers.add_parser("convert-dtld")
    dtld.add_argument("--labels", type=Path, required=True)
    dtld.add_argument("--images", type=Path, required=True)
    dtld.add_argument("--output", type=Path, required=True)
    _add_limit_and_dimensions(dtld)
    dtld.set_defaults(handler=_convert)

    for command, label in (
        ("convert-atlas", "ATLAS"),
        ("convert-lisa", "LISA"),
    ):
        convert = subparsers.add_parser(command, help=f"convert {label}")
        convert.add_argument("--root", type=Path, required=True)
        convert.add_argument("--output", type=Path, required=True)
        _add_limit_and_dimensions(convert)
        convert.set_defaults(handler=_convert)

    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--hash-images", action="store_true")
    split.set_defaults(handler=_split)

    qa = subparsers.add_parser("qa")
    qa.add_argument("--input", type=Path, required=True)
    qa.add_argument("--output", type=Path, required=True)
    qa.add_argument("--hash-images", action="store_true")
    qa.add_argument("--overlays", type=Path, default=None)
    qa.add_argument("--overlay-fraction", type=float, default=0.01)
    qa.add_argument("--seed", type=int, default=42)
    qa.set_defaults(handler=_qa)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    args.handler(args)
