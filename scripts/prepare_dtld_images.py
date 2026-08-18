"""Convert labeled DTLD TIFF frames to clean JPEG images.

The canonical converter consumes these images together with the DTLD v2 JSON
labels.  This preparation step deliberately never renders annotations and
never deletes an existing output tree.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm


DEFAULT_SPLITS = ("train", "test")
LOGGER = logging.getLogger(__name__)


def validate_source_tree(
    data_root: str | Path,
    label_root: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> tuple[Path, Path]:
    """Resolve the raw-image and v2-label roots and validate required files."""

    data = Path(data_root).resolve()
    labels = Path(label_root).resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"DTLD image root does not exist: {data}")
    if not labels.is_dir():
        raise FileNotFoundError(f"DTLD label root does not exist: {labels}")
    for split in splits:
        manifest = labels / f"DTLD_{split}.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"DTLD label manifest does not exist: {manifest}")
    return data, labels


def prepare_output_tree(
    target_root: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> Path:
    """Create split directories without overwriting any existing content."""

    target = Path(target_root).resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"DTLD output path is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for split in splits:
        (target / split).mkdir(exist_ok=True)
    return target


def _driveu_database_class() -> Any:
    try:
        from dtld_parsing.driveu_dataset import DriveuDatabase
    except ImportError as error:
        raise RuntimeError(
            "dtld_parsing is required to prepare DTLD images; install the "
            "project requirements before running this command"
        ) from error
    return DriveuDatabase


def convert_split(
    target_root: str | Path,
    label_root: str | Path,
    data_root: str | Path,
    split: str,
) -> dict[str, int | str]:
    """Convert the labeled frames for one split and return deterministic counts."""

    target = Path(target_root).resolve() / split
    manifest = Path(label_root).resolve() / f"DTLD_{split}.json"
    database = _driveu_database_class()(str(manifest))
    if not database.open(str(Path(data_root).resolve())):
        raise RuntimeError(f"DriveU database could not be opened for split {split}")

    converted = 0
    skipped_without_labels = 0
    for image in tqdm(database.images, desc=f"DTLD {split}", total=len(database.images)):
        if not image.objects:
            skipped_without_labels += 1
            continue
        decoded = image.get_image()
        if not isinstance(decoded, tuple) or len(decoded) < 2 or decoded[1] is None:
            raise RuntimeError(f"could not decode DTLD image: {image.file_path}")
        destination = target / f"{Path(image.file_path).stem}.jpg"
        if destination.exists():
            raise FileExistsError(f"duplicate DTLD output image: {destination}")
        Image.fromarray(decoded[1][..., ::-1]).save(destination)
        converted += 1

    return {
        "split": split,
        "converted": converted,
        "skipped_without_labels": skipped_without_labels,
    }


def prepare_dtld_images(
    target_root: str | Path,
    data_root: str | Path,
    label_root: str | Path,
    *,
    workers: int = 2,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, object]:
    """Prepare clean JPEGs for every requested DTLD split."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    data, labels = validate_source_tree(data_root, label_root, splits)
    target = Path(target_root).resolve()
    if target in {data, labels}:
        raise ValueError("DTLD output must differ from image and label roots")
    target = prepare_output_tree(target, splits)

    if workers == 1:
        results = [convert_split(target, labels, data, split) for split in splits]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(splits))) as executor:
            futures = [
                executor.submit(convert_split, target, labels, data, split)
                for split in splits
            ]
            results = [future.result() for future in futures]

    by_split = {str(result["split"]): result for result in results}
    return {
        "schema": "TLR-YOLO-MTL DTLD clean-image preparation v1",
        "target_root": str(target),
        "data_root": str(data),
        "label_root": str(labels),
        "splits": by_split,
        "converted": sum(int(result["converted"]) for result in results),
        "skipped_without_labels": sum(
            int(result["skipped_without_labels"]) for result in results
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert labeled DTLD TIFF frames to clean JPEGs."
    )
    parser.add_argument(
        "--target-path",
        "--target_path",
        dest="target_path",
        type=Path,
        required=True,
        help="new or empty output directory containing train/ and test/",
    )
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        type=Path,
        required=True,
        help="raw DTLD root containing the city directories",
    )
    parser.add_argument(
        "--label-path",
        "--label_path",
        dest="label_path",
        type=Path,
        required=True,
        help="DTLD v2 label directory containing DTLD_train.json and DTLD_test.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel split workers (default: 2; use 1 for sequential conversion)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    report = prepare_dtld_images(
        args.target_path,
        args.data_path,
        args.label_path,
        workers=args.workers,
    )
    LOGGER.info("DTLD conversion completed: %s images", report["converted"])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
