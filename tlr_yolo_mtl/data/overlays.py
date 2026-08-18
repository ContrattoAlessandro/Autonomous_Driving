"""Sample loss-aware annotation overlays for manual QA."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import ImageRecord


def _sample(records: list[ImageRecord], fraction: float, seed: int) -> list[ImageRecord]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    by_source: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_dataset].append(record)
    selected: list[ImageRecord] = []
    for source_records in by_source.values():
        count = max(1, math.ceil(len(source_records) * fraction))
        selected.extend(rng.sample(source_records, min(count, len(source_records))))
    return selected


def generate_overlays(
    records: Iterable[ImageRecord],
    output_dir: str | Path,
    *,
    fraction: float = 0.01,
    seed: int = 42,
    max_size: tuple[int, int] = (1600, 1000),
) -> list[Path]:
    """Render at least ``fraction`` per source; invalid attributes show as ``?``."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for overlays; install the project requirements"
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = _sample(list(records), fraction, seed)
    written: list[Path] = []

    for record in selected:
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        image.thumbnail(max_size)
        scale_x = image.width / record.original_width
        scale_y = image.height / record.original_height
        draw = ImageDraw.Draw(image)

        def scaled(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            return (
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            )

        for item in record.ignore_regions:
            box = scaled(item.bbox_xyxy)
            draw.rectangle(box, outline=(255, 160, 0), width=3)
            draw.line((box[0], box[1], box[2], box[3]), fill=(255, 160, 0), width=2)
            draw.line((box[0], box[3], box[2], box[1]), fill=(255, 160, 0), width=2)
            draw.text((box[0], max(0, box[1] - 12)), f"IGNORE {item.reason}", fill=(255, 160, 0))

        for item in record.traffic_lights:
            box = scaled(item.bbox_xyxy)
            color = (40, 220, 80) if item.relevance == 1 else (255, 70, 70)
            draw.rectangle(box, outline=color, width=2)
            state = item.state if item.valid_state else "state?"
            pictogram = item.pictogram if item.valid_pictogram else "picto?"
            relevance = str(item.relevance) if item.valid_relevance else "?"
            draw.text(
                (box[0], max(0, box[1] - 12)),
                f"TL {state}/{pictogram} rel={relevance}",
                fill=color,
            )

        for item in record.road_arrows:
            box = scaled(item.bbox_xyxy)
            draw.rectangle(box, outline=(40, 160, 255), width=3)
            direction = "".join(str(value) for value in item.direction_multihot)
            draw.text((box[0], max(0, box[1] - 12)), f"AR {direction}", fill=(40, 160, 255))

        safe_id = record.image_id.replace("/", "__").replace("\\", "__")
        destination = output / f"{safe_id}.jpg"
        image.save(destination, quality=92)
        written.append(destination)
    return written

