"""Small dependency-free helpers shared by dataset converters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..geometry import image_size
from ..schema import ImageRecord

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(slots=True)
class ConversionResult:
    records: list[ImageRecord]
    stats: Counter[str]


def image_index(root: Path, *, recursive: bool = False) -> dict[str, Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    return {
        path.stem: path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def dimensions(
    path: Path,
    fallback: tuple[int, int],
    *,
    verify: bool,
    prefer_actual: bool = False,
) -> tuple[int, int]:
    if verify or prefer_actual:
        try:
            actual = image_size(path)
        except Exception:
            if verify:
                raise
        else:
            if actual != fallback:
                return actual
    return fallback


def parse_yolo_row(line: str) -> tuple[int, float, float, float, float] | None:
    fields = line.split()
    if len(fields) < 5:
        return None
    try:
        class_id = int(float(fields[0]))
        cx, cy, width, height = (float(value) for value in fields[1:5])
    except ValueError:
        return None
    return class_id, cx, cy, width, height
