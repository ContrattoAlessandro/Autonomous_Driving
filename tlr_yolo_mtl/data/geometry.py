"""Geometry helpers used identically by converters, augmentation, and tests."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Sequence

from .schema import BBox


def xywh_to_xyxy(x: float, y: float, width: float, height: float) -> BBox:
    return float(x), float(y), float(x + width), float(y + height)


def xyxy_to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1, y1, x2 - x1, y2 - y1


def yolo_to_xyxy(
    cx: float,
    cy: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    bw, bh = width * image_width, height * image_height
    center_x, center_y = cx * image_width, cy * image_height
    return (
        center_x - bw / 2,
        center_y - bh / 2,
        center_x + bw / 2,
        center_y + bh / 2,
    )


def xyxy_to_yolo(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        ((x1 + x2) / 2) / image_width,
        ((y1 + y2) / 2) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def clip_box(box: Sequence[float], image_width: int, image_height: int) -> BBox | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    clipped = (
        min(max(x1, 0.0), float(image_width)),
        min(max(y1, 0.0), float(image_height)),
        min(max(x2, 0.0), float(image_width)),
        min(max(y2, 0.0), float(image_height)),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def letterbox_parameters(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[float, float, float]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image sizes must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    pad_x = (target_width - source_width * scale) / 2
    pad_y = (target_height - source_height * scale) / 2
    return scale, pad_x, pad_y


def letterbox_box(
    box: Sequence[float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> BBox:
    scale, pad_x, pad_y = letterbox_parameters(source_size, target_size)
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        x1 * scale + pad_x,
        y1 * scale + pad_y,
        x2 * scale + pad_x,
        y2 * scale + pad_y,
    )


def horizontal_flip_box(box: Sequence[float], image_width: int) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in box)
    return float(image_width) - x2, y1, float(image_width) - x1, y2


def image_size(path: str | Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without importing a training dependency."""

    image_path = Path(path)
    with image_path.open("rb") as stream:
        signature = stream.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", signature[16:24])
        if signature[:2] == b"\xff\xd8":
            stream.seek(2)
            return _jpeg_size(stream)
    raise ValueError(f"unsupported or corrupt image format: {image_path}")


def _jpeg_size(stream: BinaryIO) -> tuple[int, int]:
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while True:
        byte = stream.read(1)
        if not byte:
            break
        if byte != b"\xff":
            continue
        marker = stream.read(1)
        while marker == b"\xff":
            marker = stream.read(1)
        if not marker:
            break
        marker_id = marker[0]
        if marker_id in {0xD8, 0xD9}:
            continue
        raw_length = stream.read(2)
        if len(raw_length) != 2:
            break
        length = struct.unpack(">H", raw_length)[0]
        if length < 2:
            break
        if marker_id in sof_markers:
            payload = stream.read(5)
            if len(payload) != 5:
                break
            height, width = struct.unpack(">HH", payload[1:5])
            return width, height
        stream.seek(length - 2, 1)
    raise ValueError("JPEG has no supported SOF marker")

