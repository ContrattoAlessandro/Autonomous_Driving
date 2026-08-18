"""Streaming JSONL I/O for unified manifests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from .schema import ImageRecord


def write_records(path: str | Path, records: Iterable[ImageRecord]) -> int:
    """Atomically write one validated :class:`ImageRecord` per JSONL row."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=output.parent
    ) as stream:
        temporary = Path(stream.name)
        try:
            for record in records:
                record.validate()
                json.dump(record.to_dict(), stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
                count += 1
        except Exception:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output)
    return count


def read_records(path: str | Path) -> Iterator[ImageRecord]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield ImageRecord.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(
                        f"cannot parse {input_path}:{line_number}: {exc}"
                    ) from exc
        return

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("records", payload.get("images", ()))
    for row in rows:
        yield ImageRecord.from_dict(row)


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

