"""Group-aware split assignment and leakage auditing."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .schema import ImageRecord


def _group_key(record: ImageRecord) -> str:
    split_group = record.metadata.get("split_group")
    if isinstance(split_group, str) and split_group:
        return split_group
    return record.sequence_id or record.image_id


def _arrow_stratum(records: list[ImageRecord]) -> str:
    directions = sorted(
        {
            "".join(str(value) for value in arrow.direction_multihot)
            for record in records
            for arrow in record.road_arrows
        }
    )
    return "+".join(directions) if directions else "negative"


def assign_grouped_validation(
    records: Iterable[ImageRecord],
    source_dataset: str,
    val_fraction: float,
    *,
    seed: int = 42,
    stratify_arrows: bool = False,
) -> list[ImageRecord]:
    """Move complete training groups into validation, preserving official test."""

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    materialized = list(records)
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in materialized:
        if record.source_dataset == source_dataset and record.split == "train":
            groups[_group_key(record)].append(record)
    if not groups or val_fraction == 0:
        return materialized

    buckets: dict[str, list[str]] = defaultdict(list)
    for key, items in groups.items():
        stratum = _arrow_stratum(items) if stratify_arrows else "all"
        buckets[stratum].append(key)

    selected: set[str] = set()
    rng = random.Random(f"{seed}:{source_dataset}")
    for keys in buckets.values():
        rng.shuffle(keys)
        if len(keys) <= 1:
            continue
        target_images = max(
            1,
            int(round(sum(len(groups[key]) for key in keys) * val_fraction)),
        )
        count = 0
        for key in keys[:-1]:  # always retain at least one group in train
            selected.add(key)
            count += len(groups[key])
            if count >= target_images:
                break

    # Sparse strata may all have one group. Fill toward the global target while
    # keeping at least one source group in training.
    target_total = max(
        1, int(round(sum(len(items) for items in groups.values()) * val_fraction))
    )
    selected_count = sum(len(groups[key]) for key in selected)
    remaining = [key for key in groups if key not in selected]
    rng.shuffle(remaining)
    for key in remaining[:-1]:
        if selected_count >= target_total:
            break
        selected.add(key)
        selected_count += len(groups[key])

    return [
        replace(record, split="val")
        if (
            record.source_dataset == source_dataset
            and record.split == "train"
            and _group_key(record) in selected
        )
        else record
        for record in materialized
    ]


def protect_official_test_groups(
    records: Iterable[ImageRecord], source_dataset: str
) -> list[ImageRecord]:
    """Move every member of a test-overlapping group into the protected test.

    Some published datasets place augmented variants of one base scene across
    train and test.  Preserving the test side and quarantining the train-side
    members is the conservative correction: no official test sample enters
    optimization and the whole visual group remains evaluation-only.
    """

    materialized = list(records)
    protected_groups = {
        _group_key(record)
        for record in materialized
        if record.source_dataset == source_dataset and record.split == "test"
    }
    corrected: list[ImageRecord] = []
    for record in materialized:
        if (
            record.source_dataset == source_dataset
            and record.split != "test"
            and _group_key(record) in protected_groups
        ):
            corrected.append(
                replace(
                    record,
                    split="test",
                    metadata={
                        **record.metadata,
                        "split_adjustment": "moved_to_protected_test_group",
                    },
                )
            )
        else:
            corrected.append(record)
    return corrected


def assign_default_splits(records: Iterable[ImageRecord], *, seed: int = 42) -> list[ImageRecord]:
    """Apply the definitive validation policy to available source train sets."""

    result = assign_grouped_validation(
        records, "DTLD", 0.20, seed=seed, stratify_arrows=True
    )
    result = assign_grouped_validation(result, "ATLAS", 0.10, seed=seed)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coalesce_content_split_groups(
    records: Iterable[ImageRecord], source_dataset: str
) -> list[ImageRecord]:
    """Join source groups connected by byte-identical images.

    Published augmented datasets can contain the same encoded image under
    different scene IDs.  A union-find over source groups keeps every such
    connected component in one split.  If any member belongs to official
    test, the entire component is conservatively quarantined in test.
    """

    materialized = list(records)
    source_records = [
        record for record in materialized if record.source_dataset == source_dataset
    ]
    parent: dict[str, str] = {
        record.sequence_id or record.image_id: record.sequence_id or record.image_id
        for record in source_records
    }

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root))
        parent[high] = low

    digest_owner: dict[str, str] = {}
    for record in source_records:
        path = Path(record.image_path)
        if not path.exists():
            continue
        group = record.sequence_id or record.image_id
        digest = _sha256(path.resolve())
        owner = digest_owner.setdefault(digest, group)
        union(group, owner)

    component_splits: dict[str, set[str]] = defaultdict(set)
    component_groups: dict[str, set[str]] = defaultdict(set)
    for record in source_records:
        original_group = record.sequence_id or record.image_id
        component = find(original_group)
        component_splits[component].add(record.split)
        component_groups[component].add(original_group)

    corrected: list[ImageRecord] = []
    for record in materialized:
        if record.source_dataset != source_dataset:
            corrected.append(record)
            continue
        original_group = record.sequence_id or record.image_id
        component = find(original_group)
        target_split = "test" if "test" in component_splits[component] else record.split
        metadata = {
            **record.metadata,
            "split_group": f"{source_dataset}:content:{component}",
        }
        if len(component_groups[component]) > 1:
            metadata["content_group_merged"] = True
        if target_split != record.split:
            metadata["split_adjustment"] = "moved_to_protected_test_content_group"
        corrected.append(replace(record, split=target_split, metadata=metadata))
    return corrected


def audit_split_leakage(
    records: Iterable[ImageRecord], *, hash_images: bool = False
) -> dict[str, object]:
    """Report ID, sequence, path, and optional content overlap across splits."""

    materialized = list(records)
    issues: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter(record.split for record in materialized)

    def check(
        kind: str,
        keyed_splits: dict[str, set[str]],
        examples: dict[str, dict[str, dict[str, str]]] | None = None,
    ) -> None:
        for key, splits in keyed_splits.items():
            if len(splits) > 1:
                issue: dict[str, object] = {
                    "kind": kind,
                    "key": key,
                    "splits": sorted(splits),
                }
                if examples is not None:
                    issue["records"] = [
                        examples[key][split] for split in sorted(examples[key])
                    ]
                issues.append(issue)

    ids: dict[str, set[str]] = defaultdict(set)
    sequences: dict[str, set[str]] = defaultdict(set)
    paths: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    hash_examples: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    hash_cache: dict[Path, str] = {}
    existing: list[tuple[ImageRecord, Path, int]] = []
    sizes: dict[int, set[str]] = defaultdict(set)
    missing_images = 0

    for record in materialized:
        ids[record.image_id].add(record.split)
        if record.sequence_id:
            sequences[f"{record.source_dataset}:{record.sequence_id}"].add(record.split)
        path = Path(record.image_path)
        try:
            canonical = str(path.resolve()).casefold()
        except OSError:
            canonical = str(path.absolute()).casefold()
        paths[canonical].add(record.split)
        if hash_images:
            if not path.exists():
                missing_images += 1
                continue
            resolved = path.resolve()
            size = resolved.stat().st_size
            existing.append((record, resolved, size))
            sizes[size].add(record.split)

    check("image_id", ids)
    check("sequence_id", sequences)
    check("image_path", paths)
    if hash_images:
        candidate_sizes = {size for size, splits in sizes.items() if len(splits) > 1}
        for record, resolved, size in existing:
            if size not in candidate_sizes:
                continue
            digest = hash_cache.setdefault(resolved, _sha256(resolved))
            hashes[digest].add(record.split)
            hash_examples[digest].setdefault(
                record.split,
                {
                    "image_id": record.image_id,
                    "source_dataset": record.source_dataset,
                    "split": record.split,
                    "image_path": str(resolved),
                },
            )
        check("sha256", hashes, hash_examples)
    return {
        "ok": not issues,
        "n_records": len(materialized),
        "split_counts": dict(sorted(split_counts.items())),
        "hash_images": hash_images,
        "hashed_images": len(hash_cache),
        "missing_images": missing_images,
        "n_issues": len(issues),
        "issues": issues,
    }


def assert_no_split_leakage(
    records: Iterable[ImageRecord], *, hash_images: bool = False
) -> None:
    report = audit_split_leakage(records, hash_images=hash_images)
    if not report["ok"]:
        preview = report["issues"][:10]  # type: ignore[index]
        raise ValueError(f"split leakage detected: {preview}")
