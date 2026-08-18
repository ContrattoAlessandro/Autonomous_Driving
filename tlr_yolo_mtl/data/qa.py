"""Automatic dataset-distribution and annotation-quality report."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .geometry import letterbox_parameters
from .schema import ImageRecord, TaskValidity
from .splits import audit_split_leakage
from .taxonomy import factor_pictogram


def _size_bin(width: float) -> str:
    if width <= 4:
        return "tiny_w_le_4"
    if width <= 8:
        return "very_small_w_4_8"
    if width <= 16:
        return "small_w_8_16"
    return "normal_w_gt_16"


def _counter_dict(counter: Counter[object]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_qa_report(
    records: Iterable[ImageRecord],
    *,
    network_size: tuple[int, int] = (1600, 800),
    hash_images: bool = False,
) -> dict[str, object]:
    materialized = list(records)
    images_by_dataset: Counter[str] = Counter()
    images_by_split: Counter[str] = Counter()
    tasks_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    states: Counter[str] = Counter()
    pictograms: Counter[str] = Counter()
    round_targets: Counter[str] = Counter()
    maneuvers: Counter[str] = Counter()
    relevance: Counter[str] = Counter()
    occlusion: Counter[str] = Counter()
    arrow_directions: Counter[str] = Counter()
    arrow_ego_lane: Counter[str] = Counter()
    size_bins: Counter[str] = Counter()
    ignore_reasons: Counter[str] = Counter()
    invalid_attributes: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    duplicate_annotations = 0

    for record in materialized:
        images_by_dataset[record.source_dataset] += 1
        images_by_split[record.split] += 1
        for task_name in TaskValidity.__slots__:
            if getattr(record.task_valid, task_name):
                tasks_by_dataset[record.source_dataset][task_name] += 1
        scale, _, _ = letterbox_parameters(
            (record.original_width, record.original_height), network_size
        )
        seen_boxes: set[tuple[str, tuple[float, float, float, float]]] = set()

        for item in record.traffic_lights:
            key = ("tl", item.bbox_xyxy)
            if key in seen_boxes:
                duplicate_annotations += 1
            seen_boxes.add(key)
            x1, y1, x2, y2 = item.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                anomaly_counts["zero_area_boxes"] += 1
            if x1 < 0 or y1 < 0 or x2 > record.original_width or y2 > record.original_height:
                anomaly_counts["out_of_image_boxes"] += 1
            size_bins[_size_bin((x2 - x1) * scale)] += 1
            states[item.state if item.valid_state else "masked"] += 1
            pictograms[item.pictogram if item.valid_pictogram else "masked"] += 1
            factorized = factor_pictogram(
                item.pictogram
                if item.pictogram is not None
                else item.source_attributes.get("pictogram")
            )
            valid_round = item.valid_round or factorized.valid_round
            round_value = item.round_target if item.valid_round else factorized.round
            round_targets[str(round_value) if valid_round else "masked"] += 1
            valid_maneuver = item.valid_maneuver or factorized.valid_maneuver
            maneuver_value = (
                item.maneuver_multihot
                if item.valid_maneuver
                else factorized.maneuver
            )
            maneuvers[
                "".join(str(value) for value in maneuver_value)
                if valid_maneuver and maneuver_value is not None
                else "masked"
            ] += 1
            occlusion[item.occlusion] += 1
            if item.valid_relevance:
                relevance[str(item.relevance)] += 1
            if not item.valid_state:
                raw = item.source_attributes.get("state", "unknown")
                invalid_attributes[f"state/{raw}"] += 1
            if not item.valid_pictogram:
                raw = item.source_attributes.get(
                    "pictogram", item.source_attributes.get("class_name", "unknown")
                )
                invalid_attributes[f"pictogram/{raw}"] += 1

        for item in record.road_arrows:
            key = ("arrow", item.bbox_xyxy)
            if key in seen_boxes:
                duplicate_annotations += 1
            seen_boxes.add(key)
            arrow_directions[
                "".join(str(value) for value in item.direction_multihot)
            ] += 1
            arrow_ego_lane[
                str(item.is_ego_lane) if item.valid_ego_lane else "masked"
            ] += 1
        for item in record.ignore_regions:
            ignore_reasons[item.reason] += 1

    leakage = audit_split_leakage(materialized, hash_images=hash_images)
    return {
        "schema": "TLR-YOLO-MTL unified dataset QA v1",
        "network_size": list(network_size),
        "n_images": len(materialized),
        "images_by_dataset": _counter_dict(images_by_dataset),
        "images_by_split": _counter_dict(images_by_split),
        "task_valid_images_by_dataset": {
            source: _counter_dict(counter)
            for source, counter in sorted(tasks_by_dataset.items())
        },
        "traffic_lights": {
            "count": sum(states.values()),
            "states": _counter_dict(states),
            "pictograms": _counter_dict(pictograms),
            "round": _counter_dict(round_targets),
            "maneuvers": _counter_dict(maneuvers),
            "relevance": _counter_dict(relevance),
            "occlusion": _counter_dict(occlusion),
            "network_width_bins": _counter_dict(size_bins),
        },
        "road_arrows": {
            "count": sum(arrow_directions.values()),
            "directions": _counter_dict(arrow_directions),
            "ego_lane": _counter_dict(arrow_ego_lane),
        },
        "ignore_regions": {
            "count": sum(ignore_reasons.values()),
            "reasons": _counter_dict(ignore_reasons),
        },
        "invalid_source_attributes": _counter_dict(invalid_attributes),
        "annotation_anomalies": {
            **_counter_dict(anomaly_counts),
            "duplicate_annotations": duplicate_annotations,
        },
        "split_leakage": leakage,
    }
