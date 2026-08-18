"""Greedy score-ordered IoU matching with explicit retained indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def pairwise_iou(
    boxes_a: Sequence[Sequence[float]] | np.ndarray,
    boxes_b: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    prediction_index: int
    target_index: int
    iou: float


def greedy_iou_match(
    predicted_boxes: Sequence[Sequence[float]] | np.ndarray,
    prediction_scores: Sequence[float] | np.ndarray,
    target_boxes: Sequence[Sequence[float]] | np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> tuple[list[DetectionMatch], list[int], list[int]]:
    """Match predictions by descending score, never reusing a target."""

    predicted = np.asarray(predicted_boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(prediction_scores, dtype=float).reshape(-1)
    targets = np.asarray(target_boxes, dtype=float).reshape(-1, 4)
    if len(predicted) != len(scores):
        raise ValueError("prediction boxes and scores differ in length")
    overlaps = pairwise_iou(predicted, targets)
    remaining = set(range(len(targets)))
    matches: list[DetectionMatch] = []
    unmatched_predictions: list[int] = []
    for prediction_index in np.argsort(-scores, kind="stable"):
        if not remaining:
            unmatched_predictions.append(int(prediction_index))
            continue
        candidates = np.asarray(sorted(remaining), dtype=np.int64)
        values = overlaps[prediction_index, candidates]
        best_position = int(np.argmax(values))
        target_index = int(candidates[best_position])
        iou = float(values[best_position])
        if iou >= iou_threshold:
            matches.append(DetectionMatch(int(prediction_index), target_index, iou))
            remaining.remove(target_index)
        else:
            unmatched_predictions.append(int(prediction_index))
    return matches, unmatched_predictions, sorted(remaining)
