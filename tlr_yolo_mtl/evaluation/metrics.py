"""Dependency-light metrics for detection, attributes, arrows, and relevance."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .matching import greedy_iou_match


def _array(values: Sequence[float] | np.ndarray, *, dtype: object = float) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def binary_average_precision(
    targets: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Non-interpolated area under the precision-recall staircase."""

    y = _array(targets, dtype=np.int64).reshape(-1)
    s = _array(scores).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("binary targets and scores must have the same shape")
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    sorted_y = y[order]
    true_positives = np.cumsum(sorted_y == 1)
    false_positives = np.cumsum(sorted_y == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    return float(precision[sorted_y == 1].sum() / positives)


def binary_roc_auc(
    targets: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Tie-aware AUROC computed from average ranks."""

    y = _array(targets, dtype=np.int64).reshape(-1)
    s = _array(scores).reshape(-1)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def expected_calibration_error(
    targets: Sequence[int] | np.ndarray,
    confidences: Sequence[float] | np.ndarray,
    *,
    bins: int = 15,
) -> float:
    y = _array(targets, dtype=np.int64).reshape(-1)
    confidence = _array(confidences).reshape(-1)
    if y.shape != confidence.shape:
        raise ValueError("calibration targets and confidences must have equal shape")
    if bins < 1:
        raise ValueError("ECE bins must be positive")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidences must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = bucket == index
        if not selected.any():
            continue
        error += selected.mean() * abs(confidence[selected].mean() - y[selected].mean())
    return float(error)


def brier_score(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float:
    y = _array(targets).reshape(-1)
    p = _array(probabilities).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("Brier targets and probabilities must have equal shape")
    return float(np.square(p - y).mean()) if y.size else float("nan")


def binary_classification_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y = _array(targets, dtype=np.int64).reshape(-1)
    p = _array(probabilities).reshape(-1)
    predicted = p >= threshold
    positive = y == 1
    negative = y == 0
    tp = int((predicted & positive).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "auroc": binary_roc_auc(y, p),
        "auprc": binary_average_precision(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
    }


def multiclass_confusion_matrix(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    *,
    classes: int,
) -> np.ndarray:
    y = _array(targets, dtype=np.int64).reshape(-1)
    predicted = _array(predictions, dtype=np.int64).reshape(-1)
    if y.shape != predicted.shape:
        raise ValueError("multiclass targets and predictions must have equal shape")
    if np.any((y < 0) | (y >= classes) | (predicted < 0) | (predicted >= classes)):
        raise ValueError("multiclass label is outside the configured range")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    return matrix


def multiclass_metrics(confusion: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square")
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(1).astype(float)
    predicted = matrix.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = matrix.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "balanced_accuracy": float(recall[support > 0].mean()) if np.any(support > 0) else float("nan"),
        "macro_f1": float(f1[support > 0].mean()) if np.any(support > 0) else float("nan"),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def multilabel_metrics(
    targets: Sequence[Sequence[int]] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    y = _array(targets, dtype=np.int64)
    p = _array(probabilities)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("multi-label targets/probabilities need equal [N, C] shapes")
    per_class = [
        binary_classification_metrics(y[:, index], p[:, index], threshold=threshold)
        for index in range(y.shape[1])
    ]
    return {
        "macro_f1": float(np.mean([value["f1"] for value in per_class])),
        "macro_precision": float(np.mean([value["precision"] for value in per_class])),
        "macro_recall": float(np.mean([value["recall"] for value in per_class])),
        "exact_match_accuracy": float(np.all((p >= threshold) == y, axis=1).mean()),
        "per_class": per_class,
    }


def threshold_for_minimum_recall(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    minimum_recall: float,
) -> dict[str, float]:
    """Choose the highest-precision threshold satisfying a recall constraint."""

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum recall must be in [0, 1]")
    y = _array(targets, dtype=np.int64).reshape(-1)
    p = _array(probabilities).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metric = binary_classification_metrics(y, p, threshold=float(threshold))
        if metric["recall"] >= minimum_recall:
            feasible.append((float(metric["precision"]), float(threshold), float(metric["recall"])))
    if not feasible:
        raise ValueError("no threshold satisfies the recall constraint")
    precision, threshold, recall = max(feasible, key=lambda value: (value[0], value[1]))
    return {"threshold": threshold, "precision": precision, "recall": recall}


def validation_selection_score(metrics: Mapping[str, float]) -> float:
    """Composite checkpoint score defined by the thesis plan."""

    unified_required = {
        "traffic_light_tiny_ap",
        "arrow_ap",
        "state_macro_f1",
        "round_f1",
        "maneuver_macro_f1",
        "relevance_auprc",
    }
    if unified_required.issubset(metrics):
        return float(
            0.20 * metrics["traffic_light_tiny_ap"]
            + 0.15 * metrics["arrow_ap"]
            + 0.10 * metrics["state_macro_f1"]
            + 0.05 * metrics["round_f1"]
            + 0.15 * metrics["maneuver_macro_f1"]
            + 0.35 * metrics["relevance_auprc"]
        )
    # Historical checkpoints remain scoreable for reproducibility.
    required = {
        "traffic_light_tiny_ap",
        "state_macro_f1",
        "pictogram_macro_f1",
        "arrow_ap",
        "relevance_auprc",
    }
    missing = required.difference(metrics)
    if missing:
        raise KeyError(f"missing validation metrics: {sorted(missing)}")
    return float(
        0.25 * metrics["traffic_light_tiny_ap"]
        + 0.15 * metrics["state_macro_f1"]
        + 0.15 * metrics["pictogram_macro_f1"]
        + 0.15 * metrics["arrow_ap"]
        + 0.30 * metrics["relevance_auprc"]
    )


def compute_ap_from_matches(tp_list: np.ndarray, scores_list: np.ndarray, num_gt: int) -> float:
    """101-point COCO-standard interpolated Average Precision."""
    if num_gt == 0 or len(scores_list) == 0:
        return 0.0
    order = np.argsort(-scores_list, kind="stable")
    tp = tp_list[order]
    fp = 1 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / max(num_gt, 1)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    x = np.linspace(0, 1, 101)
    if hasattr(np, "trapezoid"):
        ap = float(np.trapezoid(np.interp(x, mrec, mpre), x))
    else:
        ap = float(np.trapz(np.interp(x, mrec, mpre), x))
    return ap


def compute_detection_and_attribute_map(
    pred_boxes_list: Sequence[np.ndarray],
    pred_scores_list: Sequence[np.ndarray],
    pred_classes_list: Sequence[np.ndarray],
    gt_boxes_list: Sequence[np.ndarray],
    gt_classes_list: Sequence[np.ndarray],
    pred_states_list: Sequence[np.ndarray] | None = None,
    gt_states_list: Sequence[np.ndarray] | None = None,
    image_shape: tuple[int, int] = (800, 1600),
    iou_thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
) -> dict[str, float]:
    """Computes mAP50, mAP50-95, AP_small, AP_medium, and mAP_state."""
    h, w = image_shape
    num_images = len(pred_boxes_list)
    classes = [0, 1]  # 0: Traffic Light, 1: Road Arrow

    class_iou_tp: dict[tuple[int, float], list[int]] = {(c, iou): [] for c in classes for iou in iou_thresholds}
    class_iou_scores: dict[tuple[int, float], list[float]] = {(c, iou): [] for c in classes for iou in iou_thresholds}
    class_gt_counts: dict[int, int] = {c: 0 for c in classes}

    scale_tp: dict[str, list[int]] = {"small": [], "medium": [], "large": []}
    scale_scores: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    scale_gt_counts: dict[str, int] = {"small": 0, "medium": 0, "large": 0}

    state_tp: dict[int, list[int]] = {s: [] for s in range(4)}
    state_scores: dict[int, list[float]] = {s: [] for s in range(4)}
    state_gt_counts: dict[int, int] = {s: 0 for s in range(4)}

    for img_idx in range(num_images):
        p_b = np.asarray(pred_boxes_list[img_idx], dtype=float).reshape(-1, 4)
        p_s = np.asarray(pred_scores_list[img_idx], dtype=float).reshape(-1)
        p_c = np.asarray(pred_classes_list[img_idx], dtype=np.int64).reshape(-1)
        g_b = np.asarray(gt_boxes_list[img_idx], dtype=float).reshape(-1, 4)
        g_c = np.asarray(gt_classes_list[img_idx], dtype=np.int64).reshape(-1)

        p_state = np.asarray(pred_states_list[img_idx], dtype=np.int64).reshape(-1) if pred_states_list is not None else None
        g_state = np.asarray(gt_states_list[img_idx], dtype=np.int64).reshape(-1) if gt_states_list is not None else None

        for c in classes:
            c_p_mask = (p_c == c)
            c_g_mask = (g_c == c)

            c_p_boxes = p_b[c_p_mask]
            c_p_scores = p_s[c_p_mask]
            c_g_boxes = g_b[c_g_mask]

            class_gt_counts[c] += len(c_g_boxes)

            for iou_thresh in iou_thresholds:
                if len(c_p_boxes) == 0:
                    continue
                if len(c_g_boxes) == 0:
                    for s in c_p_scores:
                        class_iou_tp[(c, iou_thresh)].append(0)
                        class_iou_scores[(c, iou_thresh)].append(float(s))
                    continue

                matches, unmatched_preds, _ = greedy_iou_match(
                    c_p_boxes, c_p_scores, c_g_boxes, iou_threshold=iou_thresh
                )
                matched_pred_indices = {m.prediction_index for m in matches}
                for p_idx, score in enumerate(c_p_scores):
                    is_tp = 1 if p_idx in matched_pred_indices else 0
                    class_iou_tp[(c, iou_thresh)].append(is_tp)
                    class_iou_scores[(c, iou_thresh)].append(float(score))

        # Size bins evaluation (IoU=0.50)
        for g_idx, g_box in enumerate(g_b):
            area = max(g_box[2] - g_box[0], 0.0) * w * max(g_box[3] - g_box[1], 0.0) * h
            side = np.sqrt(max(area, 0.0))
            if side < 32:
                scale_gt_counts["small"] += 1
            elif side < 96:
                scale_gt_counts["medium"] += 1
            else:
                scale_gt_counts["large"] += 1

        if len(p_b) > 0 and len(g_b) > 0:
            matches_50, _, _ = greedy_iou_match(p_b, p_s, g_b, iou_threshold=0.50)
            matched_gts = {m.prediction_index: m.target_index for m in matches_50}
            for p_idx, p_box in enumerate(p_b):
                score = float(p_s[p_idx])
                if p_idx in matched_gts:
                    t_idx = matched_gts[p_idx]
                    g_box = g_b[t_idx]
                    area = max(g_box[2] - g_box[0], 0.0) * w * max(g_box[3] - g_box[1], 0.0) * h
                    side = np.sqrt(max(area, 0.0))
                    bin_name = "small" if side < 32 else ("medium" if side < 96 else "large")
                    scale_tp[bin_name].append(1)
                    scale_scores[bin_name].append(score)
                else:
                    area = max(p_box[2] - p_box[0], 0.0) * w * max(p_box[3] - p_box[1], 0.0) * h
                    side = np.sqrt(max(area, 0.0))
                    bin_name = "small" if side < 32 else ("medium" if side < 96 else "large")
                    scale_tp[bin_name].append(0)
                    scale_scores[bin_name].append(score)

        # State mAP (for TL c=0 at IoU=0.50)
        if p_state is not None and g_state is not None:
            c0_p_mask = p_c == 0
            c0_g_mask = g_c == 0
            c0_p_boxes = p_b[c0_p_mask]
            c0_p_scores = p_s[c0_p_mask]
            c0_p_states = p_state[c0_p_mask]
            c0_g_boxes = g_b[c0_g_mask]
            c0_g_states = g_state[c0_g_mask]

            for s_gt in c0_g_states:
                if 0 <= s_gt < 4:
                    state_gt_counts[int(s_gt)] += 1

            if len(c0_p_boxes) > 0 and len(c0_g_boxes) > 0:
                tl_matches, _, _ = greedy_iou_match(c0_p_boxes, c0_p_scores, c0_g_boxes, iou_threshold=0.50)
                tl_match_dict = {m.prediction_index: m.target_index for m in tl_matches}
                for p_i, score in enumerate(c0_p_scores):
                    pred_st = int(c0_p_states[p_i])
                    if p_i in tl_match_dict:
                        gt_st = int(c0_g_states[tl_match_dict[p_i]])
                        for s_cls in range(4):
                            if pred_st == s_cls and gt_st == s_cls:
                                state_tp[s_cls].append(1)
                                state_scores[s_cls].append(float(score))
                            elif pred_st == s_cls and gt_st != s_cls:
                                state_tp[s_cls].append(0)
                                state_scores[s_cls].append(float(score))
                    else:
                        for s_cls in range(4):
                            if pred_st == s_cls:
                                state_tp[s_cls].append(0)
                                state_scores[s_cls].append(float(score))

    class_ap50 = {}
    class_ap50_95 = {}
    for c in classes:
        ap_list = []
        for iou in iou_thresholds:
            tp_arr = np.array(class_iou_tp[(c, iou)], dtype=float)
            sc_arr = np.array(class_iou_scores[(c, iou)], dtype=float)
            ap = compute_ap_from_matches(tp_arr, sc_arr, class_gt_counts[c])
            ap_list.append(ap)
            if iou == 0.50:
                class_ap50[c] = ap
        class_ap50_95[c] = float(np.mean(ap_list))

    map50 = float(np.mean(list(class_ap50.values())))
    map50_95 = float(np.mean(list(class_ap50_95.values())))

    ap_small = compute_ap_from_matches(
        np.array(scale_tp["small"], dtype=float),
        np.array(scale_scores["small"], dtype=float),
        scale_gt_counts["small"],
    )
    ap_medium = compute_ap_from_matches(
        np.array(scale_tp["medium"], dtype=float),
        np.array(scale_scores["medium"], dtype=float),
        scale_gt_counts["medium"],
    )

    state_aps = []
    for s in range(4):
        ap_s = compute_ap_from_matches(
            np.array(state_tp[s], dtype=float),
            np.array(state_scores[s], dtype=float),
            state_gt_counts[s],
        )
        state_aps.append(ap_s)
    map_state = float(np.mean(state_aps))

    return {
        "map50": map50,
        "map50_95": map50_95,
        "ap_tl_50": class_ap50.get(0, 0.0),
        "ap_arrow_50": class_ap50.get(1, 0.0),
        "ap_small": ap_small,
        "ap_medium": ap_medium,
        "map_state": map_state,
    }


AREA_BUCKETS: dict[str, tuple[float, float]] = {
    "<32": (0.0, 32.0),
    "32-64": (32.0, 64.0),
    "64-128": (64.0, 128.0),
    "128-256": (128.0, 256.0),
    "256-512": (256.0, 512.0),
    ">512": (512.0, float("inf")),
}

SIDE_BUCKETS: dict[str, tuple[float, float]] = {
    "<4": (0.0, 4.0),
    "4-6": (4.0, 6.0),
    "6-8": (6.0, 8.0),
    "8-12": (8.0, 12.0),
    ">12": (12.0, float("inf")),
}


def _get_bucket_name(value: float, buckets: Mapping[str, tuple[float, float]]) -> str | None:
    for name, (low, high) in buckets.items():
        if low <= value < high:
            return name
    return None


def compute_granular_scale_metrics(
    pred_boxes_list: Sequence[np.ndarray],
    pred_scores_list: Sequence[np.ndarray],
    pred_classes_list: Sequence[np.ndarray],
    gt_boxes_list: Sequence[np.ndarray],
    gt_classes_list: Sequence[np.ndarray],
    *,
    target_class: int = 0,
    image_shape: tuple[int, int] = (800, 1600),
    iou_thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    conf_threshold: float = 0.05,
    p3_stride: float = 8.0,
) -> dict[str, Any]:
    """Calculate fine-grained detection, recall, AP, and localization metrics across size buckets.

    Supports both area buckets (<32, 32-64, 64-128, 128-256, 256-512, >512 px^2)
    and minimum side buckets (<4, 4-6, 6-8, 8-12, >12 px).
    """
    h, w = image_shape
    num_images = len(pred_boxes_list)
    p3_cell_area = p3_stride * p3_stride

    def _init_bucket_stats(bucket_dict: Mapping[str, tuple[float, float]]):
        return {
            b_name: {
                "n_gt": 0,
                "n_tp_conf": 0,
                "n_fp_conf": 0,
                "iou_tps": {iou: [] for iou in iou_thresholds},
                "iou_scores": {iou: [] for iou in iou_thresholds},
                "delta_x": [],
                "delta_y": [],
                "delta_r": [],
                "delta_w": [],
                "delta_h": [],
                "rel_delta_w": [],
                "rel_delta_h": [],
                "gt_areas": [],
                "gt_min_sides": [],
            }
            for b_name in bucket_dict
        }

    area_stats = _init_bucket_stats(AREA_BUCKETS)
    side_stats = _init_bucket_stats(SIDE_BUCKETS)

    for img_idx in range(num_images):
        p_b = np.asarray(pred_boxes_list[img_idx], dtype=float).reshape(-1, 4)
        p_s = np.asarray(pred_scores_list[img_idx], dtype=float).reshape(-1)
        p_c = np.asarray(pred_classes_list[img_idx], dtype=np.int64).reshape(-1)
        g_b = np.asarray(gt_boxes_list[img_idx], dtype=float).reshape(-1, 4)
        g_c = np.asarray(gt_classes_list[img_idx], dtype=np.int64).reshape(-1)

        # Filter target class
        p_mask = (p_c == target_class)
        g_mask = (g_c == target_class)

        c_p_boxes = p_b[p_mask]
        c_p_scores = p_s[p_mask]
        c_g_boxes = g_b[g_mask]

        # Calculate GT dimensions
        gt_areas = []
        gt_min_sides = []
        gt_area_bucket = []
        gt_side_bucket = []
        for gb in c_g_boxes:
            gw = max(gb[2] - gb[0], 0.0) * w
            gh = max(gb[3] - gb[1], 0.0) * h
            ga = max(gw * gh, 0.0)
            gm = max(min(gw, gh), 0.0)
            gt_areas.append(ga)
            gt_min_sides.append(gm)
            ab = _get_bucket_name(ga, AREA_BUCKETS)
            sb = _get_bucket_name(gm, SIDE_BUCKETS)
            gt_area_bucket.append(ab)
            gt_side_bucket.append(sb)
            if ab in area_stats:
                area_stats[ab]["n_gt"] += 1
                area_stats[ab]["gt_areas"].append(ga)
            if sb in side_stats:
                side_stats[sb]["n_gt"] += 1
                side_stats[sb]["gt_min_sides"].append(gm)

        # Calculate Pred dimensions
        pred_areas = []
        pred_min_sides = []
        pred_area_bucket = []
        pred_side_bucket = []
        for pb in c_p_boxes:
            pw = max(pb[2] - pb[0], 0.0) * w
            ph = max(pb[3] - pb[1], 0.0) * h
            pa = max(pw * ph, 0.0)
            pm = max(min(pw, ph), 0.0)
            pred_areas.append(pa)
            pred_min_sides.append(pm)
            pred_area_bucket.append(_get_bucket_name(pa, AREA_BUCKETS))
            pred_side_bucket.append(_get_bucket_name(pm, SIDE_BUCKETS))

        # Evaluate across IoU thresholds
        for iou_thresh in iou_thresholds:
            if len(c_p_boxes) == 0:
                continue
            if len(c_g_boxes) == 0:
                for p_i, score in enumerate(c_p_scores):
                    ab = pred_area_bucket[p_i]
                    sb = pred_side_bucket[p_i]
                    if ab in area_stats:
                        area_stats[ab]["iou_tps"][iou_thresh].append(0)
                        area_stats[ab]["iou_scores"][iou_thresh].append(float(score))
                    if sb in side_stats:
                        side_stats[sb]["iou_tps"][iou_thresh].append(0)
                        side_stats[sb]["iou_scores"][iou_thresh].append(float(score))
                continue

            matches, unmatched_preds, _ = greedy_iou_match(
                c_p_boxes, c_p_scores, c_g_boxes, iou_threshold=iou_thresh
            )
            matched_pred_to_gt = {m.prediction_index: m.target_index for m in matches}

            for p_i, score in enumerate(c_p_scores):
                if p_i in matched_pred_to_gt:
                    gt_i = matched_pred_to_gt[p_i]
                    # Target GT bucket determines evaluation bucket
                    ab = gt_area_bucket[gt_i]
                    sb = gt_side_bucket[gt_i]
                    if ab in area_stats:
                        area_stats[ab]["iou_tps"][iou_thresh].append(1)
                        area_stats[ab]["iou_scores"][iou_thresh].append(float(score))
                    if sb in side_stats:
                        side_stats[sb]["iou_tps"][iou_thresh].append(1)
                        side_stats[sb]["iou_scores"][iou_thresh].append(float(score))

                    # For IoU 0.50, track operational TP / errors
                    if abs(iou_thresh - 0.50) < 1e-4:
                        if score >= conf_threshold:
                            if ab in area_stats:
                                area_stats[ab]["n_tp_conf"] += 1
                            if sb in side_stats:
                                side_stats[sb]["n_tp_conf"] += 1

                        # Localization and scale errors
                        gb = c_g_boxes[gt_i]
                        pb = c_p_boxes[p_i]
                        gt_cx = (gb[0] + gb[2]) / 2.0 * w
                        gt_cy = (gb[1] + gb[3]) / 2.0 * h
                        pred_cx = (pb[0] + pb[2]) / 2.0 * w
                        pred_cy = (pb[1] + pb[3]) / 2.0 * h

                        gt_w = max(gb[2] - gb[0], 0.0) * w
                        gt_h = max(gb[3] - gb[1], 0.0) * h
                        pred_w = max(pb[2] - pb[0], 0.0) * w
                        pred_h = max(pb[3] - pb[1], 0.0) * h

                        dx = abs(pred_cx - gt_cx)
                        dy = abs(pred_cy - gt_cy)
                        dr = float(np.sqrt(dx * dx + dy * dy))
                        dw = abs(pred_w - gt_w)
                        dh = abs(pred_h - gt_h)
                        rel_dw = dw / max(gt_w, 1e-4)
                        rel_dh = dh / max(gt_h, 1e-4)

                        if ab in area_stats:
                            area_stats[ab]["delta_x"].append(dx)
                            area_stats[ab]["delta_y"].append(dy)
                            area_stats[ab]["delta_r"].append(dr)
                            area_stats[ab]["delta_w"].append(dw)
                            area_stats[ab]["delta_h"].append(dh)
                            area_stats[ab]["rel_delta_w"].append(rel_dw)
                            area_stats[ab]["rel_delta_h"].append(rel_dh)

                        if sb in side_stats:
                            side_stats[sb]["delta_x"].append(dx)
                            side_stats[sb]["delta_y"].append(dy)
                            side_stats[sb]["delta_r"].append(dr)
                            side_stats[sb]["delta_w"].append(dw)
                            side_stats[sb]["delta_h"].append(dh)
                            side_stats[sb]["rel_delta_w"].append(rel_dw)
                            side_stats[sb]["rel_delta_h"].append(rel_dh)
                else:
                    # Unmatched prediction: bucket based on predicted box size
                    ab = pred_area_bucket[p_i]
                    sb = pred_side_bucket[p_i]
                    if ab in area_stats:
                        area_stats[ab]["iou_tps"][iou_thresh].append(0)
                        area_stats[ab]["iou_scores"][iou_thresh].append(float(score))
                        if abs(iou_thresh - 0.50) < 1e-4 and score >= conf_threshold:
                            area_stats[ab]["n_fp_conf"] += 1
                    if sb in side_stats:
                        side_stats[sb]["iou_tps"][iou_thresh].append(0)
                        side_stats[sb]["iou_scores"][iou_thresh].append(float(score))
                        if abs(iou_thresh - 0.50) < 1e-4 and score >= conf_threshold:
                            side_stats[sb]["n_fp_conf"] += 1

    def _summarize_bucket_dict(stats_dict: Mapping[str, Any], is_area: bool) -> dict[str, Any]:
        out = {}
        for b_name, stats in stats_dict.items():
            n_gt = stats["n_gt"]
            tp = stats["n_tp_conf"]
            fp = stats["n_fp_conf"]
            fn = max(0, n_gt - tp)
            recall = tp / max(n_gt, 1) if n_gt > 0 else 0.0
            precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / max(precision + recall, 1e-6) if (precision + recall) > 0 else 0.0

            # AP50 and AP50-95
            ap_list = []
            ap50 = 0.0
            for iou in iou_thresholds:
                tps = np.array(stats["iou_tps"][iou], dtype=float)
                scs = np.array(stats["iou_scores"][iou], dtype=float)
                ap_iou = compute_ap_from_matches(tps, scs, n_gt)
                ap_list.append(ap_iou)
                if abs(iou - 0.50) < 1e-4:
                    ap50 = ap_iou
            ap50_95 = float(np.mean(ap_list)) if ap_list else 0.0

            # Localization errors
            dx_arr = np.array(stats["delta_x"], dtype=float)
            dy_arr = np.array(stats["delta_y"], dtype=float)
            dr_arr = np.array(stats["delta_r"], dtype=float)
            dw_arr = np.array(stats["delta_w"], dtype=float)
            dh_arr = np.array(stats["delta_h"], dtype=float)
            rel_dw_arr = np.array(stats["rel_delta_w"], dtype=float)
            rel_dh_arr = np.array(stats["rel_delta_h"], dtype=float)

            n_matches = len(dr_arr)
            mean_dx = float(np.mean(dx_arr)) if n_matches > 0 else 0.0
            mean_dy = float(np.mean(dy_arr)) if n_matches > 0 else 0.0
            mean_dr = float(np.mean(dr_arr)) if n_matches > 0 else 0.0
            median_dr = float(np.median(dr_arr)) if n_matches > 0 else 0.0
            rmse_dr = float(np.sqrt(np.mean(dr_arr ** 2))) if n_matches > 0 else 0.0

            mean_dw = float(np.mean(dw_arr)) if n_matches > 0 else 0.0
            mean_dh = float(np.mean(dh_arr)) if n_matches > 0 else 0.0
            rel_dw = float(np.mean(rel_dw_arr)) if n_matches > 0 else 0.0
            rel_dh = float(np.mean(rel_dh_arr)) if n_matches > 0 else 0.0

            # Grid cell coverage
            if is_area:
                gt_vals = stats["gt_areas"]
                mean_size = float(np.mean(gt_vals)) if gt_vals else 0.0
                cell_coverage = mean_size / p3_cell_area
            else:
                gt_vals = stats["gt_min_sides"]
                mean_size = float(np.mean(gt_vals)) if gt_vals else 0.0
                cell_coverage = mean_size / p3_stride

            out[b_name] = {
                "n_gt": n_gt,
                "n_tp": tp,
                "n_fp": fp,
                "n_fn": fn,
                "n_matched_50": n_matches,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": ap50,
                "ap50_95": ap50_95,
                "mean_dx": mean_dx,
                "mean_dy": mean_dy,
                "mean_dr": mean_dr,
                "median_dr": median_dr,
                "rmse_dr": rmse_dr,
                "mean_dw": mean_dw,
                "mean_dh": mean_dh,
                "rel_delta_w": rel_dw,
                "rel_delta_h": rel_dh,
                "mean_ground_truth_size": mean_size,
                "p3_stride_coverage_ratio": cell_coverage,
            }
        return out

    return {
        "area_buckets": _summarize_bucket_dict(area_stats, is_area=True),
        "side_buckets": _summarize_bucket_dict(side_stats, is_area=False),
        "p3_stride": p3_stride,
        "p3_cell_area": p3_cell_area,
        "image_shape": image_shape,
        "conf_threshold": conf_threshold,
    }

