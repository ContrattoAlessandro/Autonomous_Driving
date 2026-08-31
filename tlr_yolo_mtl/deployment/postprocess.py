"""Class-aware NMS for the unified padded-set multi-task output."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torchvision.ops import nms

from ..model.arrows import gather_arrow_directions
from ..model.attributes import gather_candidate_attributes
from ..model.quality import compute_scale_conditioned_quality_scores
from ..model.relevance import (
    combine_detection_relevance_scores,
    gather_candidate_relevance,
)
from ..model.unified import ROAD_ARROW_CLASS, TRAFFIC_LIGHT_CLASS, gather_candidate_outputs


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [detections, 4]")
    center = boxes[:, :2]
    half_size = boxes[:, 2:].clamp_min(0) / 2
    return torch.cat((center - half_size, center + half_size), dim=1)



def compute_pairwise_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """Compute pairwise Intersection over Union (IoU) matrix in [0, 1]."""
    if boxes1.ndim != 2 or boxes1.shape[1] != 4 or boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise ValueError("boxes1 and boxes2 must have shape [N, 4] and [M, 4]")
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(eps)


def compute_pairwise_nwd(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    constant: float | torch.Tensor = 12.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute pairwise Normalized Wasserstein Distance (NWD) matrix in (0, 1]."""
    if isinstance(constant, (int, float)):
        if constant <= 0:
            raise ValueError("NWD normalization constant must be positive")
    elif isinstance(constant, torch.Tensor):
        if (constant <= 0).any():
            raise ValueError("NWD normalization constant tensor must contain strictly positive values")
    else:
        raise TypeError(f"constant must be a float or torch.Tensor, got {type(constant)}")
    if boxes1.ndim != 2 or boxes1.shape[1] != 4 or boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise ValueError("boxes1 and boxes2 must have shape [N, 4] and [M, 4]")
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    c1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2.0
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2.0
    s1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.0)
    s2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.0)

    d_center = c1[:, None, :] - c2[None, :, :]
    d_size = s1[:, None, :] - s2[None, :, :]
    w2 = d_center.square().sum(-1) + 0.25 * d_size.square().sum(-1)
    return torch.exp(-torch.sqrt(w2.clamp_min(eps)) / constant)


def nwd_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    nwd_threshold: float = 0.5,
    nwd_constant: float = 12.0,
) -> torch.Tensor:
    """Greedy Non-Maximum Suppression using Gaussian Normalized Wasserstein Distance."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape [N] matching boxes")
    num_boxes = boxes.shape[0]
    if num_boxes == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if num_boxes == 1:
        return torch.tensor([0], dtype=torch.long, device=boxes.device)

    is_cuda = boxes.is_cuda
    orig_device = boxes.device
    if is_cuda and num_boxes <= 500:
        boxes_work = boxes.cpu()
        scores_work = scores.cpu()
    else:
        boxes_work = boxes
        scores_work = scores

    order = scores_work.sort(descending=True).indices
    sorted_boxes = boxes_work[order]
    sim_matrix = compute_pairwise_nwd(sorted_boxes, sorted_boxes, constant=nwd_constant)
    suppress_matrix = sim_matrix >= nwd_threshold

    keep: list[torch.Tensor] = []
    suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes_work.device)
    for i in range(num_boxes):
        if suppressed[i]:
            continue
        keep.append(order[i])
        suppressed |= suppress_matrix[i]
    result = torch.stack(keep)
    return result.to(orig_device) if is_cuda and num_boxes <= 500 else result


def size_adaptive_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    classes: torch.Tensor | None = None,
    quality_scores: torch.Tensor | None = None,
    quality_alpha: float = 0.70,
    scale_conditioned: bool = False,
    alpha_min: float = 0.38,
    alpha_max: float = 0.90,
    iou_threshold: float = 0.7,
    nwd_threshold: float = 0.5,
    nwd_constant: float = 12.0,
    nwd_constant_tl: float | None = None,
    nwd_constant_arrow: float | None = None,
    nwd_constant_by_class: Mapping[int, float] | Sequence[float] | None = None,
    area_threshold: float = 64.0,
    area_threshold_tl: float | None = None,
    area_threshold_arrow: float | None = None,
    area_threshold_by_class: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    """Size-Adaptive NMS: Gaussian NWD for tiny boxes (< area_thresh), IoU for larger boxes.

    Supports class-decoupled NWD constants and area thresholds (e.g. C_TL=12.0, C_arrow=24.0).
    If quality_scores is provided, candidate ranking order is determined by the joint
    quality-aware score: s = scores^alpha * quality_scores^(1 - alpha) or continuous scale-conditioned scoring.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape [N] matching boxes")
    num_boxes = boxes.shape[0]
    if num_boxes == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if num_boxes == 1:
        return torch.tensor([0], dtype=torch.long, device=boxes.device)

    is_cuda = boxes.is_cuda
    orig_device = boxes.device
    if is_cuda and num_boxes <= 500:
        boxes_work = boxes.cpu()
        scores_work = scores.cpu()
        classes_work = classes.cpu() if classes is not None else None
        quality_work = quality_scores.cpu() if quality_scores is not None else None
    else:
        boxes_work = boxes
        scores_work = scores
        classes_work = classes
        quality_work = quality_scores

    if quality_work is not None:
        if scale_conditioned:
            ranking_scores = compute_scale_conditioned_quality_scores(
                scores,
                quality_work,
                boxes,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        elif quality_alpha < 1.0:
            p = scores.clamp(1e-7, 1.0)
            q = quality_work.clamp(1e-7, 1.0)
            ranking_scores = p.pow(quality_alpha) * q.pow(1.0 - quality_alpha)
        else:
            ranking_scores = scores
    else:
        ranking_scores = scores

    pre_cap = 3000
    if num_boxes > pre_cap:
        topk_indices = ranking_scores.topk(pre_cap).indices
        boxes_work = boxes[topk_indices].cpu()
        ranking_scores = ranking_scores[topk_indices].cpu()
        classes_work = classes[topk_indices].cpu() if classes is not None else None
        num_boxes = pre_cap
        order = ranking_scores.sort(descending=True).indices
        sorted_boxes = boxes_work[order]
        orig_indices = topk_indices[order]
    else:
        boxes_work = boxes.cpu() if is_cuda else boxes
        ranking_scores = ranking_scores.cpu() if is_cuda else ranking_scores
        classes_work = classes.cpu() if (classes is not None and is_cuda) else classes
        order = ranking_scores.sort(descending=True).indices
        sorted_boxes = boxes_work[order]
        orig_indices = order

    areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]).clamp_min(0) * (
        sorted_boxes[:, 3] - sorted_boxes[:, 1]
    ).clamp_min(0)

    # Class-aware area threshold and NWD constant handling
    if classes_work is not None:
        sorted_classes = classes_work[order].long()
        # Area threshold mapping
        a_map: dict[int, float] = {}
        if area_threshold_by_class is not None:
            if isinstance(area_threshold_by_class, Mapping):
                a_map = {int(k): float(v) for k, v in area_threshold_by_class.items()}
            else:
                a_map = {i: float(v) for i, v in enumerate(area_threshold_by_class)}
        elif area_threshold_tl is not None or area_threshold_arrow is not None:
            a_map[0] = float(area_threshold_tl if area_threshold_tl is not None else area_threshold)
            a_map[1] = float(area_threshold_arrow if area_threshold_arrow is not None else 1024.0)

        if a_map:
            box_area_thresh = torch.full((num_boxes,), area_threshold, dtype=sorted_boxes.dtype, device=sorted_boxes.device)
            for cls_k, val in a_map.items():
                box_area_thresh[sorted_classes == cls_k] = val
            is_tiny = areas < box_area_thresh
        else:
            is_tiny = areas < area_threshold

        # NWD constant mapping
        c_map: dict[int, float] = {}
        if nwd_constant_by_class is not None:
            if isinstance(nwd_constant_by_class, Mapping):
                c_map = {int(k): float(v) for k, v in nwd_constant_by_class.items()}
            else:
                c_map = {i: float(v) for i, v in enumerate(nwd_constant_by_class)}
        elif nwd_constant_tl is not None or nwd_constant_arrow is not None:
            c_map[0] = float(nwd_constant_tl if nwd_constant_tl is not None else nwd_constant)
            c_map[1] = float(nwd_constant_arrow if nwd_constant_arrow is not None else 24.0)

        if c_map:
            c_vec = torch.full((num_boxes,), nwd_constant, dtype=sorted_boxes.dtype, device=sorted_boxes.device)
            for cls_k, val in c_map.items():
                c_vec[sorted_classes == cls_k] = val
            constant_matrix = torch.maximum(c_vec[:, None], c_vec[None, :])
        else:
            constant_matrix = nwd_constant
    else:
        is_tiny = areas < area_threshold
        constant_matrix = nwd_constant

    both_tiny = is_tiny[:, None] & is_tiny[None, :]

    iou_matrix = compute_pairwise_iou(sorted_boxes, sorted_boxes)
    nwd_matrix = compute_pairwise_nwd(sorted_boxes, sorted_boxes, constant=constant_matrix)

    suppress_matrix = torch.where(
        both_tiny,
        nwd_matrix >= nwd_threshold,
        iou_matrix >= iou_threshold,
    )

    keep: list[torch.Tensor] = []
    suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes_work.device)
    for i in range(num_boxes):
        if suppressed[i]:
            continue
        keep.append(orig_indices[i])
        suppressed |= suppress_matrix[i]
    if not keep:
        return torch.empty(0, dtype=torch.long, device=orig_device)
    result = torch.stack(keep)
    return result.to(orig_device)


# First-class alias for size-adaptive NWD NMS
size_adaptive_nwd_nms = size_adaptive_nms


def retained_nms_indices(
    decoded: torch.Tensor,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    nms_type: str = "standard",
    nwd_threshold: float = 0.5,
    nwd_constant: float = 12.0,
    nwd_area_threshold: float = 64.0,
    quality_scores: torch.Tensor | None = None,
    quality_alpha: float = 0.70,
) -> list[torch.Tensor]:
    """Return original dense indices for each image after score filter/NMS."""

    if decoded.ndim != 3 or decoded.shape[1] != 5:
        raise ValueError("single-class decoded detections must have shape [B, 5, A]")
    if max_detections < 1:
        raise ValueError("max detections must be positive")
    retained: list[torch.Tensor] = []
    for image in range(decoded.shape[0]):
        scores = decoded[image, 4]
        candidates = torch.nonzero(
            scores >= confidence_threshold, as_tuple=False
        ).reshape(-1)
        if candidates.numel() == 0:
            retained.append(candidates)
            continue
        boxes = xywh_to_xyxy(decoded[image, :4, candidates].transpose(0, 1))
        cand_scores = scores[candidates]
        cand_qual = quality_scores[image, candidates] if quality_scores is not None else None
        if nms_type in ("size_adaptive", "adaptive_nwd"):
            kept_local = size_adaptive_nms(
                boxes,
                cand_scores,
                quality_scores=cand_qual,
                quality_alpha=quality_alpha,
                iou_threshold=iou_threshold,
                nwd_threshold=nwd_threshold,
                nwd_constant=nwd_constant,
                area_threshold=nwd_area_threshold,
            )[:max_detections]
        elif nms_type in ("nwd", "pure_nwd"):
            kept_local = nwd_nms(
                boxes,
                cand_scores,
                nwd_threshold=nwd_threshold,
                nwd_constant=nwd_constant,
            )[:max_detections]
        else:
            kept_local = nms(boxes, cand_scores, iou_threshold)[:max_detections]
        retained.append(candidates[kept_local])
    return retained


def _pad_indices(indices: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(indices)
    maximum = max((value.numel() for value in indices), default=0)
    device = indices[0].device if indices else torch.device("cpu")
    padded = torch.zeros((batch, maximum), dtype=torch.long, device=device)
    valid = torch.zeros((batch, maximum), dtype=torch.bool, device=device)
    for image, value in enumerate(indices):
        padded[image, : value.numel()] = value
        valid[image, : value.numel()] = True
    return padded, valid


def _selected_detection(
    decoded: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.shape[1] == 0:
        return (
            decoded.new_empty((decoded.shape[0], 0, 4)),
            decoded.new_empty((decoded.shape[0], 0)),
        )
    gathered = decoded.gather(
        2, indices[:, None, :].expand(-1, decoded.shape[1], -1)
    )
    boxes = xywh_to_xyxy(gathered[:, :4, :].permute(0, 2, 1).reshape(-1, 4))
    boxes = boxes.reshape(decoded.shape[0], indices.shape[1], 4)
    return boxes, gathered[:, 4, :]


def _retained_unified_candidates(
    decoded: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    class_index: int,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    nms_type: str = "standard",
    nwd_threshold: float = 0.5,
    nwd_constant: float = 12.0,
    nwd_area_threshold: float = 64.0,
    quality_scores: torch.Tensor | None = None,
    quality_alpha: float = 0.70,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if decoded.ndim != 3 or decoded.shape[1] != 6:
        raise ValueError("unified detections must have shape [B, 6, A]")
    if candidate_indices.shape != candidate_valid.shape:
        raise ValueError("candidate index and validity shapes differ")
    dense_result: list[torch.Tensor] = []
    slot_result: list[torch.Tensor] = []
    for image in range(decoded.shape[0]):
        slots = torch.nonzero(candidate_valid[image].bool(), as_tuple=False).reshape(-1)
        dense = candidate_indices[image, slots].long()
        scores = decoded[image, 4 + class_index, dense]
        keep_score = scores >= confidence_threshold
        slots = slots[keep_score]
        dense = dense[keep_score]
        scores = scores[keep_score]
        if dense.numel() == 0:
            dense_result.append(dense)
            slot_result.append(slots)
            continue
        boxes = xywh_to_xyxy(decoded[image, :4, dense].transpose(0, 1))
        cand_qual = None
        if quality_scores is not None:
            if quality_scores.ndim == 3:
                cand_qual = quality_scores[image, 0, dense]
            else:
                cand_qual = quality_scores[image, dense]
        if nms_type in ("size_adaptive", "adaptive_nwd"):
            kept = size_adaptive_nms(
                boxes,
                scores,
                quality_scores=cand_qual,
                quality_alpha=quality_alpha,
                iou_threshold=iou_threshold,
                nwd_threshold=nwd_threshold,
                nwd_constant=nwd_constant,
                area_threshold=nwd_area_threshold,
            )[:max_detections]
        elif nms_type in ("nwd", "pure_nwd"):
            kept = nwd_nms(
                boxes,
                scores,
                nwd_threshold=nwd_threshold,
                nwd_constant=nwd_constant,
            )[:max_detections]
        else:
            kept = nms(boxes, scores, iou_threshold)[:max_detections]
        dense_result.append(dense[kept])
        slot_result.append(slots[kept])
    return dense_result, slot_result


def _selected_unified_detection(
    decoded: torch.Tensor,
    indices: torch.Tensor,
    *,
    class_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.shape[1] == 0:
        return (
            decoded.new_empty((decoded.shape[0], 0, 4)),
            decoded.new_empty((decoded.shape[0], 0)),
        )
    gathered = decoded.gather(
        2, indices[:, None, :].expand(-1, decoded.shape[1], -1)
    )
    boxes = xywh_to_xyxy(gathered[:, :4].permute(0, 2, 1).reshape(-1, 4))
    boxes = boxes.reshape(decoded.shape[0], indices.shape[1], 4)
    return boxes, gathered[:, 4 + class_index]


def _postprocess_unified_outputs(
    outputs: Sequence[torch.Tensor],
    *,
    traffic_confidence: float,
    arrow_confidence: float,
    iou_threshold: float,
    max_traffic_lights: int,
    max_arrows: int,
    nms_type: str = "standard",
    nwd_threshold: float = 0.5,
    nwd_constant: float = 12.0,
    nwd_constant_tl: float | None = None,
    nwd_constant_arrow: float | None = None,
    nwd_area_threshold: float = 64.0,
    nwd_area_threshold_tl: float | None = None,
    nwd_area_threshold_arrow: float | None = None,
    quality_scores: torch.Tensor | None = None,
    quality_alpha: float = 0.70,
    temperature_state: float = 1.0,
    temperature_relevance: float = 1.0,
    conformal_safety_gate: Any | None = None,
    temporal_smoother: Any | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    (
        detection,
        states,
        rounds,
        maneuvers,
        ego_lane,
        traffic_candidates,
        traffic_candidate_valid,
        arrow_candidates,
        arrow_candidate_valid,
        relevance,
        attention,
    ) = outputs

    c_tl = float(nwd_constant_tl if nwd_constant_tl is not None else nwd_constant)
    c_arrow = float(nwd_constant_arrow if nwd_constant_arrow is not None else 24.0)
    a_tl = float(nwd_area_threshold_tl if nwd_area_threshold_tl is not None else nwd_area_threshold)
    a_arrow = float(nwd_area_threshold_arrow if nwd_area_threshold_arrow is not None else 1024.0)

    traffic_lists, traffic_slot_lists = _retained_unified_candidates(
        detection,
        traffic_candidates.long(),
        traffic_candidate_valid.bool(),
        class_index=TRAFFIC_LIGHT_CLASS,
        confidence_threshold=traffic_confidence,
        iou_threshold=iou_threshold,
        max_detections=max_traffic_lights,
        nms_type=nms_type,
        nwd_threshold=nwd_threshold,
        nwd_constant=c_tl,
        nwd_area_threshold=a_tl,
        quality_scores=quality_scores,
        quality_alpha=quality_alpha,
    )
    arrow_lists, arrow_slot_lists = _retained_unified_candidates(
        detection,
        arrow_candidates.long(),
        arrow_candidate_valid.bool(),
        class_index=ROAD_ARROW_CLASS,
        confidence_threshold=arrow_confidence,
        iou_threshold=iou_threshold,
        max_detections=max_arrows,
        nms_type=nms_type,
        nwd_threshold=nwd_threshold,
        nwd_constant=c_arrow,
        nwd_area_threshold=a_arrow,
    )
    traffic_indices, traffic_valid = _pad_indices(traffic_lists)
    traffic_slots, _ = _pad_indices(traffic_slot_lists)
    arrow_indices, arrow_valid = _pad_indices(arrow_lists)
    arrow_slots, _ = _pad_indices(arrow_slot_lists)
    traffic_boxes, traffic_scores = _selected_unified_detection(
        detection, traffic_indices, class_index=TRAFFIC_LIGHT_CLASS
    )
    arrow_boxes, arrow_scores = _selected_unified_detection(
        detection, arrow_indices, class_index=ROAD_ARROW_CLASS
    )

    selected_states = gather_candidate_outputs(states, traffic_indices)
    selected_rounds = gather_candidate_outputs(rounds, traffic_indices)
    selected_tl_maneuvers = gather_candidate_outputs(maneuvers, traffic_indices)
    selected_arrow_maneuvers = gather_candidate_outputs(maneuvers, arrow_indices)
    selected_ego_lane = gather_candidate_outputs(ego_lane, arrow_indices)
    selected_relevance = relevance.gather(
        2, traffic_slots[:, None, :].expand(-1, 1, -1)
    )

    scaled_states = selected_states / temperature_state if temperature_state != 1.0 else selected_states
    scaled_relevance = selected_relevance / temperature_relevance if temperature_relevance != 1.0 else selected_relevance

    score_bundle = combine_detection_relevance_scores(
        traffic_scores, scaled_relevance
    )
    selected_attention = attention.gather(
        2,
        traffic_slots[:, None, :, None].expand(
            -1, attention.shape[1], -1, attention.shape[3]
        ),
    )

    tl_dict: dict[str, torch.Tensor] = {
        "boxes_xyxy": traffic_boxes,
        "valid": traffic_valid,
        "dense_indices": traffic_indices,
        "candidate_slots": traffic_slots,
        **score_bundle,
        "relevance_logits": scaled_relevance,
        "state_logits": scaled_states,
        "state_probabilities": scaled_states.softmax(1),
        "state_indices": scaled_states.argmax(1),
        "round_logits": selected_rounds,
        "round_probabilities": selected_rounds.sigmoid(),
        "maneuver_logits": selected_tl_maneuvers,
        "maneuver_probabilities": selected_tl_maneuvers.sigmoid(),
        "maneuver_multihot": selected_tl_maneuvers.sigmoid().ge(0.5).long(),
        "attention_weights": selected_attention,
        "attention_arrow_dense_indices": arrow_candidates.long(),
        "attention_arrow_valid": arrow_candidate_valid.bool(),
    }

    if temporal_smoother is not None:
        if traffic_boxes.shape[0] == 1:
            smoothed = temporal_smoother.update(
                traffic_boxes[0],
                traffic_scores[0],
                tl_dict["state_probabilities"][0],
                score_bundle["relevance_probabilities"][0],
                valid_mask=traffic_valid[0],
            )
            tl_dict["state_probabilities"] = smoothed["state_probabilities"].unsqueeze(0)
            tl_dict["state_indices"] = smoothed["state_indices"].unsqueeze(0)
            tl_dict["relevance_probabilities"] = smoothed["relevance_probabilities"].unsqueeze(0)
            tl_dict["joint_scores"] = smoothed["joint_scores"].unsqueeze(0)
            tl_dict["detection_scores"] = smoothed["detection_scores"].unsqueeze(0)
            tl_dict["track_ids"] = smoothed["track_ids"].unsqueeze(0)
            tl_dict["flicker_suppressed"] = smoothed["flicker_suppressed"].unsqueeze(0)

    if conformal_safety_gate is not None:
        safety_eval = conformal_safety_gate.evaluate_safety_gate(
            tl_dict["state_probabilities"],
            score_bundle["relevance_probabilities"],
            traffic_scores,
        )
        tl_dict.update(safety_eval)

    return {
        "traffic_lights": tl_dict,
        "road_arrows": {
            "boxes_xyxy": arrow_boxes,
            "valid": arrow_valid,
            "dense_indices": arrow_indices,
            "candidate_slots": arrow_slots,
            "detection_scores": arrow_scores,
            "maneuver_logits": selected_arrow_maneuvers,
            "maneuver_probabilities": selected_arrow_maneuvers.sigmoid(),
            "maneuver_multihot": selected_arrow_maneuvers.sigmoid().ge(0.5).long(),
            "ego_lane_logits": selected_ego_lane,
            "ego_lane_probabilities": selected_ego_lane.sigmoid(),
        },
    }


def postprocess_multitask_outputs(
    outputs: Sequence[torch.Tensor],
    *,
    traffic_confidence: float = 0.25,
    arrow_confidence: float = 0.25,
    iou_threshold: float = 0.7,
    max_traffic_lights: int = 100,
    max_arrows: int = 50,
    nms_type: str = "standard",
    nwd_threshold: float = 0.5,
    nwd_constant: float = 12.0,
    nwd_constant_tl: float | None = None,
    nwd_constant_arrow: float | None = None,
    nwd_area_threshold: float = 64.0,
    nwd_area_threshold_tl: float | None = None,
    nwd_area_threshold_arrow: float | None = None,
    quality_scores: torch.Tensor | None = None,
    quality_alpha: float = 0.70,
    temperature_state: float = 1.0,
    temperature_relevance: float = 1.0,
    conformal_safety_gate: Any | None = None,
    temporal_smoother: Any | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Decode unified outputs; legacy six-tensor checkpoints remain readable."""

    if len(outputs) == 11:
        return _postprocess_unified_outputs(
            outputs,
            traffic_confidence=traffic_confidence,
            arrow_confidence=arrow_confidence,
            iou_threshold=iou_threshold,
            max_traffic_lights=max_traffic_lights,
            max_arrows=max_arrows,
            nms_type=nms_type,
            nwd_threshold=nwd_threshold,
            nwd_constant=nwd_constant,
            nwd_constant_tl=nwd_constant_tl,
            nwd_constant_arrow=nwd_constant_arrow,
            nwd_area_threshold=nwd_area_threshold,
            nwd_area_threshold_tl=nwd_area_threshold_tl,
            nwd_area_threshold_arrow=nwd_area_threshold_arrow,
            quality_scores=quality_scores,
            quality_alpha=quality_alpha,
            temperature_state=temperature_state,
            temperature_relevance=temperature_relevance,
            conformal_safety_gate=conformal_safety_gate,
            temporal_smoother=temporal_smoother,
        )
    if len(outputs) != 6:
        raise ValueError("full model must return 11 unified tensors")
    traffic, states, pictograms, arrows, directions, relevance = outputs
    traffic_indices_list = retained_nms_indices(
        traffic,
        confidence_threshold=traffic_confidence,
        iou_threshold=iou_threshold,
        max_detections=max_traffic_lights,
        nms_type=nms_type,
        nwd_threshold=nwd_threshold,
        nwd_constant=nwd_constant,
        nwd_area_threshold=nwd_area_threshold,
        quality_scores=quality_scores,
        quality_alpha=quality_alpha,
    )
    arrow_indices_list = retained_nms_indices(
        arrows,
        confidence_threshold=arrow_confidence,
        iou_threshold=iou_threshold,
        max_detections=max_arrows,
        nms_type=nms_type,
        nwd_threshold=nwd_threshold,
        nwd_constant=nwd_constant,
        nwd_area_threshold=nwd_area_threshold,
    )
    traffic_indices, traffic_valid = _pad_indices(traffic_indices_list)
    arrow_indices, arrow_valid = _pad_indices(arrow_indices_list)
    traffic_boxes, traffic_scores = _selected_detection(traffic, traffic_indices)
    arrow_boxes, arrow_scores = _selected_detection(arrows, arrow_indices)

    if traffic_indices.shape[1]:
        attributes = gather_candidate_attributes(states, pictograms, traffic_indices)
        selected_relevance = gather_candidate_relevance(relevance, traffic_indices)
        scores = combine_detection_relevance_scores(
            traffic_scores, selected_relevance["relevance_logits"]
        )
    else:
        attributes = {
            "state_logits": states.new_empty((states.shape[0], 4, 0)),
            "state_probabilities": states.new_empty((states.shape[0], 4, 0)),
            "state_indices": torch.empty(
                (states.shape[0], 0), dtype=torch.long, device=states.device
            ),
            "pictogram_logits": pictograms.new_empty((pictograms.shape[0], 4, 0)),
            "pictogram_probabilities": pictograms.new_empty((pictograms.shape[0], 4, 0)),
            "pictogram_indices": torch.empty(
                (pictograms.shape[0], 0), dtype=torch.long, device=pictograms.device
            ),
        }
        scores = {
            "detection_scores": traffic_scores,
            "relevance_probabilities": traffic_scores.new_empty(traffic_scores.shape),
            "joint_scores": traffic_scores.new_empty(traffic_scores.shape),
        }
        selected_relevance = {
            "relevance_logits": relevance.new_empty((relevance.shape[0], 1, 0))
        }
    if arrow_indices.shape[1]:
        selected_directions = gather_arrow_directions(directions, arrow_indices)
    else:
        selected_directions = {
            "direction_logits": directions.new_empty((directions.shape[0], 3, 0)),
            "direction_probabilities": directions.new_empty((directions.shape[0], 3, 0)),
            "direction_multihot": torch.empty(
                (directions.shape[0], 3, 0), dtype=torch.long, device=directions.device
            ),
        }
    return {
        "traffic_lights": {
            "boxes_xyxy": traffic_boxes,
            "valid": traffic_valid,
            "dense_indices": traffic_indices,
            **scores,
            **attributes,
            **selected_relevance,
        },
        "road_arrows": {
            "boxes_xyxy": arrow_boxes,
            "valid": arrow_valid,
            "dense_indices": arrow_indices,
            "detection_scores": arrow_scores,
            **selected_directions,
        },
    }


# Backward compatibility alias
size_adaptive_nwd_nms = size_adaptive_nms

