"""Unified masked losses for the five active TLR-YOLO-MTL tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from ..model.attributes import (
    _pad_targets,
    assigned_attribute_cross_entropy,
)
from ..model.relevance import assigned_relevance_focal_bce
from ..model.unified import TRAFFIC_LIGHT_CLASS


class IgnoreAwareDetectionLoss(v8DetectionLoss):
    """YOLO loss that removes background BCE inside canonical ignore boxes."""

    def _ignored_anchor_mask(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        image_size: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        anchors = anchor_points * stride_tensor
        empty = torch.zeros(
            (batch_size, anchors.shape[0]), dtype=torch.bool, device=self.device
        )
        if "ignore_batch_idx" not in batch or "ignore_bboxes" not in batch:
            return empty
        indices = batch["ignore_batch_idx"].reshape(-1).to(self.device)
        boxes = batch["ignore_bboxes"].reshape(-1, 4).to(self.device)
        if indices.numel() == 0:
            return empty
        if indices.numel() != boxes.shape[0]:
            raise ValueError("ignore batch indices and boxes differ in length")

        targets = torch.cat(
            (
                indices[:, None],
                torch.zeros((boxes.shape[0], 1), device=self.device),
                boxes,
            ),
            dim=1,
        )
        padded = self.preprocess(
            targets,
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        ignore_boxes = padded[:, :, 1:]
        valid_boxes = ignore_boxes.sum(2).gt(0)
        x = anchors[:, 0].reshape(1, -1, 1)
        y = anchors[:, 1].reshape(1, -1, 1)
        inside = (
            (x >= ignore_boxes[:, None, :, 0])
            & (x <= ignore_boxes[:, None, :, 2])
            & (y >= ignore_boxes[:, None, :, 1])
            & (y <= ignore_boxes[:, None, :, 3])
            & valid_boxes[:, None, :]
        )
        return inside.any(2)

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[Any, torch.Tensor, dict[str, torch.Tensor]]:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = (
            torch.tensor(
                preds["feats"][0].shape[2:], device=self.device, dtype=dtype
            )
            * self.stride[0]
        )
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            dim=1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, foreground, target_gt_indices = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        classification = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            classification *= self.class_weights
        ignored = self._ignored_anchor_mask(
            batch,
            batch_size=batch_size,
            image_size=image_size,
            anchor_points=anchor_points,
            stride_tensor=stride_tensor,
        ) & ~foreground
        classification = classification.masked_fill(ignored[:, :, None], 0.0)
        loss[1] = classification.sum() / target_scores_sum

        if foreground.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                foreground,
                image_size,
                stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        metrics = dict(zip(self.loss_names, loss.detach()))
        metrics["ignored_anchors"] = ignored.sum().detach()
        return (
            (
                foreground,
                target_gt_indices,
                target_bboxes,
                anchor_points,
                stride_tensor,
            ),
            loss,
            metrics,
        )


def normalized_wasserstein_loss(
    predicted_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    *,
    constant: float = 12.0,
) -> torch.Tensor:
    """NWD loss for matched boxes expressed in network-input pixels."""

    if predicted_xyxy.shape != target_xyxy.shape or predicted_xyxy.shape[-1] != 4:
        raise ValueError("NWD boxes must have matching [..., 4] shapes")
    if constant <= 0:
        raise ValueError("NWD normalization constant must be positive")
    if predicted_xyxy.numel() == 0:
        return predicted_xyxy.sum() * 0.0

    predicted_center = (predicted_xyxy[..., :2] + predicted_xyxy[..., 2:]) / 2
    target_center = (target_xyxy[..., :2] + target_xyxy[..., 2:]) / 2
    predicted_size = predicted_xyxy[..., 2:] - predicted_xyxy[..., :2]
    target_size = target_xyxy[..., 2:] - target_xyxy[..., :2]
    wasserstein_squared = (
        (predicted_center - target_center).square().sum(-1)
        + 0.25 * (predicted_size - target_size).square().sum(-1)
    )
    similarity = torch.exp(
        -torch.sqrt(wasserstein_squared.clamp_min(1e-9)) / constant
    )
    return (1.0 - similarity).mean()


@dataclass(frozen=True, slots=True)
class MultiTaskLossWeights:
    detection: float = 1.0
    state: float = 0.75
    round: float = 0.5
    maneuver: float = 1.0
    ego_lane: float = 0.5
    relevance: float = 1.0
    nwd: float = 0.5
    association: float = 0.0


@dataclass(slots=True)
class TLRMultiTaskLossResult:
    total: torch.Tensor
    detection: torch.Tensor
    state: torch.Tensor
    round: torch.Tensor
    maneuver: torch.Tensor
    ego_lane: torch.Tensor
    relevance: torch.Tensor
    nwd: torch.Tensor
    association: torch.Tensor
    state_matches: int
    round_matches: int
    maneuver_matches: int
    ego_lane_matches: int
    relevance_matches: int
    metrics: dict[str, torch.Tensor]

    # Read-only compatibility aliases for historical diagnostics.
    @property
    def traffic_detection(self) -> torch.Tensor:
        return self.detection

    @property
    def pictogram(self) -> torch.Tensor:
        return self.maneuver

    @property
    def arrow_detection(self) -> torch.Tensor:
        return self.detection * 0.0

    @property
    def arrow_direction(self) -> torch.Tensor:
        return self.maneuver

    @property
    def pictogram_matches(self) -> int:
        return self.maneuver_matches

    @property
    def arrow_direction_matches(self) -> int:
        return self.maneuver_matches

    @property
    def arrow_valid_images(self) -> int:
        return 0


def _pad_float_targets(
    values: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    *,
    width: int = 1,
) -> torch.Tensor:
    values = values.reshape(-1, width).float()
    batch_indices = batch_indices.reshape(-1).long()
    if values.shape[0] != batch_indices.numel():
        raise ValueError("targets and batch indices must have equal length")
    if batch_indices.numel() > 1 and torch.any(batch_indices[1:] < batch_indices[:-1]):
        raise ValueError("instances must be grouped by batch index")
    counts = torch.bincount(batch_indices, minlength=batch_size)
    maximum = int(counts.max().item()) if counts.numel() else 0
    padded = torch.full(
        (batch_size, maximum, width),
        -1.0,
        dtype=values.dtype,
        device=values.device,
    )
    if not values.shape[0]:
        return padded
    offsets = torch.cat(
        (torch.zeros(1, device=values.device, dtype=torch.long), counts.cumsum(0))
    )
    positions = torch.arange(values.shape[0], device=values.device) - offsets[
        batch_indices
    ]
    padded[batch_indices, positions] = values
    return padded


def assigned_binary_focal_bce(
    logits: torch.Tensor,
    padded_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    gamma: float = 0.0,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise ValueError("binary logits must have shape [batch, 1, anchors]")
    batch, _, anchors = logits.shape
    if foreground_mask.shape != (batch, anchors):
        raise ValueError("foreground mask shape does not match binary logits")
    if padded_targets.ndim == 3:
        padded_targets = padded_targets[..., 0]
    if padded_targets.shape[0] != batch:
        raise ValueError("binary targets have a different batch size")
    if padded_targets.shape[1] == 0:
        return logits.sum() * 0.0, 0
    safe = target_gt_indices.clamp(0, padded_targets.shape[1] - 1)
    targets = padded_targets.gather(1, safe)
    valid = foreground_mask.bool() & targets.ge(0)
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_logits = logits[:, 0][valid]
    selected_targets = targets[valid].to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )
    if gamma:
        probability = selected_logits.sigmoid()
        probability_target = (
            probability * selected_targets
            + (1.0 - probability) * (1.0 - selected_targets)
        )
        loss = loss * (1.0 - probability_target).pow(gamma)
    return loss.mean(), count


def assigned_multilabel_focal_bce(
    logits: torch.Tensor,
    padded_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    gamma: float = 0.0,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3:
        raise ValueError("multi-label logits must have shape [batch, labels, anchors]")
    batch, labels, anchors = logits.shape
    if padded_targets.ndim != 3 or padded_targets.shape[2] != labels:
        raise ValueError("multi-label target width does not match logits")
    if foreground_mask.shape != (batch, anchors):
        raise ValueError("foreground mask shape does not match multi-label logits")
    if padded_targets.shape[1] == 0:
        return logits.sum() * 0.0, 0
    safe = target_gt_indices.clamp(0, padded_targets.shape[1] - 1)
    targets = padded_targets.gather(
        1, safe[:, :, None].expand(-1, -1, labels)
    )
    valid = foreground_mask.bool() & targets.ge(0).all(-1)
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_logits = logits.permute(0, 2, 1)[valid]
    selected_targets = targets[valid].to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )
    if gamma:
        probabilities = selected_logits.sigmoid()
        probability_target = (
            probabilities * selected_targets
            + (1.0 - probabilities) * (1.0 - selected_targets)
        )
        loss = loss * (1.0 - probability_target).pow(gamma)
    return loss.mean(), count


class TLRMultiTaskCriterion:
    """One two-class assignment followed by conditional attribute losses."""

    def __init__(
        self,
        model: nn.Module,
        *,
        weights: MultiTaskLossWeights | None = None,
        state_class_weights: torch.Tensor | None = None,
        relevance_alpha: float | None = None,
        attribute_gamma: float = 1.5,
        maneuver_gamma: float = 2.0,
        ego_lane_gamma: float = 2.0,
        relevance_gamma: float = 2.0,
        nwd_constant: float = 12.0,
    ) -> None:
        self.weights = weights or MultiTaskLossWeights()
        self.traffic = IgnoreAwareDetectionLoss(model)
        if isinstance(self.traffic.hyp, Mapping):
            self.traffic.hyp = SimpleNamespace(**self.traffic.hyp)
        self.state_class_weights = state_class_weights
        self.relevance_alpha = relevance_alpha
        self.attribute_gamma = float(attribute_gamma)
        self.maneuver_gamma = float(maneuver_gamma)
        self.ego_lane_gamma = float(ego_lane_gamma)
        self.relevance_gamma = float(relevance_gamma)
        self.nwd_constant = float(nwd_constant)

    def __call__(
        self,
        predictions: Mapping[str, torch.Tensor] | tuple[Any, ...],
        batch: Mapping[str, torch.Tensor],
    ) -> TLRMultiTaskLossResult:
        parsed = self.traffic.parse_output(predictions)
        for name in (
            "state_logits",
            "round_logits",
            "maneuver_logits",
            "ego_lane_logits",
            "dense_local_relevance_logits",
            "relevance_logits",
            "traffic_candidate_indices",
            "traffic_candidate_valid",
            "attention_enabled_flag",
        ):
            if name not in parsed:
                raise KeyError(f"missing traffic prediction: {name}")
        required = (
            "object_batch_idx",
            "object_cls",
            "object_bboxes",
            "object_state",
            "object_round",
            "object_maneuver",
            "object_relevance",
            "object_ego_lane",
            "traffic_relevance_valid",
            "unified_detection_valid",
        )
        for name in required:
            if name not in batch:
                raise KeyError(f"missing traffic batch field: {name}")

        device = parsed["scores"].device
        detection_valid = batch["unified_detection_valid"].to(device).bool()
        if not bool(detection_valid.all()):
            raise ValueError(
                "unified detection requires exhaustive paired annotations for every training image"
            )
        detection_batch = {
            **dict(batch),
            "batch_idx": batch["object_batch_idx"],
            "cls": batch["object_cls"],
            "bboxes": batch["object_bboxes"],
        }
        assignments, detection_vector, detection_metrics = self.traffic.get_assigned_targets_and_loss(
            parsed, detection_batch
        )
        foreground, target_indices, target_boxes, anchor_points, strides = assignments
        batch_size = parsed["scores"].shape[0]
        batch_indices = batch["object_batch_idx"].to(device)
        state_targets = _pad_targets(
            batch["object_state"].to(device), batch_indices, batch_size
        )
        class_targets = _pad_targets(
            batch["object_cls"].to(device).reshape(-1).long(),
            batch_indices,
            batch_size,
        )
        relevance_targets = _pad_targets(
            batch["object_relevance"].to(device), batch_indices, batch_size
        )
        round_targets = _pad_float_targets(
            batch["object_round"].to(device), batch_indices, batch_size
        )
        maneuver_targets = _pad_float_targets(
            batch["object_maneuver"].to(device),
            batch_indices,
            batch_size,
            width=3,
        )
        ego_lane_targets = _pad_float_targets(
            batch["object_ego_lane"].to(device), batch_indices, batch_size
        )
        state_loss, state_matches = assigned_attribute_cross_entropy(
            parsed["state_logits"],
            state_targets,
            foreground,
            target_indices,
            gamma=self.attribute_gamma,
            class_weights=self.state_class_weights,
        )
        round_loss, round_matches = assigned_binary_focal_bce(
            parsed["round_logits"],
            round_targets,
            foreground,
            target_indices,
            gamma=self.attribute_gamma,
        )
        maneuver_loss, maneuver_matches = assigned_multilabel_focal_bce(
            parsed["maneuver_logits"],
            maneuver_targets,
            foreground,
            target_indices,
            gamma=self.maneuver_gamma,
        )
        ego_lane_loss, ego_lane_matches = assigned_binary_focal_bce(
            parsed["ego_lane_logits"],
            ego_lane_targets,
            foreground,
            target_indices,
            gamma=self.ego_lane_gamma,
        )
        candidate_indices = parsed["traffic_candidate_indices"].long()
        selected_foreground = foreground.gather(1, candidate_indices)
        selected_foreground = selected_foreground & parsed[
            "traffic_candidate_valid"
        ].bool()
        selected_target_indices = target_indices.gather(1, candidate_indices)
        local_relevance_loss, local_relevance_matches = assigned_relevance_focal_bce(
            parsed["dense_local_relevance_logits"],
            relevance_targets,
            foreground,
            target_indices,
            image_valid=batch["traffic_relevance_valid"].to(device),
            alpha=self.relevance_alpha,
            gamma=self.relevance_gamma,
        )
        contextual_relevance_loss, contextual_relevance_matches = assigned_relevance_focal_bce(
            parsed["relevance_logits"],
            relevance_targets,
            selected_foreground,
            selected_target_indices,
            image_valid=batch["traffic_relevance_valid"].to(device),
            alpha=self.relevance_alpha,
            gamma=self.relevance_gamma,
        )
        attention_enabled = bool(float(parsed["attention_enabled_flag"].detach()))
        if attention_enabled and contextual_relevance_matches:
            relevance_loss = 0.5 * (
                local_relevance_loss + contextual_relevance_loss
            )
            relevance_matches = contextual_relevance_matches
        else:
            relevance_loss = local_relevance_loss
            relevance_matches = local_relevance_matches

        pred_distribution = parsed["boxes"].permute(0, 2, 1).contiguous()
        predicted_boxes = self.traffic.bbox_decode(
            anchor_points, pred_distribution
        ) * strides
        if class_targets.shape[1]:
            safe_target_indices = target_indices.clamp(0, class_targets.shape[1] - 1)
            assigned_classes = class_targets.gather(1, safe_target_indices)
            traffic_foreground = foreground & assigned_classes.eq(TRAFFIC_LIGHT_CLASS)
        else:
            traffic_foreground = foreground & False
        if traffic_foreground.any():
            nwd_loss = normalized_wasserstein_loss(
                predicted_boxes[traffic_foreground],
                target_boxes[traffic_foreground],
                constant=self.nwd_constant,
            )
        else:
            nwd_loss = parsed["boxes"].sum() * 0.0

        detection_loss = detection_vector.sum() * batch_size
        association_loss = parsed["attention_weights"].sum() * 0.0
        weight = self.weights
        total = (
            weight.detection * detection_loss
            + weight.nwd * nwd_loss
            + weight.state * state_loss
            + weight.round * round_loss
            + weight.maneuver * maneuver_loss
            + weight.ego_lane * ego_lane_loss
            + weight.relevance * relevance_loss
            + weight.association * association_loss
        )
        metrics = {
            **detection_metrics,
            "state_loss": state_loss.detach(),
            "round_loss": round_loss.detach(),
            "maneuver_loss": maneuver_loss.detach(),
            "ego_lane_loss": ego_lane_loss.detach(),
            "relevance_loss": relevance_loss.detach(),
            "local_relevance_loss": local_relevance_loss.detach(),
            "contextual_relevance_loss": contextual_relevance_loss.detach(),
            "nwd_loss": nwd_loss.detach(),
            "association_loss": association_loss.detach(),
            "total_loss": total.detach(),
        }
        return TLRMultiTaskLossResult(
            total=total,
            detection=detection_loss,
            state=state_loss,
            round=round_loss,
            maneuver=maneuver_loss,
            ego_lane=ego_lane_loss,
            relevance=relevance_loss,
            nwd=nwd_loss,
            association=association_loss,
            state_matches=state_matches,
            round_matches=round_matches,
            maneuver_matches=maneuver_matches,
            ego_lane_matches=ego_lane_matches,
            relevance_matches=relevance_matches,
            metrics=metrics,
        )
