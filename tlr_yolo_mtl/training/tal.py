"""Task-Aligned Assigner implementations for TLR-YOLO-MTL.

Provides standard and scale-adaptive NWD-aware TaskAlignedAssigner modules to eliminate
anchor candidate starvation on tiny sub-grid objects while preserving rigid IoU matching
on large objects.
"""

from __future__ import annotations

from typing import Any, Mapping
import torch
from torch import nn
from ultralytics.utils.tal import TaskAlignedAssigner, xyxy2xywh, xywh2xyxy
from ultralytics.utils.ops import xyxy2xywh, xywh2xyxy


def compute_nwd_similarity(
    boxes1_xyxy: torch.Tensor,
    boxes2_xyxy: torch.Tensor,
    constant: float = 12.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute Gaussian Wasserstein similarity NWD in (0, 1] between pairs of boxes.

    Args:
        boxes1_xyxy: Bounding boxes [..., 4] in (x1, y1, x2, y2) format.
        boxes2_xyxy: Bounding boxes [..., 4] in (x1, y1, x2, y2) format.
        constant: Normalization constant C in pixels (default: 12.0).
        eps: Small epsilon for numerical stability.

    Returns:
        Tensor with values in (0, 1] of shape matching the broadcasted box dimensions.
    """
    if constant <= 0:
        raise ValueError("NWD normalization constant must be positive")
    if boxes1_xyxy.numel() == 0 or boxes2_xyxy.numel() == 0:
        return torch.zeros(boxes1_xyxy.shape[:-1], device=boxes1_xyxy.device, dtype=boxes1_xyxy.dtype)

    c1 = (boxes1_xyxy[..., :2] + boxes1_xyxy[..., 2:]) / 2.0
    c2 = (boxes2_xyxy[..., :2] + boxes2_xyxy[..., 2:]) / 2.0
    s1 = (boxes1_xyxy[..., 2:] - boxes1_xyxy[..., :2]).clamp_min(0.0)
    s2 = (boxes2_xyxy[..., 2:] - boxes2_xyxy[..., :2]).clamp_min(0.0)

    d_center = c1 - c2
    d_size = s1 - s2
    w2 = d_center.square().sum(-1) + 0.25 * d_size.square().sum(-1)
    return torch.exp(-torch.sqrt(w2.clamp_min(eps)) / constant)


class NWDAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner with Normalized Wasserstein Distance (NWD) alignment metric.

    Combines classification scores with a hybrid IoU + NWD localization overlap metric:
        t = s^alpha * (Metric_overlap)^beta

    where Metric_overlap provides continuous, non-zero gradient and assignment signal
    for sub-grid traffic lights where discrete IoU collapses to zero.

    Attributes:
        nwd_weight: Weight lambda in [0, 1] for NWD component.
        nwd_constant: Distance scaling constant C for NWD (default: 12.0).
        area_threshold: Upper area bound (px^2) below which NWD is active (default: 64.0).
        mode: Blending mode ("scale_adaptive", "convex", "additive").
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
        nwd_weight: float = 0.5,
        nwd_constant: float = 12.0,
        area_threshold: float | None = 64.0,
        mode: str = "scale_adaptive",
    ):
        super().__init__(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )
        self.nwd_weight = float(nwd_weight)
        self.nwd_constant = float(nwd_constant)
        self.area_threshold = float(area_threshold) if area_threshold is not None else None
        self.mode = str(mode)

    def get_box_metrics(self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute task alignment metric and localization overlaps with NWD enhancement.

        Args:
            pd_scores: Predicted classification scores (bs, num_total_anchors, num_classes).
            pd_bboxes: Predicted bounding boxes in pixels (bs, num_total_anchors, 4).
            gt_labels: Ground truth labels (bs, n_max_boxes, 1).
            gt_bboxes: Ground truth boxes in pixels (bs, n_max_boxes, 4).
            mask_gt: Mask for valid candidate anchor pairs (bs, n_max_boxes, num_total_anchors).

        Returns:
            align_metric: Alignment metric tensor (bs, n_max_boxes, num_total_anchors).
            overlaps: Overlap metric tensor (bs, n_max_boxes, num_total_anchors).
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, n_max_boxes, na)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]

        iou_overlaps = self.iou_calculation(gt_boxes, pd_boxes)

        if self.nwd_weight > 0:
            nwd_overlaps = compute_nwd_similarity(
                gt_boxes, pd_boxes, constant=self.nwd_constant, eps=self.eps
            )
            if self.mode == "scale_adaptive" and self.area_threshold is not None:
                gt_w = (gt_boxes[..., 2] - gt_boxes[..., 0]).clamp_min(0.0)
                gt_h = (gt_boxes[..., 3] - gt_boxes[..., 1]).clamp_min(0.0)
                gt_area = gt_w * gt_h
                scale_weight = (1.0 - gt_area / self.area_threshold).clamp(0.0, 1.0)
                effective_nwd_weight = self.nwd_weight * scale_weight
                combined_overlaps = (1.0 - effective_nwd_weight) * iou_overlaps + effective_nwd_weight * nwd_overlaps
            elif self.mode == "additive":
                combined_overlaps = iou_overlaps + self.nwd_weight * nwd_overlaps
            elif self.mode == "convex":
                combined_overlaps = (1.0 - self.nwd_weight) * iou_overlaps + self.nwd_weight * nwd_overlaps
            else:
                combined_overlaps = (1.0 - self.nwd_weight) * iou_overlaps + self.nwd_weight * nwd_overlaps
            overlaps[mask_gt] = combined_overlaps
        else:
            overlaps[mask_gt] = iou_overlaps

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps


def build_task_aligned_assigner(
    *,
    assigner_type: str = "standard",
    topk: int = 10,
    num_classes: int = 80,
    alpha: float = 0.5,
    beta: float = 6.0,
    stride: list | None = None,
    eps: float = 1e-9,
    topk2: int | None = None,
    nwd_weight: float = 0.5,
    nwd_constant: float = 12.0,
    area_threshold: float | None = 64.0,
    mode: str = "scale_adaptive",
    **kwargs: Any,
) -> TaskAlignedAssigner:
    """Factory helper to build standard or NWD-aware TaskAlignedAssigner."""
    normalized_type = str(assigner_type).lower().strip()
    # Handle aliases
    if "lambda_nwd" in kwargs and "nwd_weight" not in kwargs:
        nwd_weight = float(kwargs.pop("lambda_nwd"))
    if "tiny_transition_area" in kwargs and "area_threshold" not in kwargs:
        area_threshold = float(kwargs.pop("tiny_transition_area"))
    if normalized_type in ("nwd", "nwd_aware", "nwd_tal", "nwd_aware_tal"):
        return NWDAwareTaskAlignedAssigner(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
            nwd_weight=nwd_weight,
            nwd_constant=nwd_constant,
            area_threshold=area_threshold,
            mode=mode,
        )
    elif normalized_type in ("standard", "tal", "default"):
        return TaskAlignedAssigner(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )
    else:
        raise ValueError(f"Unknown assigner_type '{assigner_type}'. Expected 'standard' or 'nwd'.")

