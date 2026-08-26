"""Quality Loss Formulations for NWD-Quality-Aware Confidence Head (Ticket E50).

Implements scale-adaptive continuous quality loss:
- NWD target for tiny objects (<64 px^2)
- IoU target for larger objects (>=64 px^2)
- Quality Focal Binary Cross-Entropy (QFL-BCE) for smooth continuous probability supervision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..model.quality import (
    compute_nwd_quality_target,
    compute_iou_quality_target,
    compute_scale_adaptive_quality_targets,
)


@dataclass(frozen=True, slots=True)
class QualityLossWeights:
    quality: float = 0.50
    area_threshold: float = 64.0
    nwd_constant: float = 12.0
    gamma: float = 1.5


def assigned_quality_focal_loss(
    quality_logits: torch.Tensor,
    predicted_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    area_threshold: float = 64.0,
    nwd_constant: float = 12.0,
    gamma: float = 1.5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Compute scale-adaptive Quality Focal BCE Loss on assigned foreground anchors.
    
    Args:
        quality_logits: Predicted quality logits [Batch, 1, Anchors].
        predicted_boxes_xyxy: Decoded predicted bounding boxes in pixels [Batch, Anchors, 4].
        target_boxes_xyxy: Assigned ground-truth bounding boxes in pixels [Batch, Anchors, 4] or padded [Batch, M, 4].
        foreground_mask: Boolean foreground mask [Batch, Anchors].
        target_gt_indices: Index mapping from anchor to assigned ground truth box [Batch, Anchors].
        area_threshold: Cutoff area for NWD vs IoU (default: 64.0 px^2).
        nwd_constant: Distance constant C for NWD.
        gamma: Modulating focal factor exponent.
        eps: Small epsilon for numerical stability.
        
    Returns:
        loss: Scalar quality focal loss tensor.
        metrics: Dictionary of diagnostic tensors.
        num_positive: Number of valid positive anchor matches.
    """
    if quality_logits.ndim != 3 or quality_logits.shape[1] != 1:
        raise ValueError("quality_logits must have shape [Batch, 1, Anchors]")

    batch, _, num_anchors = quality_logits.shape
    if foreground_mask.shape != (batch, num_anchors):
        raise ValueError("foreground_mask shape does not match quality_logits")

    valid_mask = foreground_mask.bool()
    num_positive = int(valid_mask.sum().item())

    if num_positive == 0:
        zero_loss = quality_logits.sum() * 0.0
        return zero_loss, {
            "quality_loss": zero_loss.detach(),
            "mean_quality_target": zero_loss.detach(),
            "mean_quality_pred": zero_loss.detach(),
            "quality_matches": torch.tensor(0, device=quality_logits.device),
        }, 0

    # Extract positive predictions
    pos_logits = quality_logits[:, 0][valid_mask]
    pos_preds = pos_logits.sigmoid()

    # Extract corresponding target boxes
    if target_boxes_xyxy.shape[1] == num_anchors:
        pos_gt_boxes = target_boxes_xyxy[valid_mask]
        pos_pred_boxes = predicted_boxes_xyxy[valid_mask]
    else:
        safe_indices = target_gt_indices.clamp(0, target_boxes_xyxy.shape[1] - 1)
        expanded_targets = target_boxes_xyxy.gather(
            1, safe_indices[:, :, None].expand(-1, -1, 4)
        )
        pos_gt_boxes = expanded_targets[valid_mask]
        pos_pred_boxes = predicted_boxes_xyxy[valid_mask]

    # Compute scale-adaptive continuous quality targets in [0, 1]
    with torch.no_grad():
        quality_targets = compute_scale_adaptive_quality_targets(
            pos_pred_boxes,
            pos_gt_boxes,
            area_threshold=area_threshold,
            nwd_constant=nwd_constant,
        ).clamp(0.0, 1.0)

    # Compute Quality Focal Loss: BCE(p, q) * |p - q|^gamma
    bce = F.binary_cross_entropy_with_logits(
        pos_logits, quality_targets, reduction="none"
    )
    if gamma > 0.0:
        focal_weight = (pos_preds - quality_targets).abs().pow(gamma)
        loss = (bce * focal_weight).mean()
    else:
        loss = bce.mean()

    metrics = {
        "quality_loss": loss.detach(),
        "mean_quality_target": quality_targets.mean().detach(),
        "mean_quality_pred": pos_preds.mean().detach(),
        "quality_matches": torch.tensor(num_positive, device=quality_logits.device),
    }

    return loss, metrics, num_positive


class NWDQualityLoss(nn.Module):
    """Module wrapping scale-adaptive continuous Quality Loss."""

    def __init__(
        self,
        area_threshold: float = 64.0,
        nwd_constant: float = 12.0,
        gamma: float = 1.5,
    ) -> None:
        super().__init__()
        self.area_threshold = float(area_threshold)
        self.nwd_constant = float(nwd_constant)
        self.gamma = float(gamma)

    def forward(
        self,
        quality_logits: torch.Tensor,
        predicted_boxes_xyxy: torch.Tensor,
        target_boxes_xyxy: torch.Tensor,
        foreground_mask: torch.Tensor,
        target_gt_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
        return assigned_quality_focal_loss(
            quality_logits,
            predicted_boxes_xyxy,
            target_boxes_xyxy,
            foreground_mask,
            target_gt_indices,
            area_threshold=self.area_threshold,
            nwd_constant=self.nwd_constant,
            gamma=self.gamma,
        )
