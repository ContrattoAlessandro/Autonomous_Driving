"""Sparse Candidate Refinement Loss Formulation (Ticket E49).

Supervises virtual P1 sub-grid candidate refinement on tiny traffic light proposals.

Loss Components:
1. Box Refinement Loss (L_box):
   Gaussian Normalized Wasserstein Distance (NWD) + Smooth L1 / GIoU on (B_coarse + ΔB, B_gt).
2. State Residual Loss (L_state):
   Focal Cross-Entropy on (S_coarse + ΔS, S_gt) to refine chromatic classification.
3. Quality Delta Loss (L_qual):
   Binary Cross-Entropy on predicted quality Δc against empirical NWD localization quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..deployment.postprocess import compute_pairwise_nwd, compute_pairwise_iou


@dataclass(frozen=True, slots=True)
class RefinementLossWeights:
    """Weights for Sparse Candidate Refinement Multi-Task Loss."""
    box_refine: float = 1.0
    state_refine: float = 0.5
    quality_refine: float = 0.25
    nwd_constant: float = 12.0


class SparseRefinementLoss(nn.Module):
    """Loss module for Sparse Candidate Refinement Head."""

    def __init__(
        self,
        weights: RefinementLossWeights | None = None,
        *,
        box_refine_weight: float = 1.0,
        state_refine_weight: float = 0.5,
        quality_refine_weight: float = 0.25,
        nwd_constant: float = 12.0,
        focal_gamma: float = 1.5,
    ) -> None:
        super().__init__()
        self.weights = weights or RefinementLossWeights(
            box_refine=box_refine_weight,
            state_refine=state_refine_weight,
            quality_refine=quality_refine_weight,
            nwd_constant=nwd_constant,
        )
        self.focal_gamma = float(focal_gamma)

    def _compute_nwd_loss(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Computes Gaussian NWD loss = 1.0 - NWD(pred, target)."""
        if pred_boxes.numel() == 0:
            return pred_boxes.new_zeros(1).sum()

        c_pred = (pred_boxes[:, :2] + pred_boxes[:, 2:]) * 0.5
        c_gt = (target_boxes[:, :2] + target_boxes[:, 2:]) * 0.5
        s_pred = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp_min(1e-4)
        s_gt = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp_min(1e-4)

        d_center = (c_pred - c_gt).square().sum(-1)
        d_size = 0.25 * (s_pred - s_gt).square().sum(-1)
        w2 = (d_center + d_size).clamp_min(1e-9)

        nwd = torch.exp(-torch.sqrt(w2) / self.weights.nwd_constant)
        return (1.0 - nwd).mean()

    def _compute_focal_state_loss(
        self, pred_logits: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes multi-class Focal Loss on refined state logits."""
        if pred_logits.numel() == 0:
            return pred_logits.new_zeros(1).sum()

        ce_loss = F.cross_entropy(pred_logits, target_labels, reduction="none")
        p_t = torch.exp(-ce_loss).clamp(1e-6, 1.0 - 1e-6)
        focal_factor = (1.0 - p_t) ** self.focal_gamma
        return (focal_factor * ce_loss).mean()

    def forward(
        self,
        refinement_outputs: dict[str, torch.Tensor],
        gt_boxes_xyxy: torch.Tensor,
        gt_state_labels: torch.Tensor,
        matched_gt_indices: torch.Tensor,  # [B, K] (-1 for unmatched)
    ) -> dict[str, torch.Tensor]:
        """Calculates multi-task refinement loss on matched positive candidate proposals.

        Args:
            refinement_outputs: Output dict from SparseCandidateRefinementHead
            gt_boxes_xyxy: Ground truth bounding boxes [B, M, 4]
            gt_state_labels: Ground truth state classes [B, M] (0..3)
            matched_gt_indices: Index of assigned GT box for each candidate [B, K]

        Returns:
            Dictionary containing loss components and total refinement loss.
        """
        refine_mask = refinement_outputs["refine_mask"]  # [B, K]
        valid_matches = (matched_gt_indices >= 0) & refine_mask  # [B, K]

        if not valid_matches.any():
            zero = refinement_outputs["refined_boxes_xyxy"].sum() * 0.0
            return {
                "loss_refine_total": zero,
                "loss_refine_box": zero,
                "loss_refine_state": zero,
                "loss_refine_quality": zero,
            }

        b_idx, k_idx = torch.nonzero(valid_matches, as_tuple=True)
        m_idx = matched_gt_indices[b_idx, k_idx]

        refined_boxes = refinement_outputs["refined_boxes_xyxy"][b_idx, k_idx]
        target_boxes = gt_boxes_xyxy[b_idx, m_idx]

        refined_states = refinement_outputs["refined_state_logits"][b_idx, k_idx]
        target_states = gt_state_labels[b_idx, m_idx]

        # 1. Box Refinement Loss (NWD + Smooth L1 center offset)
        l_nwd = self._compute_nwd_loss(refined_boxes, target_boxes)
        l_l1 = F.smooth_l1_loss(refined_boxes, target_boxes, beta=1.0)
        l_box = 0.7 * l_nwd + 0.3 * l_l1

        # 2. State Residual Loss (Focal Cross-Entropy)
        l_state = self._compute_focal_state_loss(refined_states, target_states)

        # 3. Quality Delta Loss (BCE against NWD alignment)
        with torch.no_grad():
            c_p = (refined_boxes[:, :2] + refined_boxes[:, 2:]) * 0.5
            c_g = (target_boxes[:, :2] + target_boxes[:, 2:]) * 0.5
            s_p = (refined_boxes[:, 2:] - refined_boxes[:, :2]).clamp_min(1e-4)
            s_g = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp_min(1e-4)
            w2_val = ((c_p - c_g).square().sum(-1) + 0.25 * (s_p - s_g).square().sum(-1)).clamp_min(1e-9)
            target_quality = torch.exp(-torch.sqrt(w2_val) / self.weights.nwd_constant).unsqueeze(-1)

        pred_qual = refinement_outputs["quality_deltas"][b_idx, k_idx].sigmoid()
        l_qual = F.binary_cross_entropy(pred_qual, target_quality)

        # Total Weighted Refinement Loss
        total_loss = (
            self.weights.box_refine * l_box
            + self.weights.state_refine * l_state
            + self.weights.quality_refine * l_qual
        )

        return {
            "loss_refine_total": total_loss,
            "loss_refine_box": l_box,
            "loss_refine_state": l_state,
            "loss_refine_quality": l_qual,
        }
