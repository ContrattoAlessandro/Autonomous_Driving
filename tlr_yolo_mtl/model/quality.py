"""NWD-Quality-Aware Confidence Head & Tiny-Aligned Ranking (Ticket E50).

Scientific Motivation:
In standard dense object detectors, the classification head outputs candidate confidence
scores p_i = sigmoid(z_i) answering "Is there a traffic light here?". However, during
post-processing suppression (NMS) and precision-recall ranking, anchors with high classification
score but sub-pixel spatial misalignment (1-2px jitter) often suppress better-centered anchors.
For tiny objects (<8px), standard IoU collapses to zero on small offsets, making IoU-quality
supervision unstable.

This module introduces Gaussian Normalized Wasserstein Distance (NWD) as the continuous quality
target for tiny objects (<64 px^2) and IoU for larger objects, predicting a continuous quality
score q_hat_i concurrently. Post-processing ranks proposals via the joint score:
    s_i = (p_i)^alpha * (q_hat_i)^(1 - alpha)
with zero runtime inference latency overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class QualityScoringConfig:
    """Configuration for NWD-Quality-Aware Confidence Scoring."""
    enabled: bool = True
    alpha: float = 0.70                 # Ranking exponent: s = p^alpha * q^(1-alpha)
    area_threshold: float = 64.0        # px^2 (side < 8px): threshold for NWD vs IoU target
    nwd_constant: float = 12.0          # Normalization constant C in pixels
    loss_weight: float = 0.50           # Training loss multiplier for quality head
    quality_gamma: float = 1.5          # Quality Focal BCE gamma exponent
    hidden_dim: int = 64                # Hidden dimension in quality tower


def compute_nwd_quality_target(
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    *,
    constant: float = 12.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute Gaussian Wasserstein similarity NWD in (0, 1] between corresponding box pairs.
    
    Args:
        pred_boxes_xyxy: Predicted boxes [..., 4] in (x1, y1, x2, y2).
        gt_boxes_xyxy: Ground truth boxes [..., 4] in (x1, y1, x2, y2).
        constant: Normalization constant C (default: 12.0).
        eps: Small epsilon for numerical stability.
        
    Returns:
        Tensor of shape [...] with continuous NWD quality in (0, 1].
    """
    if constant <= 0:
        raise ValueError("NWD constant must be positive")
    if pred_boxes_xyxy.numel() == 0 or gt_boxes_xyxy.numel() == 0:
        return torch.zeros(pred_boxes_xyxy.shape[:-1], device=pred_boxes_xyxy.device, dtype=pred_boxes_xyxy.dtype)

    c_pred = (pred_boxes_xyxy[..., :2] + pred_boxes_xyxy[..., 2:]) / 2.0
    c_gt = (gt_boxes_xyxy[..., :2] + gt_boxes_xyxy[..., 2:]) / 2.0
    s_pred = (pred_boxes_xyxy[..., 2:] - pred_boxes_xyxy[..., :2]).clamp_min(0.0)
    s_gt = (gt_boxes_xyxy[..., 2:] - gt_boxes_xyxy[..., :2]).clamp_min(0.0)

    d_center = c_pred - c_gt
    d_size = s_pred - s_gt
    w2 = d_center.square().sum(-1) + 0.25 * d_size.square().sum(-1)
    return torch.exp(-torch.sqrt(w2.clamp_min(eps)) / constant)


def compute_iou_quality_target(
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute 1-to-1 pairwise IoU overlap in [0, 1] between corresponding box pairs."""
    if pred_boxes_xyxy.numel() == 0 or gt_boxes_xyxy.numel() == 0:
        return torch.zeros(pred_boxes_xyxy.shape[:-1], device=pred_boxes_xyxy.device, dtype=pred_boxes_xyxy.dtype)

    lt = torch.max(pred_boxes_xyxy[..., :2], gt_boxes_xyxy[..., :2])
    rb = torch.min(pred_boxes_xyxy[..., 2:], gt_boxes_xyxy[..., 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]

    area_pred = (pred_boxes_xyxy[..., 2] - pred_boxes_xyxy[..., 0]).clamp_min(0.0) * (
        pred_boxes_xyxy[..., 3] - pred_boxes_xyxy[..., 1]
    ).clamp_min(0.0)
    area_gt = (gt_boxes_xyxy[..., 2] - gt_boxes_xyxy[..., 0]).clamp_min(0.0) * (
        gt_boxes_xyxy[..., 3] - gt_boxes_xyxy[..., 1]
    ).clamp_min(0.0)

    union = area_pred + area_gt - inter
    return inter / union.clamp_min(eps)


def compute_scale_adaptive_quality_targets(
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    *,
    area_threshold: float = 64.0,
    nwd_constant: float = 12.0,
) -> torch.Tensor:
    """Compute scale-adaptive continuous quality targets: NWD for tiny (<area_thresh), IoU for macro.
    
    Args:
        pred_boxes_xyxy: Predicted boxes [..., 4] in pixel coordinates.
        gt_boxes_xyxy: Target boxes [..., 4] in pixel coordinates.
        area_threshold: Area cutoff in px^2 (default: 64.0).
        nwd_constant: Distance normalization constant C.
        
    Returns:
        Continuous quality targets in [0, 1].
    """
    gt_w = (gt_boxes_xyxy[..., 2] - gt_boxes_xyxy[..., 0]).clamp_min(0.0)
    gt_h = (gt_boxes_xyxy[..., 3] - gt_boxes_xyxy[..., 1]).clamp_min(0.0)
    gt_area = gt_w * gt_h

    is_tiny = gt_area < area_threshold
    nwd_target = compute_nwd_quality_target(pred_boxes_xyxy, gt_boxes_xyxy, constant=nwd_constant)
    iou_target = compute_iou_quality_target(pred_boxes_xyxy, gt_boxes_xyxy)

    return torch.where(is_tiny, nwd_target, iou_target)


def compute_quality_aware_scores(
    class_probabilities: torch.Tensor,
    quality_scores: torch.Tensor,
    *,
    alpha: float = 0.70,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute combined quality-aware candidate scores: s = p^alpha * q^(1 - alpha).
    
    Args:
        class_probabilities: Classification probabilities in [0, 1].
        quality_scores: Predicted continuous localization quality in [0, 1].
        alpha: Weight exponent for classification vs quality.
        eps: Small epsilon for safe exponentiation.
        
    Returns:
        Quality-aware candidate scores in [0, 1].
    """
    if alpha >= 1.0:
        return class_probabilities
    if alpha <= 0.0:
        return quality_scores

    p = class_probabilities.clamp(eps, 1.0)
    q = quality_scores.clamp(eps, 1.0)
    return p.pow(alpha) * q.pow(1.0 - alpha)


class NWDQualityPredictionTower(nn.Module):
    """Lightweight 2-layer Conv/Linear tower predicting 1-channel continuous quality logit."""

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
        
        # Initialize bias near zero (sigmoid(0) = 0.5 prior quality)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting 1-channel quality logits [B, 1, H, W]."""
        return self.conv2(self.act1(self.bn1(self.conv1(x))))


class NWDQualityConfidenceHead(nn.Module):
    """Multi-scale quality prediction head attached across feature pyramid levels."""

    def __init__(
        self,
        channels: Sequence[int],
        config: QualityScoringConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or QualityScoringConfig()
        self.towers = nn.ModuleList([
            NWDQualityPredictionTower(ch, hidden_dim=self.config.hidden_dim)
            for ch in channels
        ])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Extract and flatten quality logits across all pyramid levels.
        
        Args:
            features: Multi-scale feature maps [P2, P3, P4, P5].
            
        Returns:
            Flattened quality logits tensor [Batch, 1, Total_Anchors].
        """
        batch = features[0].shape[0]
        quality_maps = [tower(feat) for tower, feat in zip(self.towers, features)]
        flattened = torch.cat([qm.reshape(batch, 1, -1) for qm in quality_maps], dim=-1)
        return flattened
