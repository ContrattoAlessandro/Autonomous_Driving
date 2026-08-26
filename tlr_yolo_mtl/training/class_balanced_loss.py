"""Class-Balanced Focal Loss & Balanced Softmax for Long-Tail State Head (Ticket E44).

Addresses extreme class imbalance on the 4-class Traffic Light State Head:
- Red (~34.8%) and Green (~52.2%) dominate real-world urban datasets (DTLD: >85% combined).
- Yellow (~3.6%) and Off (~9.5%) represent rare long-tail classes (<15%).

Formulations:
1. Class-Balanced Loss (Cui et al., CVPR 2019):
   Modulates loss using the effective number of samples E_n = (1 - beta^n) / (1 - beta).
   Weights: W_i = (1 - beta) / (1 - beta^n_i), normalized to sum to C (number of classes).
2. Balanced Softmax (Ren et al., NeurIPS 2020):
   Corrects conditional probability estimation under class label distribution shift by
   incorporating class priors log(pi_i) into logits during training:
   L_BS = CrossEntropy(z + log(pi), y).
3. Composite Champion v3 Formulation:
   Combines Balanced Softmax prior adjustment, Class-Balanced effective weights, and Focal modulation.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

# DTLD Train Split State Class Distribution: ("red", "yellow", "green", "off")
# Red: 30179, Yellow: 3105, Green: 45258, Off: 8227 (Total: 86769)
DTLD_STATE_CLASS_COUNTS: tuple[int, int, int, int] = (30179, 3105, 45258, 8227)
STATE_CLASS_NAMES: tuple[str, str, str, str] = ("red", "yellow", "green", "off")


def compute_effective_num_weights(
    class_counts: Sequence[int] | torch.Tensor,
    beta: float = 0.9999,
    *,
    num_classes: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes normalized Class-Balanced loss weights using effective sample counts.

    E_n = (1 - beta^n_i) / (1 - beta)
    W_i = 1 / E_n = (1 - beta) / (1 - beta^n_i)
    Normalized such that sum(W_i) = C (num_classes).

    Args:
        class_counts: Sample counts per class.
        beta: Hyperparameter in [0, 1). If 0, returns uniform weights.
        num_classes: Optional number of classes.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor of shape [C] containing normalized weights.
    """
    if isinstance(class_counts, (list, tuple)):
        counts = torch.tensor(class_counts, dtype=torch.float64)
    else:
        counts = class_counts.detach().to(dtype=torch.float64, device="cpu")

    C = len(counts) if num_classes is None else int(num_classes)
    if counts.numel() != C:
        raise ValueError(f"class_counts length ({counts.numel()}) must match num_classes ({C})")

    if beta <= 0.0:
        return torch.ones(C, dtype=dtype, device=device)

    if beta >= 1.0:
        raise ValueError(f"beta must be strictly < 1.0, got {beta}")

    # Effective number of samples: (1 - beta^n) / (1 - beta)
    # Using float64 for high precision with beta close to 1.0
    effective_num = 1.0 - torch.pow(beta, counts.clamp_min(1.0))
    weights = (1.0 - beta) / effective_num.clamp_min(1e-12)

    # Normalize weights so that their sum equals C (mean weight = 1.0)
    normalized = weights / weights.sum() * float(C)
    return normalized.to(dtype=dtype, device=device)


def compute_class_priors(
    class_counts: Sequence[int] | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes class prior probabilities pi_i = n_i / sum(n_j).

    Args:
        class_counts: Sample counts per class.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor of shape [C] containing class priors pi_i summing to 1.0.
    """
    if isinstance(class_counts, (list, tuple)):
        counts = torch.tensor(class_counts, dtype=torch.float64)
    else:
        counts = class_counts.detach().to(dtype=torch.float64, device="cpu")

    total = counts.sum().clamp_min(1.0)
    priors = counts / total
    return priors.to(dtype=dtype, device=device)


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss with effective number reweighting."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor = DTLD_STATE_CLASS_COUNTS,
        beta: float = 0.9999,
        gamma: float = 1.5,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.gamma = float(gamma)
        weights = compute_effective_num_weights(class_counts, beta=self.beta, num_classes=self.num_classes)
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for unmasked, valid candidate logits.

        Args:
            logits: [N, C] unnormalized logits.
            targets: [N] class indices in [0, C-1].

        Returns:
            Scalar loss tensor.
        """
        if logits.numel() == 0 or targets.numel() == 0:
            return logits.sum() * 0.0

        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        # Compute standard cross entropy with class weights
        ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction="none")

        if self.gamma > 0.0:
            probs = logits.softmax(dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_modulator = (1.0 - p_t).clamp_min(0.0).pow(self.gamma)
            loss = focal_modulator * ce_loss
        else:
            loss = ce_loss

        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Cross-Entropy with analytical log-prior shift."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor = DTLD_STATE_CLASS_COUNTS,
        prior_scale: float = 1.0,
        gamma: float = 0.0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.prior_scale = float(prior_scale)
        self.gamma = float(gamma)
        priors = compute_class_priors(class_counts)
        log_priors = torch.log(priors.clamp_min(1e-12))
        self.register_buffer("log_priors", log_priors)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass adjusting logits with analytical label priors.

        Args:
            logits: [N, C] unnormalized logits.
            targets: [N] class indices in [0, C-1].

        Returns:
            Scalar loss tensor.
        """
        if logits.numel() == 0 or targets.numel() == 0:
            return logits.sum() * 0.0

        log_p = self.log_priors.to(device=logits.device, dtype=logits.dtype)
        adjusted_logits = logits + self.prior_scale * log_p.unsqueeze(0)

        ce_loss = F.cross_entropy(adjusted_logits, targets, reduction="none")

        if self.gamma > 0.0:
            probs = adjusted_logits.softmax(dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_modulator = (1.0 - p_t).clamp_min(0.0).pow(self.gamma)
            loss = focal_modulator * ce_loss
        else:
            loss = ce_loss

        return loss.mean()


class CompositeClassBalancedLoss(nn.Module):
    """Composite Champion v3 State Loss: Effective Number Weights + Balanced Softmax + Focal Modulation."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor = DTLD_STATE_CLASS_COUNTS,
        beta: float = 0.9999,
        prior_scale: float = 1.0,
        gamma: float = 1.5,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.prior_scale = float(prior_scale)
        self.gamma = float(gamma)

        weights = compute_effective_num_weights(class_counts, beta=self.beta, num_classes=self.num_classes)
        self.register_buffer("class_weights", weights)

        priors = compute_class_priors(class_counts)
        log_priors = torch.log(priors.clamp_min(1e-12))
        self.register_buffer("log_priors", log_priors)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass combining prior shift, effective sample reweighting, and focal decay.

        Args:
            logits: [N, C] unnormalized logits.
            targets: [N] class indices in [0, C-1].

        Returns:
            Scalar loss tensor.
        """
        if logits.numel() == 0 or targets.numel() == 0:
            return logits.sum() * 0.0

        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        log_p = self.log_priors.to(device=logits.device, dtype=logits.dtype)

        adjusted_logits = logits + self.prior_scale * log_p.unsqueeze(0)
        ce_loss = F.cross_entropy(adjusted_logits, targets, weight=weights, reduction="none")

        if self.gamma > 0.0:
            probs = adjusted_logits.softmax(dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_modulator = (1.0 - p_t).clamp_min(0.0).pow(self.gamma)
            loss = focal_modulator * ce_loss
        else:
            loss = ce_loss

        return loss.mean()


def assigned_class_balanced_state_loss(
    logits: torch.Tensor,
    padded_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    loss_type: str = "cb_focal",
    beta: float = 0.9999,
    gamma: float = 1.5,
    prior_scale: float = 1.0,
    class_counts: Sequence[int] | torch.Tensor = DTLD_STATE_CLASS_COUNTS,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Computes assigned state loss supporting all long-tail rebalancing formulations.

    Supported loss_type modes:
    - 'focal': Standard Multi-Class Focal Loss (Uniform weights).
    - 'cb_focal': Class-Balanced Focal Loss with effective number weighting.
    - 'balanced_softmax': Softmax cross entropy with analytical log priors.
    - 'balanced_focal_softmax': Balanced Softmax with focal modulation.
    - 'cb_balanced_softmax' / 'composite': Full composite CB-Focal + Balanced Softmax.

    Args:
        logits: [batch, num_classes, num_anchors]
        padded_targets: [batch, max_gt]
        foreground_mask: [batch, num_anchors]
        target_gt_indices: [batch, num_anchors]
        loss_type: Selected loss variant identifier.
        beta: Effective number parameter.
        gamma: Focal modulation power.
        prior_scale: Multiplier on log priors.
        class_counts: Sample counts per class for prior/weight calculation.
        class_weights: Explicit class weight override (if provided).

    Returns:
        tuple (loss_tensor, valid_positive_count)
    """
    if logits.ndim != 3:
        raise ValueError(f"state logits must have shape [batch, classes, anchors], got {logits.shape}")
    batch, num_classes, anchors = logits.shape
    if foreground_mask.shape != (batch, anchors):
        raise ValueError("foreground mask shape does not match logits")
    if target_gt_indices.shape != (batch, anchors):
        raise ValueError("target GT index shape does not match logits")
    if padded_targets.shape[0] != batch:
        raise ValueError("padded targets batch dimension does not match logits")
    if padded_targets.shape[1] == 0:
        return logits.sum() * 0.0, 0

    safe_indices = target_gt_indices.clamp(0, padded_targets.shape[1] - 1)
    anchor_targets = padded_targets.gather(1, safe_indices)
    valid = foreground_mask.bool() & anchor_targets.ge(0)
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0

    selected_logits = logits.permute(0, 2, 1)[valid]  # [N, C]
    selected_targets = anchor_targets[valid]           # [N]

    device = logits.device
    dtype = logits.dtype
    mode = str(loss_type).lower().strip()

    if mode in ("focal", "standard_focal", "ce"):
        weights = class_weights.to(device=device, dtype=dtype) if class_weights is not None else None
        ce = F.cross_entropy(selected_logits, selected_targets, weight=weights, reduction="none")
        if gamma > 0.0:
            p_t = selected_logits.softmax(1).gather(1, selected_targets.unsqueeze(1)).squeeze(1)
            ce = ce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
        return ce.mean(), count

    elif mode in ("cb_focal", "class_balanced_focal"):
        weights = (
            class_weights.to(device=device, dtype=dtype)
            if class_weights is not None
            else compute_effective_num_weights(class_counts, beta=beta, num_classes=num_classes, device=device, dtype=dtype)
        )
        ce = F.cross_entropy(selected_logits, selected_targets, weight=weights, reduction="none")
        if gamma > 0.0:
            p_t = selected_logits.softmax(1).gather(1, selected_targets.unsqueeze(1)).squeeze(1)
            ce = ce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
        return ce.mean(), count

    elif mode in ("balanced_softmax", "bs"):
        priors = compute_class_priors(class_counts, device=device, dtype=dtype)
        log_p = torch.log(priors.clamp_min(1e-12))
        adj_logits = selected_logits + prior_scale * log_p.unsqueeze(0)
        ce = F.cross_entropy(adj_logits, selected_targets, reduction="none")
        return ce.mean(), count

    elif mode in ("balanced_focal_softmax", "bfs"):
        priors = compute_class_priors(class_counts, device=device, dtype=dtype)
        log_p = torch.log(priors.clamp_min(1e-12))
        adj_logits = selected_logits + prior_scale * log_p.unsqueeze(0)
        ce = F.cross_entropy(adj_logits, selected_targets, reduction="none")
        if gamma > 0.0:
            p_t = adj_logits.softmax(1).gather(1, selected_targets.unsqueeze(1)).squeeze(1)
            ce = ce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
        return ce.mean(), count

    elif mode in ("cb_balanced_softmax", "class_balanced_focal_softmax", "cb_focal_softmax", "composite", "cb_bs", "champion_v3"):
        weights = (
            class_weights.to(device=device, dtype=dtype)
            if class_weights is not None
            else compute_effective_num_weights(class_counts, beta=beta, num_classes=num_classes, device=device, dtype=dtype)
        )
        priors = compute_class_priors(class_counts, device=device, dtype=dtype)
        log_p = torch.log(priors.clamp_min(1e-12))
        adj_logits = selected_logits + prior_scale * log_p.unsqueeze(0)
        ce = F.cross_entropy(adj_logits, selected_targets, weight=weights, reduction="none")
        if gamma > 0.0:
            p_t = adj_logits.softmax(1).gather(1, selected_targets.unsqueeze(1)).squeeze(1)
            ce = ce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
        return ce.mean(), count

    else:
        raise ValueError(f"Unknown state loss type: {loss_type}")
