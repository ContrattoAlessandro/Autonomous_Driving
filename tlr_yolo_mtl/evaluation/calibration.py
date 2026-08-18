"""Post-training scalar temperature fitting for frozen model logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    temperature: float
    loss_before: float
    loss_after: float
    valid_samples: int


def _valid(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets = targets.reshape(-1).long()
    if logits.ndim == 1:
        logits = logits.reshape(-1)
    elif logits.ndim == 2:
        if logits.shape[0] != targets.numel():
            raise ValueError("logit and target sample counts differ")
    else:
        raise ValueError("calibration logits must have shape [N] or [N, C]")
    mask = targets.ge(0)
    return logits[mask].float(), targets[mask]


def _nll(logits: torch.Tensor, targets: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    scaled = logits / temperature
    if logits.ndim == 1:
        return F.binary_cross_entropy_with_logits(scaled, targets.float())
    return F.cross_entropy(scaled, targets)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    grid_points: int = 400,
) -> TemperatureFit:
    """Fit one positive temperature by deterministic log-space search."""

    values, labels = _valid(logits, targets)
    if not labels.numel():
        raise ValueError("temperature fitting requires at least one valid target")
    candidates = torch.logspace(
        torch.log10(torch.tensor(minimum)),
        torch.log10(torch.tensor(maximum)),
        grid_points,
        device=values.device,
    )
    losses = torch.stack([_nll(values, labels, value) for value in candidates])
    best = int(losses.argmin())
    before = _nll(values, labels, torch.tensor(1.0, device=values.device))
    return TemperatureFit(
        temperature=float(candidates[best]),
        loss_before=float(before),
        loss_after=float(losses[best]),
        valid_samples=int(labels.numel()),
    )


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature
