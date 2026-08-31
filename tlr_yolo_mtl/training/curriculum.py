"""Dynamic Curriculum Loss Scheduling for Multi-Task Learning.

Provides smooth, continuous scheduling of multi-task loss weights (e.g. cosine, linear, sigmoid, step)
to stabilize early geometric convergence and progressively amplify fine-grained attribute classification
and contrastive relevance ranking without gradient interference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .losses import MultiTaskLossWeights, TLRMultiTaskCriterion


SUPPORTED_SCHEDULE_TYPES = ("cosine", "linear", "step", "sigmoid", "constant")


@dataclass(frozen=True, slots=True)
class CurriculumScheduleSpec:
    """Specification for single-task or multi-task dynamic weight trajectory."""

    start_epoch: float = 20.0
    end_epoch: float = 50.0
    start_weight: float = 1.0
    end_weight: float = 1.0
    schedule_type: str = "cosine"

    def __post_init__(self) -> None:
        if self.start_epoch < 0.0:
            raise ValueError(f"start_epoch must be non-negative, got {self.start_epoch}")
        if self.end_epoch < self.start_epoch:
            raise ValueError(
                f"end_epoch ({self.end_epoch}) cannot be strictly less than start_epoch ({self.start_epoch})"
            )
        if self.start_weight < 0.0 or self.end_weight < 0.0:
            raise ValueError(
                f"loss weights must be non-negative, got start={self.start_weight}, end={self.end_weight}"
            )
        if self.schedule_type.lower() not in SUPPORTED_SCHEDULE_TYPES:
            raise ValueError(
                f"unsupported schedule_type: {self.schedule_type!r}. Supported: {SUPPORTED_SCHEDULE_TYPES}"
            )

    def compute_weight(self, progress_epoch: float) -> float:
        """Compute the interpolated weight at a given continuous epoch time."""
        if progress_epoch <= self.start_epoch:
            return float(self.start_weight)
        if progress_epoch >= self.end_epoch:
            return float(self.end_weight)
        
        # Duration is positive because start_epoch < end_epoch if progress_epoch is strictly between
        duration = self.end_epoch - self.start_epoch
        if duration <= 1e-9:
            return float(self.end_weight)
        
        tau = (progress_epoch - self.start_epoch) / duration
        tau = max(0.0, min(1.0, tau))

        stype = self.schedule_type.lower()
        if stype == "cosine":
            # S-curve with zero derivative at boundaries
            alpha = (1.0 - math.cos(math.pi * tau)) / 2.0
        elif stype == "linear":
            alpha = tau
        elif stype == "sigmoid":
            # Centered sigmoid mapped from [0, 1]
            raw = 1.0 / (1.0 + math.exp(-10.0 * (tau - 0.5)))
            # Normalize so alpha(0)=0 and alpha(1)=1
            sig0 = 1.0 / (1.0 + math.exp(5.0))
            sig1 = 1.0 / (1.0 + math.exp(-5.0))
            alpha = (raw - sig0) / (sig1 - sig0)
        elif stype == "step":
            alpha = 1.0 if tau >= 1.0 else 0.0
        elif stype == "constant":
            alpha = 0.0
        else:
            alpha = (1.0 - math.cos(math.pi * tau)) / 2.0

        return float(self.start_weight + alpha * (self.end_weight - self.start_weight))


class DynamicCurriculumLossScheduler:
    """Dynamic multi-task loss weight scheduler with smooth epoch / micro-step transitions."""

    def __init__(
        self,
        base_weights: MultiTaskLossWeights | Mapping[str, float] | None = None,
        *,
        task_schedules: Mapping[str, CurriculumScheduleSpec | Mapping[str, Any]] | None = None,
        start_epoch: float = 20.0,
        end_epoch: float = 50.0,
        schedule_type: str = "cosine",
        initial_weights: MultiTaskLossWeights | Mapping[str, float] | None = None,
        target_weights: MultiTaskLossWeights | Mapping[str, float] | None = None,
        steps_per_epoch: int | None = None,
    ) -> None:
        if base_weights is None:
            if initial_weights is not None:
                self.base_weights = (
                    initial_weights
                    if isinstance(initial_weights, MultiTaskLossWeights)
                    else MultiTaskLossWeights(**initial_weights)
                )
            else:
                self.base_weights = MultiTaskLossWeights()
        elif isinstance(base_weights, MultiTaskLossWeights):
            self.base_weights = base_weights
        else:
            self.base_weights = MultiTaskLossWeights(**base_weights)

        self.steps_per_epoch = steps_per_epoch
        self.schedules: dict[str, CurriculumScheduleSpec] = {}

        # 1. Populate per-task schedules if explicitly specified
        if task_schedules:
            for task_name, spec_data in task_schedules.items():
                if isinstance(spec_data, CurriculumScheduleSpec):
                    self.schedules[task_name] = spec_data
                elif isinstance(spec_data, Mapping):
                    self.schedules[task_name] = CurriculumScheduleSpec(**spec_data)
                else:
                    raise TypeError(f"invalid schedule specification for {task_name}: {type(spec_data)}")

        # 2. If target_weights are provided, construct schedules for differing weights
        if target_weights is not None:
            target_dict = (
                asdict(target_weights)
                if isinstance(target_weights, MultiTaskLossWeights)
                else dict(target_weights)
            )
            base_dict = asdict(self.base_weights)
            init_dict = (
                asdict(initial_weights)
                if isinstance(initial_weights, MultiTaskLossWeights)
                else dict(initial_weights)
                if initial_weights is not None
                else base_dict
            )

            for task_name, target_val in target_dict.items():
                if task_name not in self.schedules:
                    start_val = float(init_dict.get(task_name, getattr(self.base_weights, task_name, 0.0)))
                    self.schedules[task_name] = CurriculumScheduleSpec(
                        start_epoch=float(start_epoch),
                        end_epoch=float(end_epoch),
                        start_weight=start_val,
                        end_weight=float(target_val),
                        schedule_type=str(schedule_type),
                    )

        self._last_epoch: float = 0.0
        self._last_step: int = 0

    def _continuous_epoch(self, epoch: float, step: int | None = None) -> float:
        """Compute the continuous epoch coordinate accounting for micro-step progression."""
        if step is None or self.steps_per_epoch is None or self.steps_per_epoch <= 0:
            return float(epoch)
        # Fractional progression within current epoch
        frac = (step % self.steps_per_epoch) / float(self.steps_per_epoch)
        return float(epoch) + frac

    def get_weights(self, epoch: float, step: int | None = None) -> MultiTaskLossWeights:
        """Calculate loss weights at a continuous epoch/step coordinate."""
        t = self._continuous_epoch(epoch, step)
        weights_dict = asdict(self.base_weights)
        for task_name, schedule in self.schedules.items():
            if task_name in weights_dict:
                weights_dict[task_name] = schedule.compute_weight(t)
        return MultiTaskLossWeights(**weights_dict)

    def get_weights_dict(self, epoch: float, step: int | None = None) -> dict[str, float]:
        """Return dict of loss weights rounded for metric logging."""
        weights = self.get_weights(epoch, step)
        return {k: round(float(v), 6) for k, v in asdict(weights).items()}

    def step(self, epoch: float, step: int | None = None) -> MultiTaskLossWeights:
        """Advance scheduler state and return current loss weights."""
        self._last_epoch = float(epoch)
        self._last_step = int(step) if step is not None else 0
        return self.get_weights(epoch, step)

    def apply_to_criterion(
        self,
        criterion: TLRMultiTaskCriterion,
        epoch: float,
        step: int | None = None,
    ) -> MultiTaskLossWeights:
        """Update criterion loss weights directly."""
        new_weights = self.step(epoch, step)
        criterion.weights = new_weights
        return new_weights

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "base_weights": asdict(self.base_weights),
            "schedules": {k: asdict(v) for k, v in self.schedules.items()},
            "steps_per_epoch": self.steps_per_epoch,
            "_last_epoch": self._last_epoch,
            "_last_step": self._last_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        if "base_weights" in state:
            self.base_weights = MultiTaskLossWeights(**state["base_weights"])
        if "schedules" in state:
            self.schedules = {
                k: CurriculumScheduleSpec(**v)
                for k, v in state["schedules"].items()
            }
        if "steps_per_epoch" in state:
            self.steps_per_epoch = state["steps_per_epoch"]
        self._last_epoch = float(state.get("_last_epoch", 0.0))
        self._last_step = int(state.get("_last_step", 0))


def build_curriculum_loss_scheduler(
    config: Mapping[str, Any],
    steps_per_epoch: int | None = None,
) -> DynamicCurriculumLossScheduler | None:
    """Factory to construct a DynamicCurriculumLossScheduler from training configuration."""
    curriculum_cfg = (
        config.get("curriculum_loss_schedule")
        or config.get("loss_curriculum")
        or config.get("loss", {}).get("curriculum_schedule")
    )
    if not curriculum_cfg or not isinstance(curriculum_cfg, Mapping):
        return None

    if not curriculum_cfg.get("enabled", True):
        return None

    base_weights_raw = config.get("loss_weights", {})
    base_weights = (
        MultiTaskLossWeights(**base_weights_raw)
        if isinstance(base_weights_raw, Mapping)
        else MultiTaskLossWeights()
    )

    initial_weights_raw = curriculum_cfg.get("initial_weights")
    target_weights_raw = curriculum_cfg.get("target_weights")
    task_schedules_raw = curriculum_cfg.get("task_schedules")

    resolved_steps = (
        steps_per_epoch
        if steps_per_epoch is not None
        else config.get("optimizer_steps_per_epoch")
    )

    return DynamicCurriculumLossScheduler(
        base_weights=base_weights,
        task_schedules=task_schedules_raw,
        start_epoch=float(curriculum_cfg.get("start_epoch", 20.0)),
        end_epoch=float(curriculum_cfg.get("end_epoch", 50.0)),
        schedule_type=str(curriculum_cfg.get("schedule_type", "cosine")),
        initial_weights=initial_weights_raw,
        target_weights=target_weights_raw,
        steps_per_epoch=int(resolved_steps) if resolved_steps is not None else None,
    )
