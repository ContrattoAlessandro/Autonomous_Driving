"""Unit and integration tests for Ticket 04: Dynamic Curriculum Loss Scheduling."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.training.curriculum import (
    CurriculumScheduleSpec,
    DynamicCurriculumLossScheduler,
    SUPPORTED_SCHEDULE_TYPES,
    build_curriculum_loss_scheduler,
)
from tlr_yolo_mtl.training.losses import MultiTaskLossWeights, TLRMultiTaskCriterion


def test_curriculum_schedule_spec_validation() -> None:
    """Verify validation and error handling in CurriculumScheduleSpec."""
    # Valid spec
    spec = CurriculumScheduleSpec(
        start_epoch=20.0,
        end_epoch=50.0,
        start_weight=1.0,
        end_weight=0.7,
        schedule_type="cosine",
    )
    assert spec.start_epoch == 20.0
    assert spec.end_epoch == 50.0
    assert spec.start_weight == 1.0
    assert spec.end_weight == 0.7

    # Negative start epoch
    with pytest.raises(ValueError, match="start_epoch must be non-negative"):
        CurriculumScheduleSpec(start_epoch=-1.0, end_epoch=10.0)

    # Inverted epochs (end < start)
    with pytest.raises(ValueError, match="cannot be strictly less"):
        CurriculumScheduleSpec(start_epoch=30.0, end_epoch=20.0)

    # Negative weights
    with pytest.raises(ValueError, match="loss weights must be non-negative"):
        CurriculumScheduleSpec(start_weight=-0.5, end_weight=1.0)

    # Unsupported schedule type
    with pytest.raises(ValueError, match="unsupported schedule_type"):
        CurriculumScheduleSpec(schedule_type="unknown_polynomial")


def test_cosine_schedule_exact_mathematical_properties() -> None:
    """Verify exact mathematical properties and boundary conditions of cosine schedule."""
    # Detection weight: 1.0 -> 0.7 from epoch 20 to 50
    det_spec = CurriculumScheduleSpec(
        start_epoch=20.0,
        end_epoch=50.0,
        start_weight=1.0,
        end_weight=0.7,
        schedule_type="cosine",
    )

    # State weight: 0.75 -> 1.25 from epoch 20 to 50
    state_spec = CurriculumScheduleSpec(
        start_epoch=20.0,
        end_epoch=50.0,
        start_weight=0.75,
        end_weight=1.25,
        schedule_type="cosine",
    )

    # 1. Early phase (epochs < 20): strictly constant initial weights
    assert det_spec.compute_weight(0.0) == 1.0
    assert det_spec.compute_weight(10.0) == 1.0
    assert det_spec.compute_weight(19.99) == 1.0
    assert state_spec.compute_weight(0.0) == 0.75
    assert state_spec.compute_weight(19.99) == 0.75

    # 2. Midpoint (epoch 35): tau = 0.5 -> alpha = (1 - cos(pi/2))/2 = 0.5
    assert pytest.approx(det_spec.compute_weight(35.0), rel=1e-5) == 0.85  # 1.0 + 0.5 * (0.7 - 1.0)
    assert pytest.approx(state_spec.compute_weight(35.0), rel=1e-5) == 1.00  # 0.75 + 0.5 * (1.25 - 0.75)

    # 3. Quarter point (epoch 27.5): tau = 0.25 -> alpha = (1 - cos(pi/4))/2 = (1 - sqrt(2)/2)/2 ≈ 0.1464466
    expected_alpha_quarter = (1.0 - math.cos(math.pi * 0.25)) / 2.0
    assert pytest.approx(det_spec.compute_weight(27.5), rel=1e-5) == 1.0 + expected_alpha_quarter * (0.7 - 1.0)
    assert pytest.approx(state_spec.compute_weight(27.5), rel=1e-5) == 0.75 + expected_alpha_quarter * (1.25 - 0.75)

    # 4. End & post-end phase (epoch >= 50): strictly target weights
    assert det_spec.compute_weight(50.0) == 0.70
    assert det_spec.compute_weight(55.0) == 0.70
    assert det_spec.compute_weight(100.0) == 0.70
    assert state_spec.compute_weight(50.0) == 1.25
    assert state_spec.compute_weight(70.0) == 1.25

    # 5. Monotonicity checks
    epochs = [20.0 + i * 0.5 for i in range(61)]
    det_weights = [det_spec.compute_weight(e) for e in epochs]
    state_weights = [state_spec.compute_weight(e) for e in epochs]

    for i in range(len(epochs) - 1):
        assert det_weights[i] >= det_weights[i + 1] - 1e-9, f"Non-monotonic det at {epochs[i]}"
        assert state_weights[i] <= state_weights[i + 1] + 1e-9, f"Non-monotonic state at {epochs[i]}"


def test_alternative_schedules() -> None:
    """Verify linear, sigmoid, step, and constant schedule behaviors."""
    # Linear schedule
    lin_spec = CurriculumScheduleSpec(
        start_epoch=10.0, end_epoch=30.0, start_weight=1.0, end_weight=2.0, schedule_type="linear"
    )
    assert lin_spec.compute_weight(10.0) == 1.0
    assert pytest.approx(lin_spec.compute_weight(20.0), rel=1e-5) == 1.5
    assert lin_spec.compute_weight(30.0) == 2.0

    # Step schedule
    step_spec = CurriculumScheduleSpec(
        start_epoch=10.0, end_epoch=30.0, start_weight=1.0, end_weight=2.0, schedule_type="step"
    )
    assert step_spec.compute_weight(29.99) == 1.0
    assert step_spec.compute_weight(30.0) == 2.0
    assert step_spec.compute_weight(35.0) == 2.0

    # Sigmoid schedule
    sig_spec = CurriculumScheduleSpec(
        start_epoch=10.0, end_epoch=30.0, start_weight=1.0, end_weight=2.0, schedule_type="sigmoid"
    )
    assert pytest.approx(sig_spec.compute_weight(10.0), abs=1e-4) == 1.0
    assert pytest.approx(sig_spec.compute_weight(20.0), rel=1e-4) == 1.5
    assert pytest.approx(sig_spec.compute_weight(30.0), abs=1e-4) == 2.0


def test_dynamic_curriculum_loss_scheduler_multi_task() -> None:
    """Verify multi-task weight coordination across epochs and micro-steps."""
    scheduler = DynamicCurriculumLossScheduler(
        start_epoch=20.0,
        end_epoch=50.0,
        schedule_type="cosine",
        initial_weights={
            "detection": 1.0,
            "state": 0.75,
            "relevance": 1.0,
            "nwd": 0.5,
            "maneuver": 1.0,
        },
        target_weights={
            "detection": 0.70,
            "state": 1.25,
            "relevance": 1.50,
            "nwd": 0.5,
            "maneuver": 1.0,
        },
        steps_per_epoch=100,
    )

    # Initial epoch 0
    w0 = scheduler.get_weights(epoch=0.0)
    assert w0.detection == 1.0
    assert w0.state == 0.75
    assert w0.relevance == 1.0
    assert w0.nwd == 0.5
    assert w0.maneuver == 1.0

    # Continuous step within epoch 35 (epoch 35, step 50 -> continuous epoch 35.5)
    w35_half = scheduler.get_weights(epoch=35, step=50)
    expected_tau = (35.5 - 20.0) / 30.0
    expected_alpha = (1.0 - math.cos(math.pi * expected_tau)) / 2.0
    assert pytest.approx(w35_half.detection, rel=1e-4) == 1.0 + expected_alpha * (0.7 - 1.0)
    assert pytest.approx(w35_half.state, rel=1e-4) == 0.75 + expected_alpha * (1.25 - 0.75)
    assert pytest.approx(w35_half.relevance, rel=1e-4) == 1.0 + expected_alpha * (1.50 - 1.0)

    # Final epoch 50
    w50 = scheduler.get_weights(epoch=50.0)
    assert pytest.approx(w50.detection, rel=1e-5) == 0.70
    assert pytest.approx(w50.state, rel=1e-5) == 1.25
    assert pytest.approx(w50.relevance, rel=1e-5) == 1.50
    assert w50.nwd == 0.5
    assert w50.maneuver == 1.0

    # Dictionary logging output
    d50 = scheduler.get_weights_dict(epoch=50.0)
    assert d50["detection"] == 0.70
    assert d50["state"] == 1.25
    assert d50["relevance"] == 1.50


def test_criterion_integration() -> None:
    """Verify application of curriculum scheduler to TLRMultiTaskCriterion."""
    from types import SimpleNamespace

    scheduler = DynamicCurriculumLossScheduler(
        start_epoch=20.0,
        end_epoch=50.0,
        schedule_type="cosine",
        initial_weights={"detection": 1.0, "state": 0.75, "relevance": 1.0},
        target_weights={"detection": 0.70, "state": 1.25, "relevance": 1.50},
    )

    class DummyDetect(nn.Module):
        def __init__(self):
            super().__init__()
            self.nc = 2
            self.reg_max = 16
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.weight = nn.Parameter(torch.zeros(1))

        def parse_output(self, preds):
            return preds

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([DummyDetect()])
            self.stride = self.model[0].stride
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    dummy_model = DummyModel()

    criterion = TLRMultiTaskCriterion(
        dummy_model,
        weights=MultiTaskLossWeights(detection=1.0, state=0.75, relevance=1.0),
    )

    assert criterion.weights.detection == 1.0
    assert criterion.weights.state == 0.75
    assert criterion.weights.relevance == 1.0

    # Apply at epoch 35
    scheduler.apply_to_criterion(criterion, epoch=35.0)
    assert pytest.approx(criterion.weights.detection, rel=1e-5) == 0.85
    assert pytest.approx(criterion.weights.state, rel=1e-5) == 1.00
    assert pytest.approx(criterion.weights.relevance, rel=1e-5) == 1.25

    # Apply at epoch 50
    scheduler.apply_to_criterion(criterion, epoch=50.0)
    assert pytest.approx(criterion.weights.detection, rel=1e-5) == 0.70
    assert pytest.approx(criterion.weights.state, rel=1e-5) == 1.25
    assert pytest.approx(criterion.weights.relevance, rel=1e-5) == 1.50


def test_scheduler_state_dict_and_checkpoint_resume() -> None:
    """Verify deterministic state serialization and restoration."""
    scheduler = DynamicCurriculumLossScheduler(
        start_epoch=20.0,
        end_epoch=50.0,
        schedule_type="cosine",
        initial_weights={"detection": 1.0, "state": 0.75},
        target_weights={"detection": 0.70, "state": 1.25},
        steps_per_epoch=100,
    )

    scheduler.step(epoch=32.0, step=42)
    state = scheduler.state_dict()

    new_scheduler = DynamicCurriculumLossScheduler()
    new_scheduler.load_state_dict(state)

    w_orig = scheduler.get_weights(epoch=37.0, step=12)
    w_restored = new_scheduler.get_weights(epoch=37.0, step=12)

    assert w_orig.detection == w_restored.detection
    assert w_orig.state == w_restored.state
    assert w_orig.relevance == w_restored.relevance


def test_build_curriculum_loss_scheduler_factory() -> None:
    """Verify factory construction from YAML configuration structures."""
    config = {
        "loss_weights": {
            "detection": 1.0,
            "state": 0.75,
            "round": 0.5,
            "maneuver": 1.0,
            "ego_lane": 0.0,
            "relevance": 1.0,
            "nwd": 0.5,
        },
        "curriculum_loss_schedule": {
            "enabled": True,
            "start_epoch": 20,
            "end_epoch": 50,
            "schedule_type": "cosine",
            "initial_weights": {
                "detection": 1.0,
                "state": 0.75,
                "relevance": 1.0,
            },
            "target_weights": {
                "detection": 0.70,
                "state": 1.25,
                "relevance": 1.50,
            },
        },
        "optimizer_steps_per_epoch": 100,
    }

    scheduler = build_curriculum_loss_scheduler(config)
    assert scheduler is not None
    assert scheduler.steps_per_epoch == 100
    assert scheduler.get_weights(0).detection == 1.0
    assert pytest.approx(scheduler.get_weights(50).detection, rel=1e-5) == 0.70
    assert pytest.approx(scheduler.get_weights(50).state, rel=1e-5) == 1.25
    assert pytest.approx(scheduler.get_weights(50).relevance, rel=1e-5) == 1.50

    # Disabled config
    disabled_config = {**config, "curriculum_loss_schedule": {"enabled": False}}
    assert build_curriculum_loss_scheduler(disabled_config) is None

    # Empty config
    empty_config = {"loss_weights": {"detection": 1.0}}
    assert build_curriculum_loss_scheduler(empty_config) is None


def test_gradient_scaling_dynamics_under_curriculum() -> None:
    """Verify that modulated loss weights scale backpropagated gradients proportionally."""
    scheduler = DynamicCurriculumLossScheduler(
        start_epoch=20.0,
        end_epoch=50.0,
        schedule_type="cosine",
        initial_weights={"detection": 1.0, "state": 0.75, "relevance": 1.0},
        target_weights={"detection": 0.70, "state": 1.25, "relevance": 1.50},
    )

    w_start = scheduler.get_weights(epoch=0.0)
    w_end = scheduler.get_weights(epoch=50.0)

    # Simulated task losses
    L_det = torch.tensor(2.0, requires_grad=True)
    L_state = torch.tensor(1.5, requires_grad=True)
    L_rel = torch.tensor(1.2, requires_grad=True)

    # Total loss at epoch 0
    total_0 = w_start.detection * L_det + w_start.state * L_state + w_start.relevance * L_rel
    total_0.backward(retain_graph=True)
    g_det_0 = L_det.grad.item()
    g_state_0 = L_state.grad.item()
    g_rel_0 = L_rel.grad.item()

    assert g_det_0 == 1.0
    assert g_state_0 == 0.75
    assert g_rel_0 == 1.0

    # Reset grads
    L_det.grad = None
    L_state.grad = None
    L_rel.grad = None

    # Total loss at epoch 50
    total_50 = w_end.detection * L_det + w_end.state * L_state + w_end.relevance * L_rel
    total_50.backward()
    g_det_50 = L_det.grad.item()
    g_state_50 = L_state.grad.item()
    g_rel_50 = L_rel.grad.item()

    assert pytest.approx(g_det_50, rel=1e-5) == 0.70
    assert pytest.approx(g_state_50, rel=1e-5) == 1.25
    assert pytest.approx(g_rel_50, rel=1e-5) == 1.50

    # Verify proportional changes
    assert pytest.approx(g_state_50 / g_state_0, rel=1e-5) == 1.25 / 0.75  # +66.67%
    assert pytest.approx(g_rel_50 / g_rel_0, rel=1e-5) == 1.50 / 1.00     # +50.00%
    assert pytest.approx(g_det_50 / g_det_0, rel=1e-5) == 0.70 / 1.00     # -30.00%

