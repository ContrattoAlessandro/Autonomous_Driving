"""Unit tests for Class-Balanced Focal Loss & Balanced Softmax (Ticket E44)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.training.class_balanced_loss import (

    DTLD_STATE_CLASS_COUNTS,
    STATE_CLASS_NAMES,
    BalancedSoftmaxLoss,
    ClassBalancedFocalLoss,
    CompositeClassBalancedLoss,
    assigned_class_balanced_state_loss,
    compute_class_priors,
    compute_effective_num_weights,
)
from tlr_yolo_mtl.training.losses import MultiTaskLossWeights, TLRMultiTaskCriterion


def test_compute_effective_num_weights():
    """Verify mathematical properties of effective number weighting."""
    counts = [30000, 3000, 45000, 8000]

    # beta = 0.0 -> uniform weights
    w_uniform = compute_effective_num_weights(counts, beta=0.0)
    assert torch.allclose(w_uniform, torch.ones(4), atol=1e-5)

    # beta = 0.9999 -> inverse rank ordering (rare classes get higher weights)
    w_cb = compute_effective_num_weights(counts, beta=0.9999)
    assert w_cb.shape == (4,)
    assert torch.isclose(w_cb.sum(), torch.tensor(4.0, dtype=w_cb.dtype), atol=1e-4)

    # Yellow (idx 1, 3000) should have the highest weight, Green (idx 2, 45000) the lowest
    assert w_cb[1] > w_cb[3] > w_cb[0] > w_cb[2]

    # Invalid beta should raise ValueError
    with pytest.raises(ValueError, match="beta must be strictly < 1.0"):
        compute_effective_num_weights(counts, beta=1.0)


def test_compute_class_priors():
    """Verify analytical class prior calculation."""
    counts = [100, 200, 300, 400]
    priors = compute_class_priors(counts)
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert torch.allclose(priors, expected, atol=1e-5)
    assert torch.isclose(priors.sum(), torch.tensor(1.0), atol=1e-5)


def test_loss_modules_forward_backward():
    """Verify that all loss modules execute forward and backward passes without error."""
    torch.manual_seed(42)
    logits = torch.randn(10, 4, requires_grad=True)
    targets = torch.randint(0, 4, (10,))

    # 1. ClassBalancedFocalLoss
    cb_focal = ClassBalancedFocalLoss(DTLD_STATE_CLASS_COUNTS, beta=0.9999, gamma=1.5)
    loss1 = cb_focal(logits, targets)
    assert loss1.ndim == 0 and loss1.item() > 0.0
    loss1.backward()
    assert logits.grad is not None
    logits.grad.zero_()

    # 2. BalancedSoftmaxLoss
    bs_loss = BalancedSoftmaxLoss(DTLD_STATE_CLASS_COUNTS, prior_scale=1.0, gamma=0.0)
    loss2 = bs_loss(logits, targets)
    assert loss2.ndim == 0 and loss2.item() > 0.0
    loss2.backward()
    assert logits.grad is not None
    logits.grad.zero_()

    # 3. CompositeClassBalancedLoss
    comp_loss = CompositeClassBalancedLoss(DTLD_STATE_CLASS_COUNTS, beta=0.9999, gamma=1.5)
    loss3 = comp_loss(logits, targets)
    assert loss3.ndim == 0 and loss3.item() > 0.0
    loss3.backward()
    assert logits.grad is not None


def test_assigned_state_loss_variants_and_masking():
    """Verify masking contract and variant consistency in assigned state loss."""
    device = torch.device("cpu")
    # Batch size 1, 4 classes, 5 anchors
    logits = torch.zeros((1, 4, 5), device=device, requires_grad=True)
    # GT targets: index 0 is class 1 (yellow), index 1 is -1 (ignore)
    padded_targets = torch.tensor([[1, -1]], device=device)
    # Anchor 1 matches GT 0 (valid), Anchor 2 matches GT 1 (masked ignore), others are background
    foreground = torch.tensor([[False, True, True, False, False]], device=device)
    gt_indices = torch.tensor([[0, 0, 1, 0, 0]], device=device)

    for loss_type in ("focal", "cb_focal", "balanced_softmax", "balanced_focal_softmax", "cb_balanced_softmax"):
        if logits.grad is not None:
            logits.grad.zero_()
        loss, count = assigned_class_balanced_state_loss(
            logits,
            padded_targets,
            foreground,
            gt_indices,
            loss_type=loss_type,
            beta=0.9999,
            gamma=1.5,
            class_counts=DTLD_STATE_CLASS_COUNTS,
        )
        assert count == 1, f"Expected 1 valid positive match for {loss_type}, got {count}"
        assert loss.ndim == 0 and loss.item() > 0.0
        loss.backward()

        grad_by_anchor = logits.grad.abs().sum(1).squeeze(0)
        active = torch.nonzero(grad_by_anchor > 0, as_tuple=False).reshape(-1).tolist()
        assert active == [1], f"Only anchor 1 should receive gradients in {loss_type}, got {active}"
        # Unassigned/ignored anchors should have exact zero gradient
        assert float(grad_by_anchor[0]) == 0.0
        assert float(grad_by_anchor[2]) == 0.0
        assert float(grad_by_anchor[3]) == 0.0
        assert float(grad_by_anchor[4]) == 0.0


def test_tlr_multitask_criterion_with_cb_state_loss():
    """Verify full TLRMultiTaskCriterion integration with state loss rebalancing."""
    from types import SimpleNamespace

    class DummyDetect(nn.Module):
        def __init__(self):
            super().__init__()
            self.nc = 2
            self.reg_max = 16
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.weight = nn.Parameter(torch.randn(2, 4))

        def parse_output(self, preds):
            return preds

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([DummyDetect()])
            self.stride = self.model[0].stride
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    model = DummyModel()

    criterion = TLRMultiTaskCriterion(
        model,
        state_loss_type="cb_balanced_softmax",
        state_beta=0.9999,
        state_prior_scale=1.0,
        attribute_gamma=1.5,
    )


    B, num_anchors = 2, (10 * 20) + (5 * 10) + (3 * 5)  # 265
    device = torch.device("cpu")
    predictions = {
        "scores": torch.randn(B, 2, num_anchors, requires_grad=True),
        "boxes": torch.randn(B, 64, num_anchors, requires_grad=True),
        "feats": [
            torch.randn(B, 64, 10, 20),
            torch.randn(B, 128, 5, 10),
            torch.randn(B, 256, 3, 5),
        ],
        "state_logits": torch.randn(B, 4, num_anchors, requires_grad=True),
        "round_logits": torch.randn(B, 1, num_anchors, requires_grad=True),
        "maneuver_logits": torch.randn(B, 3, num_anchors, requires_grad=True),
        "ego_lane_logits": torch.randn(B, 1, num_anchors, requires_grad=True),
        "dense_local_relevance_logits": torch.randn(B, 1, num_anchors, requires_grad=True),
        "relevance_logits": torch.randn(B, 1, 8, requires_grad=True),
        "traffic_candidate_indices": torch.zeros((B, 8), dtype=torch.long),
        "traffic_candidate_valid": torch.ones((B, 8), dtype=torch.bool),
        "attention_enabled_flag": torch.tensor(1.0),
        "attention_weights": torch.zeros((B, 8, 8)),
    }
    batch = {
        "object_batch_idx": torch.tensor([0, 1]),
        "object_cls": torch.tensor([[0.0], [0.0]]),
        "object_bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.2], [0.4, 0.4, 0.15, 0.25]]),
        "object_state": torch.tensor([0, 1]),
        "object_round": torch.tensor([[1.0], [0.0]]),
        "object_maneuver": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "object_relevance": torch.tensor([1, 0]),
        "object_ego_lane": torch.tensor([[1.0], [0.0]]),
        "traffic_relevance_valid": torch.tensor([True, True]),
        "unified_detection_valid": torch.tensor([True, True]),
    }

    result = criterion(predictions, batch)
    assert result.total.ndim == 0
    assert result.state.item() >= 0.0
    result.total.backward()
    assert predictions["state_logits"].grad is not None
