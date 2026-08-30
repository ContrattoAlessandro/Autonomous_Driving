"""Unit tests for Tiny-State Multi-Teacher Relation Distillation (Ticket E72)."""

import pytest
import torch
import torch.nn.functional as F

from tlr_yolo_mtl.training.distillation import (
    MultiTeacherDistillationConfig,
    MultiTeacherRelationDistillationLoss,
)


def test_multi_teacher_distillation_initialization():
    config = MultiTeacherDistillationConfig(
        temperature=3.0,
        weight_kd=1.0,
        weight_relation=0.5,
        weight_feature=0.25,
        local_teacher_weight=0.6,
        temporal_teacher_weight=0.4,
        sub4px_scale_boost=2.0,
    )
    distill_loss = MultiTeacherRelationDistillationLoss(config, student_dim=64, teacher_dim=64)

    assert distill_loss.config.temperature == 3.0
    assert distill_loss.config.weight_kd == 1.0
    assert distill_loss.config.weight_relation == 0.5
    assert distill_loss.config.weight_feature == 0.25
    assert isinstance(distill_loss.feat_proj, torch.nn.Identity)


def test_multi_teacher_consensus_weights():
    distill_loss = MultiTeacherRelationDistillationLoss()

    # Perfect agreement between teachers
    loc_logits_agree = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    temp_logits_agree = torch.tensor([[8.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]])
    consensus_high = distill_loss._compute_consensus_weights(loc_logits_agree, temp_logits_agree)

    assert consensus_high[0].item() == pytest.approx(1.0, abs=1e-3)
    assert consensus_high[1].item() == pytest.approx(1.0, abs=1e-3)

    # Disagreement between teachers
    loc_logits_disagree = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    temp_logits_disagree = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    consensus_low = distill_loss._compute_consensus_weights(loc_logits_disagree, temp_logits_disagree)

    assert consensus_low[0].item() < 0.6


def test_multi_teacher_relational_loss():
    distill_loss = MultiTeacherRelationDistillationLoss()

    N, D = 8, 64
    student_feats = torch.randn(N, D)
    teacher_feats = torch.randn(N, D)

    rel_loss = distill_loss._compute_relational_loss(student_feats, teacher_feats)
    assert rel_loss.item() >= 0.0
    assert not torch.isnan(rel_loss)

    # Identical representations should yield 0.0 relational loss
    zero_rel = distill_loss._compute_relational_loss(student_feats, student_feats)
    assert zero_rel.item() == pytest.approx(0.0, abs=1e-5)


def test_multi_teacher_distillation_forward_and_backward():
    N, D = 16, 64
    config = MultiTeacherDistillationConfig(temperature=2.5, sub4px_scale_boost=2.0)
    distill_loss = MultiTeacherRelationDistillationLoss(config, student_dim=64, teacher_dim=64)

    student_logits = torch.randn(N, 4, requires_grad=True)
    student_feats = torch.randn(N, D, requires_grad=True)

    loc_logits = torch.randn(N, 4)
    loc_feats = torch.randn(N, D)
    temp_logits = torch.randn(N, 4)
    temp_feats = torch.randn(N, D)

    # 8 sub-4px candidates (areas <= 16) and 8 macro candidates
    areas = torch.tensor([4.0] * 8 + [100.0] * 8)

    losses = distill_loss(
        student_logits=student_logits,
        student_features=student_feats,
        local_teacher_logits=loc_logits,
        local_teacher_features=loc_feats,
        temporal_teacher_logits=temp_logits,
        temporal_teacher_features=temp_feats,
        candidate_areas=areas,
    )

    assert "loss_distill_total" in losses
    assert "loss_distill_kd" in losses
    assert "loss_distill_relation" in losses
    assert "loss_distill_feature" in losses
    assert "mean_teacher_consensus" in losses

    assert losses["loss_distill_total"].item() > 0.0
    assert not torch.isnan(losses["loss_distill_total"])

    # Test backward pass
    losses["loss_distill_total"].backward()
    assert student_logits.grad is not None
    assert student_feats.grad is not None
    assert not torch.isnan(student_logits.grad).any()
    assert not torch.isnan(student_feats.grad).any()
