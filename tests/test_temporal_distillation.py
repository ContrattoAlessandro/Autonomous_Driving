"""Unit and integration tests for Ticket E52: Temporal Sequence Teacher Distillation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_e52_temporal_distillation import (
    run_e52_empirical_temporal_distillation_audit,
)
from tlr_yolo_mtl.training.temporal_distillation import (
    TemporalAttentionFusion,
    TemporalDistillationLoss,
    TemporalPositionalEncoding,
    TemporalSequenceSampler,
    TemporalSequenceTeacher,
    TemporalSequenceTriplet,
    TemporalTeacherTower,
)


def test_temporal_positional_encoding():
    """Verify temporal positional encoding table generation and slicing."""
    embed_dim = 128
    pos_enc = TemporalPositionalEncoding(embed_dim=embed_dim, max_window=7)

    for seq_len in [2, 3, 5]:
        pos = pos_enc(seq_len)
        assert pos.shape == (seq_len, embed_dim)
        assert not torch.isnan(pos).any()


def test_temporal_attention_fusion_forward_backward():
    """Verify temporal attention fusion on flat [M, C] and batched [B, N, C] tensors."""
    M, C, T = 16, 64, 3
    fusion = TemporalAttentionFusion(channels=C, num_heads=4)

    # 1. Flat [M, C] target and [M, T, C] sequence
    target_flat = torch.randn(M, C, requires_grad=True)
    seq_flat = torch.randn(M, T, C, requires_grad=True)

    out_flat = fusion(target_flat, seq_flat)
    assert out_flat.shape == (M, C)

    loss_flat = out_flat.sum()
    loss_flat.backward()
    assert target_flat.grad is not None
    assert seq_flat.grad is not None
    assert not torch.isnan(target_flat.grad).any()
    assert not torch.isnan(seq_flat.grad).any()

    # 2. Batched [B, N, C] target and [B, T, N, C] sequence
    B, N = 2, 8
    target_batched = torch.randn(B, N, C, requires_grad=True)
    seq_batched = torch.randn(B, T, N, C, requires_grad=True)

    out_batched = fusion(target_batched, seq_batched)
    assert out_batched.shape == (B, N, C)

    loss_batched = out_batched.sum()
    loss_batched.backward()
    assert target_batched.grad is not None
    assert seq_batched.grad is not None


def test_temporal_teacher_tower():
    """Verify TemporalTeacherTower feature extraction and state logits generation."""
    M, in_dim, embed_dim, T = 12, 96, 128, 3
    teacher = TemporalTeacherTower(
        in_dim=in_dim,
        embed_dim=embed_dim,
        num_states=4,
        num_heads=4,
        window_size=T,
    )

    target_feat = torch.randn(M, in_dim)
    seq_feat = torch.randn(M, T, in_dim)

    feat_out, state_logits = teacher(target_feat, seq_feat)
    assert feat_out.shape == (M, embed_dim)
    assert state_logits.shape == (M, 4)
    assert not torch.isnan(feat_out).any()
    assert not torch.isnan(state_logits).any()

    # Empty inputs handling
    empty_target = torch.empty((0, in_dim))
    empty_seq = torch.empty((0, T, in_dim))
    empty_f, empty_l = teacher(empty_target, empty_seq)
    assert empty_f.shape == (0, embed_dim)
    assert empty_l.shape == (0, 4)


def test_temporal_distillation_loss_forward_backward():
    """Verify composite temporal distillation loss computation and stop-gradient protections."""
    M, D = 10, 64
    loss_module = TemporalDistillationLoss(
        feature_weight=0.5,
        state_weight=0.5,
        flicker_weight=0.25,
        temperature=3.0,
        feature_loss_type="mse",
        teacher_supervised=True,
    )

    s_feat = torch.randn(M, D, requires_grad=True)
    t_feat = torch.randn(M, D, requires_grad=True)
    s_logits = torch.randn(M, 4, requires_grad=True)
    t_logits = torch.randn(M, 4, requires_grad=True)
    prev_s_logits = torch.randn(M, 4)
    gt_states = torch.randint(0, 4, (M,))
    same_track = torch.ones(M, dtype=torch.bool)

    loss, metrics = loss_module(
        student_features=s_feat,
        teacher_features=t_feat,
        student_state_logits=s_logits,
        teacher_state_logits=t_logits,
        prev_student_logits=prev_s_logits,
        gt_states=gt_states,
        same_track_mask=same_track,
    )

    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert "temporal_kd_loss" in metrics
    assert "temporal_feat_loss" in metrics
    assert "temporal_state_loss" in metrics
    assert "temporal_flicker_loss" in metrics
    assert "teacher_ce_loss" in metrics

    loss.backward()

    # Verify student receives clean gradients
    assert s_feat.grad is not None
    assert s_logits.grad is not None
    assert not torch.isnan(s_feat.grad).any()
    assert not torch.isnan(s_logits.grad).any()

    # Verify teacher features are protected by stop-gradient
    assert t_feat.grad is None, "Teacher features must not receive gradients via student distillation path"


def test_temporal_distillation_loss_cosine():
    """Verify cosine similarity feature loss variant."""
    M, D = 8, 32
    loss_module = TemporalDistillationLoss(
        feature_weight=0.5,
        state_weight=0.5,
        flicker_weight=0.0,
        feature_loss_type="cosine",
    )

    s_feat = torch.randn(M, D, requires_grad=True)
    t_feat = torch.randn(M, D)
    s_logits = torch.randn(M, 4, requires_grad=True)
    t_logits = torch.randn(M, 4)

    loss, metrics = loss_module(
        student_features=s_feat,
        teacher_features=t_feat,
        student_state_logits=s_logits,
        teacher_state_logits=t_logits,
    )

    assert loss.item() >= 0.0
    loss.backward()
    assert s_feat.grad is not None


def test_temporal_sequence_sampler():
    """Verify sequence sampler windowing and boundary clamping."""
    sampler = TemporalSequenceSampler(window_size=3, delta_frames=1)

    # Middle frame in sequence of length 10
    idx_mid = sampler.sample_triplet_indices(sequence_length=10, target_idx=5)
    assert idx_mid == [4, 5, 6]

    # First frame (t=0) with boundary clamping
    idx_start = sampler.sample_triplet_indices(sequence_length=10, target_idx=0)
    assert idx_start == [0, 0, 1]

    # Last frame (t=9) with boundary clamping
    idx_end = sampler.sample_triplet_indices(sequence_length=10, target_idx=9)
    assert idx_end == [8, 9, 9]

    # 5-frame window
    sampler5 = TemporalSequenceSampler(window_size=5, delta_frames=1)
    idx5_mid = sampler5.sample_triplet_indices(sequence_length=10, target_idx=5)
    assert idx5_mid == [3, 4, 5, 6, 7]


def test_e52_empirical_audit_execution(tmp_path):
    """Executes the full E52 audit and verifies all 4 confirmation criteria."""
    results = run_e52_empirical_temporal_distillation_audit(
        output_dir=tmp_path / "e52_audit",
        device="cpu",
    )

    assert results["ticket"] == "E52"
    assert results["status"] == "closed"

    crit = results["criteria_verification"]
    assert crit["criterion_1_sub8px_ap_gain"]["passed"] is True, "Criterion 1 failed"
    assert crit["criterion_2_flicker_reduction"]["passed"] is True, "Criterion 2 failed"
    assert crit["criterion_3_zero_latency"]["passed"] is True, "Criterion 3 failed"
    assert crit["criterion_4_zero_macro_degradation"]["passed"] is True, "Criterion 4 failed"
