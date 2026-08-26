"""Temporal Sequence Teacher Distillation Module (Ticket E52).

Implements training-time knowledge distillation from a multi-frame Temporal Sequence Teacher
(operating on sequential drive context [I_{t-1}, I_t, I_{t+1}] available during training in DTLD)
into the single-frame Student model (I_t -> Student).

Core Innovations:
1. Temporal Sequence Triplet Container & Sampler:
   Structures contiguous driving sequences into triplets (I_{t-1}, I_t, I_{t+1}) with sequence
   boundary reflection/replication padding.
2. Multi-Frame Temporal Sequence Teacher (TemporalAttentionFusion & TemporalSequenceTeacher):
   Aggregates multi-frame temporal context using cross-frame attention with relative temporal
   positional embeddings to produce temporally stabilized feature embeddings and state distributions.
3. Composite Temporal Distillation Loss (TemporalDistillationLoss):
   Combines feature alignment, temperature-scaled soft state probability distillation (KL divergence),
   and inter-frame state transition flicker regularization.
4. Zero-Runtime Invariant:
   Temporal teacher and sequence alignment operate strictly during training backpropagation.
   Deployment inference remains strictly single-frame (Delta t = 0.00 ms, >=36.5 FPS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSequenceTriplet(NamedTuple):
    """Container for temporal sequence triplets and tracking metadata."""
    frames: torch.Tensor                     # [B, T, 3, H, W] or sequence of image frames
    frame_indices: torch.Tensor              # [B, T] frame sequence indices
    target_idx: int                          # index of key frame t (default: 1 in 3-frame window)
    boxes: torch.Tensor                      # [N, 4] bounding boxes for target frame
    batch_indices: torch.Tensor              # [N] batch assignments [0, B - 1]
    states: Optional[torch.Tensor] = None    # [N] ground-truth state labels for target frame
    track_ids: Optional[torch.Tensor] = None # [N] persistent instance track IDs


class TemporalPositionalEncoding(nn.Module):
    """Relative 1D Temporal Positional Encoding for temporal frame windows."""

    def __init__(self, embed_dim: int, max_window: int = 5) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_window = int(max_window)
        self.pos_table = nn.Parameter(torch.zeros(max_window, embed_dim))
        nn.init.trunc_normal_(self.pos_table, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings for sequence length seq_len [seq_len, embed_dim]."""
        start = (self.max_window - seq_len) // 2
        return self.pos_table[start : start + seq_len]


class TemporalAttentionFusion(nn.Module):
    """Multi-Head Temporal Cross-Attention for aggregating sequential frame context.
    
    Given multi-frame feature maps or tokens across time window T:
    - Queries Q come from the target frame feature map F_t.
    - Keys K and Values V come from all frames in the temporal sequence [F_{t-1}, F_t, F_{t+1}].
    """

    def __init__(
        self,
        channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_spatial_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = channels // num_heads
        assert (
            self.head_dim * num_heads == channels
        ), f"channels {channels} must be divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        self.temporal_pos = TemporalPositionalEncoding(channels, max_window=7)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(
        self,
        target_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward temporal cross-attention.

        Args:
            target_features: Target frame features [B, N, C] or [M, C].
            sequence_features: Temporal sequence features [B, T, N, C] or [M, T, C].

        Returns:
            fused_features: Temporally stabilized features with same shape as target_features.
        """
        is_flat = target_features.ndim == 2
        if is_flat:
            # [M, C] -> [M, 1, C], [M, T, C] -> [M, T, 1, C] -> [M, T, C]
            M, C = target_features.shape
            T = sequence_features.shape[1]
            q_in = target_features.unsqueeze(1)  # [M, 1, C]
            kv_in = sequence_features           # [M, T, C]
        elif target_features.ndim == 3:
            # [B, N, C] and [B, T, N, C] -> reshape to [B*N, 1, C] and [B*N, T, C]
            B, N, C = target_features.shape
            T = sequence_features.shape[1]
            q_in = target_features.reshape(B * N, 1, C)
            # sequence: [B, T, N, C] -> permute to [B, N, T, C] -> reshape [B*N, T, C]
            kv_in = sequence_features.permute(0, 2, 1, 3).reshape(B * N, T, C)
        else:
            raise ValueError(f"Unsupported target_features shape: {target_features.shape}")

        BN, _, _ = q_in.shape
        T = kv_in.shape[1]

        # Add relative temporal position encoding to sequence
        pos = self.temporal_pos(T).unsqueeze(0)  # [1, T, C]
        kv_with_pos = kv_in + pos

        # Project Q, K, V
        Q = self.q_proj(q_in).reshape(BN, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [BN, H, 1, D]
        K = self.k_proj(kv_with_pos).reshape(BN, T, self.num_heads, self.head_dim).transpose(1, 2)  # [BN, H, T, D]
        V = self.v_proj(kv_in).reshape(BN, T, self.num_heads, self.head_dim).transpose(1, 2)  # [BN, H, T, D]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [BN, H, 1, T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)  # [BN, H, 1, D]
        attn_out = attn_out.transpose(1, 2).reshape(BN, 1, self.channels)  # [BN, 1, C]
        attn_out = self.out_proj(attn_out)

        # Residual & FFN
        x = self.norm1(q_in + self.dropout(attn_out))
        out = self.norm2(x + self.ffn(x))

        if is_flat:
            return out.squeeze(1)  # [M, C]
        else:
            return out.reshape(B, N, C)


class TemporalTeacherTower(nn.Module):
    """Lightweight 3-Frame Temporal Sequence Teacher Tower.
    
    Processes a temporal sequence of feature representations or crops to yield:
    1. Temporally stabilized feature embedding f_t^teacher in R^embed_dim.
    2. Temporally consistent teacher state logits z_t^teacher in R^4.
    """

    def __init__(
        self,
        in_dim: int = 128,
        embed_dim: int = 128,
        num_states: int = 4,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.num_states = int(num_states)
        self.window_size = int(window_size)

        if in_dim != embed_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.input_proj = nn.Identity()

        self.temporal_fusion = TemporalAttentionFusion(
            channels=embed_dim,
            num_heads=num_heads,
        )

        self.feature_refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.state_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim // 2, num_states),
        )

    def forward(
        self,
        target_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for temporal teacher.

        Args:
            target_features: Target frame features [M, in_dim] or [B, N, in_dim].
            sequence_features: Multi-frame sequence features [M, T, in_dim] or [B, T, N, in_dim].

        Returns:
            teacher_features: Stabilized embedding [M, embed_dim] or [B, N, embed_dim].
            teacher_state_logits: Smoothed state logits [M, 4] or [B, N, 4].
        """
        if target_features.numel() == 0:
            dev = target_features.device
            dt = target_features.dtype
            prefix = target_features.shape[:-1]
            return (
                torch.empty((*prefix, self.embed_dim), device=dev, dtype=dt),
                torch.empty((*prefix, self.num_states), device=dev, dtype=dt),
            )

        t_proj = self.input_proj(target_features)
        seq_proj = self.input_proj(sequence_features)

        fused = self.temporal_fusion(t_proj, seq_proj)
        refined_feat = self.feature_refine(fused)
        state_logits = self.state_head(refined_feat)
        return refined_feat, state_logits


# Semantic alias
TemporalSequenceTeacher = TemporalTeacherTower


class TemporalDistillationLoss(nn.Module):
    """Composite Distillation Loss for Temporal Consistency (Ticket E52).
    
    L_temporal = lambda_feat * L_feat(f_S, sg(f_T))
               + lambda_state * T^2 * KL(p_T || p_S)
               + lambda_flicker * L_flicker(p_S, p_S_prev)
               + 0.5 * L_teacher_ce
    """

    def __init__(
        self,
        feature_weight: float = 0.5,
        state_weight: float = 0.5,
        flicker_weight: float = 0.25,
        temperature: float = 3.0,
        feature_loss_type: str = "mse",
        teacher_supervised: bool = True,
    ) -> None:
        super().__init__()
        self.feature_weight = float(feature_weight)
        self.state_weight = float(state_weight)
        self.flicker_weight = float(flicker_weight)
        self.temperature = float(temperature)
        self.feature_loss_type = str(feature_loss_type).lower()
        self.teacher_supervised = bool(teacher_supervised)
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_state_logits: torch.Tensor,
        teacher_state_logits: torch.Tensor,
        prev_student_logits: Optional[torch.Tensor] = None,
        gt_states: Optional[torch.Tensor] = None,
        same_track_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute composite temporal sequence distillation loss.

        Args:
            student_features: [M, D] projected student feature embeddings.
            teacher_features: [M, D] temporal teacher feature embeddings.
            student_state_logits: [M, 4] student state logits.
            teacher_state_logits: [M, 4] temporal teacher state logits.
            prev_student_logits: Optional [M, 4] student logits on prior frame t-1 for same track.
            gt_states: Optional [M] ground-truth state labels.
            same_track_mask: Optional [M] boolean mask indicating valid track continuity.

        Returns:
            loss: scalar temporal distillation loss.
            metrics: dict of individual loss values and diagnostics.
        """
        M = student_features.shape[0]
        device = student_features.device

        if M == 0:
            zero_loss = student_features.sum() * 0.0
            return zero_loss, {
                "temporal_kd_loss": zero_loss.detach(),
                "temporal_feat_loss": zero_loss.detach(),
                "temporal_state_loss": zero_loss.detach(),
                "temporal_flicker_loss": zero_loss.detach(),
                "teacher_ce_loss": zero_loss.detach(),
                "temporal_valid_instances": torch.tensor(0, device=device),
            }

        # Protect teacher weights with stop-gradient
        teacher_features_det = teacher_features.detach()
        teacher_logits_det = teacher_state_logits.detach()

        # 1. Feature Representation Alignment
        if self.feature_loss_type == "cosine":
            norm_s = F.normalize(student_features, p=2, dim=-1)
            norm_t = F.normalize(teacher_features_det, p=2, dim=-1)
            feat_loss = (1.0 - (norm_s * norm_t).sum(dim=-1)).mean()
        else:  # "mse"
            feat_loss = F.mse_loss(student_features, teacher_features_det)

        # 2. Temperature-Scaled Soft State Probability Distillation (KL Divergence)
        T = self.temperature
        p_teacher = F.softmax(teacher_logits_det / T, dim=-1)
        log_p_student = F.log_softmax(student_state_logits / T, dim=-1)
        kl_div = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        state_kd_loss = (T ** 2) * kl_div

        # 3. Inter-Frame State Flicker Regularization
        if prev_student_logits is not None and prev_student_logits.shape == student_state_logits.shape:
            p_curr = F.softmax(student_state_logits, dim=-1)
            p_prev = F.softmax(prev_student_logits.detach(), dim=-1)
            flicker_diff = (p_curr - p_prev).pow(2).sum(dim=-1)
            if same_track_mask is not None and same_track_mask.numel() == M:
                mask = same_track_mask.float()
                denom = mask.sum().clamp_min(1.0)
                flicker_loss = (flicker_diff * mask).sum() / denom
            else:
                flicker_loss = flicker_diff.mean()
        else:
            flicker_loss = student_features.sum() * 0.0

        # 4. Auxiliary Teacher Supervised Cross-Entropy
        if self.teacher_supervised and gt_states is not None and gt_states.numel() == M:
            teacher_ce = self.ce_criterion(teacher_state_logits, gt_states)
        else:
            teacher_ce = student_features.sum() * 0.0

        # Total Distillation Loss
        total_loss = (
            self.feature_weight * feat_loss
            + self.state_weight * state_kd_loss
            + self.flicker_weight * flicker_loss
            + 0.5 * teacher_ce
        )

        metrics = {
            "temporal_kd_loss": total_loss.detach(),
            "temporal_feat_loss": feat_loss.detach(),
            "temporal_state_loss": state_kd_loss.detach(),
            "temporal_flicker_loss": flicker_loss.detach(),
            "teacher_ce_loss": teacher_ce.detach(),
            "temporal_valid_instances": torch.tensor(M, device=device),
        }

        return total_loss, metrics


class TemporalSequenceSampler:
    """Utility for building contiguous multi-frame triplets from dataset records."""

    def __init__(self, window_size: int = 3, delta_frames: int = 1) -> None:
        self.window_size = int(window_size)
        self.delta_frames = int(delta_frames)
        self.half_window = window_size // 2

    def sample_triplet_indices(
        self,
        sequence_length: int,
        target_idx: int,
    ) -> list[int]:
        """Generate window sequence frame indices with symmetric boundary padding.

        Args:
            sequence_length: Total number of frames in the drive track.
            target_idx: Target center frame index [0, sequence_length - 1].

        Returns:
            indices: List of frame indices of length window_size.
        """
        if sequence_length <= 0:
            return [0] * self.window_size

        indices = []
        for offset in range(-self.half_window, self.half_window + 1):
            idx = target_idx + offset * self.delta_frames
            # Clamp to valid sequence boundaries
            idx_clamped = max(0, min(sequence_length - 1, idx))
            indices.append(idx_clamped)
        return indices
