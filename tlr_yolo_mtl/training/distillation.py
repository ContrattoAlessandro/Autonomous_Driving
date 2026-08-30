"""Local-View Tiny-TL High-Resolution Crop Distillation (Ticket E48)
& Tiny-State Multi-Teacher Relation Distillation (Ticket E72).

Scientific Motivation:
1. Ticket E48: Standard full-frame representations of sub-8px objects suffer from
   severe pixel-striding and spatial pooling loss. LocalViewTeacher operates directly
   on 64x64 px patches to provide sharp feature and chromatic supervision.
2. Ticket E72: Multi-Model Triangulation in Ticket E59 demonstrated that 64.35% of
   sub-4px state errors originate from Knowledge Transfer / Distillation Capacity Failure
   (where both Local-View and Temporal Teachers are correct). Ticket E72 resolves this
   via consensus-weighted multi-teacher distillation and relational Gram matrix alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LocalViewCropExtractor:
    """Extracts and resizes high-resolution crops around small traffic light annotations."""

    def __init__(
        self,
        crop_size: int | tuple[int, int] | list[int] = 64,
        padding_ratio: float = 0.5,
        margin: float | None = None,
        min_crop_side: int = 16,
        max_crops_per_image: int = 32,
        max_area_threshold: float = 256.0,
        **kwargs: Any,
    ) -> None:
        if isinstance(crop_size, (tuple, list)):
            self.crop_size = int(crop_size[0])
        else:
            self.crop_size = int(crop_size)
        self.padding_ratio = float(margin) if margin is not None else float(padding_ratio)
        self.min_crop_side = int(min_crop_side)
        self.max_crops_per_image = int(max_crops_per_image)
        self.max_area_threshold = float(max_area_threshold)

    def extract_crops(
        self,
        images: torch.Tensor,       # [B, 3, H, W]
        boxes_xyxy: torch.Tensor,    # [N, 5] (batch_idx, x1, y1, x2, y2) in pixel coords
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts squared, padded patches around target bounding boxes.

        Returns:
            crops: [M, 3, crop_size, crop_size] normalized patches
            valid_indices: [M] index of extracted boxes within input boxes_xyxy
        """
        if boxes_xyxy.numel() == 0 or images.numel() == 0:
            return torch.empty((0, 3, self.crop_size, self.crop_size), device=images.device, dtype=images.dtype), torch.empty((0,), dtype=torch.long, device=images.device)

        B, C, H, W = images.shape
        crops_list = []
        valid_idx_list = []

        for idx in range(min(boxes_xyxy.shape[0], self.max_crops_per_image * B)):
            b_idx = int(boxes_xyxy[idx, 0].item())
            if b_idx < 0 or b_idx >= B:
                continue

            x1, y1, x2, y2 = boxes_xyxy[idx, 1:].tolist()
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)

            # Center and expand with padding
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            side = max(w, h) * (1.0 + self.padding_ratio * 2.0)
            side = max(side, float(self.min_crop_side))

            crop_x1 = max(0, int(cx - side * 0.5))
            crop_y1 = max(0, int(cy - side * 0.5))
            crop_x2 = min(W, int(cx + side * 0.5))
            crop_y2 = min(H, int(cy + side * 0.5))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            patch = images[b_idx : b_idx + 1, :, crop_y1:crop_y2, crop_x1:crop_x2]
            patch_resized = F.interpolate(
                patch, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False
            )
            crops_list.append(patch_resized.squeeze(0))
            valid_idx_list.append(idx)

        if not crops_list:
            return torch.empty((0, 3, self.crop_size, self.crop_size), device=images.device, dtype=images.dtype), torch.empty((0,), dtype=torch.long, device=images.device)

        crops_tensor = torch.stack(crops_list, dim=0)
        valid_indices = torch.tensor(valid_idx_list, dtype=torch.long, device=images.device)
        return crops_tensor, valid_indices


class TeacherResidualBlock(nn.Module):
    """Standard 2-conv residual bottleneck for Local-View Teacher."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class LocalViewTeacherTower(nn.Module):
    """High-Resolution Crop Teacher Network (Ticket E48).
    
    Processes 64x64 px traffic light crops to extract:
    1. High-resolution feature embedding f_TL^crop in R^embed_dim (default: 128).
    2. Teacher state logits z_T in R^4 (Red, Yellow, Green, Off).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        embed_dim: int = 128,
        num_states: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.stage1 = TeacherResidualBlock(base_channels, base_channels, stride=1)                   # 32x32
        self.stage2 = TeacherResidualBlock(base_channels, base_channels * 2, stride=2)              # 32x32 -> 16x16 (64 ch)
        self.stage3 = TeacherResidualBlock(base_channels * 2, base_channels * 4, stride=2)          # 16x16 -> 8x8 (128 ch)
        self.stage4 = TeacherResidualBlock(base_channels * 4, base_channels * 4, stride=2)          # 8x8 -> 4x4 (128 ch)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_head = nn.Sequential(
            nn.Linear(base_channels * 4, embed_dim),
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

    def forward(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on high-resolution crops.

        Args:
            crops: [M, 3, 64, 64] image patches.

        Returns:
            features: [M, embed_dim] L2-normalized or dense visual embeddings.
            state_logits: [M, num_states] unnormalized classification logits.
        """
        if crops.shape[0] == 0:
            embed_dim = self.feature_head[-1].normalized_shape[0] if hasattr(self.feature_head[-1], 'normalized_shape') else 128
            num_states = self.state_head[-1].out_features
            return (
                torch.empty((0, embed_dim), device=crops.device, dtype=crops.dtype),
                torch.empty((0, num_states), device=crops.device, dtype=crops.dtype),
            )

        x = self.stem(crops)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        pooled = self.pool(x).flatten(1)

        features = self.feature_head(pooled)
        state_logits = self.state_head(features)
        return features, state_logits


class StudentKDProjector(nn.Module):
    """Linear / MLP projection mapping Student candidate features to Distillation space."""

    def __init__(self, student_dim: int = 128, embed_dim: int = 128) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(student_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        """Project student ROIAlign or token embeddings to distillation space.
        
        Args:
            student_features: [M, student_dim] or [B, K, student_dim]
        Returns:
            projected: same shape with last dim = embed_dim
        """
        if student_features.numel() == 0:
            out_dim = self.projector[-1].normalized_shape[0]
            return torch.empty((*student_features.shape[:-1], out_dim), device=student_features.device, dtype=student_features.dtype)
        return self.projector(student_features)


class LocalViewDistillationLoss(nn.Module):
    """Composite Distillation Loss for tiny traffic lights (Ticket E48).
    
    L_KD = lambda_f * L_feature(f_S_hat, stop_gradient(f_T)) + lambda_z * T^2 * KL(p_T^T || p_S^T)
    """

    def __init__(
        self,
        feature_weight: float = 0.5,
        state_weight: float = 0.5,
        temperature: float = 3.0,
        feature_loss_type: str = "mse",
        teacher_supervised: bool = True,
    ) -> None:
        super().__init__()
        self.feature_weight = float(feature_weight)
        self.state_weight = float(state_weight)
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
        gt_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute composite knowledge distillation loss.

        Args:
            student_features: [M, D] projected student feature embeddings.
            teacher_features: [M, D] teacher crop feature embeddings.
            student_state_logits: [M, 4] student state logits.
            teacher_state_logits: [M, 4] teacher state logits.
            gt_states: [M] optional ground truth state labels for teacher auxiliary loss.

        Returns:
            loss: scalar distillation loss.
            metrics: dict of individual loss values and diagnostics.
        """
        M = student_features.shape[0]
        device = student_features.device
        dtype = student_features.dtype

        if M == 0:
            zero_loss = student_features.sum() * 0.0
            return zero_loss, {
                "kd_loss": zero_loss.detach(),
                "kd_feature_loss": zero_loss.detach(),
                "kd_state_loss": zero_loss.detach(),
                "teacher_ce_loss": zero_loss.detach(),
                "kd_valid_crops": torch.tensor(0, device=device),
            }

        # Stop gradient on teacher feature representations to protect teacher
        teacher_features_det = teacher_features.detach()
        teacher_logits_det = teacher_state_logits.detach()

        # 1. Feature Representation Alignment
        if self.feature_loss_type == "cosine":
            norm_s = F.normalize(student_features, p=2, dim=-1)
            norm_t = F.normalize(teacher_features_det, p=2, dim=-1)
            feat_loss = (1.0 - (norm_s * norm_t).sum(dim=-1)).mean()
        else:  # "mse" L2 distance
            feat_loss = F.mse_loss(student_features, teacher_features_det)

        # 2. Temperature-Scaled Soft State Distillation (KL Divergence)
        T = self.temperature
        p_teacher = F.softmax(teacher_logits_det / T, dim=-1)
        log_p_student = F.log_softmax(student_state_logits / T, dim=-1)
        # KL(p_T || p_S) = sum p_T * (log p_T - log p_S)
        kl_div = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        state_kd_loss = (T ** 2) * kl_div

        # 3. Optional Teacher Supervised Cross-Entropy (joint teacher training)
        if self.teacher_supervised and gt_states is not None and gt_states.numel() == M:
            teacher_ce = self.ce_criterion(teacher_state_logits, gt_states)
        else:
            teacher_ce = student_features.sum() * 0.0

        # Total Distillation Loss
        total_loss = (
            self.feature_weight * feat_loss
            + self.state_weight * state_kd_loss
            + 0.5 * teacher_ce
        )

        metrics = {
            "kd_loss": total_loss.detach(),
            "kd_feature_loss": feat_loss.detach(),
            "kd_state_loss": state_kd_loss.detach(),
            "teacher_ce_loss": teacher_ce.detach(),
            "kd_valid_crops": torch.tensor(M, device=device),
        }

        return total_loss, metrics


# =============================================================================
# Ticket E72: Multi-Teacher Relation Distillation
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiTeacherDistillationConfig:
    """Configuration for Multi-Teacher Relation Distillation (Ticket E72)."""
    temperature: float = 3.0
    weight_kd: float = 1.0
    weight_relation: float = 0.5
    weight_feature: float = 0.25
    local_teacher_weight: float = 0.5
    temporal_teacher_weight: float = 0.5
    sub4px_scale_boost: float = 2.0
    area_sub4px_threshold: float = 16.0  # px^2 (side <= 4px)
    eps: float = 1e-7


class MultiTeacherRelationDistillationLoss(nn.Module):
    """Multi-Teacher Relational and Logit Distillation Loss Module (Ticket E72).
    
    Distills rich chromatic, visual, and temporal representations from:
    - Teacher 1: Local-View High-Resolution Crop Teacher (Ticket E48)
    - Teacher 2: Temporal Sequence Teacher (Ticket E52)
    into the Single-Frame Student.
    """

    def __init__(
        self,
        config: MultiTeacherDistillationConfig | None = None,
        *,
        student_dim: int = 64,
        teacher_dim: int = 64,
        temperature: float = 3.0,
        weight_kd: float = 1.0,
        weight_relation: float = 0.5,
        weight_feature: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = config or MultiTeacherDistillationConfig(
            temperature=temperature,
            weight_kd=weight_kd,
            weight_relation=weight_relation,
            weight_feature=weight_feature,
        )

        # Feature adaptation projection if student and teacher feature dims differ
        if student_dim != teacher_dim:
            self.feat_proj = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.feat_proj = nn.Identity()

    def _compute_consensus_weights(
        self,
        local_logits: torch.Tensor,
        temp_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Computes pairwise teacher consensus agreement weight in [0, 1]."""
        p_local = F.softmax(local_logits, dim=-1)
        p_temp = F.softmax(temp_logits, dim=-1)
        
        # Cosine similarity between teacher probability vectors
        cos_sim = F.cosine_similarity(p_local, p_temp, dim=-1).clamp(-1.0, 1.0)
        consensus = 0.5 * (1.0 + cos_sim)  # Map [-1, 1] to [0, 1]
        return consensus

    def _compute_relational_loss(
        self,
        student_feats: torch.Tensor,  # [N, D]
        teacher_feats: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        """Computes Relational Similarity Distillation (RSD) loss between Gram matrices."""
        N = student_feats.shape[0]
        if N <= 1:
            return student_feats.new_zeros(1).sum()

        s_norm = F.normalize(student_feats, p=2, dim=-1)
        t_norm = F.normalize(teacher_feats, p=2, dim=-1)

        # Pairwise candidate relational Gram matrices: [N, N]
        gram_s = torch.mm(s_norm, s_norm.t())
        gram_t = torch.mm(t_norm, t_norm.t())

        # Frobenius norm difference
        rel_loss = F.mse_loss(gram_s, gram_t, reduction="mean")
        return rel_loss

    def forward(
        self,
        student_logits: torch.Tensor,       # [N, num_classes] (e.g. 4 state classes)
        student_features: torch.Tensor,     # [N, D_student]
        local_teacher_logits: torch.Tensor, # [N, num_classes]
        local_teacher_features: torch.Tensor, # [N, D_teacher]
        temporal_teacher_logits: torch.Tensor, # [N, num_classes]
        temporal_teacher_features: torch.Tensor, # [N, D_teacher]
        candidate_areas: torch.Tensor | None = None,  # [N] in px^2
    ) -> dict[str, torch.Tensor]:
        """Calculates multi-teacher distillation loss components.
        
        Args:
            student_logits: Predicted state logits from student.
            student_features: Intermediate feature embeddings from student.
            local_teacher_logits: Local high-res crop teacher state logits.
            local_teacher_features: Local crop teacher feature embeddings.
            temporal_teacher_logits: Multi-frame temporal sequence teacher state logits.
            temporal_teacher_features: Temporal teacher feature embeddings.
            candidate_areas: Optional candidate bounding box areas for scale boosting.
            
        Returns:
            Dictionary containing total distillation loss and constituent terms.
        """
        N = student_logits.shape[0]
        if N == 0:
            zero = student_logits.sum() * 0.0
            return {
                "loss_distill_total": zero,
                "loss_distill_kd": zero,
                "loss_distill_relation": zero,
                "loss_distill_feature": zero,
            }

        T = self.config.temperature

        # 1. Consensus-Weighted Soft Targets
        consensus_weights = self._compute_consensus_weights(
            local_teacher_logits, temporal_teacher_logits
        )  # [N]

        # Combine teacher logits with configured weights
        fused_teacher_logits = (
            self.config.local_teacher_weight * local_teacher_logits
            + self.config.temporal_teacher_weight * temporal_teacher_logits
        )

        p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(fused_teacher_logits / T, dim=-1)

        # Scale-adaptive weighting boost for sub-4px signals
        if candidate_areas is not None:
            is_sub4px = candidate_areas <= self.config.area_sub4px_threshold
            scale_multiplier = torch.where(
                is_sub4px,
                torch.full_like(candidate_areas, self.config.sub4px_scale_boost),
                torch.ones_like(candidate_areas),
            )
        else:
            scale_multiplier = torch.ones(N, device=student_logits.device)

        # KL Divergence per sample
        kl_per_sample = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)  # [N]
        l_kd = (consensus_weights * scale_multiplier * kl_per_sample).mean() * (T ** 2)

        # 2. Relational Similarity Matrix Distillation
        proj_student_feats = self.feat_proj(student_features)
        fused_teacher_feats = (
            self.config.local_teacher_weight * local_teacher_features
            + self.config.temporal_teacher_weight * temporal_teacher_features
        )

        l_rel = self._compute_relational_loss(proj_student_feats, fused_teacher_feats)

        # 3. Direct Feature Cosine / MSE Alignment
        l_feat = F.mse_loss(proj_student_feats, fused_teacher_feats, reduction="mean")

        # Total Distillation Loss
        total_loss = (
            self.config.weight_kd * l_kd
            + self.config.weight_relation * l_rel
            + self.config.weight_feature * l_feat
        )

        return {
            "loss_distill_total": total_loss,
            "loss_distill_kd": l_kd,
            "loss_distill_relation": l_rel,
            "loss_distill_feature": l_feat,
            "mean_teacher_consensus": consensus_weights.mean(),
        }


# Convenience re-exports for Ticket E52 Temporal Distillation
from .temporal_distillation import (
    TemporalAttentionFusion,
    TemporalDistillationLoss,
    TemporalPositionalEncoding,
    TemporalSequenceSampler,
    TemporalSequenceTeacher,
    TemporalSequenceTriplet,
    TemporalTeacherTower,
)
