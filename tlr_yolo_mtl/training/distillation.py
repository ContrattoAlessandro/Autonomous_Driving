"""Local-View Tiny-TL High-Resolution Crop Distillation Module (Ticket E48).

Implements training-time knowledge distillation from a high-resolution Local-View Teacher
(operating on zoomed 64x64 px crops of sub-16px traffic lights) into the Student's
full-frame P2/P3 ROIAlign representation.

Core Innovations:
1. Dynamic Batch Crop Extractor: Bilinear extraction of sub-16px traffic lights with
   margin padding and standardizing to 64x64 px patches.
2. Lightweight Local-View Teacher Tower: High-frequency convolutional encoder (64x64 -> 128D)
   and 4-class state logit predictor.
3. Student-to-Teacher Distillation Projector: Aligns student ROIAlign feature space
   with the teacher's high-resolution visual embedding manifold.
4. Composite Distillation Loss: Combines L2/Cosine feature alignment with temperature-scaled
   soft-label KL divergence.
5. Zero Runtime Overhead: Active strictly during training backpropagation (delta_t = 0.00 ms).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalViewCrops(NamedTuple):
    """Container for dynamically extracted high-resolution crops."""
    crops: torch.Tensor             # [M, 3, crop_h, crop_w]
    box_indices: torch.Tensor       # [M] index into original positive box list
    batch_indices: torch.Tensor     # [M] batch indices [0, B-1]
    gt_boxes: torch.Tensor          # [M, 4] original bounding boxes (pixels)
    gt_states: torch.Tensor         # [M] ground-truth state labels


class LocalViewCropExtractor(nn.Module):
    """Dynamic batch cropper extracting high-resolution patches around tiny traffic lights.
    
    For all ground-truth traffic lights with area < max_area_threshold (default: 256 px^2,
    i.e. <16x16 px), extracts bounding box crops with contextual margin and bilinearly
    resamples them to standard (crop_h, crop_w) resolution.
    """

    def __init__(
        self,
        crop_size: tuple[int, int] = (64, 64),
        margin: float = 0.15,
        max_area_threshold: float = 256.0,
        min_dim: float = 1.0,
    ) -> None:
        super().__init__()
        self.crop_h, self.crop_w = crop_size
        self.margin = float(margin)
        self.max_area_threshold = float(max_area_threshold)
        self.min_dim = float(min_dim)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        batch_idx: torch.Tensor,
        states: torch.Tensor | None = None,
        is_normalized: bool = False,
    ) -> LocalViewCrops:
        """Extract and resize crops for eligible tiny traffic lights.

        Args:
            images: Batch images [B, 3, H, W] in [0, 1] or [0, 255].
            boxes: Target bounding boxes [N, 4] (x1, y1, x2, y2).
            batch_idx: Batch assignment indices [N] in [0, B - 1].
            states: Optional ground truth state labels [N].
            is_normalized: Whether box coordinates are in [0, 1] relative to (H, W).

        Returns:
            LocalViewCrops container.
        """
        B, C, H, W = images.shape
        device = images.device

        if boxes.numel() == 0 or batch_idx.numel() == 0:
            return LocalViewCrops(
                crops=torch.empty((0, C, self.crop_h, self.crop_w), device=device, dtype=images.dtype),
                box_indices=torch.empty((0,), device=device, dtype=torch.long),
                batch_indices=torch.empty((0,), device=device, dtype=torch.long),
                gt_boxes=torch.empty((0, 4), device=device, dtype=images.dtype),
                gt_states=torch.empty((0,), device=device, dtype=torch.long),
            )

        boxes_px = boxes.clone()
        if is_normalized:
            scale_vec = torch.tensor([W, H, W, H], device=device, dtype=boxes.dtype)
            boxes_px = boxes_px * scale_vec

        # Calculate bounding box areas in pixels
        w = (boxes_px[:, 2] - boxes_px[:, 0]).clamp_min(self.min_dim)
        h = (boxes_px[:, 3] - boxes_px[:, 1]).clamp_min(self.min_dim)
        areas = w * h

        # Filter boxes that are strictly tiny (area < max_area_threshold)
        valid_mask = (areas > 0) & (areas <= self.max_area_threshold)
        valid_indices = torch.where(valid_mask)[0]

        if valid_indices.numel() == 0:
            return LocalViewCrops(
                crops=torch.empty((0, C, self.crop_h, self.crop_w), device=device, dtype=images.dtype),
                box_indices=torch.empty((0,), device=device, dtype=torch.long),
                batch_indices=torch.empty((0,), device=device, dtype=torch.long),
                gt_boxes=torch.empty((0, 4), device=device, dtype=images.dtype),
                gt_states=torch.empty((0,), device=device, dtype=torch.long),
            )

        sel_boxes = boxes_px[valid_indices]
        sel_b_idx = batch_idx[valid_indices].long()
        sel_states = states[valid_indices].long() if states is not None else torch.zeros_like(valid_indices)

        # Apply contextual margin around physical box
        cx = (sel_boxes[:, 0] + sel_boxes[:, 2]) * 0.5
        cy = (sel_boxes[:, 1] + sel_boxes[:, 3]) * 0.5
        sel_w = (sel_boxes[:, 2] - sel_boxes[:, 0]).clamp_min(self.min_dim)
        sel_h = (sel_boxes[:, 3] - sel_boxes[:, 1]).clamp_min(self.min_dim)

        exp_w = sel_w * (1.0 + 2.0 * self.margin)
        exp_h = sel_h * (1.0 + 2.0 * self.margin)

        exp_x1 = (cx - exp_w * 0.5).clamp(min=0.0, max=float(W - 1))
        exp_y1 = (cy - exp_h * 0.5).clamp(min=0.0, max=float(H - 1))
        exp_x2 = (cx + exp_w * 0.5).clamp(min=0.0, max=float(W - 1))
        exp_y2 = (cy + exp_h * 0.5).clamp(min=0.0, max=float(H - 1))

        # Dynamic patch extraction via grid_sample
        M = valid_indices.shape[0]
        crops = torch.empty((M, C, self.crop_h, self.crop_w), device=device, dtype=images.dtype)

        # Construct normalized sampling grids
        norm_x1 = (exp_x1 / max(W - 1, 1)) * 2.0 - 1.0
        norm_y1 = (exp_y1 / max(H - 1, 1)) * 2.0 - 1.0
        norm_x2 = (exp_x2 / max(W - 1, 1)) * 2.0 - 1.0
        norm_y2 = (exp_y2 / max(H - 1, 1)) * 2.0 - 1.0

        for i in range(M):
            b_i = int(sel_b_idx[i].item())
            gy = torch.linspace(norm_y1[i], norm_y2[i], self.crop_h, device=device, dtype=images.dtype)
            gx = torch.linspace(norm_x1[i], norm_x2[i], self.crop_w, device=device, dtype=images.dtype)
            mesh_y, mesh_x = torch.meshgrid(gy, gx, indexing="ij")
            grid = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0)  # [1, crop_h, crop_w, 2]

            sample = F.grid_sample(
                images[b_i : b_i + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            crops[i] = sample.squeeze(0)

        return LocalViewCrops(
            crops=crops,
            box_indices=valid_indices,
            batch_indices=sel_b_idx,
            gt_boxes=sel_boxes,
            gt_states=sel_states,
        )


class TeacherResidualBlock(nn.Module):
    """Residual convolutional block with BatchNorm and SiLU activation."""

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
    """Lightweight High-Resolution Local-View Teacher Network.
    
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
            embed_dim = self.feature_head[-1].out_features
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
