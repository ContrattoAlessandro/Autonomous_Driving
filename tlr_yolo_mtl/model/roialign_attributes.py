"""Candidate-Centered Multi-Scale ROIAlign (P2+P3) Attribute Head.

Replaces single-point anchor feature sampling with candidate-centered 3x3
Multi-Scale ROIAlign over P2 (stride 4) and P3 (stride 8) feature maps for
the top K_TL = 32 candidates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torchvision.ops import roi_align
except ImportError:  # Fallback for environments without compiled torchvision ops
    def roi_align(
        input: torch.Tensor,
        boxes: torch.Tensor | list[torch.Tensor],
        output_size: tuple[int, int] | int,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
        aligned: bool = True,
    ) -> torch.Tensor:
        # Native PyTorch grid_sample fallback
        if isinstance(output_size, int):
            out_h, out_w = output_size, output_size
        else:
            out_h, out_w = output_size

        if isinstance(boxes, list):
            flat_boxes = []
            for b_idx, b_tensor in enumerate(boxes):
                if b_tensor.numel() > 0:
                    idx_col = torch.full((b_tensor.shape[0], 1), b_idx, dtype=b_tensor.dtype, device=b_tensor.device)
                    flat_boxes.append(torch.cat([idx_col, b_tensor], dim=1))
            if flat_boxes:
                boxes_tensor = torch.cat(flat_boxes, dim=0)
            else:
                boxes_tensor = torch.empty((0, 5), device=input.device, dtype=input.dtype)
        else:
            boxes_tensor = boxes

        if boxes_tensor.shape[0] == 0:
            return torch.empty((0, input.shape[1], out_h, out_w), device=input.device, dtype=input.dtype)

        N = boxes_tensor.shape[0]
        C = input.shape[1]
        out = torch.zeros((N, C, out_h, out_w), device=input.device, dtype=input.dtype)

        in_h, in_w = input.shape[2], input.shape[3]
        for i in range(N):
            b_idx = int(boxes_tensor[i, 0].item())
            x1 = boxes_tensor[i, 1] * spatial_scale
            y1 = boxes_tensor[i, 2] * spatial_scale
            x2 = boxes_tensor[i, 3] * spatial_scale
            y2 = boxes_tensor[i, 4] * spatial_scale

            grid_y = torch.linspace(y1, y2, out_h, device=input.device)
            grid_x = torch.linspace(x1, x2, out_w, device=input.device)
            mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

            # Normalize to [-1, 1] for grid_sample
            norm_x = (mesh_x / max(in_w - 1, 1)) * 2.0 - 1.0
            norm_y = (mesh_y / max(in_h - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, out_h, out_w, 2]

            sample = F.grid_sample(input[b_idx : b_idx + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            out[i] = sample.squeeze(0)
        return out


class CandidateMultiScaleROIAlign(nn.Module):
    """Bilinear 3x3 ROIAlign feature extractor over multi-scale P2 and P3 maps."""

    def __init__(
        self,
        channels_p2: int = 64,
        channels_p3: int = 128,
        roi_size: tuple[int, int] = (3, 3),
        embed_dim: int = 128,
        stride_p2: float = 4.0,
        stride_p3: float = 8.0,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.spatial_scale_p2 = 1.0 / stride_p2
        self.spatial_scale_p3 = 1.0 / stride_p3

        in_dim = (channels_p2 + channels_p3) * roi_size[0] * roi_size[1]
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        p2_feat: torch.Tensor,
        p3_feat: torch.Tensor,
        candidate_boxes_xyxy: torch.Tensor,  # [B, K, 4] in pixel coordinates
        img_shape: tuple[int, int] | None = None,  # Optional [H, W] if boxes are normalized
    ) -> torch.Tensor:
        """Extracts and fuses 3x3 ROIAlign candidate tokens.

        Returns:
            fused_tokens: [B, K, embed_dim]
        """
        B, K, _ = candidate_boxes_xyxy.shape
        boxes = candidate_boxes_xyxy
        if img_shape is not None and boxes.numel() > 0 and boxes.max() <= 1.05:
            # Boxes are normalized in [0, 1], scale to pixel coordinates
            h, w = img_shape
            scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes * scale

        # Prepare list of boxes for torchvision roi_align
        boxes_list = [boxes[b] for b in range(B)]

        # Extract P2 ROIAlign features: [B*K, C_p2, 3, 3]
        roi_p2 = roi_align(
            p2_feat,
            boxes_list,
            output_size=self.roi_size,
            spatial_scale=self.spatial_scale_p2,
            aligned=True,
        )
        # Extract P3 ROIAlign features: [B*K, C_p3, 3, 3]
        roi_p3 = roi_align(
            p3_feat,
            boxes_list,
            output_size=self.roi_size,
            spatial_scale=self.spatial_scale_p3,
            aligned=True,
        )

        flat_p2 = roi_p2.flatten(1)  # [B*K, C_p2 * 9]
        flat_p3 = roi_p3.flatten(1)  # [B*K, C_p3 * 9]
        concatenated = torch.cat([flat_p2, flat_p3], dim=1)  # [B*K, (C_p2 + C_p3) * 9]

        fused = self.fusion(concatenated)  # [B*K, embed_dim]
        return fused.view(B, K, -1)


class CandidateAttributeTower(nn.Module):
    """Fine-grained multi-task attribute tower operating on ROIAlign candidate features."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_states: int = 4,
        num_maneuvers: int = 3,
    ) -> None:
        super().__init__()
        self.state_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, num_states),
        )
        self.round_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        self.maneuver_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, num_maneuvers),
        )

    def forward(self, candidate_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produces candidate-level attribute predictions.

        candidate_tokens: [B, K, embed_dim]
        """
        state_logits = self.state_head(candidate_tokens)  # [B, K, 4]
        round_logits = self.round_head(candidate_tokens).squeeze(-1)  # [B, K]
        maneuver_logits = self.maneuver_head(candidate_tokens)  # [B, K, 3]

        return {
            "state_logits": state_logits,
            "round_logits": round_logits,
            "maneuver_logits": maneuver_logits,
            "state_probs": state_logits.softmax(dim=-1),
            "round_probs": round_logits.sigmoid(),
            "maneuver_probs": maneuver_logits.sigmoid(),
        }


class CandidateMultiScaleROIAlignPipeline(nn.Module):
    """Combined end-to-end pipeline: extracts ROIAlign candidate tokens and predicts attributes."""

    def __init__(
        self,
        channels_p2: int = 64,
        channels_p3: int = 128,
        roi_size: tuple[int, int] = (3, 3),
        embed_dim: int = 128,
        stride_p2: float = 4.0,
        stride_p3: float = 8.0,
        num_states: int = 4,
        num_maneuvers: int = 3,
    ) -> None:
        super().__init__()
        self.extractor = CandidateMultiScaleROIAlign(
            channels_p2=channels_p2,
            channels_p3=channels_p3,
            roi_size=roi_size,
            embed_dim=embed_dim,
            stride_p2=stride_p2,
            stride_p3=stride_p3,
        )
        self.tower = CandidateAttributeTower(
            embed_dim=embed_dim,
            num_states=num_states,
            num_maneuvers=num_maneuvers,
        )

    def forward(
        self,
        p2_feat: torch.Tensor,
        p3_feat: torch.Tensor,
        candidate_boxes_xyxy: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.extractor(p2_feat, p3_feat, candidate_boxes_xyxy, img_shape=img_shape)
        attrs = self.tower(tokens)
        attrs["candidate_tokens"] = tokens
        return attrs
