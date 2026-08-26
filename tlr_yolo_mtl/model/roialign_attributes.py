"""Candidate-Centered Multi-Scale ROIAlign Attribute Head & Task-Specific Gated Fusion (Ticket E28 & E41).

Replaces single-point anchor feature sampling with candidate-centered ROIAlign
over P2 (stride 4) and P3 (stride 8) feature maps for the top K_TL = 32 candidates.

Ticket E41 Innovations:
1. Decoupled Spatial Resolution: Selective 5x5 ROIAlign for the State Head (25 sampling points)
   to resolve 3-lamp vertical stacked subdivisions, while maintaining efficient 3x3 ROIAlign
   for Roundness, Maneuver, and Relevance candidate tokens.
2. Learnable Task-Specific Feature Gating (alpha_t): Decouples P2 (fine-grained chromatic texture)
   and P3 (semantic contextual layout) blending for each multi-task prediction head.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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


class TaskSpecificGatedROIAlign(nn.Module):
    """Decoupled Task-Specific Gated ROIAlign Feature Extractor (Ticket E41).

    Extracts multi-scale ROIAlign features across P2 (stride 4) and P3 (stride 8) with:
    - Selective 5x5 ROIAlign grid for State classification (25 spatial points)
    - Lightweight 3x3 ROIAlign grid for Roundness, Maneuver, and Relevance candidate tokens (9 points)
    - Learnable task-specific gate parameters (alpha_state, alpha_round, alpha_man, alpha_rel)
      decoupling fine-grained chromatic acuity (P2) from contextual receptive field semantics (P3).
    """

    def __init__(
        self,
        channels_p2: int = 64,
        channels_p3: int = 128,
        state_roi_size: tuple[int, int] = (5, 5),
        aux_roi_size: tuple[int, int] = (3, 3),
        embed_dim: int = 128,
        stride_p2: float = 4.0,
        stride_p3: float = 8.0,
        use_task_gating: bool = True,
    ) -> None:
        super().__init__()
        self.channels_p2 = int(channels_p2)
        self.channels_p3 = int(channels_p3)
        self.state_roi_size = state_roi_size
        self.aux_roi_size = aux_roi_size
        self.embed_dim = int(embed_dim)
        self.spatial_scale_p2 = 1.0 / stride_p2
        self.spatial_scale_p3 = 1.0 / stride_p3
        self.use_task_gating = bool(use_task_gating)

        # 1. State feature projection (5x5 ROIAlign grid)
        in_dim_state = (channels_p2 + channels_p3) * state_roi_size[0] * state_roi_size[1]
        self.fusion_state = nn.Sequential(
            nn.Linear(in_dim_state, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 2. Auxiliary feature projections (3x3 ROIAlign grid)
        in_dim_aux = (channels_p2 + channels_p3) * aux_roi_size[0] * aux_roi_size[1]
        self.fusion_round = nn.Sequential(
            nn.Linear(in_dim_aux, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.fusion_man = nn.Sequential(
            nn.Linear(in_dim_aux, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.fusion_candidate = nn.Sequential(
            nn.Linear(in_dim_aux, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 3. Learnable task-specific gate parameters (logits)
        # raw_gate > 0 biases towards P2 (high-res texture); raw_gate < 0 biases towards P3 (context)
        if self.use_task_gating:
            self.raw_gate_state = nn.Parameter(torch.tensor(1.2))   # sigma(1.2) ~ 0.77 (P2 dominant)
            self.raw_gate_round = nn.Parameter(torch.tensor(0.5))   # sigma(0.5) ~ 0.62 (P2 favored)
            self.raw_gate_man = nn.Parameter(torch.tensor(0.0))     # sigma(0.0) = 0.50 (balanced)
            self.raw_gate_rel = nn.Parameter(torch.tensor(-0.85))  # sigma(-0.85) ~ 0.30 (P3 context dominant)
        else:
            self.register_parameter("raw_gate_state", None)
            self.register_parameter("raw_gate_round", None)
            self.register_parameter("raw_gate_man", None)
            self.register_parameter("raw_gate_rel", None)

    @property
    def task_gates(self) -> dict[str, float]:
        """Returns the current evaluated gate weights alpha_t in [0, 1]."""
        if not self.use_task_gating:
            return {"state": 0.5, "round": 0.5, "maneuver": 0.5, "relevance": 0.5}
        return {
            "state": float(torch.sigmoid(self.raw_gate_state).item()),
            "round": float(torch.sigmoid(self.raw_gate_round).item()),
            "maneuver": float(torch.sigmoid(self.raw_gate_man).item()),
            "relevance": float(torch.sigmoid(self.raw_gate_rel).item()),
        }

    def forward(
        self,
        p2_feat: torch.Tensor,
        p3_feat: torch.Tensor,
        candidate_boxes_xyxy: torch.Tensor,  # [B, K, 4]
        img_shape: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extracts task-decoupled candidate embeddings across 5x5 State and 3x3 Auxiliary grids.

        Returns:
            Dictionary containing:
                - state_tokens: [B, K, embed_dim]
                - round_tokens: [B, K, embed_dim]
                - man_tokens: [B, K, embed_dim]
                - candidate_tokens: [B, K, embed_dim] (relevance/general token)
                - task_gates: dictionary of scalar gate tensors
        """
        B, K, _ = candidate_boxes_xyxy.shape
        boxes = candidate_boxes_xyxy
        if img_shape is not None and boxes.numel() > 0 and boxes.max() <= 1.05:
            h, w = img_shape
            scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes * scale

        boxes_list = [boxes[b] for b in range(B)]

        # 1. State 5x5 ROIAlign extraction
        roi_p2_state = roi_align(
            p2_feat,
            boxes_list,
            output_size=self.state_roi_size,
            spatial_scale=self.spatial_scale_p2,
            aligned=True,
        )
        roi_p3_state = roi_align(
            p3_feat,
            boxes_list,
            output_size=self.state_roi_size,
            spatial_scale=self.spatial_scale_p3,
            aligned=True,
        )
        flat_p2_state = roi_p2_state.flatten(1)  # [B*K, C_p2 * 25]
        flat_p3_state = roi_p3_state.flatten(1)  # [B*K, C_p3 * 25]

        # 2. Auxiliary 3x3 ROIAlign extraction
        roi_p2_aux = roi_align(
            p2_feat,
            boxes_list,
            output_size=self.aux_roi_size,
            spatial_scale=self.spatial_scale_p2,
            aligned=True,
        )
        roi_p3_aux = roi_align(
            p3_feat,
            boxes_list,
            output_size=self.aux_roi_size,
            spatial_scale=self.spatial_scale_p3,
            aligned=True,
        )
        flat_p2_aux = roi_p2_aux.flatten(1)  # [B*K, C_p2 * 9]
        flat_p3_aux = roi_p3_aux.flatten(1)  # [B*K, C_p3 * 9]

        # 3. Gating computations
        if self.use_task_gating and self.raw_gate_state is not None:
            alpha_state = torch.sigmoid(self.raw_gate_state)
            alpha_round = torch.sigmoid(self.raw_gate_round)
            alpha_man = torch.sigmoid(self.raw_gate_man)
            alpha_rel = torch.sigmoid(self.raw_gate_rel)
        else:
            alpha_state = alpha_round = alpha_man = alpha_rel = torch.tensor(
                0.5, device=p2_feat.device, dtype=p2_feat.dtype
            )

        # 4. Gated feature concatenation & projections
        # State: 5x5 grid
        state_cat = torch.cat(
            [flat_p2_state * (2.0 * alpha_state), flat_p3_state * (2.0 * (1.0 - alpha_state))],
            dim=1,
        )
        state_tokens = self.fusion_state(state_cat).view(B, K, self.embed_dim)

        # Roundness: 3x3 grid
        round_cat = torch.cat(
            [flat_p2_aux * (2.0 * alpha_round), flat_p3_aux * (2.0 * (1.0 - alpha_round))],
            dim=1,
        )
        round_tokens = self.fusion_round(round_cat).view(B, K, self.embed_dim)

        # Maneuver: 3x3 grid
        man_cat = torch.cat(
            [flat_p2_aux * (2.0 * alpha_man), flat_p3_aux * (2.0 * (1.0 - alpha_man))],
            dim=1,
        )
        man_tokens = self.fusion_man(man_cat).view(B, K, self.embed_dim)

        # General / Relevance candidate token: 3x3 grid
        rel_cat = torch.cat(
            [flat_p2_aux * (2.0 * alpha_rel), flat_p3_aux * (2.0 * (1.0 - alpha_rel))],
            dim=1,
        )
        candidate_tokens = self.fusion_candidate(rel_cat).view(B, K, self.embed_dim)

        return {
            "state_tokens": state_tokens,
            "round_tokens": round_tokens,
            "man_tokens": man_tokens,
            "candidate_tokens": candidate_tokens,
            "alpha_state": alpha_state,
            "alpha_round": alpha_round,
            "alpha_man": alpha_man,
            "alpha_rel": alpha_rel,
        }


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


class TaskSpecificAttributeTower(nn.Module):
    """Task-Specific Attribute Tower with dedicated embeddings per task branch (Ticket E41)."""

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

    def forward(
        self,
        state_tokens: torch.Tensor,
        round_tokens: torch.Tensor,
        man_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Produces candidate-level attribute predictions using task-decoupled tokens."""
        state_logits = self.state_head(state_tokens)          # [B, K, 4]
        round_logits = self.round_head(round_tokens).squeeze(-1)  # [B, K]
        maneuver_logits = self.maneuver_head(man_tokens)      # [B, K, 3]

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


class TaskSpecificROIAlignPipeline(nn.Module):
    """Complete Task-Specific Gated Multi-Scale ROIAlign Pipeline (Ticket E41 Champion v2)."""

    def __init__(
        self,
        channels_p2: int = 64,
        channels_p3: int = 128,
        state_roi_size: tuple[int, int] = (5, 5),
        aux_roi_size: tuple[int, int] = (3, 3),
        embed_dim: int = 128,
        stride_p2: float = 4.0,
        stride_p3: float = 8.0,
        num_states: int = 4,
        num_maneuvers: int = 3,
        use_task_gating: bool = True,
    ) -> None:
        super().__init__()
        self.extractor = TaskSpecificGatedROIAlign(
            channels_p2=channels_p2,
            channels_p3=channels_p3,
            state_roi_size=state_roi_size,
            aux_roi_size=aux_roi_size,
            embed_dim=embed_dim,
            stride_p2=stride_p2,
            stride_p3=stride_p3,
            use_task_gating=use_task_gating,
        )
        self.tower = TaskSpecificAttributeTower(
            embed_dim=embed_dim,
            num_states=num_states,
            num_maneuvers=num_maneuvers,
        )

    @property
    def task_gates(self) -> dict[str, float]:
        return self.extractor.task_gates

    def forward(
        self,
        p2_feat: torch.Tensor,
        p3_feat: torch.Tensor,
        candidate_boxes_xyxy: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        extracted = self.extractor(p2_feat, p3_feat, candidate_boxes_xyxy, img_shape=img_shape)
        attrs = self.tower(
            state_tokens=extracted["state_tokens"],
            round_tokens=extracted["round_tokens"],
            man_tokens=extracted["man_tokens"],
        )
        attrs["candidate_tokens"] = extracted["candidate_tokens"]
        attrs["state_tokens"] = extracted["state_tokens"]
        attrs["round_tokens"] = extracted["round_tokens"]
        attrs["man_tokens"] = extracted["man_tokens"]
        attrs["task_gates"] = {
            "state": extracted["alpha_state"],
            "round": extracted["alpha_round"],
            "maneuver": extracted["alpha_man"],
            "relevance": extracted["alpha_rel"],
        }
        return attrs
