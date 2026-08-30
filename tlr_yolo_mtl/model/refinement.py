"""Sparse Candidate Refinement Head on Top-32 Sub-Grid Regions (Ticket E49).

Introduces a lightweight, coarse-to-fine candidate refinement module that acts
as a virtual P1 (stride 2) local feature stage for tiny traffic lights (<256 px^2).

Scientific Principles:
1. Selective Sparse Routing: Only routes candidate boxes with area < 256 px^2
   (side < 16 px) among Top-K=32 proposals to the refinement branch. Macro objects
   and road arrows bypass this stage with zero compute overhead.
2. High-Resolution Multi-Scale Sampling: Extracts 7x7 ROIAlign features over
   concatenated P2 (stride 4, 64-ch) and shallow projection C2 (stride 4, 64-ch) maps.
3. Residual Sub-Grid Parameterization:
   - Box Deltas (Δx, Δy, Δw, Δh): Fine-grained sub-pixel center offset and scale refinement.
   - State Residuals (Δq_state): High-frequency chromatic logit corrections.
   - Quality Confidence Delta (Δc): Localization alignment score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torchvision.ops import roi_align
except ImportError:
    def roi_align(
        input: torch.Tensor,
        boxes: torch.Tensor | list[torch.Tensor],
        output_size: tuple[int, int] | int,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
        aligned: bool = True,
    ) -> torch.Tensor:
        if isinstance(output_size, int):
            out_h, out_w = output_size, output_size
        else:
            out_h, out_w = output_size

        if isinstance(boxes, list):
            flat_boxes = []
            for b_idx, b_tensor in enumerate(boxes):
                if b_tensor.numel() > 0:
                    idx_col = torch.full(
                        (b_tensor.shape[0], 1),
                        b_idx,
                        dtype=b_tensor.dtype,
                        device=b_tensor.device,
                    )
                    flat_boxes.append(torch.cat([idx_col, b_tensor], dim=1))
            boxes_tensor = (
                torch.cat(flat_boxes, dim=0)
                if flat_boxes
                else torch.empty((0, 5), device=input.device, dtype=input.dtype)
            )
        else:
            boxes_tensor = boxes

        if boxes_tensor.shape[0] == 0:
            return torch.empty(
                (0, input.shape[1], out_h, out_w),
                device=input.device,
                dtype=input.dtype,
            )

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

            norm_x = (mesh_x / max(in_w - 1, 1)) * 2.0 - 1.0
            norm_y = (mesh_y / max(in_h - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

            sample = F.grid_sample(
                input[b_idx : b_idx + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            out[i] = sample.squeeze(0)
        return out


@dataclass(frozen=True, slots=True)
class SparseRefinementConfig:
    """Configuration for Sparse Candidate Refinement (Tickets E49, E68 & E69)."""
    channels_p2: int = 64
    channels_c2: int = 64
    hidden_dim: int = 64
    roi_size: tuple[int, int] = (7, 7)
    area_threshold: float = 256.0  # side < 16 px
    max_candidates: int = 32
    stride: float = 4.0
    state_classes: int = 4
    enable_box_refine: bool = True
    enable_state_refine: bool = True
    enable_quality_refine: bool = True
    # Ticket E68 Dynamic Budgeting
    dynamic_budget: bool = True
    budget_tiers: tuple[int, ...] = (8, 16, 32, 48, 64)
    min_budget: int = 8
    max_budget: int = 64
    # Ticket E69 Distributional Refinement
    distributional: bool = True
    reg_max: int = 16
    delta_range: tuple[float, float] = (-1.5, 1.5)


def select_dynamic_refinement_budget(
    active_count: int | torch.Tensor,
    tiers: tuple[int, ...] = (8, 16, 32, 48, 64),
    min_k: int = 8,
    max_k: int = 64,
) -> int:
    """Select smallest budget tier >= active candidate count (Ticket E68)."""
    if isinstance(active_count, torch.Tensor):
        n = int(active_count.max().item()) if active_count.numel() > 0 else min_k
    else:
        n = int(active_count)
    
    for t in sorted(tiers):
        if t >= n:
            return max(min_k, min(t, max_k))
    return max_k


class SparseCandidateRefinementHead(nn.Module):
    """Virtual P1 sub-grid candidate refinement head for tiny objects (Tickets E49, E68 & E69).
    
    Processes only the Top-K candidate proposals with area < area_threshold.
    Supports:
    - Dynamic scene-adaptive candidate budget K in {8, 16, 32, 48, 64} (Ticket E68).
    - Continuous Gaussian distributional box regression with DFL expectations (Ticket E69).
    """

    def __init__(
        self,
        config: SparseRefinementConfig | None = None,
        *,
        channels_p2: int = 64,
        channels_c2: int = 64,
        hidden_dim: int = 64,
        roi_size: tuple[int, int] = (7, 7),
        area_threshold: float = 256.0,
        max_candidates: int = 32,
        stride: float = 4.0,
        state_classes: int = 4,
        dynamic_budget: bool = True,
        budget_tiers: tuple[int, ...] = (8, 16, 32, 48, 64),
        distributional: bool = True,
        reg_max: int = 16,
        delta_range: tuple[float, float] = (-1.5, 1.5),
    ) -> None:
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = SparseRefinementConfig(
                channels_p2=channels_p2,
                channels_c2=channels_c2,
                hidden_dim=hidden_dim,
                roi_size=roi_size,
                area_threshold=area_threshold,
                max_candidates=max_candidates,
                stride=stride,
                state_classes=state_classes,
                dynamic_budget=dynamic_budget,
                budget_tiers=budget_tiers,
                distributional=distributional,
                reg_max=reg_max,
                delta_range=delta_range,
            )

        in_channels = self.config.channels_p2 + self.config.channels_c2
        self.spatial_scale = 1.0 / self.config.stride
        self.roi_size = self.config.roi_size
        self.area_threshold = float(self.config.area_threshold)
        self.max_candidates = int(self.config.max_candidates)

        # 1. Convolutional Local Feature Tower (7x7 ROI)
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.config.hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Refinement Prediction Heads
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(inplace=True),
        )

        # Box regression head: either continuous distributional (E69) or deterministic (E49)
        self.reg_max = int(self.config.reg_max)
        if self.config.distributional:
            self.box_dist_head = nn.Linear(self.config.hidden_dim, 4 * self.reg_max)
            nn.init.zeros_(self.box_dist_head.weight)
            nn.init.zeros_(self.box_dist_head.bias)
            bin_vals = torch.linspace(
                self.config.delta_range[0], self.config.delta_range[1], self.reg_max
            )
            self.register_buffer("bin_values", bin_vals)
            self.box_delta_head = None
        else:
            self.box_delta_head = nn.Linear(self.config.hidden_dim, 4)
            nn.init.zeros_(self.box_delta_head.weight)
            nn.init.zeros_(self.box_delta_head.bias)
            self.box_dist_head = None
            self.register_buffer("bin_values", None)

        # State logit residual head: Δq_state
        self.state_delta_head = nn.Linear(self.config.hidden_dim, self.config.state_classes)
        nn.init.zeros_(self.state_delta_head.weight)
        nn.init.zeros_(self.state_delta_head.bias)

        # Localization quality delta: Δc
        self.quality_delta_head = nn.Linear(self.config.hidden_dim, 1)
        nn.init.zeros_(self.quality_delta_head.weight)
        nn.init.zeros_(self.quality_delta_head.bias)

    def forward(
        self,
        p2_feat: torch.Tensor,
        c2_feat: torch.Tensor | None = None,
        candidate_boxes_xyxy: torch.Tensor | None = None,
        candidate_scores: torch.Tensor | None = None,
        coarse_state_logits: torch.Tensor | None = None,
        img_shape: tuple[int, int] | None = None,
        is_normalized: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Refines small candidate proposals with virtual P1 sub-grid features."""
        if c2_feat is None:
            c2_feat = p2_feat

        if candidate_boxes_xyxy is None:
            raise ValueError("candidate_boxes_xyxy must be provided")

        B, K_orig, _ = candidate_boxes_xyxy.shape
        boxes = candidate_boxes_xyxy

        # Scale normalized boxes if specified
        if img_shape is not None and is_normalized:
            h, w = img_shape
            scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
            boxes_px = boxes * scale
        else:
            boxes_px = boxes

        # Calculate bounding box areas in pixels
        w_px = (boxes_px[..., 2] - boxes_px[..., 0]).clamp_min(0.0)
        h_px = (boxes_px[..., 3] - boxes_px[..., 1]).clamp_min(0.0)
        areas = w_px * h_px

        # Active mask: tiny traffic light candidates (area < threshold)
        refine_mask = (areas > 0.0) & (areas < self.area_threshold)

        # E68 Dynamic Budgeting: slice to dynamic K if enabled
        if self.config.dynamic_budget and K_orig > self.config.min_budget:
            active_per_img = refine_mask.sum(dim=1)  # [B]
            k_dyn = select_dynamic_refinement_budget(
                active_per_img,
                tiers=self.config.budget_tiers,
                min_k=self.config.min_budget,
                max_k=min(K_orig, self.config.max_budget),
            )
            k_active = min(k_dyn, K_orig)
            boxes_px_active = boxes_px[:, :k_active]
            refine_mask_active = refine_mask[:, :k_active]
        else:
            k_active = K_orig
            boxes_px_active = boxes_px
            refine_mask_active = refine_mask

        # Flatten boxes for ROIAlign batch indexing: [B*k_active, 5]
        batch_ids = torch.arange(B, device=boxes.device, dtype=boxes.dtype).unsqueeze(1).repeat(1, k_active).reshape(-1, 1)
        rois = torch.cat([batch_ids, boxes_px_active.reshape(-1, 4)], dim=1)

        # Direct, vectorized ROIAlign extraction without whole-map concatenation
        roi_p2 = roi_align(
            p2_feat, rois, output_size=self.roi_size, spatial_scale=self.spatial_scale, aligned=True
        )
        roi_c2 = roi_align(
            c2_feat, rois, output_size=self.roi_size, spatial_scale=self.spatial_scale, aligned=True
        )
        fused_rois = torch.cat([roi_p2, roi_c2], dim=1)  # [B*k_active, C_p2 + C_c2, 7, 7]

        # Convolutional refinement feature extraction
        conv_out = self.refinement_conv(fused_rois)
        features = self.fc(self.pool(conv_out).flatten(1))  # [B*k_active, hidden_dim]

        # Predict box deltas (Distributional vs Deterministic)
        if self.config.distributional:
            raw_box_dist = self.box_dist_head(features).reshape(B, k_active, 4, self.reg_max)
            prob_dist = F.softmax(raw_box_dist, dim=-1)
            # Expectation over discrete bin values: [B, k_active, 4]
            raw_box_deltas = torch.sum(prob_dist * self.bin_values, dim=-1)
            # Predictive spatial uncertainty (variance): [B, k_active, 4]
            raw_uncertainty = torch.sum(
                prob_dist * (self.bin_values - raw_box_deltas.unsqueeze(-1)).square(), dim=-1
            )
        else:
            raw_box_deltas = self.box_delta_head(features).reshape(B, k_active, 4)
            raw_box_dist = None
            raw_uncertainty = None

        raw_state_res = self.state_delta_head(features).reshape(B, k_active, self.config.state_classes)
        raw_qual_deltas = self.quality_delta_head(features).reshape(B, k_active, 1)

        # Mask inactive (macro) candidate updates to zero
        mask_4d = refine_mask_active.unsqueeze(-1)
        box_deltas_act = torch.where(mask_4d, raw_box_deltas, torch.zeros_like(raw_box_deltas))
        state_res_act = torch.where(mask_4d, raw_state_res, torch.zeros_like(raw_state_res))
        quality_deltas_act = torch.where(mask_4d, raw_qual_deltas, torch.zeros_like(raw_qual_deltas))

        # Pad back to K_orig if dynamically sliced
        if k_active < K_orig:
            pad_k = K_orig - k_active
            box_deltas = F.pad(box_deltas_act, (0, 0, 0, pad_k), value=0.0)
            state_residuals = F.pad(state_res_act, (0, 0, 0, pad_k), value=0.0)
            quality_deltas = F.pad(quality_deltas_act, (0, 0, 0, pad_k), value=0.0)
            if raw_box_dist is not None:
                box_dist = F.pad(raw_box_dist, (0, 0, 0, 0, 0, pad_k), value=0.0)
                box_uncertainty = F.pad(raw_uncertainty, (0, 0, 0, pad_k), value=0.0)
            else:
                box_dist = None
                box_uncertainty = None
        else:
            box_deltas = box_deltas_act
            state_residuals = state_res_act
            quality_deltas = quality_deltas_act
            box_dist = raw_box_dist
            box_uncertainty = raw_uncertainty

        # Apply box sub-pixel refinement
        if self.config.enable_box_refine:
            cx = (boxes_px[..., 0] + boxes_px[..., 2]) * 0.5
            cy = (boxes_px[..., 1] + boxes_px[..., 3]) * 0.5
            bw = (boxes_px[..., 2] - boxes_px[..., 0]).clamp_min(1e-4)
            bh = (boxes_px[..., 3] - boxes_px[..., 1]).clamp_min(1e-4)

            cx_new = cx + box_deltas[..., 0] * bw
            cy_new = cy + box_deltas[..., 1] * bh
            bw_new = bw * torch.exp(box_deltas[..., 2].clamp(-1.0, 1.0))
            bh_new = bh * torch.exp(box_deltas[..., 3].clamp(-1.0, 1.0))

            x1_new = cx_new - bw_new * 0.5
            y1_new = cy_new - bh_new * 0.5
            x2_new = cx_new + bw_new * 0.5
            y2_new = cy_new + bh_new * 0.5

            refined_boxes = torch.stack([x1_new, y1_new, x2_new, y2_new], dim=-1)
        else:
            refined_boxes = boxes_px.clone()

        # Apply state logit residual refinement
        if self.config.enable_state_refine and coarse_state_logits is not None:
            refined_state = coarse_state_logits + state_residuals
        elif coarse_state_logits is not None:
            refined_state = coarse_state_logits.clone()
        else:
            refined_state = state_residuals

        # Re-normalize boxes if original input was normalized
        if img_shape is not None and is_normalized:
            refined_boxes = refined_boxes / scale

        res = {
            "refined_boxes_xyxy": refined_boxes,
            "refined_state_logits": refined_state,
            "box_deltas": box_deltas,
            "state_residuals": state_residuals,
            "quality_deltas": quality_deltas,
            "refine_mask": refine_mask,
        }
        if box_dist is not None:
            res["box_distribution"] = box_dist
            res["box_uncertainty"] = box_uncertainty
        return res
