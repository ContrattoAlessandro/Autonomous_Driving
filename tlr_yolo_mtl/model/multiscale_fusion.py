"""Multi-Scale P2 + P3 Candidate Token Feature Fusion for TLR-YOLO-MTL (Ticket E22).

Extracts and fuses sub-grid edge/chroma details from P2 (stride 4) with
contextual receptive field features from P3 (stride 8) into a unified
multi-scale candidate token representation:

    f_TL = Linear(LayerNorm([f_P2, f_P3]))

This eliminates the trade-off between sharp local spatial acuity (P2)
and stable contextual semantic receptive fields (P3).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect

from .unified import (
    TRAFFIC_LIGHT_CLASS,
    ROAD_ARROW_CLASS,
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    fixed_topk_candidates,
    scale_relevance_gradient,
    _gather_dense,
)


class MultiScaleCandidateFeatureExtractor(nn.Module):
    """Bilinear multi-scale feature sampler and fusion projection for candidate detections."""

    def __init__(
        self,
        token_feature_dim: int = 64,
        out_feature_dim: int = 64,
        mode: str = "p2_p3_fused",
    ) -> None:
        super().__init__()
        if mode not in {"p2_only", "p3_only", "p2_p3_fused", "p2_p3_p4_fused", "task_gated_p2_p3"}:
            raise ValueError(f"unknown multi-scale extraction mode: {mode}")
        self.token_feature_dim = int(token_feature_dim)
        self.out_feature_dim = int(out_feature_dim)
        self.mode = mode

        num_levels = 2 if mode in {"p2_p3_fused", "task_gated_p2_p3"} else (3 if mode == "p2_p3_p4_fused" else 1)
        in_dim = num_levels * self.token_feature_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_feature_dim),
            nn.SiLU(inplace=True),
        )
        if mode == "task_gated_p2_p3":
            self.gate_param = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("gate_param", None)

    def _sample_level(self, feature_map: torch.Tensor, normalized_centers: torch.Tensor) -> torch.Tensor:
        """Sample feature map at normalized center coordinates [0, 1] using bilinear grid_sample.

        Args:
            feature_map: [B, C, H, W]
            normalized_centers: [B, K, 2] in [0, 1]
        Returns:
            sampled: [B, K, C]
        """
        # Convert [0, 1] to [-1, 1] for grid_sample
        grid = (normalized_centers.unsqueeze(2) * 2.0 - 1.0).to(
            dtype=feature_map.dtype, device=feature_map.device
        )  # [B, K, 1, 2]
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [B, C, K, 1]
        return sampled.squeeze(-1).transpose(1, 2)  # [B, K, C]

    def forward(
        self,
        pyramid_features: Sequence[torch.Tensor],
        normalized_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and fuse multi-scale features for candidate boxes.

        Args:
            pyramid_features: list of [B, C, H_i, W_i] for P2, P3, P4, P5
            normalized_boxes: [B, K, 4] normalized (cx, cy, w, h) in [0, 1]
        Returns:
            fused_features: [B, K, out_feature_dim]
        """
        centers = normalized_boxes[:, :, :2].clamp(0.0, 1.0)
        p2_map = pyramid_features[0]
        p3_map = pyramid_features[1]

        if self.mode == "p2_only":
            f_p2 = self._sample_level(p2_map, centers)
            return self.fusion(f_p2)
        elif self.mode == "p3_only":
            f_p3 = self._sample_level(p3_map, centers)
            return self.fusion(f_p3)
        elif self.mode == "p2_p3_fused":
            f_p2 = self._sample_level(p2_map, centers)
            f_p3 = self._sample_level(p3_map, centers)
            concatenated = torch.cat((f_p2, f_p3), dim=-1)
            return self.fusion(concatenated)
        elif self.mode == "task_gated_p2_p3":
            f_p2 = self._sample_level(p2_map, centers)
            f_p3 = self._sample_level(p3_map, centers)
            gate = torch.sigmoid(self.gate_param) if self.gate_param is not None else 0.5
            concatenated = torch.cat((f_p2 * (2.0 * gate), f_p3 * (2.0 * (1.0 - gate))), dim=-1)
            return self.fusion(concatenated)
        elif self.mode == "p2_p3_p4_fused":
            p4_map = pyramid_features[2]
            f_p2 = self._sample_level(p2_map, centers)
            f_p3 = self._sample_level(p3_map, centers)
            f_p4 = self._sample_level(p4_map, centers)
            concatenated = torch.cat((f_p2, f_p3, f_p4), dim=-1)
            return self.fusion(concatenated)
        raise RuntimeError(f"unhandled mode {self.mode}")


class MultiScaleUnifiedTrafficControlDetect(UnifiedTrafficControlDetect):
    """Unified detector with Multi-Scale P2+P3 candidate token representation."""

    def __init__(
        self,
        base: Detect,
        *,
        config: UnifiedHeadConfig | None = None,
        fusion_mode: str = "p2_p3_fused",
    ) -> None:
        super().__init__(base, config=config)
        self.fusion_mode = fusion_mode

        token_feature_dim = self.head_config.token_feature_dim
        self.tl_feature_extractor = MultiScaleCandidateFeatureExtractor(
            token_feature_dim=token_feature_dim,
            out_feature_dim=token_feature_dim,
            mode=fusion_mode,
        )
        self.arrow_feature_extractor = MultiScaleCandidateFeatureExtractor(
            token_feature_dim=token_feature_dim,
            out_feature_dim=token_feature_dim,
            mode=fusion_mode,
        )

    def _build_tokens(
        self,
        *,
        decoded: torch.Tensor,
        token_features: torch.Tensor,
        round_logits: torch.Tensor,
        maneuver_logits: torch.Tensor,
        ego_lane_logits: torch.Tensor,
        dense_local_relevance_logits: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        type_scores = decoded[:, 4 : 4 + 2]
        traffic_indices, traffic_scores, traffic_valid = fixed_topk_candidates(
            type_scores[:, TRAFFIC_LIGHT_CLASS],
            self.head_config.max_traffic_lights,
            threshold=(0.0 if self.training else self.head_config.traffic_score_threshold),
        )
        arrow_indices, arrow_scores, arrow_valid = fixed_topk_candidates(
            type_scores[:, ROAD_ARROW_CLASS],
            self.head_config.max_arrows,
            threshold=self.head_config.arrow_score_threshold,
        )
        boxes = self._normalized_boxes(decoded[:, :4], features)
        traffic_boxes = _gather_dense(boxes, traffic_indices).transpose(1, 2)
        arrow_boxes = _gather_dense(boxes, arrow_indices).transpose(1, 2)

        # Multi-scale feature sampling (E22 Innovation)
        level_token_feature_maps = [
            head(val) for head, val in zip(self.token_feature_heads, features)
        ]
        traffic_features = self.tl_feature_extractor(level_token_feature_maps, traffic_boxes)
        arrow_features = self.arrow_feature_extractor(level_token_feature_maps, arrow_boxes)

        traffic_round = _gather_dense(round_logits.sigmoid(), traffic_indices)[:, 0]
        traffic_maneuver = _gather_dense(
            maneuver_logits.sigmoid(), traffic_indices
        ).transpose(1, 2)
        arrow_maneuver = _gather_dense(
            maneuver_logits.sigmoid(), arrow_indices
        ).transpose(1, 2)
        arrow_ego_lane = _gather_dense(
            ego_lane_logits.sigmoid(), arrow_indices
        )[:, 0]
        if not self.head_config.ego_lane_enabled:
            arrow_ego_lane = arrow_ego_lane * 0.0 + 0.5
        selected_local_relevance = _gather_dense(
            dense_local_relevance_logits, traffic_indices
        )

        traffic_scale = self._perception_gradient_scale
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            arrow_scale: float | torch.Tensor = 1.0
        else:
            context_scale = (
                self._context_gradient_scale.to(
                    device=arrow_features.device, dtype=arrow_features.dtype
                ).reshape(-1)
                if isinstance(self._context_gradient_scale, torch.Tensor)
                else arrow_features.new_full(
                    (1,), float(self._context_gradient_scale)
                )
            )
            perception_scale = (
                self._perception_gradient_scale.to(
                    device=arrow_features.device, dtype=arrow_features.dtype
                ).reshape(-1)
                if isinstance(self._perception_gradient_scale, torch.Tensor)
                else arrow_features.new_full(
                    (1,), float(self._perception_gradient_scale)
                )
            )
            if context_scale.numel() == 1 and perception_scale.numel() > 1:
                context_scale = context_scale.expand_as(perception_scale)
            if perception_scale.numel() == 1 and context_scale.numel() > 1:
                perception_scale = perception_scale.expand_as(context_scale)
            arrow_scale = context_scale * perception_scale

        if self.export:
            traffic_boxes_source = traffic_boxes
            traffic_feature_source = traffic_features
            traffic_round_source = traffic_round
            traffic_maneuver_source = traffic_maneuver
            traffic_score_source = traffic_scores
            arrow_boxes_source = arrow_boxes
            arrow_feature_source = arrow_features
            arrow_maneuver_source = arrow_maneuver
            arrow_ego_source = arrow_ego_lane
            arrow_score_source = arrow_scores
        else:
            traffic_boxes_source = scale_relevance_gradient(
                traffic_boxes, traffic_scale
            )
            traffic_feature_source = scale_relevance_gradient(
                traffic_features, traffic_scale
            )
            traffic_round_source = scale_relevance_gradient(
                traffic_round, traffic_scale
            )
            traffic_maneuver_source = scale_relevance_gradient(
                traffic_maneuver, traffic_scale
            )
            traffic_score_source = scale_relevance_gradient(
                traffic_scores, traffic_scale
            )
            arrow_boxes_source = scale_relevance_gradient(arrow_boxes, arrow_scale)
            arrow_feature_source = scale_relevance_gradient(
                arrow_features, arrow_scale
            )
            arrow_maneuver_source = scale_relevance_gradient(
                arrow_maneuver, arrow_scale
            )
            arrow_ego_source = scale_relevance_gradient(
                arrow_ego_lane, arrow_scale
            )
            arrow_score_source = scale_relevance_gradient(
                arrow_scores, arrow_scale
            )

        traffic_position = self.position_encoding(traffic_boxes_source)
        arrow_position = self.position_encoding(arrow_boxes_source)
        traffic_source = torch.cat(
            (
                traffic_feature_source,
                traffic_position,
                traffic_round_source[..., None],
                traffic_maneuver_source,
                traffic_score_source[..., None],
            ),
            dim=-1,
        )
        arrow_source = torch.cat(
            (
                arrow_feature_source,
                arrow_position,
                arrow_maneuver_source,
                arrow_ego_source[..., None],
                arrow_score_source[..., None],
            ),
            dim=-1,
        )
        traffic_tokens = self.traffic_token_projection(traffic_source)
        arrow_tokens = self.arrow_token_projection(arrow_source)
        conditioned, attention, geometry_bias = self.cross_attention(
            traffic_tokens,
            arrow_tokens,
            traffic_boxes=traffic_boxes_source,
            arrow_boxes=arrow_boxes_source,
            traffic_round=traffic_round_source,
            traffic_maneuver=traffic_maneuver_source,
            arrow_maneuver=arrow_maneuver_source,
            arrow_ego_lane=arrow_ego_source,
            arrow_valid=arrow_valid,
            enabled=self.attention_enabled,
        )
        local_conditioned = self.cross_attention.normalization(traffic_tokens)
        local_context_delta = self.relevance_head(
            torch.cat((traffic_tokens, local_conditioned), dim=-1)
        ).transpose(1, 2)
        conditioned_context_delta = self.relevance_head(
            torch.cat((traffic_tokens, conditioned), dim=-1)
        ).transpose(1, 2)
        relevance = selected_local_relevance + (
            conditioned_context_delta - local_context_delta
        )
        return {
            "traffic_candidate_indices": traffic_indices,
            "traffic_candidate_scores": traffic_scores,
            "traffic_candidate_valid": traffic_valid,
            "arrow_candidate_indices": arrow_indices,
            "arrow_candidate_scores": arrow_scores,
            "arrow_candidate_valid": arrow_valid,
            "traffic_candidate_boxes": traffic_boxes,
            "arrow_candidate_boxes": arrow_boxes,
            "local_relevance_logits": selected_local_relevance,
            "relevance_logits": relevance,
            "attention_weights": attention,
            "attention_geometry_bias": geometry_bias,
        }


def attach_multiscale_unified_relevance_head(
    model_wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
    fusion_mode: str = "p2_p3_fused",
) -> MultiScaleUnifiedTrafficControlDetect:
    """Attach MultiScaleUnifiedTrafficControlDetect in place of final Detect module."""
    base = model_wrapper.model.model[-1]
    if isinstance(base, MultiScaleUnifiedTrafficControlDetect):
        return base
    if isinstance(base, UnifiedTrafficControlDetect):
        base_detect = base
    elif isinstance(base, Detect):
        base_detect = base
    else:
        raise TypeError(f"expected Detect or UnifiedTrafficControlDetect, got {type(base).__name__}")

    unified = MultiScaleUnifiedTrafficControlDetect(
        base_detect, config=config, fusion_mode=fusion_mode
    )
    model_wrapper.model.model[-1] = unified
    return unified
