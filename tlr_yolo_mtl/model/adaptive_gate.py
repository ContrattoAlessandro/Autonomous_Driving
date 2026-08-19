"""Per-Query Adaptive Contextual Gating for TLR-YOLO-MTL (Ticket E23).

Replaces the global scalar fusion parameter alpha with a dynamic,
per-TL query-adaptive contextual gate g_i in [0, 1] conditioned on:
1. Visual candidate token representation (f_TL, i)
2. Roundness probability P(round_i)
3. Cross-attention entropy H(a_i)
4. Null token attention mass m_null, i
5. Maximum detected arrow score max_j s_arrow, j
6. Count of valid road arrow candidates N_valid
7. Local vs contextual conflict magnitude |logit_local - Delta_ctx|

Mathematical Formulation:
    g_i = (1 - P(round_i)) * sigma(MLP(z_i))
    R_i = logit_local, i + g_i * (Delta_ctx, i - Delta_null, i)

This prevents contextual noise corruption on round lights (g_i -> 0)
while maximizing directional cross-modal reasoning capacity.
"""

from __future__ import annotations

import math
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


class AdaptiveContextualGate(nn.Module):
    """Computes dynamic per-query contextual gate g_i in [0, 1] for each traffic light candidate."""

    def __init__(self, token_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

        # Feature vector z_i has dimension: token_dim + 6
        # [f_TL (token_dim), P(round), Entropy, Null_Mass, Max_Arrow_Score, Valid_Arrow_Ratio, Conflict_Mag]
        in_dim = self.token_dim + 6

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        # Initialize final linear layer to output small positive gate initially
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(
        self,
        traffic_tokens: torch.Tensor,
        traffic_round: torch.Tensor,
        attention_weights: torch.Tensor,
        arrow_scores: torch.Tensor,
        arrow_valid: torch.Tensor,
        local_context_delta: torch.Tensor,
        conditioned_context_delta: torch.Tensor,
        *,
        enforce_round_fallback: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-query gate g_i in [0, 1].

        Args:
            traffic_tokens: [B, K_TL, D]
            traffic_round: [B, K_TL] in [0, 1]
            attention_weights: [B, H, K_TL, K_Arrow + 1]
            arrow_scores: [B, K_Arrow] in [0, 1]
            arrow_valid: [B, K_Arrow] bool
            local_context_delta: [B, 1, K_TL]
            conditioned_context_delta: [B, 1, K_TL]
            enforce_round_fallback: if True, scale by (1 - P(round))
        Returns:
            gates: [B, 1, K_TL] in [0, 1]
            telemetry: dictionary of gate diagnostic statistics
        """
        B, K_TL, D = traffic_tokens.shape
        K_Arrow = arrow_scores.shape[1]

        # 1. Average attention weights across heads -> [B, K_TL, K_Arrow + 1]
        mean_attn = attention_weights.mean(dim=1)

        # 2. Attention entropy H(a_i) across all arrow keys + null
        eps = 1e-7
        entropy = -(mean_attn * torch.log(mean_attn + eps)).sum(dim=-1, keepdim=True)  # [B, K_TL, 1]

        # 3. Null token mass m_null, i
        null_mass = mean_attn[:, :, -1:].clamp(0.0, 1.0)  # [B, K_TL, 1]

        # 4. Max detected arrow score in scene
        max_arrow_score = (arrow_scores * arrow_valid.float()).max(dim=-1, keepdim=True)[0]  # [B, 1]
        max_arrow_score = max_arrow_score.unsqueeze(1).expand(-1, K_TL, 1)  # [B, K_TL, 1]

        # 5. Valid arrow ratio
        valid_arrow_ratio = (arrow_valid.float().sum(dim=-1, keepdim=True) / float(max(1, K_Arrow))).unsqueeze(1).expand(-1, K_TL, 1)

        # 6. Conflict magnitude |conditioned_delta - local_delta|
        local_d = local_context_delta.transpose(1, 2)  # [B, K_TL, 1]
        cond_d = conditioned_context_delta.transpose(1, 2)  # [B, K_TL, 1]
        conflict_mag = torch.abs(cond_d - local_d).clamp(0.0, 10.0)  # [B, K_TL, 1]

        # Assemble gate input vector z_i
        p_round = traffic_round.unsqueeze(-1).clamp(0.0, 1.0)  # [B, K_TL, 1]
        z = torch.cat(
            (
                traffic_tokens,
                p_round,
                entropy,
                null_mass,
                max_arrow_score,
                valid_arrow_ratio,
                conflict_mag,
            ),
            dim=-1,
        )

        raw_gate_logits = self.mlp(z)  # [B, K_TL, 1]
        base_gate = raw_gate_logits.sigmoid()  # [B, K_TL, 1] in [0, 1]

        if enforce_round_fallback:
            effective_gate = (1.0 - p_round) * base_gate
        else:
            effective_gate = base_gate

        gate_output = effective_gate.transpose(1, 2)  # [B, 1, K_TL]

        telemetry = {
            "mean_gate_overall": effective_gate.mean(),
            "mean_gate_directional": (effective_gate * (1.0 - p_round)).sum() / (1.0 - p_round).sum().clamp_min(1.0),
            "mean_gate_round": (effective_gate * p_round).sum() / p_round.sum().clamp_min(1.0),
            "mean_entropy": entropy.mean(),
            "mean_null_mass": null_mass.mean(),
        }

        return gate_output, telemetry


class AdaptiveGatedUnifiedDetect(UnifiedTrafficControlDetect):
    """Unified detector with per-query adaptive contextual gating g_i."""

    def __init__(
        self,
        base: Detect,
        *,
        config: UnifiedHeadConfig | None = None,
        enforce_round_fallback: bool = True,
    ) -> None:
        super().__init__(base, config=config)
        self.enforce_round_fallback = enforce_round_fallback
        self.adaptive_gate = AdaptiveContextualGate(
            token_dim=self.head_config.token_dim,
            hidden_dim=64,
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
        traffic_features = _gather_dense(token_features, traffic_indices).transpose(1, 2)
        arrow_features = _gather_dense(token_features, arrow_indices).transpose(1, 2)
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

        # Dynamic Per-Query Gating (E23 Innovation)
        query_gates, gate_telemetry = self.adaptive_gate(
            traffic_tokens=traffic_tokens,
            traffic_round=traffic_round,
            attention_weights=attention,
            arrow_scores=arrow_scores,
            arrow_valid=arrow_valid,
            local_context_delta=local_context_delta,
            conditioned_context_delta=conditioned_context_delta,
            enforce_round_fallback=self.enforce_round_fallback,
        )

        context_delta = conditioned_context_delta - local_context_delta
        relevance = selected_local_relevance + query_gates * context_delta

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
            "adaptive_gates": query_gates,
            "attention_weights": attention,
            "attention_geometry_bias": geometry_bias,
        }


def attach_adaptive_gated_unified_relevance_head(
    model_wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
    enforce_round_fallback: bool = True,
) -> AdaptiveGatedUnifiedDetect:
    """Attach AdaptiveGatedUnifiedDetect in place of final Detect module."""
    base = model_wrapper.model.model[-1]
    if isinstance(base, AdaptiveGatedUnifiedDetect):
        return base
    if isinstance(base, (Detect, UnifiedTrafficControlDetect)):
        base_detect = base
    else:
        raise TypeError(f"expected Detect or UnifiedTrafficControlDetect, got {type(base).__name__}")

    unified = AdaptiveGatedUnifiedDetect(
        base_detect, config=config, enforce_round_fallback=enforce_round_fallback
    )
    model_wrapper.model.model[-1] = unified
    return unified
