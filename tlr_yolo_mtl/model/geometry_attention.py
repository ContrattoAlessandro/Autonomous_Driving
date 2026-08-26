"""Geometry-Aware Cross-Attention with Explicit Relative Spatial Bias (Ticket E42).

Injects an explicit relative spatial-geometric inductive bias directly into the
TL <-> Road Arrow cross-attention matrix:

    A_ij = softmax( (q_i^T k_j) / sqrt(d) + MLP(phi_ij) )

where phi_ij encodes:
1. Relative spatial offsets (Delta x / W, Delta y / H)
2. Log scale and aspect ratios log(w_TL / w_Arr), log(h_TL / h_Arr), log(Area_TL / Area_Arr)
3. Perspective depth / vertical positions (y_TL / H, y_Arr / H)
4. Lateral ego-lane offset (x_Arr - x_ego) / W
5. Road arrow directional semantics (3-class maneuver: Left, Straight, Right)
6. Intrinsic candidate detection confidence scores (s_TL, s_Arr)
7. Semantic roundness compatibility (wildcard vs directional alignment)

This inductive bias allows the cross-attention mechanism to structurally reject visually
plausible traffic lights that physically govern adjacent turn-bays or cross-street lanes,
directly resolving the Relevance Precision (83.7%) vs Recall (87.4%) gap.
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


class ExplicitRelativeGeometryEncoder(nn.Module):
    """Encodes rich normalized relative geometry and spatial perspective features for TL-Arrow pairs."""

    def __init__(self, ego_x: float = 0.5, p_drop: float = 0.0) -> None:
        super().__init__()
        self.ego_x = float(ego_x)
        self.p_drop = float(p_drop)

    def forward(
        self,
        tl_boxes: torch.Tensor,
        arrow_boxes: torch.Tensor,
        tl_scores: torch.Tensor,
        arrow_scores: torch.Tensor,
        tl_round: torch.Tensor,
        tl_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
        arrow_ego_lane: torch.Tensor | None = None,
        tl_valid: torch.Tensor | None = None,
        arrow_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute relative geometric feature tensor phi_ij in [B, K_TL, K_Arrow, 14].

        Args:
            tl_boxes: [B, K_TL, 4] normalized (cx, cy, w, h)
            arrow_boxes: [B, K_Arrow, 4] normalized (cx, cy, w, h)
            tl_scores: [B, K_TL] detection confidence
            arrow_scores: [B, K_Arrow] detection confidence
            tl_round: [B, K_TL] round signal probability in [0, 1]
            tl_maneuver: [B, K_TL, 3] maneuver directional probabilities
            arrow_maneuver: [B, K_Arrow, 3] maneuver directional probabilities
            arrow_ego_lane: [B, K_Arrow] ego-lane prior in [0, 1] (optional)
            tl_valid: [B, K_TL] bool
            arrow_valid: [B, K_Arrow] bool
        Returns:
            geom_feats: [B, K_TL, K_Arrow, 14]
        """
        B, K_TL, _ = tl_boxes.shape
        K_Arrow = arrow_boxes.shape[1]

        tl_x = tl_boxes[:, :, None, 0]  # [B, K_TL, 1]
        tl_y = tl_boxes[:, :, None, 1]  # [B, K_TL, 1]
        tl_w = tl_boxes[:, :, None, 2].clamp_min(1e-5)
        tl_h = tl_boxes[:, :, None, 3].clamp_min(1e-5)

        ar_x = arrow_boxes[:, None, :, 0]  # [B, 1, K_Arrow]
        ar_y = arrow_boxes[:, None, :, 1]  # [B, 1, K_Arrow]
        ar_w = arrow_boxes[:, None, :, 2].clamp_min(1e-5)
        ar_h = arrow_boxes[:, None, :, 3].clamp_min(1e-5)

        # 1. Normalized Relative Offsets (dx/W, dy/H)
        delta_x = ar_x - tl_x  # [B, K_TL, K_Arrow]
        delta_y = ar_y - tl_y  # [B, K_TL, K_Arrow]

        # 2. Scale-Normalized Offsets
        norm_dx = (ar_x - tl_x) / tl_w  # [B, K_TL, K_Arrow]
        norm_dy = (ar_y - tl_y) / tl_h  # [B, K_TL, K_Arrow]

        # 3. Log Scale & Area Ratios
        log_w_ratio = torch.log(tl_w / ar_w).clamp(-5.0, 5.0).expand(-1, -1, K_Arrow)
        log_h_ratio = torch.log(tl_h / ar_h).clamp(-5.0, 5.0).expand(-1, -1, K_Arrow)
        log_area_tl = torch.log(tl_w * tl_h + 1e-7).expand(-1, -1, K_Arrow)
        log_area_ar = torch.log(ar_w * ar_h + 1e-7).expand(-1, K_TL, -1)
        log_area_ratio = (log_area_tl - log_area_ar).clamp(-6.0, 6.0)

        # 4. Vertical Perspective Coordinates
        tl_y_exp = tl_y.expand(-1, -1, K_Arrow)
        ar_y_exp = ar_y.expand(-1, K_TL, -1)

        # 5. Lateral Ego Perspective Offset
        ego_dx_ar = (ar_x - self.ego_x).expand(-1, K_TL, -1)  # [B, K_TL, K_Arrow]

        # 6. Directional & Maneuver Semantic Affinity
        # Directional dot product: [B, K_TL, K_Arrow]
        tl_man_norm = tl_maneuver / tl_maneuver.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        directional_affinity = (
            tl_man_norm[:, :, None, :] * arrow_maneuver[:, None, :, :]
        ).sum(dim=-1)  # [B, K_TL, K_Arrow]

        # 7. Candidate Detection Confidences
        s_tl_exp = tl_scores[:, :, None].expand(-1, -1, K_Arrow)
        s_ar_exp = arrow_scores[:, None, :].expand(-1, K_TL, -1)

        # Assemble normalized feature descriptor phi_ij (14 dimensions)
        phi = torch.stack(
            (
                delta_x.clamp(-1.0, 1.0),
                delta_y.clamp(-1.0, 1.0),
                norm_dx.clamp(-10.0, 10.0),
                norm_dy.clamp(-10.0, 10.0),
                log_w_ratio,
                log_h_ratio,
                log_area_ratio,
                tl_y_exp.clamp(0.0, 1.0),
                ar_y_exp.clamp(0.0, 1.0),
                ego_dx_ar.clamp(-1.0, 1.0),
                directional_affinity.clamp(0.0, 1.0),
                tl_round[:, :, None].expand(-1, -1, K_Arrow).clamp(0.0, 1.0),
                s_tl_exp.clamp(0.0, 1.0),
                s_ar_exp.clamp(0.0, 1.0),
            ),
            dim=-1,
        )  # [B, K_TL, K_Arrow, 14]

        # Apply Contextual Geometry Dropout if enabled during training
        if self.training and self.p_drop > 0.0:
            drop_mask = (torch.rand(B, K_TL, K_Arrow, 1, device=phi.device) > self.p_drop).float()
            phi = phi * drop_mask

        return phi


class GeometryAttentionBiasMLP(nn.Module):
    """Lightweight 2-layer MLP projecting relative geometric vectors to multi-head attention biases."""

    def __init__(self, in_features: int = 14, hidden_dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, heads),
        )
        # Neutral identity initialization: zero bias ensures neutral start
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute per-head attention bias matrix B_ij in [B, heads, K_TL, K_Arrow].

        Args:
            phi: [B, K_TL, K_Arrow, in_features]
        Returns:
            bias: [B, heads, K_TL, K_Arrow]
        """
        bias = self.network(phi)  # [B, K_TL, K_Arrow, heads]
        return bias.permute(0, 3, 1, 2)  # [B, heads, K_TL, K_Arrow]


class GeometryAwareCrossAttention(nn.Module):
    """Multi-Head Cross-Attention with Explicit Relative Spatial Bias (Ticket E42)."""

    def __init__(
        self,
        dimension: int = 128,
        heads: int = 4,
        hidden_dim: int = 32,
        p_drop: float = 0.0,
        use_confidence_gating: bool = True,
    ) -> None:
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dimension = self.dimension // self.heads
        self.use_confidence_gating = bool(use_confidence_gating)

        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

        self.geometry_encoder = ExplicitRelativeGeometryEncoder(ego_x=0.5, p_drop=p_drop)
        self.geometry_mlp = GeometryAttentionBiasMLP(in_features=14, hidden_dim=hidden_dim, heads=heads)

        self.null_token = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.normal_(self.null_token, std=0.02)
        self.round_wildcard_logit = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.zeros(1))
        self.normalization = nn.LayerNorm(dimension)

    def _semantic_compatibility(
        self,
        tl_round: torch.Tensor,
        tl_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
    ) -> torch.Tensor:
        directional = (
            tl_maneuver[:, :, None, :] * arrow_maneuver[:, None, :, :]
        ).sum(-1) / tl_maneuver.sum(-1, keepdim=True).clamp_min(1e-6)
        wildcard = self.round_wildcard_logit.sigmoid()
        round_probability = tl_round[:, :, None]
        return round_probability * wildcard + (1.0 - round_probability) * directional

    def forward(
        self,
        traffic_tokens: torch.Tensor,
        arrow_tokens: torch.Tensor,
        *,
        traffic_boxes: torch.Tensor,
        arrow_boxes: torch.Tensor,
        traffic_scores: torch.Tensor,
        arrow_scores: torch.Tensor,
        traffic_round: torch.Tensor,
        traffic_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
        arrow_ego_lane: torch.Tensor | None = None,
        arrow_valid: torch.Tensor,
        traffic_valid: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute geometry-aware cross attention.

        Args:
            traffic_tokens: [B, K_TL, D]
            arrow_tokens: [B, K_Arrow, D]
            traffic_boxes: [B, K_TL, 4] normalized
            arrow_boxes: [B, K_Arrow, 4] normalized
            traffic_scores: [B, K_TL]
            arrow_scores: [B, K_Arrow]
            traffic_round: [B, K_TL] in [0, 1]
            traffic_maneuver: [B, K_TL, 3]
            arrow_maneuver: [B, K_Arrow, 3]
            arrow_ego_lane: [B, K_Arrow]
            arrow_valid: [B, K_Arrow] bool
            traffic_valid: [B, K_TL] bool
            enabled: whether residual attention gate is applied
        Returns:
            conditioned_tokens: [B, K_TL, D]
            attention_weights: [B, H, K_TL, K_Arrow + 1]
            geometry_bias: [B, H, K_TL, K_Arrow + 1]
        """
        batch, traffic_count, _ = traffic_tokens.shape
        arrow_count = arrow_tokens.shape[1]

        if traffic_valid is None:
            traffic_valid = torch.ones((batch, traffic_count), dtype=torch.bool, device=traffic_tokens.device)

        null = self.null_token.expand(batch, -1, -1)
        keys_values = torch.cat((arrow_tokens, null), dim=1)

        # Standard Multi-Head Q, K, V Projections
        query = self.query(traffic_tokens).reshape(
            batch, traffic_count, self.heads, self.head_dimension
        ).transpose(1, 2)  # [B, H, K_TL, d_h]
        key = self.key(keys_values).reshape(
            batch, arrow_count + 1, self.heads, self.head_dimension
        ).transpose(1, 2)  # [B, H, K_Arrow + 1, d_h]
        value = self.value(keys_values).reshape(
            batch, arrow_count + 1, self.heads, self.head_dimension
        ).transpose(1, 2)  # [B, H, K_Arrow + 1, d_h]

        # Appearance Attention Logits
        logits = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dimension ** 0.5)  # [B, H, K_TL, K_Arrow + 1]

        # 1. Compute Explicit Spatial-Geometric Descriptor phi_ij
        phi = self.geometry_encoder(
            tl_boxes=traffic_boxes,
            arrow_boxes=arrow_boxes,
            tl_scores=traffic_scores,
            arrow_scores=arrow_scores,
            tl_round=traffic_round,
            tl_maneuver=traffic_maneuver,
            arrow_maneuver=arrow_maneuver,
            arrow_ego_lane=arrow_ego_lane,
            tl_valid=traffic_valid,
            arrow_valid=arrow_valid,
        )  # [B, K_TL, K_Arrow, 14]

        # 2. Compute Multi-Head Geometric Bias B_ij via MLP
        geom_bias = self.geometry_mlp(phi)  # [B, H, K_TL, K_Arrow]

        # 3. Optional Candidate Confidence Score Modulation
        if self.use_confidence_gating:
            score_gate = (traffic_scores[:, :, None] * arrow_scores[:, None, :]).clamp(0.0, 1.0).unsqueeze(1)
            geom_bias = geom_bias * (0.5 + 0.5 * score_gate)

        # 4. Semantic Compatibility Bonus
        compat = self._semantic_compatibility(
            traffic_round, traffic_maneuver, arrow_maneuver
        ).unsqueeze(1)  # [B, 1, K_TL, K_Arrow]

        pair_bias = geom_bias + compat
        null_bias = pair_bias.new_zeros((batch, self.heads, traffic_count, 1))
        full_bias = torch.cat((pair_bias, null_bias), dim=-1)  # [B, H, K_TL, K_Arrow + 1]

        # Inject Explicit Bias into Attention Logits
        logits = logits + full_bias

        # Mask Invalid Arrow Keys
        null_valid = torch.ones((batch, 1), dtype=torch.bool, device=arrow_valid.device)
        key_valid = torch.cat((arrow_valid.bool(), null_valid), dim=1)
        logits = logits.masked_fill(~key_valid[:, None, None, :], torch.finfo(logits.dtype).min)

        # Softmax Attention Weights
        weights = logits.softmax(dim=-1)

        # Aggregate Attended Context Values
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, traffic_count, self.dimension
        )
        attended = self.output(attended)

        # Gated Residual Connection
        residual = self.gate.to(attended.dtype) * attended if enabled else attended * 0.0
        return self.normalization(traffic_tokens + residual), weights, full_bias


class GeometryAwareUnifiedDetect(UnifiedTrafficControlDetect):
    """Unified detector with Geometry-Aware Cross-Attention (Ticket E42)."""

    def __init__(
        self,
        base: Detect,
        *,
        config: UnifiedHeadConfig | None = None,
        hidden_dim: int = 32,
        p_drop: float = 0.0,
        use_confidence_gating: bool = True,
    ) -> None:
        super().__init__(base, config=config)
        self.cross_attention = GeometryAwareCrossAttention(
            dimension=self.head_config.token_dim,
            heads=self.head_config.attention_heads,
            hidden_dim=hidden_dim,
            p_drop=p_drop,
            use_confidence_gating=use_confidence_gating,
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
            traffic_scores=traffic_score_source,
            arrow_scores=arrow_score_source,
            traffic_round=traffic_round_source,
            traffic_maneuver=traffic_maneuver_source,
            arrow_maneuver=arrow_maneuver_source,
            arrow_ego_lane=arrow_ego_source,
            arrow_valid=arrow_valid,
            traffic_valid=traffic_valid,
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


def attach_geometry_aware_unified_relevance_head(
    model_wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
    hidden_dim: int = 32,
    p_drop: float = 0.0,
    use_confidence_gating: bool = True,
) -> GeometryAwareUnifiedDetect:
    """Attach GeometryAwareUnifiedDetect in place of final Detect module."""
    base = model_wrapper.model.model[-1]
    if isinstance(base, GeometryAwareUnifiedDetect):
        return base
    if isinstance(base, (Detect, UnifiedTrafficControlDetect)):
        base_detect = base
    else:
        raise TypeError(f"expected Detect or UnifiedTrafficControlDetect, got {type(base).__name__}")

    unified = GeometryAwareUnifiedDetect(
        base_detect,
        config=config,
        hidden_dim=hidden_dim,
        p_drop=p_drop,
        use_confidence_gating=use_confidence_gating,
    )
    model_wrapper.model.model[-1] = unified
    return unified
