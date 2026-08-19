"""Normalized Relative Geometry Encoding & Relation MLP for TLR-YOLO-MTL (Ticket E25).

Replaces naive geometric offsets with normalized relative spatial vectors and
scene-level ordinal ranking features, processed through a dedicated Relation MLP:

    g_ij = [
        (x_A - x_TL) / w_TL,
        (y_A - y_TL) / h_TL,
        (x_A - x_ego) / W,
        y_A / H,
        log(Area_A),
        log(Area_TL),
        Rank_x,
        Rank_y,
        Rank_Area
    ]
    r_ij = RelationMLP(g_ij)

Features contextual geometry dropout (p_geom in [0.1, 0.3]) to regularize cross-attention
and prevent memorization of fixed camera mounting priors.
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


def compute_normalized_scene_ranks(values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute normalized ordinal scene ranks in [0, 1] along the candidate dimension.

    Args:
        values: [B, K]
        valid_mask: [B, K] bool (optional)
    Returns:
        ranks: [B, K] in [0, 1]
    """
    B, K = values.shape
    if K <= 1:
        return values.new_zeros((B, K))

    # argsort twice gives the ordinal rank (0 to K-1)
    ranks = torch.argsort(torch.argsort(values, dim=-1), dim=-1).to(dtype=values.dtype)
    return ranks / float(max(1, K - 1))


class NormalizedRelativeGeometryEncoder(nn.Module):
    """Encodes normalized relative geometry and ordinal scene ranking features for TL-Arrow pairs."""

    def __init__(self, ego_x: float = 0.5, p_drop: float = 0.0) -> None:
        super().__init__()
        self.ego_x = float(ego_x)
        self.p_drop = float(p_drop)

    def forward(
        self,
        tl_boxes: torch.Tensor,
        arrow_boxes: torch.Tensor,
        tl_valid: torch.Tensor,
        arrow_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative geometric feature tensor G in [B, K_TL, K_Arrow, 10].

        Args:
            tl_boxes: [B, K_TL, 4] normalized (cx, cy, w, h)
            arrow_boxes: [B, K_Arrow, 4] normalized (cx, cy, w, h)
            tl_valid: [B, K_TL] bool
            arrow_valid: [B, K_Arrow] bool
        Returns:
            geom_feats: [B, K_TL, K_Arrow, 10]
        """
        B, K_TL, _ = tl_boxes.shape
        K_Arrow = arrow_boxes.shape[1]

        tl_x = tl_boxes[:, :, None, 0]  # [B, K_TL, 1]
        tl_y = tl_boxes[:, :, None, 1]
        tl_w = tl_boxes[:, :, None, 2].clamp_min(1e-5)
        tl_h = tl_boxes[:, :, None, 3].clamp_min(1e-5)

        ar_x = arrow_boxes[:, None, :, 0]  # [B, 1, K_Arrow]
        ar_y = arrow_boxes[:, None, :, 1]
        ar_w = arrow_boxes[:, None, :, 2].clamp_min(1e-5)
        ar_h = arrow_boxes[:, None, :, 3].clamp_min(1e-5)

        # 1. Scale-normalized relative offsets
        norm_dx = (ar_x - tl_x) / tl_w  # [B, K_TL, K_Arrow]
        norm_dy = (ar_y - tl_y) / tl_h  # [B, K_TL, K_Arrow]

        # 2. Ego perspective offsets
        ego_dx = (ar_x - self.ego_x)     # [B, 1, K_Arrow] -> expand
        ego_dx_exp = ego_dx.expand(-1, K_TL, -1)
        ar_y_exp = ar_y.expand(-1, K_TL, -1)  # Longitudinal perspective depth

        # 3. Log area
        log_area_tl = torch.log(tl_w * tl_h + 1e-7).expand(-1, -1, K_Arrow)
        log_area_ar = torch.log(ar_w * ar_h + 1e-7).expand(-1, K_TL, -1)

        # 4. Ordinal scene ranks
        rank_x_tl = compute_normalized_scene_ranks(tl_boxes[..., 0], tl_valid)[:, :, None].expand(-1, -1, K_Arrow)
        rank_y_ar = compute_normalized_scene_ranks(arrow_boxes[..., 1], arrow_valid)[:, None, :].expand(-1, K_TL, -1)
        rank_area_tl = compute_normalized_scene_ranks(tl_boxes[..., 2] * tl_boxes[..., 3], tl_valid)[:, :, None].expand(-1, -1, K_Arrow)
        rank_area_ar = compute_normalized_scene_ranks(arrow_boxes[..., 2] * arrow_boxes[..., 3], arrow_valid)[:, None, :].expand(-1, K_TL, -1)

        geom = torch.stack(
            (
                norm_dx.clamp(-10.0, 10.0),
                norm_dy.clamp(-10.0, 10.0),
                ego_dx_exp.clamp(-1.0, 1.0),
                ar_y_exp.clamp(0.0, 1.0),
                log_area_tl,
                log_area_ar,
                rank_x_tl,
                rank_y_ar,
                rank_area_tl,
                rank_area_ar,
            ),
            dim=-1,
        )  # [B, K_TL, K_Arrow, 10]

        # Apply Contextual Geometry Dropout if enabled
        if self.training and self.p_drop > 0.0:
            drop_mask = (torch.rand(B, K_TL, K_Arrow, 1, device=geom.device) > self.p_drop).float()
            geom = geom * drop_mask

        return geom


class RelationMLP(nn.Module):
    """2-Layer Relation MLP projecting relative geometric vectors to per-head attention bias."""

    def __init__(self, in_features: int = 10, hidden_dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, heads),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, geometric_features: torch.Tensor) -> torch.Tensor:
        """Compute per-head relation bias.

        Args:
            geometric_features: [B, K_TL, K_Arrow, in_features]
        Returns:
            relation_bias: [B, heads, K_TL, K_Arrow]
        """
        bias = self.network(geometric_features)  # [B, K_TL, K_Arrow, heads]
        return bias.permute(0, 3, 1, 2)         # [B, heads, K_TL, K_Arrow]


class RelationGeometryCrossAttention(nn.Module):
    """Cross-attention layer powered by Normalized Relative Geometry and Relation MLP."""

    def __init__(
        self,
        dimension: int = 128,
        heads: int = 4,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dimension = self.dimension // self.heads

        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

        self.geom_encoder = NormalizedRelativeGeometryEncoder(ego_x=0.5, p_drop=p_drop)
        self.relation_mlp = RelationMLP(in_features=10, hidden_dim=32, heads=heads)

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
        traffic_round: torch.Tensor,
        traffic_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
        arrow_ego_lane: torch.Tensor,
        arrow_valid: torch.Tensor,
        traffic_valid: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, traffic_count, _ = traffic_tokens.shape
        arrow_count = arrow_tokens.shape[1]

        if traffic_valid is None:
            traffic_valid = torch.ones((batch, traffic_count), dtype=torch.bool, device=traffic_tokens.device)

        null = self.null_token.expand(batch, -1, -1)
        keys_values = torch.cat((arrow_tokens, null), dim=1)
        query = self.query(traffic_tokens).reshape(
            batch, traffic_count, self.heads, self.head_dimension
        ).transpose(1, 2)
        key = self.key(keys_values).reshape(
            batch, arrow_count + 1, self.heads, self.head_dimension
        ).transpose(1, 2)
        value = self.value(keys_values).reshape(
            batch, arrow_count + 1, self.heads, self.head_dimension
        ).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dimension ** 0.5)

        # 1. Normalized Relative Geometry & Relation MLP (E25)
        geom_feats = self.geom_encoder(
            traffic_boxes, arrow_boxes, traffic_valid, arrow_valid
        )  # [B, K_TL, K_Arrow, 10]
        relation_bias = self.relation_mlp(geom_feats)  # [B, heads, K_TL, K_Arrow]

        # 2. Semantic Compatibility
        compat = self._semantic_compatibility(
            traffic_round, traffic_maneuver, arrow_maneuver
        ).unsqueeze(1)  # [B, 1, K_TL, K_Arrow]

        pair_bias = relation_bias + compat
        null_bias = pair_bias.new_zeros((batch, self.heads, traffic_count, 1))
        full_bias = torch.cat((pair_bias, null_bias), dim=-1)
        logits = logits + full_bias

        null_valid = torch.ones((batch, 1), dtype=torch.bool, device=arrow_valid.device)
        key_valid = torch.cat((arrow_valid.bool(), null_valid), dim=1)
        logits = logits.masked_fill(~key_valid[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=-1)

        attended = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, traffic_count, self.dimension
        )
        attended = self.output(attended)
        residual = self.gate.to(attended.dtype) * attended if enabled else attended * 0.0
        return self.normalization(traffic_tokens + residual), weights, full_bias


class RelationGeometryUnifiedDetect(UnifiedTrafficControlDetect):
    """Unified detector with Normalized Relative Geometry & Relation MLP (Ticket E25)."""

    def __init__(
        self,
        base: Detect,
        *,
        config: UnifiedHeadConfig | None = None,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__(base, config=config)
        self.cross_attention = RelationGeometryCrossAttention(
            dimension=self.head_config.token_dim,
            heads=self.head_config.attention_heads,
            p_drop=p_drop,
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


def attach_relation_geometry_unified_relevance_head(
    model_wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
    p_drop: float = 0.2,
) -> RelationGeometryUnifiedDetect:
    """Attach RelationGeometryUnifiedDetect in place of final Detect module."""
    base = model_wrapper.model.model[-1]
    if isinstance(base, RelationGeometryUnifiedDetect):
        return base
    if isinstance(base, (Detect, UnifiedTrafficControlDetect)):
        base_detect = base
    else:
        raise TypeError(f"expected Detect or UnifiedTrafficControlDetect, got {type(base).__name__}")

    unified = RelationGeometryUnifiedDetect(
        base_detect, config=config, p_drop=p_drop
    )
    model_wrapper.model.model[-1] = unified
    return unified
