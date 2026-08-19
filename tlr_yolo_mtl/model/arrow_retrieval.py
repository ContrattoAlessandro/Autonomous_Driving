"""Query-Conditioned Road Arrow Selection (Top-M per TL Query) for TLR-YOLO-MTL (Ticket E24).

Dynamically retrieves the top M most relevant road arrows for each specific
traffic light query (e.g. M=8 selected from a global candidate pool of K=32)
via a learned matching MLP, replacing unconditioned global cross-attention.

Mathematical Formulation:
    q_ij = MLP([Delta x_ij, Delta y_ij, w_i, h_i, w_j, h_j, score_j, cos_sim(f_TL,i, f_A,j)])
    S_i = TopK_{j in {1...K_Arrow}}(q_ij, k=M)
    attended_i = CrossAttention(Query=f_TL,i, Keys/Values=f_Arrow,S_i union {NullToken})

Advantages:
1. Eliminates distant/opposite-lane arrow distractor noise.
2. Sharper cross-attention entropy and interpretability.
3. Compute efficiency on larger global candidate pools (K_Arrow >= 32).
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


class QueryConditionedArrowMatcher(nn.Module):
    """Lightweight pairwise matching network scoring compatibility between TL queries and Arrow candidates."""

    def __init__(self, token_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

        # Input feature per (TL_i, Arrow_j) pair:
        # [Delta_x, Delta_y, w_tl, h_tl, w_arrow, h_arrow, log_area_tl, log_area_arrow, arrow_score, token_sim]
        in_dim = 10

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        traffic_tokens: torch.Tensor,
        arrow_tokens: torch.Tensor,
        traffic_boxes: torch.Tensor,
        arrow_boxes: torch.Tensor,
        arrow_scores: torch.Tensor,
        arrow_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute matching score matrix Q in [B, K_TL, K_Arrow].

        Args:
            traffic_tokens: [B, K_TL, D]
            arrow_tokens: [B, K_Arrow, D]
            traffic_boxes: [B, K_TL, 4] normalized (cx, cy, w, h)
            arrow_boxes: [B, K_Arrow, 4] normalized (cx, cy, w, h)
            arrow_scores: [B, K_Arrow] in [0, 1]
            arrow_valid: [B, K_Arrow] bool
        Returns:
            matching_scores: [B, K_TL, K_Arrow]
        """
        B, K_TL, D = traffic_tokens.shape
        K_Arrow = arrow_tokens.shape[1]

        # 1. Spatial relative offsets
        tl_c = traffic_boxes[:, :, None, :2]  # [B, K_TL, 1, 2]
        ar_c = arrow_boxes[:, None, :, :2]   # [B, 1, K_Arrow, 2]
        delta_xy = ar_c - tl_c               # [B, K_TL, K_Arrow, 2]

        tl_wh = traffic_boxes[:, :, None, 2:].clamp_min(1e-6)  # [B, K_TL, 1, 2]
        ar_wh = arrow_boxes[:, None, :, 2:].clamp_min(1e-6)   # [B, 1, K_Arrow, 2]
        tl_wh_exp = tl_wh.expand(-1, -1, K_Arrow, -1)
        ar_wh_exp = ar_wh.expand(-1, K_TL, -1, -1)

        log_area_tl = torch.log(tl_wh_exp[..., 0:1] * tl_wh_exp[..., 1:2] + 1e-7)
        log_area_ar = torch.log(ar_wh_exp[..., 0:1] * ar_wh_exp[..., 1:2] + 1e-7)

        # 2. Arrow detection score
        ar_score_exp = arrow_scores[:, None, :, None].expand(-1, K_TL, -1, -1)  # [B, K_TL, K_Arrow, 1]

        # 3. Token similarity (cosine similarity)
        tl_norm = F.normalize(traffic_tokens, p=2, dim=-1)  # [B, K_TL, D]
        ar_norm = F.normalize(arrow_tokens, p=2, dim=-1)   # [B, K_Arrow, D]
        cos_sim = torch.bmm(tl_norm, ar_norm.transpose(1, 2)).unsqueeze(-1)  # [B, K_TL, K_Arrow, 1]

        pair_feats = torch.cat(
            (
                delta_xy,
                tl_wh_exp,
                ar_wh_exp,
                log_area_tl,
                log_area_ar,
                ar_score_exp,
                cos_sim,
            ),
            dim=-1,
        )  # [B, K_TL, K_Arrow, 10]

        scores = self.mlp(pair_feats).squeeze(-1)  # [B, K_TL, K_Arrow]

        # Mask invalid arrows with large negative score safe for float16
        invalid_mask = (~arrow_valid)[:, None, :].expand(-1, K_TL, -1)
        min_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(invalid_mask, min_val)
        return scores


class QueryConditionedCrossAttention(nn.Module):
    """Query-conditioned cross-attention where each TL query attends only to its Top-M arrows + Null Token."""

    def __init__(
        self,
        dimension: int = 128,
        heads: int = 4,
        top_m: int = 8,
    ) -> None:
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dimension = self.dimension // self.heads
        self.top_m = int(top_m)

        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

        self.geometry_bias = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, heads),
        )
        nn.init.zeros_(self.geometry_bias[-1].weight)
        nn.init.zeros_(self.geometry_bias[-1].bias)

        self.null_token = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.normal_(self.null_token, std=0.02)
        self.round_wildcard_logit = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.zeros(1))
        self.normalization = nn.LayerNorm(dimension)

    def _pair_bias_query(
        self,
        tl_boxes: torch.Tensor,
        arrow_boxes: torch.Tensor,
        tl_round: torch.Tensor,
        tl_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
        arrow_ego_lane: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pair bias for per-query retrieved arrows.

        Args:
            tl_boxes: [B, K_TL, 4]
            arrow_boxes: [B, K_TL, M, 4]
            tl_round: [B, K_TL]
            tl_maneuver: [B, K_TL, 3]
            arrow_maneuver: [B, K_TL, M, 3]
            arrow_ego_lane: [B, K_TL, M]
        Returns:
            bias: [B, H, K_TL, M]
        """
        tl_center = tl_boxes[:, :, None, :2]  # [B, K_TL, 1, 2]
        ar_center = arrow_boxes[:, :, :, :2]  # [B, K_TL, M, 2]
        delta = ar_center - tl_center         # [B, K_TL, M, 2]

        tl_size = tl_boxes[:, :, None, 2:].clamp_min(1e-6)
        ar_size = arrow_boxes[:, :, :, 2:].clamp_min(1e-6)
        log_ratio = torch.log(tl_size / ar_size)  # [B, K_TL, M, 2]

        ego = arrow_ego_lane[:, :, :, None]  # [B, K_TL, M, 1]

        # Semantic compatibility
        directional = (
            tl_maneuver[:, :, None, :] * arrow_maneuver
        ).sum(-1, keepdim=True) / tl_maneuver.sum(-1, keepdim=True)[:, :, None].clamp_min(1e-6)
        wildcard = self.round_wildcard_logit.sigmoid()
        round_p = tl_round[:, :, None, None]
        compatibility = round_p * wildcard + (1.0 - round_p) * directional  # [B, K_TL, M, 1]

        geom = torch.cat((delta, log_ratio, ego, compatibility), dim=-1)  # [B, K_TL, M, 6]
        bias = self.geometry_bias(geom)  # [B, K_TL, M, H]
        return bias.permute(0, 3, 1, 2)  # [B, H, K_TL, M]

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
        retrieval_indices: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute cross-attention per query over Top-M retrieved arrows.

        Args:
            traffic_tokens: [B, K_TL, D]
            arrow_tokens: [B, K_Arrow, D]
            traffic_boxes: [B, K_TL, 4]
            arrow_boxes: [B, K_Arrow, 4]
            traffic_round: [B, K_TL]
            traffic_maneuver: [B, K_TL, 3]
            arrow_maneuver: [B, K_Arrow, 3]
            arrow_ego_lane: [B, K_Arrow]
            arrow_valid: [B, K_Arrow]
            retrieval_indices: [B, K_TL, M]
        Returns:
            conditioned_tokens: [B, K_TL, D]
            attention_weights: [B, H, K_TL, M + 1]
            bias: [B, H, K_TL, M + 1]
        """
        B, K_TL, D = traffic_tokens.shape
        M = min(self.top_m, retrieval_indices.shape[-1])

        # Gather per-query arrow representations:
        # Expand indices for gather: [B, K_TL, M, D]
        idx_tokens = retrieval_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        arrow_tokens_exp = arrow_tokens.unsqueeze(1).expand(-1, K_TL, -1, -1)
        selected_arrow_tokens = torch.gather(arrow_tokens_exp, 2, idx_tokens)  # [B, K_TL, M, D]

        # Gather per-query arrow boxes:
        idx_boxes = retrieval_indices.unsqueeze(-1).expand(-1, -1, -1, 4)
        arrow_boxes_exp = arrow_boxes.unsqueeze(1).expand(-1, K_TL, -1, -1)
        selected_arrow_boxes = torch.gather(arrow_boxes_exp, 2, idx_boxes)  # [B, K_TL, M, 4]

        # Gather per-query arrow maneuvers:
        idx_man = retrieval_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        arrow_man_exp = arrow_maneuver.unsqueeze(1).expand(-1, K_TL, -1, -1)
        selected_arrow_maneuvers = torch.gather(arrow_man_exp, 2, idx_man)  # [B, K_TL, M, 3]

        # Gather per-query arrow ego lane:
        arrow_ego_exp = arrow_ego_lane.unsqueeze(1).expand(-1, K_TL, -1)
        selected_arrow_ego = torch.gather(arrow_ego_exp, 2, retrieval_indices)  # [B, K_TL, M]

        # Gather per-query arrow valid:
        arrow_valid_exp = arrow_valid.unsqueeze(1).expand(-1, K_TL, -1)
        selected_arrow_valid = torch.gather(arrow_valid_exp, 2, retrieval_indices)  # [B, K_TL, M]

        # Append Null Token to keys/values: [B, K_TL, M + 1, D]
        null = self.null_token.expand(B, K_TL, 1, -1)
        keys_values = torch.cat((selected_arrow_tokens, null), dim=2)  # [B, K_TL, M + 1, D]

        # Query projection: [B, H, K_TL, 1, d_h]
        query = self.query(traffic_tokens).reshape(
            B, K_TL, self.heads, self.head_dimension
        ).permute(0, 2, 1, 3).unsqueeze(3)  # [B, H, K_TL, 1, d_h]

        # Key & Value projections: [B, H, K_TL, M + 1, d_h]
        key = self.key(keys_values).reshape(
            B, K_TL, M + 1, self.heads, self.head_dimension
        ).permute(0, 3, 1, 2, 4)  # [B, H, K_TL, M + 1, d_h]

        value = self.value(keys_values).reshape(
            B, K_TL, M + 1, self.heads, self.head_dimension
        ).permute(0, 3, 1, 2, 4)  # [B, H, K_TL, M + 1, d_h]

        # Compute raw attention logits: [B, H, K_TL, 1, M + 1] -> squeeze dim 3 -> [B, H, K_TL, M + 1]
        logits = torch.matmul(query, key.transpose(-2, -1)).squeeze(3) / (self.head_dimension ** 0.5)

        # Pair bias over M retrieved arrows
        pair_bias = self._pair_bias_query(
            traffic_boxes,
            selected_arrow_boxes,
            traffic_round,
            traffic_maneuver,
            selected_arrow_maneuvers,
            selected_arrow_ego,
        )  # [B, H, K_TL, M]
        null_bias = pair_bias.new_zeros((B, self.heads, K_TL, 1))
        full_bias = torch.cat((pair_bias, null_bias), dim=-1)  # [B, H, K_TL, M + 1]
        logits = logits + full_bias

        # Mask invalid arrows
        null_valid = torch.ones((B, K_TL, 1), dtype=torch.bool, device=selected_arrow_valid.device)
        key_valid = torch.cat((selected_arrow_valid.bool(), null_valid), dim=-1).unsqueeze(1)  # [B, 1, K_TL, M + 1]
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)

        weights = logits.softmax(dim=-1)  # [B, H, K_TL, M + 1]

        # Weighted value sum: [B, H, K_TL, 1, M + 1] @ [B, H, K_TL, M + 1, d_h] -> [B, H, K_TL, 1, d_h]
        attended = torch.matmul(weights.unsqueeze(3), value).squeeze(3)  # [B, H, K_TL, d_h]
        attended = attended.permute(0, 2, 1, 3).reshape(B, K_TL, self.dimension)  # [B, K_TL, D]
        attended = self.output(attended)

        residual = self.gate.to(attended.dtype) * attended if enabled else attended * 0.0
        return self.normalization(traffic_tokens + residual), weights, full_bias


class QueryConditionedUnifiedDetect(UnifiedTrafficControlDetect):
    """Unified detector with Top-M Query-Conditioned Road Arrow Selection (Ticket E24)."""

    def __init__(
        self,
        base: Detect,
        *,
        config: UnifiedHeadConfig | None = None,
        top_m: int = 8,
    ) -> None:
        super().__init__(base, config=config)
        self.top_m = int(top_m)
        self.arrow_matcher = QueryConditionedArrowMatcher(
            token_dim=self.head_config.token_dim,
            hidden_dim=64,
        )
        self.cross_attention = QueryConditionedCrossAttention(
            dimension=self.head_config.token_dim,
            heads=self.head_config.attention_heads,
            top_m=self.top_m,
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

        # 1. E24 Query-Conditioned Top-M Matching & Retrieval
        matching_scores = self.arrow_matcher(
            traffic_tokens=traffic_tokens,
            arrow_tokens=arrow_tokens,
            traffic_boxes=traffic_boxes_source,
            arrow_boxes=arrow_boxes_source,
            arrow_scores=arrow_score_source,
            arrow_valid=arrow_valid,
        )  # [B, K_TL, K_Arrow]

        M = min(self.top_m, arrow_tokens.shape[1])
        retrieval_scores, retrieval_indices = matching_scores.topk(
            M, dim=-1, largest=True, sorted=True
        )  # [B, K_TL, M]

        # 2. Query-Conditioned Cross Attention
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
            retrieval_indices=retrieval_indices,
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
            "traffic_tokens": traffic_tokens,
            "arrow_tokens": arrow_tokens,
            "traffic_candidate_round": traffic_round_source,
            "traffic_candidate_maneuver": traffic_maneuver_source,
            "arrow_candidate_maneuver": arrow_maneuver_source,
            "arrow_matching_scores": matching_scores,
            "retrieval_indices": retrieval_indices,
            "retrieval_scores": retrieval_scores,
            "local_relevance_logits": selected_local_relevance,
            "relevance_logits": relevance,
            "attention_weights": attention,
            "attention_geometry_bias": geometry_bias,
        }


def attach_query_conditioned_unified_relevance_head(
    model_wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
    top_m: int = 8,
) -> QueryConditionedUnifiedDetect:
    """Attach QueryConditionedUnifiedDetect in place of final Detect module."""
    base = model_wrapper.model.model[-1]
    if isinstance(base, QueryConditionedUnifiedDetect):
        return base
    if isinstance(base, (Detect, UnifiedTrafficControlDetect)):
        base_detect = base
    else:
        raise TypeError(f"expected Detect or UnifiedTrafficControlDetect, got {type(base).__name__}")

    unified = QueryConditionedUnifiedDetect(
        base_detect, config=config, top_m=top_m
    )
    model_wrapper.model.model[-1] = unified
    return unified
