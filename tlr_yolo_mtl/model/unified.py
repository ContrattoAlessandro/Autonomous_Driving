"""Unified traffic-control detector and lane-aware TL-to-arrow attention.

The active architecture uses one two-class YOLO head for traffic lights and
road arrows.  State, roundness and maneuver are factorized attributes attached
to the same dense candidates.  Relevance is produced only for a fixed padded
set of traffic-light candidates by one gated cross-attention block over a
fixed padded set of arrow candidates plus a learned null token.

Candidate selection is intentionally top-k and therefore the model is jointly
trainable, not fully differentiable with respect to the selected indices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect

from ..data.schema import ImageRecord, TrafficLightAnnotation
from ..data.taxonomy import factor_pictogram, normalize_label
from .attributes import STATE_CLASSES, STATE_TO_INDEX, _AttributeTower

TYPE_CLASSES = ("traffic_light", "road_arrow")
TRAFFIC_LIGHT_CLASS = 0
ROAD_ARROW_CLASS = 1
MANEUVER_CLASSES = ("left", "straight", "right")

DEFAULT_TOKEN_DIM = 128
DEFAULT_TOKEN_FEATURE_DIM = 64
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_MAX_TRAFFIC_LIGHTS = 32
DEFAULT_MAX_ARROWS = 16
DEFAULT_ARROW_SCORE_THRESHOLD = 0.05
DEFAULT_TRAFFIC_SCORE_THRESHOLD = 0.01


@dataclass(frozen=True, slots=True)
class UnifiedHeadConfig:
    token_dim: int = DEFAULT_TOKEN_DIM
    token_feature_dim: int = DEFAULT_TOKEN_FEATURE_DIM
    attention_heads: int = DEFAULT_ATTENTION_HEADS
    max_traffic_lights: int = DEFAULT_MAX_TRAFFIC_LIGHTS
    max_arrows: int = DEFAULT_MAX_ARROWS
    traffic_score_threshold: float = DEFAULT_TRAFFIC_SCORE_THRESHOLD
    arrow_score_threshold: float = DEFAULT_ARROW_SCORE_THRESHOLD
    ego_lane_enabled: bool = False

    def validate(self) -> None:
        if self.token_dim <= 0 or self.token_feature_dim <= 0:
            raise ValueError("token dimensions must be positive")
        if self.attention_heads <= 0 or self.token_dim % self.attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads")
        if self.max_traffic_lights <= 0 or self.max_arrows <= 0:
            raise ValueError("candidate-set sizes must be positive")
        for name, value in (
            ("traffic_score_threshold", self.traffic_score_threshold),
            ("arrow_score_threshold", self.arrow_score_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def scale_relevance_gradient(
    values: torch.Tensor, scales: float | torch.Tensor
) -> torch.Tensor:
    """Preserve forward values while scaling gradients into perception.

    Projection and attention weights remain trainable when ``scales`` is zero;
    only the upstream tensor producer is protected.
    """

    if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
        return values
    scale = (
        scales.to(device=values.device, dtype=values.dtype).reshape(-1)
        if isinstance(scales, torch.Tensor)
        else values.new_full((1,), float(scales))
    )
    if scale.numel() == 1:
        scale = scale.expand(values.shape[0])
    if scale.numel() != values.shape[0]:
        raise ValueError("gradient scale must be scalar or one value per image")
    if not torch.onnx.is_in_onnx_export() and (
        torch.any(scale < 0) or torch.any(scale > 1)
    ):
        raise ValueError("gradient scales must be in [0, 1]")
    shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    scale = scale.reshape(shape)
    detached = values.detach()
    return detached + scale * (values - detached)


def _gather_dense(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or indices.ndim != 2:
        raise ValueError("dense values and indices must have shapes [B,C,A] and [B,K]")
    if values.shape[0] != indices.shape[0]:
        raise ValueError("dense values and indices have different batch sizes")
    if indices.numel() and not torch.onnx.is_in_onnx_export() and (
        torch.any(indices < 0) or torch.any(indices >= values.shape[2])
    ):
        raise ValueError("candidate index is outside the dense tensor")
    return values.gather(2, indices[:, None, :].expand(-1, values.shape[1], -1))


def fixed_topk_candidates(
    scores: torch.Tensor,
    k: int,
    *,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-width top-k indices, scores and a padding/score mask."""

    if scores.ndim != 2:
        raise ValueError("candidate scores must have shape [batch, anchors]")
    if k <= 0:
        raise ValueError("k must be positive")
    count = min(k, scores.shape[1])
    selected_scores, indices = scores.topk(count, dim=1, largest=True, sorted=True)
    valid = selected_scores >= threshold
    if count < k:
        padding = k - count
        indices = F.pad(indices, (0, padding), value=0)
        selected_scores = F.pad(selected_scores, (0, padding), value=0.0)
        valid = F.pad(valid, (0, padding), value=False)
    return indices, selected_scores, valid


class BoxPositionEncoding(nn.Module):
    """Learned encoding of normalized ``cx, cy, width, height`` geometry."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, dimension),
            nn.SiLU(inplace=True),
            nn.Linear(dimension, dimension),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.network(boxes)


class GatedLaneAwareCrossAttention(nn.Module):
    """One TL-query/arrow-key-value attention block with learned pair bias."""

    def __init__(self, dimension: int = 128, heads: int = 4) -> None:
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
        self.geometry_bias = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, heads),
        )
        # Neutral at initialization: learned geometry and cross-attention cannot
        # perturb the local-only path before receiving evidence from training.
        nn.init.zeros_(self.geometry_bias[-1].weight)
        nn.init.zeros_(self.geometry_bias[-1].bias)
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

    def _pair_bias(
        self,
        tl_boxes: torch.Tensor,
        arrow_boxes: torch.Tensor,
        tl_round: torch.Tensor,
        tl_maneuver: torch.Tensor,
        arrow_maneuver: torch.Tensor,
        arrow_ego_lane: torch.Tensor,
    ) -> torch.Tensor:
        tl_center = tl_boxes[:, :, None, :2]
        arrow_center = arrow_boxes[:, None, :, :2]
        delta = arrow_center - tl_center
        tl_size = tl_boxes[:, :, None, 2:].clamp_min(1e-6)
        arrow_size = arrow_boxes[:, None, :, 2:].clamp_min(1e-6)
        log_ratio = torch.log(tl_size / arrow_size)
        ego = arrow_ego_lane[:, None, :, None].expand(
            -1, tl_boxes.shape[1], -1, -1
        )
        compatibility = self._semantic_compatibility(
            tl_round, tl_maneuver, arrow_maneuver
        )[..., None]
        geometry = torch.cat((delta, log_ratio, ego, compatibility), dim=-1)
        return self.geometry_bias(geometry).permute(0, 3, 1, 2)

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
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, traffic_count, _ = traffic_tokens.shape
        arrow_count = arrow_tokens.shape[1]
        if arrow_valid.shape != (batch, arrow_count):
            raise ValueError("arrow_valid shape does not match arrow tokens")

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
        logits = torch.matmul(query, key.transpose(-2, -1)) / self.head_dimension**0.5
        pair_bias = self._pair_bias(
            traffic_boxes,
            arrow_boxes,
            traffic_round,
            traffic_maneuver,
            arrow_maneuver,
            arrow_ego_lane,
        )
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


class UnifiedTrafficControlDetect(Detect):
    """Two-type detector with conditional attributes and TL→arrow relevance."""

    def __init__(
        self,
        base: Detect,
        *,
        config: UnifiedHeadConfig | None = None,
    ) -> None:
        self.head_config = config or UnifiedHeadConfig()
        self.head_config.validate()
        if int(base.nc) != len(TYPE_CLASSES):
            raise ValueError(
                f"unified detector requires nc={len(TYPE_CLASSES)}, got {base.nc}"
            )
        channels = tuple(branch[0].conv.in_channels for branch in base.cv2)
        super().__init__(
            nc=base.nc,
            reg_max=base.reg_max,
            end2end=base.end2end,
            ch=channels,
        )
        self.cv2 = base.cv2
        self.cv3 = base.cv3
        self.dfl = base.dfl
        if base.end2end:
            self.one2one_cv2 = base.one2one_cv2
            self.one2one_cv3 = base.one2one_cv3
        self.stride = base.stride
        self.anchors = base.anchors
        self.strides = base.strides
        self.legacy = base.legacy
        self.dynamic = base.dynamic
        self.xyxy = base.xyxy

        self.state_heads = nn.ModuleList(_AttributeTower(value, 4) for value in channels)
        self.round_heads = nn.ModuleList(_AttributeTower(value, 1) for value in channels)
        # This branch is shared by directional traffic lights and road arrows.
        self.maneuver_heads = nn.ModuleList(
            _AttributeTower(value, 3) for value in channels
        )
        self.ego_lane_heads = nn.ModuleList(
            _AttributeTower(value, 1) for value in channels
        )
        self.local_relevance_heads = nn.ModuleList(
            _AttributeTower(value, 1) for value in channels
        )
        self.token_feature_heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(value, self.head_config.token_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.head_config.token_feature_dim),
                nn.SiLU(inplace=True),
            )
            for value in channels
        )
        position_dim = 32
        self.position_encoding = BoxPositionEncoding(position_dim)
        token_input = self.head_config.token_feature_dim + position_dim + 5
        self.traffic_token_projection = nn.Sequential(
            nn.Linear(token_input, self.head_config.token_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(self.head_config.token_dim),
        )
        self.arrow_token_projection = nn.Sequential(
            nn.Linear(token_input, self.head_config.token_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(self.head_config.token_dim),
        )
        self.cross_attention = GatedLaneAwareCrossAttention(
            self.head_config.token_dim, self.head_config.attention_heads
        )
        self.relevance_head = nn.Sequential(
            nn.Linear(2 * self.head_config.token_dim, self.head_config.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.head_config.token_dim, 1),
        )
        self.attribute_channels = channels
        self.attention_enabled = True
        self._context_gradient_scale: float | torch.Tensor = 0.0
        self._perception_gradient_scale: float | torch.Tensor = 0.0

        for name in ("i", "f", "type", "np"):
            if hasattr(base, name):
                setattr(self, name, getattr(base, name))
        self.train(base.training)

    @staticmethod
    def _flatten(maps: Sequence[torch.Tensor]) -> torch.Tensor:
        batch = maps[0].shape[0]
        return torch.cat(
            [value.reshape(batch, value.shape[1], -1) for value in maps], dim=-1
        )

    def set_attention_enabled(self, enabled: bool) -> None:
        self.attention_enabled = bool(enabled)

    def set_context_gradient_scale(self, scales: float | torch.Tensor) -> None:
        self._context_gradient_scale = scales

    def set_perception_gradient_scale(self, scales: float | torch.Tensor) -> None:
        self._perception_gradient_scale = scales

    def context_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.position_encoding,
            self.traffic_token_projection,
            self.arrow_token_projection,
            self.cross_attention,
            self.relevance_head,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def _normalized_boxes(
        self, decoded_boxes: torch.Tensor, features: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        unit = decoded_boxes[:, :1, :1] * 0.0 + 1.0
        height = unit * features[0].shape[-2] * self.stride[0]
        width = unit * features[0].shape[-1] * self.stride[0]
        if self.xyxy:
            center = (decoded_boxes[:, :2] + decoded_boxes[:, 2:]) / 2
            size = (decoded_boxes[:, 2:] - decoded_boxes[:, :2]).clamp_min(0)
            boxes = torch.cat((center, size), dim=1)
        else:
            boxes = decoded_boxes
        normalization = torch.cat((width, height, width, height), dim=1)
        return (boxes / normalization).clamp(0.0, 1.0)

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
        type_scores = decoded[:, 4 : 4 + len(TYPE_CLASSES)]
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
            # Gradient controls have no inference semantics and legacy ONNX
            # tracing otherwise promotes their scalar constants to float64.
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

    def forward(
        self, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | tuple[Any, ...] | torch.Tensor:
        state_logits = self._flatten(
            [head(value) for head, value in zip(self.state_heads, features)]
        )
        round_logits = self._flatten(
            [head(value) for head, value in zip(self.round_heads, features)]
        )
        maneuver_logits = self._flatten(
            [head(value) for head, value in zip(self.maneuver_heads, features)]
        )
        ego_lane_logits = self._flatten(
            [head(value) for head, value in zip(self.ego_lane_heads, features)]
        )
        local_relevance_features = (
            features
            if self.export
            else [
                scale_relevance_gradient(value, self._perception_gradient_scale)
                for value in features
            ]
        )
        dense_local_relevance_logits = self._flatten(
            [
                head(value)
                for head, value in zip(
                    self.local_relevance_heads, local_relevance_features
                )
            ]
        )
        token_features = self._flatten(
            [head(value) for head, value in zip(self.token_feature_heads, features)]
        )
        detection = super().forward(features)
        if self.training:
            if not isinstance(detection, dict):
                raise TypeError("training Detect output must be a dictionary")
            raw = detection
            decoded = self._inference(raw)
        elif self.export:
            if not torch.is_tensor(detection):
                raise TypeError("export Detect output must be a tensor")
            decoded = detection
            raw = None
        else:
            decoded, raw = detection

        candidate_outputs = self._build_tokens(
            decoded=decoded,
            token_features=token_features,
            round_logits=round_logits,
            maneuver_logits=maneuver_logits,
            ego_lane_logits=ego_lane_logits,
            dense_local_relevance_logits=dense_local_relevance_logits,
            features=features,
        )
        if self.training:
            raw.update(
                {
                    "state_logits": state_logits,
                    "round_logits": round_logits,
                    "maneuver_logits": maneuver_logits,
                    "ego_lane_logits": ego_lane_logits,
                    "dense_local_relevance_logits": dense_local_relevance_logits,
                    "token_features": token_features,
                    "attention_enabled_flag": decoded.new_tensor(
                        1.0 if self.attention_enabled else 0.0
                    ),
                    **candidate_outputs,
                }
            )
            return raw
        if self.export:
            return (
                decoded,
                state_logits,
                round_logits,
                maneuver_logits,
                ego_lane_logits,
                candidate_outputs["traffic_candidate_indices"],
                candidate_outputs["traffic_candidate_valid"],
                candidate_outputs["arrow_candidate_indices"],
                candidate_outputs["arrow_candidate_valid"],
                candidate_outputs["relevance_logits"],
                candidate_outputs["attention_weights"],
            )
        raw.update(
            {
                "state_logits": state_logits,
                "round_logits": round_logits,
                "maneuver_logits": maneuver_logits,
                "ego_lane_logits": ego_lane_logits,
                "dense_local_relevance_logits": dense_local_relevance_logits,
                "token_features": token_features,
                "attention_enabled_flag": decoded.new_tensor(
                    1.0 if self.attention_enabled else 0.0
                ),
                **candidate_outputs,
            }
        )
        return decoded, raw


def attach_unified_relevance_head(
    wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
) -> Any:
    """Replace the final two-class Detect module with the active unified head."""

    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    base = model.model[-1]
    if isinstance(base, UnifiedTrafficControlDetect):
        return wrapper
    if not isinstance(base, Detect):
        raise TypeError(f"expected an Ultralytics Detect head, got {type(base)!r}")
    head = UnifiedTrafficControlDetect(base, config=config)
    model.model[-1] = head
    model.stride = head.stride
    return wrapper


def _unified_head(wrapper: Any) -> UnifiedTrafficControlDetect:
    if isinstance(wrapper, UnifiedTrafficControlDetect):
        return wrapper
    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    if isinstance(model, UnifiedTrafficControlDetect):
        return model
    if hasattr(model, "model") and isinstance(model.model, (nn.Sequential, list, tuple)):
        head = model.model[-1]
    elif isinstance(model, (nn.Sequential, list, tuple)):
        head = model[-1]
    else:
        head = model
    if not isinstance(head, UnifiedTrafficControlDetect):
        raise TypeError("the model does not contain the unified relevance head")
    return head


def set_cross_attention_enabled(wrapper: Any, enabled: bool) -> None:
    _unified_head(wrapper).set_attention_enabled(enabled)


def set_context_gradient_scale(
    wrapper: Any, scales: float | torch.Tensor
) -> None:
    _unified_head(wrapper).set_context_gradient_scale(scales)


def set_relevance_perception_gradient_scale(
    wrapper: Any, scales: float | torch.Tensor
) -> None:
    _unified_head(wrapper).set_perception_gradient_scale(scales)


def _factorized_tl_targets(
    item: TrafficLightAnnotation,
) -> tuple[int, tuple[int, int, int] | None, bool, bool]:
    if item.valid_round or item.valid_maneuver:
        return (
            int(item.round_target or 0),
            item.maneuver_multihot,
            item.valid_round,
            item.valid_maneuver,
        )
    raw = item.pictogram
    if raw is None:
        raw = item.source_attributes.get("pictogram")
    factorized = factor_pictogram(raw)
    return (
        int(factorized.round or 0),
        factorized.maneuver,
        factorized.valid_round,
        factorized.valid_maneuver,
    )


def _ego_lane_target(value: Any) -> int | None:
    label = normalize_label(value)
    if label in {"1", "true", "yes", "relevant", "ego", "ego_lane"}:
        return 1
    if label in {"0", "false", "no", "irrelevant", "other", "other_lane"}:
        return 0
    return None


def encode_record_unified(record: ImageRecord) -> dict[str, torch.Tensor]:
    """Encode both object types in one GT order for one TaskAlignedAssigner."""

    boxes = [item.bbox_xyxy for item in record.traffic_lights]
    boxes.extend(item.bbox_xyxy for item in record.road_arrows)
    classes = [TRAFFIC_LIGHT_CLASS] * len(record.traffic_lights)
    classes.extend([ROAD_ARROW_CLASS] * len(record.road_arrows))
    states: list[int] = []
    rounds: list[float] = []
    maneuvers: list[tuple[float, float, float]] = []
    relevance: list[int] = []
    ego_lane: list[float] = []

    for item in record.traffic_lights:
        states.append(
            STATE_TO_INDEX[item.state]
            if item.valid_state and item.state in STATE_TO_INDEX
            else -1
        )
        round_target, maneuver, valid_round, valid_maneuver = _factorized_tl_targets(item)
        rounds.append(float(round_target) if valid_round else -1.0)
        maneuvers.append(
            tuple(float(value) for value in maneuver)
            if valid_maneuver and maneuver is not None
            else (-1.0, -1.0, -1.0)
        )
        relevance.append(
            int(item.relevance)
            if item.valid_relevance and item.relevance in (0, 1)
            else -1
        )
        ego_lane.append(-1.0)

    for item in record.road_arrows:
        states.append(-1)
        rounds.append(-1.0)
        maneuvers.append(tuple(float(value) for value in item.direction_multihot))
        relevance.append(-1)
        target = item.is_ego_lane if item.valid_ego_lane else None
        if target is None:
            target = _ego_lane_target(
                item.source_attributes.get(
                    "is_ego_lane", item.source_attributes.get("arrow_relevance")
                )
            )
        ego_lane.append(float(target) if target in (0, 1) else -1.0)

    return {
        "object_boxes_xyxy": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "object_cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
        "object_state": torch.tensor(states, dtype=torch.long),
        "object_round": torch.tensor(rounds, dtype=torch.float32),
        "object_maneuver": torch.tensor(maneuvers, dtype=torch.float32).reshape(-1, 3),
        "object_relevance": torch.tensor(relevance, dtype=torch.long),
        "object_ego_lane": torch.tensor(ego_lane, dtype=torch.float32),
        "unified_detection_valid": torch.tensor(
            record.task_valid.traffic_light_detection
            and record.task_valid.arrow_detection,
            dtype=torch.bool,
        ),
        "traffic_relevance_valid": torch.tensor(
            record.task_valid.traffic_light_relevance, dtype=torch.bool
        ),
    }


def gather_candidate_outputs(
    values: torch.Tensor, candidate_indices: torch.Tensor
) -> torch.Tensor:
    """Public candidate gather used by deployment and tests."""

    return _gather_dense(values, candidate_indices)


def run_unified_forward_smoke(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (800, 1600),
    device: str = "cuda",
    half: bool = True,
) -> dict[str, Any]:
    """Validate shapes, fixed candidate sets and neutral initialization."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = wrapper.model.to(resolved).eval()
    use_half = bool(half and resolved.type == "cuda")
    model = model.half() if use_half else model.float()
    head = model.model[-1]
    if not isinstance(head, UnifiedTrafficControlDetect):
        raise TypeError("unified smoke requires UnifiedTrafficControlDetect")
    height, width = input_size
    dtype = torch.float16 if use_half else torch.float32
    sample = torch.zeros((1, 3, height, width), device=resolved, dtype=dtype)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    with torch.inference_mode():
        decoded, raw = model(sample)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    locations = sum(
        (height // int(stride)) * (width // int(stride)) for stride in head.stride
    )
    expected = {
        "detection": (1, 6, locations),
        "state": (1, 4, locations),
        "round": (1, 1, locations),
        "maneuver": (1, 3, locations),
        "ego_lane": (1, 1, locations),
        "traffic_candidates": (1, head.head_config.max_traffic_lights),
        "arrow_candidates": (1, head.head_config.max_arrows),
        "relevance": (1, 1, head.head_config.max_traffic_lights),
        "attention": (
            1,
            head.head_config.attention_heads,
            head.head_config.max_traffic_lights,
            head.head_config.max_arrows + 1,
        ),
    }
    actual = {
        "detection": tuple(decoded.shape),
        "state": tuple(raw["state_logits"].shape),
        "round": tuple(raw["round_logits"].shape),
        "maneuver": tuple(raw["maneuver_logits"].shape),
        "ego_lane": tuple(raw["ego_lane_logits"].shape),
        "traffic_candidates": tuple(raw["traffic_candidate_indices"].shape),
        "arrow_candidates": tuple(raw["arrow_candidate_indices"].shape),
        "relevance": tuple(raw["relevance_logits"].shape),
        "attention": tuple(raw["attention_weights"].shape),
    }
    if actual != expected:
        raise AssertionError(f"unified shape contract failed: {actual} != {expected}")
    if not torch.allclose(raw["relevance_logits"], raw["local_relevance_logits"]):
        raise AssertionError("zero gate did not preserve local relevance")
    if not torch.allclose(
        raw["attention_weights"].sum(-1),
        torch.ones_like(raw["attention_weights"].sum(-1)),
    ):
        raise AssertionError("attention weights do not sum to one")

    groups = {
        "unified_detect": list(head.cv2.parameters()) + list(head.cv3.parameters()),
        "factorized_attributes": list(head.state_heads.parameters())
        + list(head.round_heads.parameters())
        + list(head.maneuver_heads.parameters())
        + list(head.ego_lane_heads.parameters()),
        "token_projections": list(head.token_feature_heads.parameters())
        + list(head.position_encoding.parameters())
        + list(head.traffic_token_projection.parameters())
        + list(head.arrow_token_projection.parameters()),
        "cross_attention": list(head.cross_attention.parameters()),
        "relevance": list(head.local_relevance_heads.parameters())
        + list(head.relevance_head.parameters()),
    }
    return {
        "schema": "TLR-YOLO-MTL unified attention smoke v1",
        "input_shape": list(sample.shape),
        "dtype": str(dtype).removeprefix("torch."),
        "strides": [int(value) for value in head.stride],
        "dense_locations": locations,
        "shapes": {name: list(value) for name, value in actual.items()},
        "type_classes": list(TYPE_CLASSES),
        "state_classes": list(STATE_CLASSES),
        "maneuver_classes": list(MANEUVER_CLASSES),
        "round_is_learned_wildcard": True,
        "ego_lane_bias_enabled": head.head_config.ego_lane_enabled,
        "null_token": True,
        "attention_gate_initial_value": float(head.cross_attention.gate.detach()),
        "local_fallback_exact_at_initialization": True,
        "separate_arrow_detector": False,
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "parameter_groups": {
            name: int(sum(value.numel() for value in parameters))
            for name, parameters in groups.items()
        },
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved))
            if resolved.type == "cuda"
            else None
        ),
    }
