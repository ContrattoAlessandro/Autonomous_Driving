"""Local+ (Capacity-Matched) Relevance Architecture for TLR-YOLO-MTL.

This module implements the parameter-matched Local+ baseline for Ticket E16.
Local+ adds a deep Residual MLP branch directly to the traffic-light candidate
features (visual token f_64, box position encoding PE_32, state, round, maneuver,
score -> 101 dims) without consuming any road arrow tokens or cross-attention.

Parameter parity:
- Cross-Attention Context Branch: 127,655 parameters
- Local+ Residual MLP Branch:     127,617 parameters (99.97% parity, Delta = -38)
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

from .attributes import STATE_CLASSES, STATE_TO_INDEX, _AttributeTower
from .unified import (
    ROAD_ARROW_CLASS,
    TRAFFIC_LIGHT_CLASS,
    TYPE_CLASSES,
    MANEUVER_CLASSES,
    DEFAULT_TOKEN_DIM,
    DEFAULT_TOKEN_FEATURE_DIM,
    DEFAULT_MAX_TRAFFIC_LIGHTS,
    DEFAULT_MAX_ARROWS,
    DEFAULT_ARROW_SCORE_THRESHOLD,
    DEFAULT_TRAFFIC_SCORE_THRESHOLD,
    BoxPositionEncoding,
    UnifiedHeadConfig,
    _gather_dense,
    fixed_topk_candidates,
    scale_relevance_gradient,
)


class LocalPlusResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm and SiLU activations.
    
    Contains 2 x Linear(dim, dim) + 2 x LayerNorm(dim), yielding
    2 * (dim * dim + dim) + 2 * (2 * dim) parameters (33,536 for dim=128).
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(self.fc1(x)))
        out = self.act2(self.norm2(self.fc2(out)))
        return out + residual


class LocalPlusRelevanceBranch(nn.Module):
    """Capacity-matched Local+ relevance processing branch.
    
    Architecture:
    1. BoxPositionEncoding(32):                       1,216 params
    2. Input Projection: Linear(101, 128) + LN(128): 13,312 params
    3. 3 x LocalPlusResidualBlock(128):             100,608 params
    4. Relevance Head: Linear(128, 96) + Linear(96, 1): 12,481 params
    Total:                                          127,617 params
    """

    def __init__(
        self,
        token_feature_dim: int = 64,
        position_dim: int = 32,
        hidden_dim: int = 128,
        head_hidden_dim: int = 96,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.token_feature_dim = token_feature_dim
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.position_encoding = BoxPositionEncoding(position_dim)
        # 64 (feature) + 32 (pos) + 1 (round) + 3 (maneuver) + 1 (score) = 101
        token_input = token_feature_dim + position_dim + 5
        self.input_projection = nn.Sequential(
            nn.Linear(token_input, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_blocks = nn.ModuleList(
            [LocalPlusResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(head_hidden_dim, 1),
        )
        # Scalar gate initialized to zero for exact identity/neutral fallback at step 0
        self.gate = nn.Parameter(torch.zeros(1))

    def count_parameters(self) -> dict[str, int]:
        """Return breakdown of parameters across submodules."""
        return {
            "position_encoding": sum(p.numel() for p in self.position_encoding.parameters()),
            "input_projection": sum(p.numel() for p in self.input_projection.parameters()),
            "residual_blocks": sum(p.numel() for p in self.residual_blocks.parameters()),
            "relevance_head": sum(p.numel() for p in self.relevance_head.parameters()),
            "gate": self.gate.numel(),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def forward(
        self,
        *,
        traffic_features: torch.Tensor,
        traffic_boxes: torch.Tensor,
        traffic_round: torch.Tensor,
        traffic_maneuver: torch.Tensor,
        traffic_scores: torch.Tensor,
        use_gate: bool = True,
    ) -> torch.Tensor:
        """Compute Local+ relevance delta for traffic-light candidate tokens.
        
        Args:
            traffic_features: [B, K_TL, 64]
            traffic_boxes: [B, K_TL, 4] normalized cx, cy, w, h
            traffic_round: [B, K_TL]
            traffic_maneuver: [B, K_TL, 3]
            traffic_scores: [B, K_TL]
            use_gate: whether to scale output by self.gate

        Returns:
            delta: [B, 1, K_TL] relevance logit delta
        """
        traffic_position = self.position_encoding(traffic_boxes)
        traffic_source = torch.cat(
            (
                traffic_features,
                traffic_position,
                traffic_round[..., None],
                traffic_maneuver,
                traffic_scores[..., None],
            ),
            dim=-1,
        )
        x = self.input_projection(traffic_source)
        for block in self.residual_blocks:
            x = block(x)
        raw_delta = self.relevance_head(x).transpose(1, 2)  # [B, 1, K_TL]
        if use_gate:
            return self.gate.to(raw_delta.dtype) * raw_delta
        return raw_delta


class LocalPlusTrafficControlDetect(Detect):
    """Two-type detector with factorized attributes and Capacity-Matched Local+ relevance."""

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
        self.maneuver_heads = nn.ModuleList(_AttributeTower(value, 3) for value in channels)
        self.ego_lane_heads = nn.ModuleList(_AttributeTower(value, 1) for value in channels)
        self.local_relevance_heads = nn.ModuleList(_AttributeTower(value, 1) for value in channels)
        self.token_feature_heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(value, self.head_config.token_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.head_config.token_feature_dim),
                nn.SiLU(inplace=True),
            )
            for value in channels
        )

        self.local_plus_branch = LocalPlusRelevanceBranch(
            token_feature_dim=self.head_config.token_feature_dim,
            position_dim=32,
            hidden_dim=self.head_config.token_dim,
            head_hidden_dim=96,
            num_blocks=3,
        )

        self.attribute_channels = channels
        self.local_plus_enabled = True
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

    def set_local_plus_enabled(self, enabled: bool) -> None:
        self.local_plus_enabled = bool(enabled)

    def set_perception_gradient_scale(self, scales: float | torch.Tensor) -> None:
        self._perception_gradient_scale = scales

    def local_plus_parameters(self) -> list[nn.Parameter]:
        return list(self.local_plus_branch.parameters())

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
        if self.export:
            traffic_boxes_source = traffic_boxes
            traffic_feature_source = traffic_features
            traffic_round_source = traffic_round
            traffic_maneuver_source = traffic_maneuver
            traffic_score_source = traffic_scores
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

        if self.local_plus_enabled:
            local_plus_delta = self.local_plus_branch(
                traffic_features=traffic_feature_source,
                traffic_boxes=traffic_boxes_source,
                traffic_round=traffic_round_source,
                traffic_maneuver=traffic_maneuver_source,
                traffic_scores=traffic_score_source,
                use_gate=True,
            )
            relevance = selected_local_relevance + local_plus_delta
        else:
            relevance = selected_local_relevance
            local_plus_delta = selected_local_relevance * 0.0

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
            "local_plus_delta": local_plus_delta,
            "relevance_logits": relevance,
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
                    "local_plus_enabled_flag": decoded.new_tensor(
                        1.0 if self.local_plus_enabled else 0.0
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
                candidate_outputs["relevance_logits"],
            )
        raw.update(
            {
                "state_logits": state_logits,
                "round_logits": round_logits,
                "maneuver_logits": maneuver_logits,
                "ego_lane_logits": ego_lane_logits,
                "dense_local_relevance_logits": dense_local_relevance_logits,
                "token_features": token_features,
                "local_plus_enabled_flag": decoded.new_tensor(
                    1.0 if self.local_plus_enabled else 0.0
                ),
                **candidate_outputs,
            }
        )
        return decoded, raw


def attach_local_plus_relevance_head(
    wrapper: Any,
    *,
    config: UnifiedHeadConfig | None = None,
) -> Any:
    """Replace the final Detect module with LocalPlusTrafficControlDetect."""
    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    base = model.model[-1]
    if isinstance(base, LocalPlusTrafficControlDetect):
        return wrapper
    if not isinstance(base, Detect):
        raise TypeError(f"expected an Ultralytics Detect head, got {type(base)!r}")
    head = LocalPlusTrafficControlDetect(base, config=config)
    model.model[-1] = head
    model.stride = head.stride
    return wrapper
