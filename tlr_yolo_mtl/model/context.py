"""Dense arrow context and FiLM-conditioned traffic-light relevance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..data.schema import ImageRecord
from .arrows import RoadArrowMultiTaskDetect
from .relevance import (
    LocalRelevanceDetect,
    MaskedRelevanceCriterion,
    attach_local_relevance_head,
)

CONTEXT_DIRECTIONS = 3
CONTEXT_GRID = (4, 8)
CONTEXT_DIMENSION = 64
DEFAULT_PAIRED_GRADIENT_SCALE = 0.25


def build_soft_arrow_heatmaps(
    arrow_scores: torch.Tensor,
    direction_logits: torch.Tensor,
    feature_shapes: Sequence[tuple[int, int]],
    *,
    scores_are_probabilities: bool = False,
) -> list[torch.Tensor]:
    """Build differentiable pre-NMS direction maps for each pyramid level."""

    if arrow_scores.ndim != 3 or arrow_scores.shape[1] != 1:
        raise ValueError("arrow scores must have shape [batch, 1, anchors]")
    if direction_logits.ndim != 3 or direction_logits.shape[1] != 3:
        raise ValueError("direction logits must have shape [batch, 3, anchors]")
    if arrow_scores.shape[0] != direction_logits.shape[0]:
        raise ValueError("arrow score and direction batch dimensions differ")
    if arrow_scores.shape[2] != direction_logits.shape[2]:
        raise ValueError("arrow score and direction anchor dimensions differ")
    expected = sum(height * width for height, width in feature_shapes)
    if expected != arrow_scores.shape[2]:
        raise ValueError(
            f"feature maps contain {expected} locations, predictions contain "
            f"{arrow_scores.shape[2]}"
        )

    score_probabilities = (
        arrow_scores if scores_are_probabilities else arrow_scores.sigmoid()
    )
    if scores_are_probabilities and not torch.onnx.is_in_onnx_export() and (
        torch.any(score_probabilities < 0) or torch.any(score_probabilities > 1)
    ):
        raise ValueError("decoded arrow scores must be probabilities in [0, 1]")
    soft_directions = score_probabilities * direction_logits.sigmoid()
    maps: list[torch.Tensor] = []
    offset = 0
    for height, width in feature_shapes:
        locations = height * width
        maps.append(
            soft_directions[:, :, offset : offset + locations].reshape(
                soft_directions.shape[0], CONTEXT_DIRECTIONS, height, width
            )
        )
        offset += locations
    return maps


def scale_arrow_context_gradient(
    values: torch.Tensor, scales: float | torch.Tensor
) -> torch.Tensor:
    """Keep forward values intact while scaling gradients per image."""

    scale = torch.as_tensor(scales, dtype=values.dtype, device=values.device).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(values.shape[0])
    if scale.numel() != values.shape[0]:
        raise ValueError("context gradient scale must be scalar or one value per image")
    if torch.any(scale < 0) or torch.any(scale > 1):
        raise ValueError("context gradient scales must be in [0, 1]")
    scale = scale.reshape(values.shape[0], *([1] * (values.ndim - 1)))
    detached = values.detach()
    return detached + scale * (values - detached)


def encode_record_context_gradient(
    record: ImageRecord,
    *,
    paired_scale: float = DEFAULT_PAIRED_GRADIENT_SCALE,
) -> dict[str, torch.Tensor]:
    """Enable relevance-to-arrow gradients only for genuinely paired images."""

    paired = (
        record.task_valid.traffic_light_relevance
        and record.task_valid.arrow_detection
    )
    return {
        "relevance_arrow_context_scale": torch.tensor(
            paired_scale if paired else 0.0, dtype=torch.float32
        ),
        "relevance_arrow_context_paired": torch.tensor(paired, dtype=torch.bool),
    }


class ArrowContextRelevanceDetect(LocalRelevanceDetect):
    """Condition local per-TL relevance with dense soft arrow evidence."""

    def __init__(self, base: LocalRelevanceDetect) -> None:
        super().__init__(base)

        # Preserve all trained branches from Milestones 3-5.
        self.state_heads = base.state_heads
        self.pictogram_heads = base.pictogram_heads
        self.arrow_detect = base.arrow_detect
        self.arrow_direction_heads = base.arrow_direction_heads
        self.relevance_heads = base.relevance_heads
        channels = tuple(base.attribute_channels)

        self.context_fuse = nn.Conv2d(
            len(channels) * CONTEXT_DIRECTIONS,
            CONTEXT_DIRECTIONS,
            kernel_size=1,
        )
        self._initialize_direction_fusion(len(channels))
        self.context_pool = nn.AdaptiveAvgPool2d(CONTEXT_GRID)
        self.context_mlp = nn.Sequential(
            nn.Linear(CONTEXT_DIRECTIONS * CONTEXT_GRID[0] * CONTEXT_GRID[1], 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, CONTEXT_DIMENSION),
            nn.SiLU(inplace=True),
        )
        self.film_layers = nn.ModuleList(
            nn.Linear(CONTEXT_DIMENSION, 2 * value) for value in channels
        )
        self.relevance_aux_projections = nn.ModuleList(
            nn.Conv2d(8, value, kernel_size=1) for value in channels
        )
        for layer in self.film_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.relevance_aux_projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Keep the safe default blocked until the training loop supplies the
        # per-image paired mask derived from the canonical dataset contract.
        self._arrow_context_gradient_scale: float | torch.Tensor = 0.0
        self.context_enabled = True
        self.train(base.training)

    def _initialize_direction_fusion(self, levels: int) -> None:
        nn.init.zeros_(self.context_fuse.weight)
        nn.init.zeros_(self.context_fuse.bias)
        with torch.no_grad():
            for direction in range(CONTEXT_DIRECTIONS):
                for level in range(levels):
                    channel = level * CONTEXT_DIRECTIONS + direction
                    self.context_fuse.weight[direction, channel, 0, 0] = 1.0 / levels

    def set_arrow_context_gradient_scale(
        self, scales: float | torch.Tensor
    ) -> None:
        """Set scalar or per-image gradient scale for the next forward passes."""

        self._arrow_context_gradient_scale = scales

    def set_context_enabled(self, enabled: bool) -> None:
        """Enable FiLM context or use the pure local relevance fallback."""

        self.context_enabled = bool(enabled)

    @staticmethod
    def _feature_shapes(features: Sequence[torch.Tensor]) -> list[tuple[int, int]]:
        return [(int(value.shape[-2]), int(value.shape[-1])) for value in features]

    @staticmethod
    def _split_dense(
        values: torch.Tensor, shapes: Sequence[tuple[int, int]]
    ) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        offset = 0
        for height, width in shapes:
            locations = height * width
            result.append(
                values[:, :, offset : offset + locations].reshape(
                    values.shape[0], values.shape[1], height, width
                )
            )
            offset += locations
        if offset != values.shape[2]:
            raise ValueError("dense prediction count does not match feature shapes")
        return result

    def _arrow_context(
        self,
        arrow_scores: torch.Tensor,
        direction_logits: torch.Tensor,
        shapes: Sequence[tuple[int, int]],
        *,
        scores_are_probabilities: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.context_enabled:
            batch = arrow_scores.shape[0]
            height, width = shapes[0]
            zero = arrow_scores.sum() * 0.0
            heatmap = zero + torch.zeros(
                (batch, CONTEXT_DIRECTIONS, height, width),
                dtype=arrow_scores.dtype,
                device=arrow_scores.device,
            )
            pooled = zero + torch.zeros(
                (batch, CONTEXT_DIRECTIONS, *CONTEXT_GRID),
                dtype=arrow_scores.dtype,
                device=arrow_scores.device,
            )
            embedding = zero + torch.zeros(
                (batch, CONTEXT_DIMENSION),
                dtype=arrow_scores.dtype,
                device=arrow_scores.device,
            )
            return heatmap, pooled, embedding
        maps = build_soft_arrow_heatmaps(
            arrow_scores,
            direction_logits,
            shapes,
            scores_are_probabilities=scores_are_probabilities,
        )
        target_shape = shapes[0]
        aligned = [
            value
            if value.shape[-2:] == target_shape
            else F.interpolate(
                value, size=target_shape, mode="bilinear", align_corners=False
            )
            for value in maps
        ]
        stacked = torch.cat(aligned, dim=1)
        controlled = scale_arrow_context_gradient(
            stacked, self._arrow_context_gradient_scale
        )
        heatmap = self.context_fuse(controlled)
        pooled = self.context_pool(heatmap)
        embedding = self.context_mlp(pooled.flatten(1))
        return heatmap, pooled, embedding

    def _conditioned_relevance(
        self,
        features: Sequence[torch.Tensor],
        context_embedding: torch.Tensor,
        pictogram_logits: torch.Tensor,
        decoded_traffic: torch.Tensor,
    ) -> torch.Tensor:
        shapes = self._feature_shapes(features)
        pictogram_maps = self._split_dense(pictogram_logits.softmax(1), shapes)
        box_maps = self._split_dense(decoded_traffic[:, :4, :], shapes)
        outputs: list[torch.Tensor] = []

        for level, (feature, pictograms, boxes) in enumerate(
            zip(features, pictogram_maps, box_maps)
        ):
            batch, channels, height, width = feature.shape
            if self.context_enabled:
                film = self.film_layers[level](context_embedding).reshape(
                    batch, 2, channels, 1, 1
                )
                gamma, beta = film[:, 0], film[:, 1]
                modulated = feature * (1.0 + gamma) + beta
            else:
                modulated = feature

            x = (
                torch.arange(width, device=feature.device, dtype=torch.float32)
                + 0.5
            ) / width
            y = (
                torch.arange(height, device=feature.device, dtype=torch.float32)
                + 0.5
            ) / height
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack((grid_x, grid_y), dim=0).expand(
                batch, -1, -1, -1
            ).to(feature.dtype)

            if self.xyxy:
                box_width = boxes[:, 2:3] - boxes[:, 0:1]
                box_height = boxes[:, 3:4] - boxes[:, 1:2]
            else:
                box_width = boxes[:, 2:3]
                box_height = boxes[:, 3:4]
            stride = float(self.stride[level])
            input_width = width * stride
            input_height = height * stride
            normalized_size = torch.cat(
                (
                    box_width.clamp_min(0) / input_width,
                    box_height.clamp_min(0) / input_height,
                ),
                dim=1,
            ).to(feature.dtype)
            auxiliary = torch.cat(
                (coordinates, normalized_size, pictograms.to(feature.dtype)), dim=1
            )
            modulated = modulated + self.relevance_aux_projections[level](auxiliary)
            outputs.append(self.relevance_heads[level](modulated))
        return self._flatten(outputs)

    def forward(
        self, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | tuple[Any, ...] | torch.Tensor:
        # Explicitly skip LocalRelevanceDetect.forward: the local tower is run
        # once below after FiLM and auxiliary conditioning.
        output = RoadArrowMultiTaskDetect.forward(self, features)
        shapes = self._feature_shapes(features)

        if self.training:
            if not isinstance(output, dict):
                raise TypeError("training multi-task head must return a dictionary")
            decoded_traffic = self._inference(output)
            arrow_scores = output["arrow_scores"]
            directions = output["arrow_direction_logits"]
            pictograms = output["pictogram_logits"]
            scores_are_probabilities = False
        elif self.export:
            if not isinstance(output, tuple) or len(output) != 5:
                raise TypeError("export arrow head must return five tensors")
            decoded_traffic, _, pictograms, decoded_arrow, directions = output
            arrow_scores = decoded_arrow[:, 4:5, :]
            scores_are_probabilities = True
        else:
            decoded_traffic, raw = output
            arrow_scores = raw["arrow_scores"]
            directions = raw["arrow_direction_logits"]
            pictograms = raw["pictogram_logits"]
            scores_are_probabilities = False

        heatmap, pooled, embedding = self._arrow_context(
            arrow_scores,
            directions,
            shapes,
            scores_are_probabilities=scores_are_probabilities,
        )
        relevance_logits = self._conditioned_relevance(
            features, embedding, pictograms, decoded_traffic
        )

        if self.training:
            output["arrow_context_heatmap"] = heatmap
            output["arrow_context_pooled"] = pooled
            output["arrow_context_embedding"] = embedding
            output["relevance_logits"] = relevance_logits
            return output
        if self.export:
            return (*output, relevance_logits)
        raw["arrow_context_heatmap"] = heatmap
        raw["arrow_context_pooled"] = pooled
        raw["arrow_context_embedding"] = embedding
        raw["relevance_logits"] = relevance_logits
        return decoded_traffic, raw


def attach_arrow_context_relevance(wrapper: Any) -> Any:
    """Attach the complete dense-context relevance head."""

    attach_local_relevance_head(wrapper)
    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    base = model.model[-1]
    if isinstance(base, ArrowContextRelevanceDetect):
        return wrapper
    if not isinstance(base, LocalRelevanceDetect):
        raise TypeError(f"expected local relevance head, got {type(base)!r}")
    head = ArrowContextRelevanceDetect(base)
    model.model[-1] = head
    model.stride = head.stride
    return wrapper


def set_arrow_context_gradient_scale(
    wrapper: Any, scales: float | torch.Tensor
) -> None:
    """Configure relevance-to-arrow gradient flow before a training forward."""

    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    head = model.model[-1]
    if not isinstance(head, ArrowContextRelevanceDetect):
        raise TypeError("model does not contain an arrow-context relevance head")
    head.set_arrow_context_gradient_scale(scales)


def set_arrow_context_enabled(wrapper: Any, enabled: bool) -> None:
    """Switch between local-only and FiLM-conditioned relevance."""

    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    head = model.model[-1]
    if not isinstance(head, ArrowContextRelevanceDetect):
        raise TypeError("model does not contain an arrow-context relevance head")
    head.set_context_enabled(enabled)


def summarize_context_pairing(records_path: str | Path) -> dict[str, int]:
    """Count relevance, arrow, and truly paired images in canonical JSONL."""

    relevance_images = 0
    arrow_images = 0
    paired_images = 0
    total_images = 0
    with Path(records_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            total_images += 1
            tasks = json.loads(line)["task_valid"]
            relevance = bool(tasks.get("traffic_light_relevance", False))
            arrows = bool(tasks.get("arrow_detection", False))
            relevance_images += int(relevance)
            arrow_images += int(arrows)
            paired_images += int(relevance and arrows)
    return {
        "total_images": total_images,
        "relevance_images": relevance_images,
        "arrow_images": arrow_images,
        "paired_images": paired_images,
        "unpaired_context_gradient_scale": 0,
        "paired_context_gradient_scale_percent": int(
            DEFAULT_PAIRED_GRADIENT_SCALE * 100
        ),
    }


def run_context_forward_smoke(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (800, 1600),
    device: str = "cuda",
    half: bool = True,
) -> dict[str, Any]:
    """Verify fixed-shape dense context, FiLM, and local fallback."""

    height, width = input_size
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = wrapper.model.to(resolved).eval()
    head = model.model[-1]
    head.set_arrow_context_gradient_scale(0.0)
    use_half = half and resolved.type == "cuda"
    model = model.half() if use_half else model.float()
    dtype = torch.float16 if use_half else torch.float32
    sample = torch.zeros((1, 3, height, width), device=resolved, dtype=dtype)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    with torch.inference_mode():
        decoded, raw = model(sample)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)

    expected_locations = sum(
        (height // int(stride)) * (width // int(stride)) for stride in head.stride
    )
    expected_heatmap = (
        1,
        CONTEXT_DIRECTIONS,
        height // int(head.stride[0]),
        width // int(head.stride[0]),
    )
    checks = {
        "relevance": tuple(raw["relevance_logits"].shape)
        == (1, 1, expected_locations),
        "heatmap": tuple(raw["arrow_context_heatmap"].shape) == expected_heatmap,
        "pooled": tuple(raw["arrow_context_pooled"].shape)
        == (1, CONTEXT_DIRECTIONS, *CONTEXT_GRID),
        "embedding": tuple(raw["arrow_context_embedding"].shape)
        == (1, CONTEXT_DIMENSION),
    }
    if not all(checks.values()):
        raise AssertionError(f"unexpected arrow-context output shapes: {checks}")

    film_zero = max(
        float(parameter.detach().abs().max())
        for layer in head.film_layers
        for parameter in layer.parameters()
    )
    auxiliary_zero = max(
        float(parameter.detach().abs().max())
        for layer in head.relevance_aux_projections
        for parameter in layer.parameters()
    )
    if film_zero != 0.0 or auxiliary_zero != 0.0:
        raise AssertionError("FiLM/local auxiliary projections must start at zero")
    context_parameters = sum(
        parameter.numel()
        for name, parameter in head.named_parameters()
        if name.startswith("context_")
        or name.startswith("film_layers")
        or name.startswith("relevance_aux_projections")
    )
    return {
        "schema": "TLR-YOLO-MTL Milestone 6 arrow context smoke v1",
        "input_shape": [1, 3, height, width],
        "dtype": str(dtype).removeprefix("torch."),
        "levels": ["P3", "P4", "P5"],
        "p2_enabled": False,
        "traffic_detection_shape": list(decoded.shape),
        "relevance_shape": list(raw["relevance_logits"].shape),
        "context_heatmap_shape": list(raw["arrow_context_heatmap"].shape),
        "context_pool_shape": list(raw["arrow_context_pooled"].shape),
        "context_embedding_shape": list(raw["arrow_context_embedding"].shape),
        "context_dimension": CONTEXT_DIMENSION,
        "film_zero_initialized": True,
        "local_auxiliary_zero_initialized": True,
        "context_parameters": int(context_parameters),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved))
            if resolved.type == "cuda"
            else None
        ),
    }


def run_context_gradient_smoke(
    wrapper: Any,
    *,
    device: str = "cuda",
    image_size: int = 320,
) -> dict[str, Any]:
    """Verify stopped and 0.25-scaled gradients through real dense predictions."""

    resolved = torch.device(device)
    model = wrapper.model.to(resolved).float().train()
    head = model.model[-1]
    if not isinstance(head, ArrowContextRelevanceDetect):
        raise TypeError("expected arrow-context relevance head")
    zero_initialized = all(
        torch.count_nonzero(parameter).item() == 0
        for layer in head.film_layers
        for parameter in layer.parameters()
    )
    if not zero_initialized:
        raise AssertionError("FiLM layers must be zero-initialized")

    # At initialization the model is exactly the local Milestone 5 model. Open
    # a tiny FiLM path only for this gradient-contract smoke test.
    with torch.no_grad():
        for layer in head.film_layers:
            layer.weight.fill_(1e-3)

    criterion = MaskedRelevanceCriterion(model)
    image = torch.zeros((1, 3, image_size, image_size), device=resolved)
    batch = {
        "batch_idx": torch.tensor([0], device=resolved),
        "cls": torch.tensor([[0.0]], device=resolved),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=resolved),
        "tl_relevance": torch.tensor([1], device=resolved),
        "tl_relevance_valid": torch.tensor([True], device=resolved),
    }

    def backward_with_scale(scale: float) -> tuple[float, float, int]:
        model.zero_grad(set_to_none=True)
        head.set_arrow_context_gradient_scale(scale)
        result = criterion(model(image), batch)
        result.relevance.backward()
        arrow_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in (
                list(head.arrow_detect.parameters())
                + list(head.arrow_direction_heads.parameters())
            )
            if parameter.grad is not None
        )
        context_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in (
                list(head.context_fuse.parameters())
                + list(head.context_mlp.parameters())
                + list(head.film_layers.parameters())
            )
            if parameter.grad is not None
        )
        return arrow_gradient, context_gradient, result.relevance_matches

    stopped_arrow, stopped_context, stopped_matches = backward_with_scale(0.0)
    scaled_arrow, scaled_context, scaled_matches = backward_with_scale(
        DEFAULT_PAIRED_GRADIENT_SCALE
    )
    head.set_arrow_context_gradient_scale(0.0)
    if not (
        stopped_matches > 0
        and scaled_matches > 0
        and stopped_arrow == 0.0
        and stopped_context > 0
        and scaled_arrow > 0
        and scaled_context > 0
    ):
        raise AssertionError("arrow-context controlled-gradient smoke failed")
    return {
        "film_initially_zero": zero_initialized,
        "positive_relevance_matches": scaled_matches,
        "unpaired_gradient_scale": 0.0,
        "unpaired_arrow_gradient_sum": stopped_arrow,
        "unpaired_context_gradient_sum": stopped_context,
        "paired_gradient_scale": DEFAULT_PAIRED_GRADIENT_SCALE,
        "paired_arrow_gradient_sum": scaled_arrow,
        "paired_context_gradient_sum": scaled_context,
        "dense_pre_nms_path": True,
        "hard_matching_used": False,
        "nms_used_in_training_path": False,
        "controlled_gradient_ok": True,
    }
