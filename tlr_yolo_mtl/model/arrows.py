"""Separate P3-P5 road-arrow detector with masked multi-label directions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect

from ..data.schema import ImageRecord
from .attributes import (
    _AttributeTower,
    TrafficLightAttributeDetect,
    attach_attribute_heads,
)

ARROW_DIRECTIONS = ("left", "straight", "right")


class RoadArrowMultiTaskDetect(TrafficLightAttributeDetect):
    """Traffic-light head plus an independent arrow Detect and direction head."""

    def __init__(self, base: TrafficLightAttributeDetect) -> None:
        super().__init__(base)
        # Preserve attribute towers if the model was already augmented.
        self.state_heads = base.state_heads
        self.pictogram_heads = base.pictogram_heads
        channels = tuple(base.attribute_channels)

        self.arrow_detect = Detect(
            nc=1,
            reg_max=base.reg_max,
            end2end=False,
            ch=channels,
        )
        self.arrow_detect.stride = base.stride.detach().clone()
        self.arrow_detect.bias_init()
        self.arrow_direction_heads = nn.ModuleList(
            _AttributeTower(value, classes=3) for value in channels
        )
        self.train(base.training)

    def _apply(self, function: Any) -> "RoadArrowMultiTaskDetect":
        super()._apply(function)
        self.arrow_detect.stride = function(self.arrow_detect.stride)
        self.arrow_detect.anchors = function(self.arrow_detect.anchors)
        self.arrow_detect.strides = function(self.arrow_detect.strides)
        return self

    def forward(
        self, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | tuple[Any, ...] | torch.Tensor:
        traffic = super().forward(features)
        direction_maps = [
            head(value) for head, value in zip(self.arrow_direction_heads, features)
        ]
        direction_logits = self._flatten(direction_maps)
        self.arrow_detect.export = self.export
        arrow = self.arrow_detect(features)

        if self.training:
            if not isinstance(traffic, dict) or not isinstance(arrow, dict):
                raise TypeError("training heads must return prediction dictionaries")
            traffic["arrow_boxes"] = arrow["boxes"]
            traffic["arrow_scores"] = arrow["scores"]
            traffic["arrow_feats"] = arrow["feats"]
            traffic["arrow_direction_logits"] = direction_logits
            return traffic

        if self.export:
            traffic_detection, states, pictograms = traffic
            return (
                traffic_detection,
                states,
                pictograms,
                arrow,
                direction_logits,
            )

        traffic_decoded, raw = traffic
        arrow_decoded, arrow_raw = arrow
        raw["arrow_decoded"] = arrow_decoded
        raw["arrow_boxes"] = arrow_raw["boxes"]
        raw["arrow_scores"] = arrow_raw["scores"]
        raw["arrow_feats"] = arrow_raw["feats"]
        raw["arrow_direction_logits"] = direction_logits
        return traffic_decoded, raw


def attach_arrow_heads(wrapper: Any) -> Any:
    """Attach attributes if needed, then add the independent arrow branches."""

    attach_attribute_heads(wrapper)
    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    base = model.model[-1]
    if isinstance(base, RoadArrowMultiTaskDetect):
        return wrapper
    if not isinstance(base, TrafficLightAttributeDetect):
        raise TypeError(f"expected attribute Detect head, got {type(base)!r}")
    head = RoadArrowMultiTaskDetect(base)
    model.model[-1] = head
    model.stride = head.stride
    return wrapper


def encode_record_arrows(record: ImageRecord) -> dict[str, torch.Tensor]:
    """Encode arrow directions and the image-level exhaustiveness mask."""

    return {
        "arrow_direction": torch.tensor(
            [item.direction_multihot for item in record.road_arrows],
            dtype=torch.float32,
        ).reshape(-1, 3),
        "arrow_detection_valid": torch.tensor(
            record.task_valid.arrow_detection, dtype=torch.bool
        ),
    }


def _pad_multihot_targets(
    values: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    values = values.reshape(-1, 3).float()
    batch_indices = batch_indices.reshape(-1).long()
    if values.shape[0] != batch_indices.numel():
        raise ValueError("arrow directions and batch indices must have equal length")
    if batch_indices.numel() > 1 and torch.any(batch_indices[1:] < batch_indices[:-1]):
        raise ValueError("arrow instances must be grouped by batch index")
    counts = torch.bincount(batch_indices, minlength=batch_size)
    maximum = int(counts.max().item()) if counts.numel() else 0
    padded = torch.zeros(
        (batch_size, maximum, 3), dtype=values.dtype, device=values.device
    )
    if not values.shape[0]:
        return padded
    offsets = torch.cat(
        (torch.zeros(1, device=values.device, dtype=torch.long), counts.cumsum(0))
    )
    positions = torch.arange(values.shape[0], device=values.device) - offsets[
        batch_indices
    ]
    padded[batch_indices, positions] = values
    return padded


def assigned_direction_bce(
    logits: torch.Tensor,
    padded_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    gamma: float = 0.0,
    alpha: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Optionally focal multi-label BCE for positive arrow matches."""

    if logits.ndim != 3 or logits.shape[1] != 3:
        raise ValueError("direction logits must have shape [batch, 3, anchors]")
    batch, _, anchors = logits.shape
    if foreground_mask.shape != (batch, anchors):
        raise ValueError("foreground mask shape does not match direction logits")
    if target_gt_indices.shape != (batch, anchors):
        raise ValueError("target GT index shape does not match direction logits")
    if padded_targets.shape[0] != batch:
        raise ValueError("direction target batch dimension does not match logits")
    if padded_targets.shape[1] == 0:
        return logits.sum() * 0.0, 0
    safe_indices = target_gt_indices.clamp(0, padded_targets.shape[1] - 1)
    anchor_targets = padded_targets.gather(
        1, safe_indices[:, :, None].expand(-1, -1, 3)
    )
    valid = foreground_mask.bool()
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_logits = logits.permute(0, 2, 1)[valid]
    selected_targets = anchor_targets[valid]
    if gamma < 0:
        raise ValueError("focal gamma must be non-negative")
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError("focal alpha must be in [0, 1]")
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )
    if gamma:
        probabilities = selected_logits.sigmoid()
        probability_target = (
            probabilities * selected_targets
            + (1.0 - probabilities) * (1.0 - selected_targets)
        )
        loss = loss * (1.0 - probability_target).pow(gamma)
    if alpha is not None:
        alpha_target = (
            alpha * selected_targets + (1.0 - alpha) * (1.0 - selected_targets)
        )
        loss = loss * alpha_target
    return loss.mean(), count


def gather_arrow_directions(
    direction_logits: torch.Tensor, candidate_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Gather multi-label direction scores with retained arrow candidates."""

    if direction_logits.ndim != 3 or direction_logits.shape[1] != 3:
        raise ValueError("direction logits must have shape [batch, 3, anchors]")
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != direction_logits.shape[0]:
        raise ValueError("candidate indices must have shape [batch, detections]")
    if candidate_indices.numel() and (
        candidate_indices.min() < 0
        or candidate_indices.max() >= direction_logits.shape[2]
    ):
        raise ValueError("candidate index is outside direction logits")
    index = candidate_indices[:, None, :].expand(-1, 3, -1)
    selected = direction_logits.gather(2, index)
    return {
        "direction_logits": selected,
        "direction_probabilities": selected.sigmoid(),
        "direction_multihot": selected.sigmoid().ge(0.5).to(torch.int64),
    }


@dataclass(slots=True)
class ArrowLossResult:
    total: torch.Tensor
    detection: torch.Tensor
    direction: torch.Tensor
    valid_images: int
    direction_matches: int
    metrics: dict[str, torch.Tensor]


class MaskedArrowCriterion:
    """Arrow detection/direction loss with an exhaustive image-level mask."""

    def __init__(
        self,
        model: nn.Module,
        *,
        direction_weight: float = 1.0,
        direction_gamma: float = 0.0,
        direction_alpha: float | None = None,
    ) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        self.detection = v8DetectionLoss(model)
        if isinstance(self.detection.hyp, Mapping):
            self.detection.hyp = SimpleNamespace(**self.detection.hyp)
        self.direction_weight = float(direction_weight)
        self.direction_gamma = float(direction_gamma)
        self.direction_alpha = direction_alpha

    @staticmethod
    def _connected_zero(parsed: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            parsed["arrow_boxes"].sum()
            + parsed["arrow_scores"].sum()
            + parsed["arrow_direction_logits"].sum()
        ) * 0.0

    def __call__(
        self,
        predictions: Mapping[str, torch.Tensor] | tuple[Any, ...],
        batch: Mapping[str, torch.Tensor],
    ) -> ArrowLossResult:
        parsed = self.detection.parse_output(predictions)
        for name in (
            "arrow_boxes",
            "arrow_scores",
            "arrow_feats",
            "arrow_direction_logits",
        ):
            if name not in parsed:
                raise KeyError(f"missing arrow prediction: {name}")
        for name in (
            "arrow_detection_valid",
            "arrow_batch_idx",
            "arrow_bboxes",
            "arrow_direction",
        ):
            if name not in batch:
                raise KeyError(f"missing arrow batch field: {name}")

        device = parsed["arrow_scores"].device
        image_mask = batch["arrow_detection_valid"].to(device).reshape(-1).bool()
        full_batch = parsed["arrow_scores"].shape[0]
        if image_mask.numel() != full_batch:
            raise ValueError("arrow detection mask must contain one value per image")
        valid_indices = torch.nonzero(image_mask, as_tuple=False).reshape(-1)
        valid_images = int(valid_indices.numel())
        if valid_images == 0:
            zero = self._connected_zero(parsed)
            return ArrowLossResult(
                total=zero,
                detection=zero,
                direction=zero,
                valid_images=0,
                direction_matches=0,
                metrics={
                    "arrow_detection_loss": zero.detach(),
                    "arrow_direction_loss": zero.detach(),
                },
            )

        arrow_predictions = {
            "boxes": parsed["arrow_boxes"].index_select(0, valid_indices),
            "scores": parsed["arrow_scores"].index_select(0, valid_indices),
            "feats": [
                value.index_select(0, valid_indices) for value in parsed["arrow_feats"]
            ],
        }
        original_instance_batch = batch["arrow_batch_idx"].to(device).reshape(-1).long()
        if original_instance_batch.numel() != batch["arrow_bboxes"].shape[0]:
            raise ValueError("arrow instance batch indices and boxes differ in length")
        instance_mask = image_mask[original_instance_batch]
        remap = torch.full((full_batch,), -1, dtype=torch.long, device=device)
        remap[valid_indices] = torch.arange(valid_images, device=device)
        filtered_batch_indices = remap[original_instance_batch[instance_mask]]
        filtered_boxes = batch["arrow_bboxes"].to(device)[instance_mask]
        filtered_directions = batch["arrow_direction"].to(device)[instance_mask]
        detection_batch = {
            "batch_idx": filtered_batch_indices,
            "cls": torch.zeros(
                (filtered_boxes.shape[0], 1), dtype=torch.float32, device=device
            ),
            "bboxes": filtered_boxes,
        }
        assignments, detection_vector, detection_metrics = (
            self.detection.get_assigned_targets_and_loss(
                arrow_predictions, detection_batch
            )
        )
        foreground_mask, target_gt_indices = assignments[:2]
        padded_directions = _pad_multihot_targets(
            filtered_directions, filtered_batch_indices, valid_images
        )
        direction_logits = parsed["arrow_direction_logits"].index_select(
            0, valid_indices
        )
        direction_loss, direction_matches = assigned_direction_bce(
            direction_logits,
            padded_directions,
            foreground_mask,
            target_gt_indices,
            gamma=self.direction_gamma,
            alpha=self.direction_alpha,
        )
        detection_total = detection_vector.sum() * valid_images
        total = detection_total + self.direction_weight * direction_loss
        return ArrowLossResult(
            total=total,
            detection=detection_total,
            direction=direction_loss,
            valid_images=valid_images,
            direction_matches=direction_matches,
            metrics={
                **{f"arrow_{name}": value for name, value in detection_metrics.items()},
                "arrow_direction_loss": direction_loss.detach(),
            },
        )


def run_arrow_forward_smoke(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (800, 1600),
    device: str = "cuda",
    half: bool = True,
) -> dict[str, Any]:
    height, width = input_size
    resolved = torch.device(device)
    model = wrapper.model.to(resolved).eval()
    use_half = half and resolved.type == "cuda"
    model = model.half() if use_half else model.float()
    dtype = torch.float16 if use_half else torch.float32
    sample = torch.zeros((1, 3, height, width), device=resolved, dtype=dtype)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    with torch.inference_mode():
        _, raw = model(sample)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    head = model.model[-1]
    expected_locations = sum(
        (height // int(stride)) * (width // int(stride)) for stride in head.stride
    )
    expected_detection = (1, 5, expected_locations)
    expected_direction = (1, 3, expected_locations)
    if tuple(raw["arrow_decoded"].shape) != expected_detection:
        raise AssertionError(f"unexpected arrow detection shape: {raw['arrow_decoded'].shape}")
    if tuple(raw["arrow_direction_logits"].shape) != expected_direction:
        raise AssertionError(
            f"unexpected arrow direction shape: {raw['arrow_direction_logits'].shape}"
        )
    arrow_parameters = sum(
        parameter.numel()
        for name, parameter in head.named_parameters()
        if name.startswith("arrow_detect")
        or name.startswith("arrow_direction_heads")
    )
    return {
        "schema": "TLR-YOLO-MTL Milestone 4 arrow smoke v1",
        "input_shape": [1, 3, height, width],
        "dtype": str(dtype).removeprefix("torch."),
        "levels": ["P3", "P4", "P5"],
        "p2_enabled": False,
        "arrow_detection_shape": list(raw["arrow_decoded"].shape),
        "arrow_direction_shape": list(raw["arrow_direction_logits"].shape),
        "direction_classes": list(ARROW_DIRECTIONS),
        "direction_activation": "independent_sigmoid",
        "arrow_parameters": int(arrow_parameters),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved))
            if resolved.type == "cuda"
            else None
        ),
    }


def run_arrow_assignment_smoke(
    wrapper: Any,
    *,
    device: str = "cuda",
    image_size: int = 320,
) -> dict[str, Any]:
    """Verify compound directions and complete task masking with real TAL."""

    resolved = torch.device(device)
    model = wrapper.model.to(resolved).float().train()
    criterion = MaskedArrowCriterion(model)
    image = torch.zeros((1, 3, image_size, image_size), device=resolved)

    model.zero_grad(set_to_none=True)
    predictions = model(image)
    valid_batch = {
        "arrow_detection_valid": torch.tensor([True], device=resolved),
        "arrow_batch_idx": torch.tensor([0], device=resolved),
        "arrow_bboxes": torch.tensor(
            [[0.5, 0.65, 0.25, 0.20]], device=resolved
        ),
        "arrow_direction": torch.tensor([[1.0, 1.0, 0.0]], device=resolved),
    }
    valid_result = criterion(predictions, valid_batch)
    valid_result.total.backward()
    head = model.model[-1]
    detection_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.arrow_detect.parameters()
        if parameter.grad is not None
    )
    direction_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.arrow_direction_heads.parameters()
        if parameter.grad is not None
    )

    model.zero_grad(set_to_none=True)
    masked_predictions = model(image)
    masked_batch = {
        "arrow_detection_valid": torch.tensor([False], device=resolved),
        "arrow_batch_idx": torch.empty(0, dtype=torch.long, device=resolved),
        "arrow_bboxes": torch.empty((0, 4), device=resolved),
        "arrow_direction": torch.empty((0, 3), device=resolved),
    }
    masked_result = criterion(masked_predictions, masked_batch)
    masked_result.total.backward()
    masked_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in (
            list(head.arrow_detect.parameters())
            + list(head.arrow_direction_heads.parameters())
        )
        if parameter.grad is not None
    )
    if not (
        valid_result.direction_matches > 0
        and detection_gradient > 0
        and direction_gradient > 0
        and masked_result.valid_images == 0
        and masked_gradient == 0
    ):
        raise AssertionError("arrow assignment or task masking smoke failed")
    return {
        "compound_target": [1, 1, 0],
        "positive_direction_matches": valid_result.direction_matches,
        "arrow_detection_gradient_sum": detection_gradient,
        "arrow_direction_gradient_sum": direction_gradient,
        "fully_masked_valid_images": masked_result.valid_images,
        "fully_masked_arrow_gradient_sum": masked_gradient,
        "arrow_masking_ok": True,
    }
