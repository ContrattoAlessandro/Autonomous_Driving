"""Factorized traffic-light attribute heads and positive-match losses.

The detection class remains a single ``vehicle_traffic_light`` class.  State
and pictogram logits are produced at the same P3-P5 candidate locations and
are supervised only where the detector's task-aligned assigner marks a
positive candidate.  Missing instance labels stay masked and therefore
produce exactly zero gradient.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect

from ..data.schema import ImageRecord

STATE_CLASSES = ("red", "yellow", "green", "off")
PICTOGRAM_CLASSES = ("round", "left", "straight", "right")
STATE_TO_INDEX = {name: index for index, name in enumerate(STATE_CLASSES)}
PICTOGRAM_TO_INDEX = {
    name: index for index, name in enumerate(PICTOGRAM_CLASSES)
}


class _AttributeTower(nn.Sequential):
    """Depthwise 3x3 followed by a pointwise four-logit projection."""

    def __init__(self, channels: int, classes: int = 4) -> None:
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, classes, kernel_size=1),
        )


class TrafficLightAttributeDetect(Detect):
    """Standard YOLO Detect head augmented with state and pictogram branches."""

    def __init__(self, base: Detect) -> None:
        channels = tuple(branch[0].conv.in_channels for branch in base.cv2)
        super().__init__(
            nc=base.nc,
            reg_max=base.reg_max,
            end2end=base.end2end,
            ch=channels,
        )

        # Preserve the already warm-started standard detection head.
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

        self.state_heads = nn.ModuleList(_AttributeTower(value) for value in channels)
        self.pictogram_heads = nn.ModuleList(
            _AttributeTower(value) for value in channels
        )
        self.attribute_channels = channels

        # Ultralytics graph metadata lives on each parsed YAML module.
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

    def forward(
        self, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | tuple[Any, ...] | torch.Tensor:
        state_maps = [head(value) for head, value in zip(self.state_heads, features)]
        pictogram_maps = [
            head(value) for head, value in zip(self.pictogram_heads, features)
        ]
        state_logits = self._flatten(state_maps)
        pictogram_logits = self._flatten(pictogram_maps)
        detection = super().forward(features)

        if self.training:
            if not isinstance(detection, dict):
                raise TypeError("training Detect output must be a prediction dictionary")
            detection["state_logits"] = state_logits
            detection["pictogram_logits"] = pictogram_logits
            return detection

        if self.export:
            return detection, state_logits, pictogram_logits

        decoded, raw = detection
        raw["state_logits"] = state_logits
        raw["pictogram_logits"] = pictogram_logits
        return decoded, raw


def attach_attribute_heads(wrapper: Any) -> Any:
    """Replace the final Detect module while preserving its COCO parameters."""

    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    base = model.model[-1]
    if isinstance(base, TrafficLightAttributeDetect):
        return wrapper
    if not isinstance(base, Detect):
        raise TypeError(f"expected an Ultralytics Detect head, got {type(base)!r}")
    head = TrafficLightAttributeDetect(base)
    model.model[-1] = head
    model.stride = head.stride
    return wrapper


def encode_record_attributes(record: ImageRecord) -> dict[str, torch.Tensor]:
    """Encode attributes in the exact order of the record's TL boxes."""

    states: list[int] = []
    pictograms: list[int] = []
    for item in record.traffic_lights:
        states.append(
            STATE_TO_INDEX[item.state]
            if item.valid_state and item.state in STATE_TO_INDEX
            else -1
        )
        pictograms.append(
            PICTOGRAM_TO_INDEX[item.pictogram]
            if item.valid_pictogram and item.pictogram in PICTOGRAM_TO_INDEX
            else -1
        )
    return {
        "tl_state": torch.tensor(states, dtype=torch.long),
        "tl_pictogram": torch.tensor(pictograms, dtype=torch.long),
    }


def _pad_targets(
    values: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Pad flat per-instance targets in the same order as YOLO preprocessing."""

    values = values.reshape(-1).long()
    batch_indices = batch_indices.reshape(-1).long()
    if values.numel() != batch_indices.numel():
        raise ValueError("attribute targets and batch indices must have equal length")
    if batch_indices.numel() and (
        batch_indices.min() < 0 or batch_indices.max() >= batch_size
    ):
        raise ValueError("batch index is outside the current batch")
    if batch_indices.numel() > 1 and torch.any(batch_indices[1:] < batch_indices[:-1]):
        raise ValueError("YOLO batch instances must be grouped by batch index")

    counts = torch.bincount(batch_indices, minlength=batch_size)
    maximum = int(counts.max().item()) if counts.numel() else 0
    padded = torch.full(
        (batch_size, maximum),
        -1,
        dtype=torch.long,
        device=values.device,
    )
    if not values.numel():
        return padded
    offsets = torch.cat(
        (torch.zeros(1, device=values.device, dtype=torch.long), counts.cumsum(0))
    )
    positions = torch.arange(values.numel(), device=values.device) - offsets[
        batch_indices
    ]
    padded[batch_indices, positions] = values
    return padded


def assigned_attribute_cross_entropy(
    logits: torch.Tensor,
    padded_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    gamma: float = 0.0,
    class_weights: torch.Tensor | None = None,
    log_priors: torch.Tensor | None = None,
    prior_scale: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Optionally focal cross-entropy on valid detector-positive matches with optional log priors and class weights."""

    if logits.ndim != 3:
        raise ValueError("attribute logits must have shape [batch, classes, anchors]")
    batch, _, anchors = logits.shape
    if foreground_mask.shape != (batch, anchors):
        raise ValueError("foreground mask shape does not match logits")
    if target_gt_indices.shape != (batch, anchors):
        raise ValueError("target GT index shape does not match logits")
    if padded_targets.shape[0] != batch:
        raise ValueError("padded targets batch dimension does not match logits")
    if padded_targets.shape[1] == 0:
        return logits.sum() * 0.0, 0

    safe_indices = target_gt_indices.clamp(0, padded_targets.shape[1] - 1)
    anchor_targets = padded_targets.gather(1, safe_indices)
    valid = foreground_mask.bool() & anchor_targets.ge(0)
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_logits = logits.permute(0, 2, 1)[valid]
    selected_targets = anchor_targets[valid]
    if gamma < 0:
        raise ValueError("focal gamma must be non-negative")
    weights = None
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        if weights.numel() != logits.shape[1]:
            raise ValueError("class weights must contain one value per class")
    if log_priors is not None:
        lp = log_priors.to(device=logits.device, dtype=logits.dtype)
        if lp.numel() != logits.shape[1]:
            raise ValueError("log priors must contain one value per class")
        selected_logits = selected_logits + float(prior_scale) * lp.unsqueeze(0)
    cross_entropy = F.cross_entropy(
        selected_logits, selected_targets, weight=weights, reduction="none"
    )
    if gamma:
        target_probabilities = selected_logits.softmax(1).gather(
            1, selected_targets[:, None]
        ).squeeze(1)
        cross_entropy = cross_entropy * (1.0 - target_probabilities).pow(gamma)
    return cross_entropy.mean(), count



def gather_candidate_attributes(
    state_logits: torch.Tensor,
    pictogram_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Keep attributes aligned when candidate/NMS indices are selected."""

    if state_logits.shape[0] != pictogram_logits.shape[0]:
        raise ValueError("state and pictogram batch dimensions differ")
    if state_logits.shape[2] != pictogram_logits.shape[2]:
        raise ValueError("state and pictogram anchor dimensions differ")
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != state_logits.shape[0]:
        raise ValueError("candidate indices must have shape [batch, detections]")
    if candidate_indices.numel() and (
        candidate_indices.min() < 0
        or candidate_indices.max() >= state_logits.shape[2]
    ):
        raise ValueError("candidate index is outside the dense prediction tensor")

    def gather(logits: torch.Tensor) -> torch.Tensor:
        index = candidate_indices[:, None, :].expand(-1, logits.shape[1], -1)
        return logits.gather(2, index)

    selected_state = gather(state_logits)
    selected_pictogram = gather(pictogram_logits)
    return {
        "state_logits": selected_state,
        "state_probabilities": selected_state.softmax(1),
        "state_indices": selected_state.argmax(1),
        "pictogram_logits": selected_pictogram,
        "pictogram_probabilities": selected_pictogram.softmax(1),
        "pictogram_indices": selected_pictogram.argmax(1),
    }


@dataclass(slots=True)
class MultiTaskLossResult:
    total: torch.Tensor
    detection: torch.Tensor
    state: torch.Tensor
    pictogram: torch.Tensor
    state_matches: int
    pictogram_matches: int
    metrics: dict[str, torch.Tensor]


class FactorizedAttributeCriterion:
    """YOLO detection loss plus masked state/pictogram positive-match losses."""

    def __init__(
        self,
        model: nn.Module,
        *,
        state_weight: float = 1.0,
        pictogram_weight: float = 1.0,
    ) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        self.detection = v8DetectionLoss(model)
        if isinstance(self.detection.hyp, Mapping):
            self.detection.hyp = SimpleNamespace(**self.detection.hyp)
        self.state_weight = float(state_weight)
        self.pictogram_weight = float(pictogram_weight)

    def __call__(
        self,
        predictions: Mapping[str, torch.Tensor] | tuple[Any, ...],
        batch: Mapping[str, torch.Tensor],
    ) -> MultiTaskLossResult:
        parsed = self.detection.parse_output(predictions)
        required_predictions = {"state_logits", "pictogram_logits"}
        missing_predictions = required_predictions.difference(parsed)
        if missing_predictions:
            raise KeyError(f"missing attribute predictions: {sorted(missing_predictions)}")
        for name in ("batch_idx", "tl_state", "tl_pictogram"):
            if name not in batch:
                raise KeyError(f"missing batch field: {name}")

        assignments, detection_vector, detection_metrics = (
            self.detection.get_assigned_targets_and_loss(parsed, dict(batch))
        )
        foreground_mask, target_gt_indices = assignments[:2]
        batch_size = parsed["scores"].shape[0]
        batch_indices = batch["batch_idx"].to(parsed["scores"].device)
        state_targets = _pad_targets(
            batch["tl_state"].to(parsed["scores"].device),
            batch_indices,
            batch_size,
        )
        pictogram_targets = _pad_targets(
            batch["tl_pictogram"].to(parsed["scores"].device),
            batch_indices,
            batch_size,
        )
        state_loss, state_matches = assigned_attribute_cross_entropy(
            parsed["state_logits"],
            state_targets,
            foreground_mask,
            target_gt_indices,
        )
        pictogram_loss, pictogram_matches = assigned_attribute_cross_entropy(
            parsed["pictogram_logits"],
            pictogram_targets,
            foreground_mask,
            target_gt_indices,
        )
        detection_total = detection_vector.sum() * batch_size
        total = (
            detection_total
            + self.state_weight * state_loss
            + self.pictogram_weight * pictogram_loss
        )
        metrics = {
            **detection_metrics,
            "state_loss": state_loss.detach(),
            "pictogram_loss": pictogram_loss.detach(),
        }
        return MultiTaskLossResult(
            total=total,
            detection=detection_total,
            state=state_loss,
            pictogram=pictogram_loss,
            state_matches=state_matches,
            pictogram_matches=pictogram_matches,
            metrics=metrics,
        )


def run_attribute_forward_smoke(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (800, 1600),
    device: str = "cuda",
    half: bool = True,
) -> dict[str, Any]:
    """Verify the single-model detection and factorized attribute outputs."""

    height, width = input_size
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = wrapper.model.to(resolved).eval()
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
    head = model.model[-1]
    expected_locations = sum(
        (height // int(stride)) * (width // int(stride)) for stride in head.stride
    )
    if raw["state_logits"].shape != (1, 4, expected_locations):
        raise AssertionError(f"unexpected state shape: {raw['state_logits'].shape}")
    if raw["pictogram_logits"].shape != (1, 4, expected_locations):
        raise AssertionError(
            f"unexpected pictogram shape: {raw['pictogram_logits'].shape}"
        )

    attribute_parameters = sum(
        parameter.numel()
        for name, parameter in head.named_parameters()
        if name.startswith("state_heads") or name.startswith("pictogram_heads")
    )
    return {
        "schema": "TLR-YOLO-MTL Milestone 3 attribute smoke v1",
        "input_shape": [1, 3, height, width],
        "dtype": str(dtype).removeprefix("torch."),
        "strides": [int(value) for value in head.stride.tolist()],
        "detection_shape": list(decoded.shape),
        "state_shape": list(raw["state_logits"].shape),
        "pictogram_shape": list(raw["pictogram_logits"].shape),
        "state_classes": list(STATE_CLASSES),
        "pictogram_classes": list(PICTOGRAM_CLASSES),
        "attribute_parameters": int(attribute_parameters),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved))
            if resolved.type == "cuda"
            else None
        ),
    }


def run_masking_gradient_smoke(device: str = "cpu") -> dict[str, Any]:
    """Prove that positives receive gradients while unknown/background do not."""

    resolved = torch.device(device)
    logits = torch.zeros((1, 4, 5), device=resolved, requires_grad=True)
    targets = torch.tensor([[2, -1]], device=resolved)
    foreground = torch.tensor([[False, True, True, False, False]], device=resolved)
    gt_indices = torch.tensor([[0, 0, 1, 0, 0]], device=resolved)
    loss, count = assigned_attribute_cross_entropy(
        logits, targets, foreground, gt_indices
    )
    loss.backward()
    gradient_by_anchor = logits.grad.abs().sum(1).squeeze(0)
    active = torch.nonzero(gradient_by_anchor > 0, as_tuple=False).reshape(-1).tolist()
    if active != [1] or count != 1:
        raise AssertionError(
            f"masked attribute gradient contract failed: active={active}, count={count}"
        )
    return {
        "valid_positive_matches": count,
        "anchors_with_gradient": active,
        "unknown_anchor_gradient": float(gradient_by_anchor[2]),
        "background_gradient_sum": float(
            gradient_by_anchor[[0, 3, 4]].sum()
        ),
        "masking_ok": True,
    }


def run_assignment_gradient_smoke(
    wrapper: Any,
    *,
    device: str = "cuda",
    image_size: int = 320,
) -> dict[str, Any]:
    """Backpropagate through YOLO's real task-aligned positive assignments."""

    resolved = torch.device(device)
    model = wrapper.model.to(resolved).float().train()
    model.zero_grad(set_to_none=True)
    criterion = FactorizedAttributeCriterion(model)
    image = torch.zeros((1, 3, image_size, image_size), device=resolved)
    predictions = model(image)
    batch = {
        "batch_idx": torch.tensor([0], device=resolved),
        "cls": torch.tensor([[0.0]], device=resolved),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=resolved),
        "tl_state": torch.tensor([STATE_TO_INDEX["red"]], device=resolved),
        "tl_pictogram": torch.tensor([-1], device=resolved),
    }
    result = criterion(predictions, batch)
    result.total.backward()
    head = model.model[-1]
    state_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.state_heads.parameters()
        if parameter.grad is not None
    )
    pictogram_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.pictogram_heads.parameters()
        if parameter.grad is not None
    )
    if not (
        result.state_matches > 0
        and result.pictogram_matches == 0
        and state_gradient > 0
        and pictogram_gradient == 0
    ):
        raise AssertionError(
            "task-aligned attribute gradient contract failed: "
            f"state_matches={result.state_matches}, "
            f"pictogram_matches={result.pictogram_matches}, "
            f"state_gradient={state_gradient}, "
            f"pictogram_gradient={pictogram_gradient}"
        )
    return {
        "image_shape": [1, 3, image_size, image_size],
        "state_positive_matches": result.state_matches,
        "masked_pictogram_matches": result.pictogram_matches,
        "state_head_gradient_sum": state_gradient,
        "masked_pictogram_gradient_sum": pictogram_gradient,
        "assignment_masking_ok": True,
    }
