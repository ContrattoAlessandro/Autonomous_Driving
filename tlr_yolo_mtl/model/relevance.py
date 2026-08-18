"""Per-traffic-light local relevance head and masked positive-match loss."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..data.schema import ImageRecord
from .arrows import RoadArrowMultiTaskDetect, attach_arrow_heads
from .attributes import _AttributeTower, _pad_targets


class LocalRelevanceDetect(RoadArrowMultiTaskDetect):
    """Add one independent relevance logit at every dense TL candidate."""

    def __init__(self, base: RoadArrowMultiTaskDetect) -> None:
        super().__init__(base)

        # ``super`` rebuilds the composite head. Preserve every branch already
        # attached to the source model before adding the new local branch.
        self.state_heads = base.state_heads
        self.pictogram_heads = base.pictogram_heads
        self.arrow_detect = base.arrow_detect
        self.arrow_direction_heads = base.arrow_direction_heads
        channels = tuple(base.attribute_channels)
        self.relevance_heads = nn.ModuleList(
            _AttributeTower(value, classes=1) for value in channels
        )
        self.train(base.training)

    def forward(
        self, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | tuple[Any, ...] | torch.Tensor:
        relevance_maps = [
            head(value) for head, value in zip(self.relevance_heads, features)
        ]
        relevance_logits = self._flatten(relevance_maps)
        output = super().forward(features)

        if self.training:
            if not isinstance(output, dict):
                raise TypeError("training multi-task head must return a dictionary")
            output["relevance_logits"] = relevance_logits
            return output

        if self.export:
            if not isinstance(output, tuple) or len(output) != 5:
                raise TypeError("export arrow head must return five tensors")
            return (*output, relevance_logits)

        decoded, raw = output
        raw["relevance_logits"] = relevance_logits
        return decoded, raw


def attach_local_relevance_head(wrapper: Any) -> Any:
    """Attach all earlier heads and then add local TL relevance."""

    attach_arrow_heads(wrapper)
    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    base = model.model[-1]
    if isinstance(base, LocalRelevanceDetect):
        return wrapper
    if not isinstance(base, RoadArrowMultiTaskDetect):
        raise TypeError(f"expected road-arrow multi-task head, got {type(base)!r}")
    head = LocalRelevanceDetect(base)
    model.model[-1] = head
    model.stride = head.stride
    return wrapper


def encode_record_relevance(record: ImageRecord) -> dict[str, torch.Tensor]:
    """Encode binary relevance in the exact order of the TL instances."""

    image_valid = record.task_valid.traffic_light_relevance
    values = [
        int(item.relevance)
        if image_valid and item.valid_relevance and item.relevance in (0, 1)
        else -1
        for item in record.traffic_lights
    ]
    return {
        "tl_relevance": torch.tensor(values, dtype=torch.long),
        "tl_relevance_valid": torch.tensor(image_valid, dtype=torch.bool),
    }


def assigned_relevance_focal_bce(
    logits: torch.Tensor,
    padded_targets: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_indices: torch.Tensor,
    *,
    image_valid: torch.Tensor | None = None,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> tuple[torch.Tensor, int]:
    """Binary focal loss on valid detector-positive TL matches only."""

    if logits.ndim != 3 or logits.shape[1] != 1:
        raise ValueError("relevance logits must have shape [batch, 1, anchors]")
    batch, _, anchors = logits.shape
    if foreground_mask.shape != (batch, anchors):
        raise ValueError("foreground mask shape does not match relevance logits")
    if target_gt_indices.shape != (batch, anchors):
        raise ValueError("target GT index shape does not match relevance logits")
    if padded_targets.shape[0] != batch:
        raise ValueError("relevance target batch dimension does not match logits")
    if gamma < 0:
        raise ValueError("focal gamma must be non-negative")
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError("focal alpha must be in [0, 1]")
    if image_valid is not None and image_valid.reshape(-1).numel() != batch:
        raise ValueError("relevance image mask must contain one value per image")
    if padded_targets.shape[1] == 0:
        return logits.sum() * 0.0, 0

    safe_indices = target_gt_indices.clamp(0, padded_targets.shape[1] - 1)
    anchor_targets = padded_targets.gather(1, safe_indices)
    valid = foreground_mask.bool() & anchor_targets.ge(0)
    if image_valid is not None:
        valid &= image_valid.to(logits.device).reshape(batch, 1).bool()
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0

    selected_logits = logits[:, 0, :][valid]
    selected_targets = anchor_targets[valid].to(logits.dtype)
    cross_entropy = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, reduction="none"
    )
    probabilities = selected_logits.sigmoid()
    probability_target = (
        probabilities * selected_targets
        + (1.0 - probabilities) * (1.0 - selected_targets)
    )
    loss = cross_entropy * (1.0 - probability_target).pow(gamma)
    if alpha is not None:
        alpha_target = (
            alpha * selected_targets + (1.0 - alpha) * (1.0 - selected_targets)
        )
        loss = loss * alpha_target
    return loss.mean(), count


def gather_candidate_relevance(
    relevance_logits: torch.Tensor, candidate_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Keep relevance aligned with retained traffic-light candidates."""

    if relevance_logits.ndim != 3 or relevance_logits.shape[1] != 1:
        raise ValueError("relevance logits must have shape [batch, 1, anchors]")
    if (
        candidate_indices.ndim != 2
        or candidate_indices.shape[0] != relevance_logits.shape[0]
    ):
        raise ValueError("candidate indices must have shape [batch, detections]")
    if candidate_indices.numel() and (
        candidate_indices.min() < 0
        or candidate_indices.max() >= relevance_logits.shape[2]
    ):
        raise ValueError("candidate index is outside relevance logits")
    selected = relevance_logits.gather(2, candidate_indices[:, None, :])
    return {
        "relevance_logits": selected,
        "relevance_probabilities": selected.sigmoid(),
    }


def combine_detection_relevance_scores(
    detection_scores: torch.Tensor, relevance_logits: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Expose detection, relevance, and their product as distinct scores."""

    if relevance_logits.ndim != 3 or relevance_logits.shape[1] != 1:
        raise ValueError("relevance logits must have shape [batch, 1, detections]")
    probabilities = relevance_logits.sigmoid()
    if detection_scores.ndim == 2:
        relevance = probabilities[:, 0, :]
    elif detection_scores.ndim == 3 and detection_scores.shape[1] == 1:
        relevance = probabilities
    else:
        raise ValueError("detection scores must have shape [batch, detections] or [batch, 1, detections]")
    if detection_scores.shape != relevance.shape:
        raise ValueError("detection and relevance score shapes differ")
    return {
        "detection_scores": detection_scores,
        "relevance_probabilities": relevance,
        "joint_scores": detection_scores * relevance,
    }


@dataclass(slots=True)
class RelevanceLossResult:
    total: torch.Tensor
    detection: torch.Tensor
    relevance: torch.Tensor
    relevance_matches: int
    metrics: dict[str, torch.Tensor]


class MaskedRelevanceCriterion:
    """Traffic-light detection plus masked per-instance relevance loss."""

    def __init__(
        self,
        model: nn.Module,
        *,
        relevance_weight: float = 1.0,
        alpha: float | None = None,
        gamma: float = 2.0,
    ) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        self.detection = v8DetectionLoss(model)
        if isinstance(self.detection.hyp, Mapping):
            self.detection.hyp = SimpleNamespace(**self.detection.hyp)
        self.relevance_weight = float(relevance_weight)
        self.alpha = alpha
        self.gamma = float(gamma)

    def __call__(
        self,
        predictions: Mapping[str, torch.Tensor] | tuple[Any, ...],
        batch: Mapping[str, torch.Tensor],
    ) -> RelevanceLossResult:
        parsed = self.detection.parse_output(predictions)
        if "relevance_logits" not in parsed:
            raise KeyError("missing relevance prediction: relevance_logits")
        for name in ("batch_idx", "tl_relevance", "tl_relevance_valid"):
            if name not in batch:
                raise KeyError(f"missing relevance batch field: {name}")

        assignments, detection_vector, detection_metrics = (
            self.detection.get_assigned_targets_and_loss(parsed, dict(batch))
        )
        foreground_mask, target_gt_indices = assignments[:2]
        device = parsed["scores"].device
        batch_size = parsed["scores"].shape[0]
        batch_indices = batch["batch_idx"].to(device)
        targets = _pad_targets(
            batch["tl_relevance"].to(device), batch_indices, batch_size
        )
        relevance_loss, relevance_matches = assigned_relevance_focal_bce(
            parsed["relevance_logits"],
            targets,
            foreground_mask,
            target_gt_indices,
            image_valid=batch["tl_relevance_valid"].to(device),
            alpha=self.alpha,
            gamma=self.gamma,
        )
        detection_total = detection_vector.sum() * batch_size
        total = detection_total + self.relevance_weight * relevance_loss
        return RelevanceLossResult(
            total=total,
            detection=detection_total,
            relevance=relevance_loss,
            relevance_matches=relevance_matches,
            metrics={
                **detection_metrics,
                "relevance_loss": relevance_loss.detach(),
            },
        )


def run_relevance_forward_smoke(
    wrapper: Any,
    *,
    input_size: tuple[int, int] = (800, 1600),
    device: str = "cuda",
    half: bool = True,
) -> dict[str, Any]:
    """Verify the full fixed-shape local-relevance forward path."""

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
    expected_shape = (1, 1, expected_locations)
    if tuple(raw["relevance_logits"].shape) != expected_shape:
        raise AssertionError(
            f"unexpected relevance shape: {raw['relevance_logits'].shape}"
        )
    relevance_parameters = sum(
        parameter.numel()
        for name, parameter in head.named_parameters()
        if name.startswith("relevance_heads")
    )
    return {
        "schema": "TLR-YOLO-MTL Milestone 5 local relevance smoke v1",
        "input_shape": [1, 3, height, width],
        "dtype": str(dtype).removeprefix("torch."),
        "levels": ["P3", "P4", "P5"],
        "p2_enabled": False,
        "traffic_detection_shape": list(decoded.shape),
        "relevance_shape": list(raw["relevance_logits"].shape),
        "relevance_activation": "independent_sigmoid",
        "relevance_parameters": int(relevance_parameters),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved))
            if resolved.type == "cuda"
            else None
        ),
    }


def run_relevance_assignment_smoke(
    wrapper: Any,
    *,
    device: str = "cuda",
    image_size: int = 320,
) -> dict[str, Any]:
    """Verify real TAL supervision and exact masking of local relevance."""

    resolved = torch.device(device)
    model = wrapper.model.to(resolved).float().train()
    criterion = MaskedRelevanceCriterion(model)
    image = torch.zeros((1, 3, image_size, image_size), device=resolved)
    common = {
        "batch_idx": torch.tensor([0], device=resolved),
        "cls": torch.tensor([[0.0]], device=resolved),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=resolved),
    }

    model.zero_grad(set_to_none=True)
    valid = criterion(
        model(image),
        {
            **common,
            "tl_relevance": torch.tensor([1], device=resolved),
            "tl_relevance_valid": torch.tensor([True], device=resolved),
        },
    )
    valid.total.backward()
    head = model.model[-1]
    valid_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.relevance_heads.parameters()
        if parameter.grad is not None
    )

    model.zero_grad(set_to_none=True)
    masked = criterion(
        model(image),
        {
            **common,
            "tl_relevance": torch.tensor([-1], device=resolved),
            "tl_relevance_valid": torch.tensor([False], device=resolved),
        },
    )
    masked.total.backward()
    masked_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.relevance_heads.parameters()
        if parameter.grad is not None
    )
    if not (
        valid.relevance_matches > 0
        and valid_gradient > 0
        and masked.relevance_matches == 0
        and masked_gradient == 0
    ):
        raise AssertionError("local relevance assignment or masking smoke failed")

    sample_scores = combine_detection_relevance_scores(
        torch.tensor([[0.8, 0.6]], device=resolved),
        torch.tensor([[[2.0, 1.0]]], device=resolved),
    )
    independent = sample_scores["relevance_probabilities"]
    return {
        "positive_relevance_matches": valid.relevance_matches,
        "relevance_head_gradient_sum": valid_gradient,
        "masked_relevance_matches": masked.relevance_matches,
        "masked_relevance_gradient_sum": masked_gradient,
        "multiple_relevant_probabilities": independent.detach().cpu().tolist()[0],
        "multiple_relevant_probability_sum": float(independent.sum()),
        "detection_scores_preserved": sample_scores["detection_scores"].detach().cpu().tolist()[0],
        "joint_scores": sample_scores["joint_scores"].detach().cpu().tolist()[0],
        "assignment_masking_ok": True,
    }
