"""Runtime diagnostics for the active paired multi-task training pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from ..model.context import DEFAULT_PAIRED_GRADIENT_SCALE
from ..model.unified import (
    UnifiedTrafficControlDetect,
    set_context_gradient_scale,
    set_cross_attention_enabled,
    set_relevance_perception_gradient_scale,
)
from .data import (
    CanonicalMultiTaskDataset,
    canonical_multitask_collate,
    source_group,
)
from .engine import (
    ExponentialMovingAverage,
    apply_training_overrides,
    build_adamw,
    build_multitask_criterion,
    build_training_components,
    configure_phase,
    load_training_config,
    move_batch_to_device,
    parse_phases,
    seed_everything,
)


def validate_smoke_context_batch(
    batch: Mapping[str, Any], *, require_all_paired: bool = False
) -> dict[str, Any]:
    """Validate legacy mixed or active DTLD-only context batches."""

    paired_value = batch.get("relevance_arrow_context_paired")
    scale_value = batch.get("relevance_arrow_context_scale")
    if not isinstance(paired_value, torch.Tensor) or not isinstance(
        scale_value, torch.Tensor
    ):
        raise TypeError("smoke batch must contain tensor context fields")

    paired = paired_value.reshape(-1).bool()
    scales = scale_value.reshape(-1).float()
    if paired.numel() != scales.numel():
        raise AssertionError("paired flags and context scales differ in length")
    paired_count = int(paired.sum().item())
    expected_count = paired.numel() if require_all_paired else 1
    if paired_count != expected_count:
        raise AssertionError(
            (
                "every DTLD-only smoke image must be paired"
                if require_all_paired
                else f"smoke batch must contain exactly one paired DTLD image, got {paired_count}"
            )
        )

    expected = torch.where(
        paired,
        torch.full_like(scales, DEFAULT_PAIRED_GRADIENT_SCALE),
        torch.zeros_like(scales),
    )
    if not torch.allclose(scales, expected):
        raise AssertionError(
            "smoke context scales must be 0.25 for paired DTLD and zero otherwise"
        )
    return {
        "context_paired_images": paired_count,
        "context_gradient_scales": scales.detach().cpu().tolist(),
    }


def _find_smoke_indices(dataset: CanonicalMultiTaskDataset) -> list[int]:
    selected: list[tuple[float, int]] = []
    for index, entry in enumerate(dataset.entries):
        group = source_group(entry.source_dataset)
        if group != "DTLD":
            continue
        record = dataset._record(index)
        if group == "DTLD":
            valid_lights = [
                item
                for item in record.traffic_lights
                if item.valid_relevance
                and item.valid_state
                and item.valid_pictogram
            ]
            if (
                not valid_lights
                or not record.task_valid.arrow_detection
                or not record.road_arrows
            ):
                continue
            normalized_area = max(
                (item.bbox_xyxy[2] - item.bbox_xyxy[0])
                * (item.bbox_xyxy[3] - item.bbox_xyxy[1])
                / (record.original_width * record.original_height)
                for item in valid_lights
            )
            selected.append((normalized_area, index))
    selected.sort(reverse=True)
    if len(selected) < 2:
        raise ValueError("could not find two paired DTLD smoke examples")
    return [index for _, index in selected[:2]]


def _find_memory_probe_indices(
    dataset: CanonicalMultiTaskDataset,
    micro_batch_size: int,
    quotas: Mapping[str, int],
) -> list[int]:
    """Choose a quota-proportional, task-bearing physical batch."""

    if micro_batch_size <= 0:
        raise ValueError("memory-probe batch must be positive")
    groups = tuple(quotas)
    quota_total = sum(int(quotas[group]) for group in groups)
    exact = {
        group: micro_batch_size * int(quotas[group]) / quota_total for group in groups
    }
    counts = {group: int(math.floor(exact[group])) for group in groups}
    remaining = micro_batch_size - sum(counts.values())
    ranked = sorted(
        groups,
        key=lambda group: (-(exact[group] - counts[group]), groups.index(group)),
    )
    for group in ranked[:remaining]:
        counts[group] += 1

    selected: dict[str, list[int]] = {group: [] for group in groups}
    for index, entry in enumerate(dataset.entries):
        group = source_group(entry.source_dataset)
        if group not in counts or len(selected[group]) >= counts[group]:
            continue
        record = dataset._record(index)
        valid = (
            (
                any(item.valid_relevance for item in record.traffic_lights)
                and record.task_valid.arrow_detection
                and bool(record.road_arrows)
            )
            if group == "DTLD"
            else bool(record.traffic_lights)
        )
        if valid:
            selected[group].append(index)
        if all(len(selected[name]) >= counts[name] for name in groups):
            break

    missing = {
        group: counts[group] - len(selected[group])
        for group in groups
        if len(selected[group]) < counts[group]
    }
    if missing:
        raise ValueError(f"not enough task-bearing samples for memory probe: {missing}")
    return [index for group in groups for index in selected[group]]


def run_multitask_training_smoke(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    device: str = "cuda",
    image_size: int = 320,
) -> dict[str, Any]:
    """Run one paired DTLD-only forward/backward without optimizer mutation."""

    config = load_training_config(config_path)
    seed_everything(int(config["seed"]))
    resolved = torch.device(device)
    wrapper, dataset, _, _ = build_training_components(
        config,
        weights_path=weights_path or config["warmstart_weights"],
        target_size=(image_size, image_size),
        quotas={"DTLD": 2},
        micro_batch_size=2,
        windows_per_epoch=1,
        training_augmentations=False,
    )
    indices = _find_smoke_indices(dataset)
    batch = canonical_multitask_collate([dataset[index] for index in indices])
    batch = move_batch_to_device(batch, resolved)
    model = wrapper.model.to(resolved).float().train()
    set_cross_attention_enabled(wrapper, True)
    set_relevance_perception_gradient_scale(wrapper, 1.0)
    set_context_gradient_scale(
        wrapper, batch["relevance_arrow_context_scale"]
    )
    head = model.model[-1]
    if not isinstance(head, UnifiedTrafficControlDetect):
        raise TypeError("smoke model does not have the complete head")
    with torch.no_grad():
        head.cross_attention.gate.fill_(0.1)
    criterion = build_multitask_criterion(model, config)
    model.zero_grad(set_to_none=True)
    result = criterion(model(batch["img"]), batch)
    if not torch.isfinite(result.total):
        raise AssertionError("multi-task smoke loss is not finite")
    result.total.backward()
    def gradient_sum(parameters: Iterable[nn.Parameter]) -> float:
        return sum(
            float(parameter.grad.abs().sum())
            for parameter in parameters
            if parameter.grad is not None
        )

    gradients = {
        "detection": gradient_sum(
            list(head.cv2.parameters()) + list(head.cv3.parameters())
        ),
        "state": gradient_sum(head.state_heads.parameters()),
        "round": gradient_sum(head.round_heads.parameters()),
        "maneuver": gradient_sum(head.maneuver_heads.parameters()),
        "relevance": gradient_sum(
            list(head.local_relevance_heads.parameters())
            + list(head.relevance_head.parameters())
        ),
        "cross_attention": gradient_sum(head.cross_attention.parameters()),
    }
    supervised_gradients = {
        name: value for name, value in gradients.items() if name != "cross_attention"
    }
    if not all(value > 0 for value in supervised_gradients.values()):
        raise AssertionError(f"one or more active heads received no gradient: {gradients}")
    context_summary = validate_smoke_context_batch(batch, require_all_paired=True)
    return {
        "schema": "TLR-YOLO-MTL unified paired training smoke v3",
        "input_shape": list(batch["img"].shape),
        "sources": batch["source_datasets"],
        "image_ids": batch["image_ids"],
        "model_variant": str(config["model_variant"]),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "p2_enabled": False,
        **context_summary,
        "losses": {
            "total": float(result.total.detach()),
            "detection": float(result.detection.detach()),
            "state": float(result.state.detach()),
            "round": float(result.round.detach()),
            "maneuver": float(result.maneuver.detach()),
            "ego_lane": float(result.ego_lane.detach()),
            "relevance": float(result.relevance.detach()),
            "nwd": float(result.nwd.detach()),
        },
        "matches": {
            "state": result.state_matches,
            "round": result.round_matches,
            "maneuver": result.maneuver_matches,
            "ego_lane": result.ego_lane_matches,
            "relevance": result.relevance_matches,
        },
        "gradient_sums": gradients,
        "all_supervised_heads_receive_gradient": True,
        "cross_attention_gradient_requires_selected_positive": True,
    }


def run_full_resolution_memory_probe(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    device: str = "cuda",
    micro_batch_size: int = 1,
    amp_initial_scale: float | None = None,
) -> dict[str, Any]:
    """Measure worst-phase AMP train memory at the final rectangular shape."""

    config = apply_training_overrides(
        load_training_config(config_path), micro_batch_size=micro_batch_size
    )
    if amp_initial_scale is not None:
        config["amp_initial_scale"] = float(amp_initial_scale)
        config = apply_training_overrides(config)
    seed_everything(int(config["seed"]))
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the full-resolution memory probe requires CUDA")
    wrapper, dataset, _, _ = build_training_components(
        config,
        weights_path=weights_path or config["warmstart_weights"],
        target_size=tuple(int(value) for value in config["input_size"]),
        windows_per_epoch=1,
        training_augmentations=False,
    )
    indices = _find_memory_probe_indices(
        dataset,
        int(config["micro_batch_size"]),
        config["source_quotas_per_effective_batch"],
    )
    batch = canonical_multitask_collate([dataset[index] for index in indices])
    batch = move_batch_to_device(batch, resolved)
    model = wrapper.model.to(resolved).float()
    joint_phase = next(
        phase for phase in parse_phases(config) if phase.name == "joint_finetuning"
    )
    configure_phase(model, joint_phase, wrapper)
    set_context_gradient_scale(
        wrapper, batch["relevance_arrow_context_scale"]
    )
    criterion = build_multitask_criterion(model, config)
    optimizer = build_adamw(
        model, joint_phase, weight_decay=float(config["weight_decay"])
    )
    ema = ExponentialMovingAverage(model, float(config["ema_decay"]))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=float(config.get("amp_initial_scale", 65536.0)),
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(resolved)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = criterion(model(batch["img"]), batch)
    if not torch.isfinite(result.total):
        raise AssertionError("full-resolution training loss is not finite")
    scaler.scale(result.total).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(config["gradient_clip_norm"]),
    )
    amp_scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    amp_scale_after = float(scaler.get_scale())
    optimizer_step_applied = amp_scale_after >= amp_scale_before
    if optimizer_step_applied:
        ema.update(model)
    torch.cuda.synchronize(resolved)
    peak_allocated = int(torch.cuda.max_memory_allocated(resolved))
    peak_reserved = int(torch.cuda.max_memory_reserved(resolved))
    total_memory = int(torch.cuda.get_device_properties(resolved).total_memory)
    source_counts = {
        source: batch["source_datasets"].count(source)
        for source in sorted(set(batch["source_datasets"]))
    }
    return {
        "schema": "TLR-YOLO-MTL unified full-resolution AMP memory probe v3",
        "image_ids": batch["image_ids"],
        "sources": batch["source_datasets"],
        "source_counts": source_counts,
        "input_shape": list(batch["img"].shape),
        "micro_batch_size": int(config["micro_batch_size"]),
        "effective_batch_via_accumulation": int(config["effective_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "dtype": "autocast_float16",
        "phase_profile": joint_phase.name,
        "model_variant": str(config["model_variant"]),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "p2_enabled": False,
        "loss": float(result.total.detach()),
        "gradient_norm": (
            float(gradient_norm) if bool(torch.isfinite(gradient_norm)) else None
        ),
        "amp_scale_before": amp_scale_before,
        "amp_scale_after": amp_scale_after,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": total_memory,
        "allocated_fraction_of_device": peak_allocated / total_memory,
        "optimizer_step_attempted": True,
        "optimizer_step_applied": optimizer_step_applied,
        "ema_update_executed": optimizer_step_applied,
        "checkpoint_written": False,
        "fits_memory": True,
        "finite_optimizer_step": optimizer_step_applied
        and bool(torch.isfinite(gradient_norm)),
    }
