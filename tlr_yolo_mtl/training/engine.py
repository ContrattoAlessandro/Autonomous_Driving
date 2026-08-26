"""Single-phase end-to-end and schedule-based training for unified TL/arrow perception."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation import evaluate_validation_epoch
from ..model.geometry_attention import (
    GeometryAwareUnifiedDetect,
    attach_geometry_aware_unified_relevance_head,
)
from ..model.milestone2 import build_detection_model, load_coco_warmstart
from ..model.unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
    set_context_gradient_scale,
    set_cross_attention_enabled,
    set_relevance_perception_gradient_scale,
)
from .data import (
    BalancedEffectiveBatchSampler,
    CanonicalMultiTaskDataset,
    canonical_multitask_collate,
)
from .losses import MultiTaskLossWeights, TLRMultiTaskCriterion


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    name: str
    epochs: int
    freeze_backbone: bool
    context_enabled: bool
    backbone_lr: float
    head_lr: float
    freeze_batchnorm: bool = False
    freeze_neck: bool = False
    freeze_perception: bool = False
    relevance_perception_gradient_scale: float = 0.0
    relevance_perception_gradient_scale_end: float | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"training config must contain a mapping: {config_path}")
    _validate_training_config(config)
    return config


def _validate_training_config(config: Mapping[str, Any]) -> None:
    if not str(config.get("model_variant", "")).strip():
        raise ValueError("model_variant must identify the active model scale")
    if not str(config.get("model_config", "")).strip():
        raise ValueError("model_config must select an active model YAML")
    if not str(config.get("warmstart_weights", "")).strip():
        raise ValueError("warmstart_weights must select matching weights")
    input_size = tuple(config.get("input_size", ()))
    if len(input_size) != 2 or min(int(value) for value in input_size) <= 0:
        raise ValueError("input_size must contain positive [height, width]")
    effective = int(config["effective_batch_size"])
    micro = int(config["micro_batch_size"])
    accumulation = int(config["gradient_accumulation_steps"])
    if effective != micro * accumulation:
        raise ValueError("effective batch must equal micro batch × accumulation steps")
    quotas = config["source_quotas_per_effective_batch"]
    training_sources = set(config.get("training_sources", ()))
    active_quota_groups = {"DTLD"} if training_sources == {"DTLD"} else {"DTLD", "AUX_TL"}
    if set(quotas) != active_quota_groups:
        raise ValueError(
            "active source quota groups must be exactly "
            f"{sorted(active_quota_groups)}, got {sorted(quotas)}"
        )
    if sum(int(value) for value in quotas.values()) != effective:
        raise ValueError("source quotas must sum to the effective batch size")
    if int(config["workers"]) < 0:
        raise ValueError("workers must be non-negative")
    if int(config.get("prefetch_factor", 2)) <= 0:
        raise ValueError("prefetch_factor must be positive")
    if float(config.get("amp_initial_scale", 65536.0)) <= 0:
        raise ValueError("amp_initial_scale must be positive")
    steps_per_epoch = config.get("optimizer_steps_per_epoch")
    if steps_per_epoch is not None and int(steps_per_epoch) <= 0:
        raise ValueError("optimizer_steps_per_epoch must be positive or null")
    if training_sources and training_sources != {"DTLD"}:
        raise ValueError("the active unified architecture is DTLD-only")
    architecture = config.get("architecture", {})
    if architecture:
        head_kwargs = {
            k: v for k, v in architecture.items()
            if k in UnifiedHeadConfig.__dataclass_fields__
        }
        UnifiedHeadConfig(**head_kwargs).validate()
    for phase in config.get("phases", ()):
        start = float(phase.get("relevance_perception_gradient_scale", 0.0))
        end = float(phase.get("relevance_perception_gradient_scale_end", start))
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            raise ValueError("relevance perception gradient scales must be in [0, 1]")


def apply_training_overrides(
    config: Mapping[str, Any],
    *,
    micro_batch_size: int | None = None,
    optimizer_steps_per_epoch: int | None = None,
    workers: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Apply safe runtime overrides while preserving the effective batch."""

    resolved = dict(config)
    if micro_batch_size is not None:
        micro = int(micro_batch_size)
        effective = int(resolved["effective_batch_size"])
        if micro <= 0:
            raise ValueError("micro batch size must be positive")
        if effective % micro:
            raise ValueError(
                f"effective batch {effective} must be divisible by micro batch {micro}"
            )
        resolved["micro_batch_size"] = micro
        resolved["gradient_accumulation_steps"] = effective // micro
    if optimizer_steps_per_epoch is not None:
        resolved["optimizer_steps_per_epoch"] = int(optimizer_steps_per_epoch)
    if workers is not None:
        resolved["workers"] = int(workers)
    if device is not None:
        resolved["device"] = str(device)
    _validate_training_config(resolved)
    return resolved


def parse_phases(config: Mapping[str, Any]) -> list[PhaseSpec]:
    phases = [PhaseSpec(**value) for value in config["phases"]]
    if not phases or any(phase.epochs <= 0 for phase in phases):
        raise ValueError("training phases must have positive epoch counts")
    return phases


def assert_active_pyramid(model: nn.Module, *, p2_enabled: bool = False) -> None:
    head = model.model[-1]
    strides = tuple(int(value) for value in head.stride.tolist())
    expected = (4, 8, 16, 32) if (p2_enabled or len(strides) == 4) else (8, 16, 32)
    if strides != expected:
        raise AssertionError(f"active training requires {expected} strides, got {strides}")


def configure_phase(model: nn.Module, phase: PhaseSpec, wrapper: Any) -> None:
    """Apply feature, perception, attention and BatchNorm policy for a phase."""

    final_index = len(model.model) - 1
    backbone_end = 10
    for idx, mod in enumerate(model.model):
        if "Upsample" in type(mod).__name__:
            backbone_end = idx - 1
            break
    for index, module in enumerate(model.model):
        backbone_frozen = phase.freeze_backbone and index <= backbone_end
        neck_frozen = phase.freeze_neck and backbone_end < index < final_index
        trainable = not backbone_frozen and not neck_frozen
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)
    head = model.model[-1]
    if phase.freeze_perception:
        if not isinstance(head, UnifiedTrafficControlDetect):
            raise TypeError("freeze_perception requires the unified head")
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        for parameter in head.context_parameters():
            parameter.requires_grad_(True)
    set_cross_attention_enabled(wrapper, phase.context_enabled)
    set_relevance_perception_gradient_scale(
        wrapper, phase.relevance_perception_gradient_scale
    )
    model.train()
    if phase.freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def _is_feature_extractor_parameter(name: str, final_index: int = 23) -> bool:
    fields = name.split(".")
    if len(fields) >= 2 and fields[0] == "model":
        try:
            return int(fields[1]) < final_index
        except ValueError:
            pass
    return False


def build_adamw(
    model: nn.Module,
    phase: PhaseSpec,
    *,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("feature_extractor", True): [],
        ("feature_extractor", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    final_index = len(model.model) - 1
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        region = (
            "feature_extractor"
            if _is_feature_extractor_parameter(name, final_index)
            else "head"
        )
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(region, use_decay)].append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    for (region, use_decay), parameters in groups.items():
        if not parameters:
            continue
        learning_rate = (
            phase.backbone_lr if region == "feature_extractor" else phase.head_lr
        )
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay if use_decay else 0.0,
                "region": region,
            }
        )
    return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999))


def compute_module_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Compute Frobenius gradient norms for functional submodules."""

    categories: dict[str, list[torch.Tensor]] = {
        "backbone": [],
        "neck": [],
        "detect": [],
        "attributes": [],
        "cross_attention": [],
        "relevance": [],
    }
    inner_model = getattr(model, "model", None)
    final_index = len(inner_model) - 1 if inner_model is not None else -1
    backbone_end = 10
    if inner_model is not None:
        for idx, mod in enumerate(inner_model):
            if "Upsample" in type(mod).__name__:
                backbone_end = idx - 1
                break

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if "model." in name:
            fields = name.split(".")
            idx = -1
            for f in fields:
                if f.isdigit():
                    idx = int(f)
                    break
            if 0 <= idx <= backbone_end:
                categories["backbone"].append(grad)
            elif backbone_end < idx < final_index:
                categories["neck"].append(grad)
            elif idx == final_index:
                if any(
                    attr in name
                    for attr in (
                        "state_heads",
                        "round_heads",
                        "maneuver_heads",
                        "ego_lane_heads",
                    )
                ):
                    categories["attributes"].append(grad)
                elif any(
                    ctx in name
                    for ctx in (
                        "cross_attention",
                        "token_projection",
                        "traffic_token",
                        "arrow_token",
                    )
                ):
                    categories["cross_attention"].append(grad)
                elif "relevance" in name:
                    categories["relevance"].append(grad)
                else:
                    categories["detect"].append(grad)
            else:
                categories["detect"].append(grad)
        else:
            categories["detect"].append(grad)

    norms: dict[str, float] = {}
    for cat_name, grads in categories.items():
        if grads:
            total_norm_sq = sum(float(g.float().norm(2).item() ** 2) for g in grads)
            norms[cat_name] = round(math.sqrt(total_norm_sq), 4)
        else:
            norms[cat_name] = 0.0
    return norms


class ExponentialMovingAverage:
    """Small state-dict EMA kept on the model device for affordable updates."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA checkpoint does not contain a shadow mapping")
        missing = set(self.shadow).difference(shadow)
        unexpected = set(shadow).difference(self.shadow)
        if missing or unexpected:
            raise ValueError(
                "EMA checkpoint keys differ from the active model: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.shadow = {
            name: shadow[name].detach().to(
                device=current.device, dtype=current.dtype
            ).clone()
            for name, current in self.shadow.items()
        }


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor | list[str]], device: torch.device
) -> dict[str, torch.Tensor | list[str]]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def _loss_weights(config: Mapping[str, Any]) -> MultiTaskLossWeights:
    return MultiTaskLossWeights(**config["loss_weights"])


def build_multitask_criterion(
    model: nn.Module, config: Mapping[str, Any]
) -> TLRMultiTaskCriterion:
    loss = config.get("loss", {})
    tal_assigner_cfg = config.get("tal_assigner", {})
    tal_assigner_type = tal_assigner_cfg.get("type", "standard") if isinstance(tal_assigner_cfg, Mapping) else "standard"
    state_loss_type = loss.get("state_loss_type", "focal")
    state_beta = float(loss.get("state_beta", 0.9999))
    state_prior_scale = float(loss.get("state_prior_scale", 1.0))
    distillation_cfg = config.get("distillation")
    quality_cfg = (
        config.get("quality_head")
        or config.get("architecture", {}).get("quality_head")
        or config.get("architecture", {}).get("nwd_quality_head")
    )
    refinement_cfg = (
        config.get("sparse_refinement")
        or config.get("architecture", {}).get("sparse_refinement")
    )
    temporal_cfg = config.get("temporal_distillation")
    return TLRMultiTaskCriterion(
        model,
        weights=_loss_weights(config),
        state_loss_type=state_loss_type,
        state_beta=state_beta,
        state_prior_scale=state_prior_scale,
        attribute_gamma=float(loss.get("attribute_focal_gamma", 1.5)),
        maneuver_gamma=float(loss.get("maneuver_focal_gamma", 2.0)),
        ego_lane_gamma=float(loss.get("ego_lane_focal_gamma", 2.0)),
        relevance_gamma=float(loss.get("relevance_focal_gamma", 2.0)),
        nwd_constant=float(loss.get("nwd_constant", 12.0)),
        tal_assigner_type=tal_assigner_type,
        tal_assigner_config=tal_assigner_cfg if isinstance(tal_assigner_cfg, Mapping) and tal_assigner_cfg else None,
        distillation_config=distillation_cfg if isinstance(distillation_cfg, Mapping) and distillation_cfg else None,
        quality_config=quality_cfg if isinstance(quality_cfg, Mapping) and quality_cfg else None,
        refinement_config=refinement_cfg if isinstance(refinement_cfg, Mapping) and refinement_cfg else None,
        temporal_distillation_config=temporal_cfg if isinstance(temporal_cfg, Mapping) and temporal_cfg else None,
    )




_LOSS_COMPONENT_NAMES = (
    "detection",
    "state",
    "round",
    "maneuver",
    "ego_lane",
    "relevance",
    "nwd",
    "association",
    "contrastive",
    "distillation",
    "quality",
    "refinement",
    "temporal_distillation",
    "total",
)


def _loss_snapshot(losses: Any) -> dict[str, float]:
    return {
        name: float(getattr(losses, name).detach().float())
        for name in _LOSS_COMPONENT_NAMES
        if hasattr(losses, name) and getattr(losses, name) is not None
    }


def _raise_for_nonfinite_loss(
    losses: Any,
    batch: Mapping[str, Any],
    *,
    output: Path,
    phase: PhaseSpec,
    global_epoch: int,
    micro_step: int,
    optimizer_steps: int,
) -> dict[str, float]:
    snapshot = _loss_snapshot(losses)
    nonfinite = [name for name, value in snapshot.items() if not math.isfinite(value)]
    if not nonfinite:
        return snapshot
    failure = {
        "schema": "TLR-YOLO-MTL non-finite loss failure v1",
        "phase": phase.name,
        "global_epoch": global_epoch,
        "micro_step": micro_step,
        "optimizer_steps": optimizer_steps,
        "nonfinite_components": nonfinite,
        "losses": {
            name: value if math.isfinite(value) else str(value)
            for name, value in snapshot.items()
        },
        "image_ids": list(batch["image_ids"]),
        "sources": list(batch["source_datasets"]),
    }
    failure_path = output / "failure.json"
    failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    raise FloatingPointError(
        "non-finite forward loss; training stopped before backward. "
        f"components={nonfinite}, diagnostic={failure_path}"
    )


def build_training_components(
    config: Mapping[str, Any],
    *,
    weights_path: str | Path | None = "yolo11n.pt",
    target_size: tuple[int, int] | None = None,
    quotas: Mapping[str, int] | None = None,
    micro_batch_size: int | None = None,
    windows_per_epoch: int | None = None,
    training_augmentations: bool = True,
) -> tuple[Any, CanonicalMultiTaskDataset, BalancedEffectiveBatchSampler, DataLoader]:
    wrapper = build_detection_model(config["model_config"])
    if weights_path is not None:
        load_coco_warmstart(wrapper, weights_path)
    arch = config.get("architecture", {})
    head_kwargs = {
        k: v for k, v in arch.items()
        if k in UnifiedHeadConfig.__dataclass_fields__
    }
    if arch.get("geometry_attention", {}).get("enabled", False):
        geom_cfg = arch.get("geometry_attention", {})
        attach_geometry_aware_unified_relevance_head(
            wrapper,
            config=UnifiedHeadConfig(**head_kwargs),
            hidden_dim=int(geom_cfg.get("hidden_dim", 32)),
            p_drop=float(geom_cfg.get("p_drop", 0.0)),
            use_confidence_gating=bool(geom_cfg.get("use_confidence_gate", True)),
        )
    else:
        attach_unified_relevance_head(
            wrapper,
            config=UnifiedHeadConfig(**head_kwargs),
        )
    assert_active_pyramid(
        wrapper.model, p2_enabled=bool(config.get("p2_enabled", False))
    )

    resolved_size = target_size or tuple(int(value) for value in config["input_size"])
    dataset = CanonicalMultiTaskDataset(
        config["records"],
        split="train",
        target_size=resolved_size,
        training=training_augmentations,
        horizontal_flip=bool(config.get("horizontal_flip", False)),
        seed=int(config["seed"]),
        allowed_sources=tuple(config.get("training_sources", ("DTLD",))),
        require_paired=bool(config.get("require_paired", True)),
    )
    resolved_quotas = quotas or {
        name: int(value)
        for name, value in config["source_quotas_per_effective_batch"].items()
    }
    resolved_windows_per_epoch = (
        windows_per_epoch
        if windows_per_epoch is not None
        else config.get("optimizer_steps_per_epoch")
    )
    sampler = BalancedEffectiveBatchSampler(
        dataset.entries,
        micro_batch_size=(
            int(micro_batch_size)
            if micro_batch_size is not None
            else int(config["micro_batch_size"])
        ),
        quotas=resolved_quotas,
        seed=int(config["seed"]),
        windows_per_epoch=(
            int(resolved_windows_per_epoch)
            if resolved_windows_per_epoch is not None
            else None
        ),
    )
    workers = int(config["workers"])
    loader_options: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "collate_fn": canonical_multitask_collate,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_options)
    return wrapper, dataset, sampler, loader


def build_validation_loader(
    config: Mapping[str, Any],
    *,
    target_size: tuple[int, int] | None = None,
    val_batch_size: int | None = None,
    workers: int | None = None,
) -> tuple[CanonicalMultiTaskDataset | None, DataLoader | None]:
    records_path = Path(config["records"]).resolve()
    resolved_size = target_size or tuple(int(value) for value in config["input_size"])
    try:
        dataset = CanonicalMultiTaskDataset(
            records_path,
            split="val",
            target_size=resolved_size,
            training=False,
            horizontal_flip=False,
            seed=int(config["seed"]),
            allowed_sources=tuple(config.get("training_sources", ("DTLD",))),
            require_paired=bool(config.get("require_paired", True)),
        )
    except Exception:
        return None, None
    if not dataset.entries:
        return None, None
    resolved_workers = int(workers if workers is not None else config.get("workers", 2))
    batch_size = (
        int(val_batch_size)
        if val_batch_size is not None
        else int(config.get("val_batch_size", config["micro_batch_size"]))
    )
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": resolved_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": resolved_workers > 0,
        "collate_fn": canonical_multitask_collate,
        "shuffle": False,
    }
    if resolved_workers > 0:
        loader_options["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_options)
    return dataset, loader


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


CHECKPOINT_SCHEMA = "TLR-YOLO-MTL unified-attention training checkpoint v3"
_RESUME_CONTRACT_KEYS = (
    "seed",
    "records",
    "model_variant",
    "model_config",
    "warmstart_weights",
    "input_size",
    "amp",
    "amp_initial_scale",
    "effective_batch_size",
    "gradient_clip_norm",
    "weight_decay",
    "ema_decay",
    "p2_enabled",
    "source_quotas_per_effective_batch",
    "training_sources",
    "require_paired",
    "architecture",
    "loss_weights",
    "loss",
    "tal_assigner",
    "phases",
)


def _load_training_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"training checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        schema = payload.get("schema") if isinstance(payload, dict) else type(payload)
        raise ValueError(
            f"resume requires {CHECKPOINT_SCHEMA!r}, found {schema!r}; "
            "old or bounded-trial checkpoints cannot be resumed safely"
        )
    if not bool(payload.get("epoch_complete", False)):
        raise ValueError(
            "the checkpoint was written in the middle of an epoch; use it only "
            "as a bounded smoke result, not as a resume point"
        )
    return checkpoint_path, payload


def _assert_resume_contract(
    current: Mapping[str, Any], saved: Mapping[str, Any]
) -> None:
    differences = [
        key
        for key in _RESUME_CONTRACT_KEYS
        if current.get(key) != saved.get(key)
    ]
    if differences:
        raise ValueError(
            "the current config differs from the checkpoint training contract in: "
            + ", ".join(differences)
        )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def train_from_config(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    output_dir: str | Path = "runs/tlr_yolo_mtl",
    only_phases: Sequence[str] | None = None,
    max_optimizer_steps: int | None = None,
    micro_batch_size: int | None = None,
    optimizer_steps_per_epoch: int | None = None,
    workers: int | None = None,
    device: str | None = None,
    resume_checkpoint: str | Path | None = None,
    overwrite: bool = False,
    val_steps_per_epoch: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Execute the configured training schedule with safe runtime overrides and epoch resume."""

    base_config = load_training_config(config_path)
    resume_path: Path | None = None
    resume: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        if only_phases:
            raise ValueError("--resume cannot be combined with phase filtering")
        resume_path, resume = _load_training_checkpoint(resume_checkpoint)
        if not bool(resume.get("full_schedule", False)):
            raise ValueError(
                "a checkpoint produced with phase filtering is not a valid full-run "
                "resume point"
            )
        saved_config = resume.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("checkpoint does not contain a valid training config")
        _validate_training_config(saved_config)
        _assert_resume_contract(base_config, saved_config)
        saved_micro_batch = int(saved_config["micro_batch_size"])
        if (
            micro_batch_size is not None
            and int(micro_batch_size) != saved_micro_batch
        ):
            raise ValueError(
                "the resumed micro batch must match the checkpoint: "
                f"requested {micro_batch_size}, saved {saved_micro_batch}"
            )
        saved_steps_per_epoch = saved_config.get("optimizer_steps_per_epoch")
        if (
            optimizer_steps_per_epoch is not None
            and int(optimizer_steps_per_epoch) != saved_steps_per_epoch
        ):
            raise ValueError(
                "the resumed steps per epoch must match the checkpoint: "
                f"requested {optimizer_steps_per_epoch}, "
                f"saved {saved_steps_per_epoch}"
            )
        config = apply_training_overrides(
            base_config,
            micro_batch_size=saved_micro_batch,
            optimizer_steps_per_epoch=(
                int(saved_steps_per_epoch)
                if saved_steps_per_epoch is not None
                else None
            ),
            workers=workers,
            device=device,
        )
        config["optimizer_steps_per_epoch"] = saved_steps_per_epoch
        _validate_training_config(config)
    else:
        config = apply_training_overrides(
            base_config,
            micro_batch_size=micro_batch_size,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            workers=workers,
            device=device,
        )

    seed_everything(int(config["seed"]))
    resolved_device = torch.device(config["device"])
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")
    if resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output = Path(output_dir).resolve()
    existing_last = output / "weights" / "last.pt"
    existing_metrics = output / "metrics.jsonl"
    if resume is None and (existing_last.exists() or existing_metrics.exists()):
        if overwrite:
            if existing_last.exists():
                existing_last.unlink()
            if existing_metrics.exists():
                existing_metrics.unlink()
        else:
            raise FileExistsError(
                f"run directory already contains training artifacts: {output}; "
                "use --resume for a valid checkpoint, pass --overwrite, or choose a new --output-dir"
            )

    resolved_weights = (
        None
        if resume is not None
        else weights_path
        if weights_path is not None
        else config["warmstart_weights"]
    )
    wrapper, dataset, sampler, loader = build_training_components(
        config, weights_path=resolved_weights
    )
    val_dataset, val_loader = build_validation_loader(
        config,
        workers=workers,
    )
    model = wrapper.model.to(resolved_device)
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
    phases = parse_phases(config)
    if only_phases:
        selected = set(only_phases)
        phases = [phase for phase in phases if phase.name in selected]
        unknown = selected.difference(phase.name for phase in phases)
        if unknown:
            raise ValueError(f"unknown training phases: {sorted(unknown)}")
    criterion = build_multitask_criterion(model, config)
    amp_enabled = bool(config["amp"]) and resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(config.get("amp_initial_scale", 65536.0)),
    )
    ema = ExponentialMovingAverage(model, float(config["ema_decay"]))
    if resume is not None:
        ema.load_state_dict(resume["ema"])
        scaler.load_state_dict(resume["scaler"])

    log_path = output / "metrics.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    optimizer_steps = int(resume["optimizer_steps"]) if resume is not None else 0
    global_epoch = int(resume["global_epoch"]) + 1 if resume is not None else 0
    best_val_score = (
        float(resume.get("best_val_score", -float("inf")))
        if resume is not None and resume.get("best_val_score") is not None
        else -float("inf")
    )
    best_tl_det_score = (
        float(resume.get("best_tl_det_score", -float("inf")))
        if resume is not None and resume.get("best_tl_det_score") is not None
        else -float("inf")
    )
    best_relevance_score = (
        float(resume.get("best_relevance_score", -float("inf")))
        if resume is not None and resume.get("best_relevance_score") is not None
        else -float("inf")
    )
    best_relevant_red_score = (
        float(resume.get("best_relevant_red_score", -float("inf")))
        if resume is not None and resume.get("best_relevant_red_score") is not None
        else -float("inf")
    )
    amp_overflow_count = (
        int(resume.get("amp_overflow_count", 0)) if resume is not None else 0
    )
    grad_clipped_count = (
        int(resume.get("grad_clipped_count", 0)) if resume is not None else 0
    )
    resume_phase_index = -1
    resume_phase_epoch = -1
    if resume is not None:
        saved_phase_name = str(resume["phase"]["name"])
        matching = [
            index for index, phase in enumerate(phases) if phase.name == saved_phase_name
        ]
        if len(matching) != 1:
            raise ValueError(
                f"checkpoint phase {saved_phase_name!r} is absent or ambiguous"
            )
        resume_phase_index = matching[0]
        resume_phase_epoch = int(resume["phase_epoch"])
        expected_global_epoch = sum(
            phase.epochs for phase in phases[:resume_phase_index]
        ) + resume_phase_epoch
        if int(resume["global_epoch"]) != expected_global_epoch:
            raise ValueError(
                "checkpoint epoch metadata is inconsistent with the configured phases"
            )
        _restore_rng_state(resume["rng_state"])

    started = time.perf_counter()
    last_console_update = started
    epochs_entered = 0
    epochs_completed = 0
    stopped_by_step_cap = False

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        for phase_index, phase in enumerate(phases):
            if resume is not None and phase_index < resume_phase_index:
                continue
            start_phase_epoch = (
                resume_phase_epoch + 1
                if resume is not None and phase_index == resume_phase_index
                else 0
            )
            if start_phase_epoch >= phase.epochs:
                continue
            configure_phase(model, phase, wrapper)
            optimizer = build_adamw(
                model, phase, weight_decay=float(config["weight_decay"])
            )
            accumulation = sampler.accumulation_steps
            total_phase_steps = max(1, phase.epochs * len(loader) // accumulation)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_phase_steps, eta_min=1e-6
            )
            if resume is not None and phase_index == resume_phase_index:
                optimizer.load_state_dict(resume["optimizer"])
                scheduler.load_state_dict(resume["scheduler"])
            optimizer.zero_grad(set_to_none=True)

            for epoch in range(start_phase_epoch, phase.epochs):
                scale_end = (
                    phase.relevance_perception_gradient_scale
                    if phase.relevance_perception_gradient_scale_end is None
                    else phase.relevance_perception_gradient_scale_end
                )
                progress = 1.0 if phase.epochs == 1 else epoch / (phase.epochs - 1)
                phase_gradient_scale = (
                    phase.relevance_perception_gradient_scale
                    + (scale_end - phase.relevance_perception_gradient_scale) * progress
                )
                set_relevance_perception_gradient_scale(wrapper, phase_gradient_scale)
                epochs_entered += 1
                dataset.set_epoch(global_epoch)
                sampler.set_epoch(global_epoch)
                micro_steps = 0
                window_loss = 0.0
                window_micro_steps = 0
                window_task_matches = {
                    "state": 0,
                    "round": 0,
                    "maneuver": 0,
                    "ego_lane": 0,
                    "relevance": 0,
                }
                window_loss_components = {
                    name: 0.0 for name in _LOSS_COMPONENT_NAMES
                }
                reached_step_cap = False
                for micro_step, raw_batch in enumerate(loader, 1):
                    batch = move_batch_to_device(raw_batch, resolved_device)
                    set_context_gradient_scale(
                        wrapper, batch["relevance_arrow_context_scale"]
                    )
                    with torch.autocast(
                        device_type=resolved_device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        predictions = model(batch["img"])
                        losses = criterion(predictions, batch)
                    loss_snapshot = _raise_for_nonfinite_loss(
                        losses,
                        batch,
                        output=output,
                        phase=phase,
                        global_epoch=global_epoch,
                        micro_step=micro_step,
                        optimizer_steps=optimizer_steps,
                    )
                    scaled_loss = losses.total / accumulation
                    scaler.scale(scaled_loss).backward()
                    micro_steps += 1
                    window_loss += loss_snapshot["total"]
                    window_micro_steps += 1
                    for name, value in loss_snapshot.items():
                        window_loss_components[name] += value
                    window_task_matches["state"] += losses.state_matches
                    window_task_matches["round"] += losses.round_matches
                    window_task_matches["maneuver"] += losses.maneuver_matches
                    window_task_matches["ego_lane"] += losses.ego_lane_matches
                    window_task_matches["relevance"] += losses.relevance_matches

                    if micro_step % accumulation == 0:
                        scaler.unscale_(optimizer)
                        raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                            [
                                parameter
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ],
                            float(config["gradient_clip_norm"]),
                        )
                        finite_gradient_norm = bool(torch.isfinite(raw_grad_norm))
                        if finite_gradient_norm and float(raw_grad_norm) > float(
                            config["gradient_clip_norm"]
                        ):
                            grad_clipped_count += 1
                        module_grad_norms = (
                            compute_module_gradient_norms(model)
                            if finite_gradient_norm
                            else {}
                        )

                        amp_scale_before = float(scaler.get_scale())
                        scaler.step(optimizer)
                        scaler.update()
                        amp_scale_after = float(scaler.get_scale())
                        optimizer.zero_grad(set_to_none=True)
                        step_applied = (
                            not amp_enabled or amp_scale_after >= amp_scale_before
                        )
                        if not step_applied:
                            amp_overflow_count += 1
                        if step_applied:
                            scheduler.step()
                            ema.update(model)
                            optimizer_steps += 1
                        event = {
                            "phase": phase.name,
                            "epoch": global_epoch,
                            "optimizer_step": optimizer_steps,
                            "optimizer_step_applied": step_applied,
                            "amp_scale_before": amp_scale_before,
                            "amp_scale_after": amp_scale_after,
                            "amp_overflow_count": amp_overflow_count,
                            "grad_clipped_rate": round(
                                grad_clipped_count / max(1, optimizer_steps), 4
                            ),
                            "mean_micro_loss": window_loss / window_micro_steps,
                            "mean_loss_components": {
                                name: value / window_micro_steps
                                for name, value in window_loss_components.items()
                            },
                            "gradient_norm": (
                                float(raw_grad_norm) if finite_gradient_norm else None
                            ),
                            "module_gradient_norms": module_grad_norms,
                            "learning_rates": [
                                float(group["lr"]) for group in optimizer.param_groups
                            ],
                            "task_matches": dict(window_task_matches),
                        }
                        log.write(json.dumps(event) + "\n")
                        log.flush()
                        now = time.perf_counter()
                        if (
                            not step_applied
                            or optimizer_steps == 1
                            or now - last_console_update >= 30.0
                        ):
                            status = "train" if step_applied else "amp-overflow"
                            norm_text = (
                                f"{float(raw_grad_norm):.3f}"
                                if finite_gradient_norm
                                else "non-finite"
                            )
                            print(
                                f"[{status}] phase={phase.name} "
                                f"epoch={epoch + 1}/{phase.epochs} "
                                f"step={optimizer_steps} "
                                f"loss={window_loss / window_micro_steps:.4f} "
                                f"grad_norm={norm_text} amp_scale={amp_scale_after:g}",
                                flush=True,
                            )
                            last_console_update = now
                        if (
                            step_applied
                            and max_optimizer_steps is not None
                            and optimizer_steps >= max_optimizer_steps
                        ):
                            reached_step_cap = True
                            break
                        window_loss = 0.0
                        window_micro_steps = 0
                        window_task_matches = {
                            name: 0 for name in window_task_matches
                        }
                        window_loss_components = {
                            name: 0.0 for name in window_loss_components
                        }
                epoch_complete = micro_steps == len(loader)
                checkpoint = {
                    "schema": CHECKPOINT_SCHEMA,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "phase": asdict(phase),
                    "phase_epoch": epoch,
                    "global_epoch": global_epoch,
                    "optimizer_steps": optimizer_steps,
                    "micro_steps_completed": micro_steps,
                    "epoch_complete": epoch_complete,
                    "full_schedule": only_phases is None,
                    "best_val_score": (
                        best_val_score if math.isfinite(best_val_score) else None
                    ),
                    "best_tl_det_score": (
                        best_tl_det_score if math.isfinite(best_tl_det_score) else None
                    ),
                    "best_relevance_score": (
                        best_relevance_score
                        if math.isfinite(best_relevance_score)
                        else None
                    ),
                    "best_relevant_red_score": (
                        best_relevant_red_score
                        if math.isfinite(best_relevant_red_score)
                        else None
                    ),
                    "amp_overflow_count": amp_overflow_count,
                    "grad_clipped_count": grad_clipped_count,
                    "rng_state": _capture_rng_state(),
                    "config": config,
                }
                _atomic_checkpoint(output / "weights" / "last.pt", checkpoint)
                if epoch_complete and (
                    global_epoch + 1
                ) % int(config["checkpoint_every"]) == 0:
                    _atomic_checkpoint(
                        output / "weights" / f"epoch_{global_epoch + 1:03d}.pt",
                        checkpoint,
                    )

                if (
                    epoch_complete
                    and not skip_validation
                    and val_loader is not None
                    and val_dataset is not None
                    and len(val_dataset) > 0
                ):
                    print(
                        f"\n[VAL] Epoch {epoch + 1}/{phase.epochs} ({phase.name}) | Global Epoch {global_epoch + 1} | Inizio validazione...",
                        flush=True,
                    )
                    val_res = evaluate_validation_epoch(
                        model,
                        val_loader,
                        criterion=criterion,
                        device=resolved_device,
                        amp_enabled=amp_enabled,
                        max_batches=(
                            int(val_steps_per_epoch)
                            if val_steps_per_epoch is not None
                            else (
                                int(config["val_steps_per_epoch"])
                                if config.get("val_steps_per_epoch") is not None
                                else None
                            )
                        ),
                    )
                    det = val_res["detection"]
                    rel = val_res["relevance"]
                    att = val_res["attributes"]
                    v_loss = val_res["mean_losses"]
                    score = float(val_res["selection_score"])
                    tl_det_score = float(det.get("ap_tl_50", det.get("map50", 0.0)))
                    rel_score = float(rel.get("auprc", 0.0))
                    rel_red_score = float(rel.get("relevant_red_recall", 0.0))

                    is_best = score > best_val_score
                    if is_best:
                        best_val_score = score
                        best_checkpoint = dict(checkpoint)
                        best_checkpoint["best_val_score"] = best_val_score
                        best_checkpoint["val_metrics"] = val_res
                        _atomic_checkpoint(output / "weights" / "best.pt", best_checkpoint)
                        _atomic_checkpoint(
                            output / "weights" / "best_composite.pt", best_checkpoint
                        )

                    if tl_det_score > best_tl_det_score:
                        best_tl_det_score = tl_det_score
                        ckpt_det = dict(checkpoint)
                        ckpt_det["best_tl_det_score"] = best_tl_det_score
                        ckpt_det["val_metrics"] = val_res
                        _atomic_checkpoint(
                            output / "weights" / "best_tl_detection.pt", ckpt_det
                        )

                    if rel_score > best_relevance_score:
                        best_relevance_score = rel_score
                        ckpt_rel = dict(checkpoint)
                        ckpt_rel["best_relevance_score"] = best_relevance_score
                        ckpt_rel["val_metrics"] = val_res
                        _atomic_checkpoint(
                            output / "weights" / "best_relevance.pt", ckpt_rel
                        )

                    if rel_red_score > best_relevant_red_score:
                        best_relevant_red_score = rel_red_score
                        ckpt_rrr = dict(checkpoint)
                        ckpt_rrr["best_relevant_red_score"] = best_relevant_red_score
                        ckpt_rrr["val_metrics"] = val_res
                        _atomic_checkpoint(
                            output / "weights" / "best_relevant_red_recall.pt", ckpt_rrr
                        )

                    best_tag = (
                        "★ [BEST COMPOSITE] Nuovo checkpoint migliore salvato in weights/best.pt"
                        if is_best
                        else ""
                    )

                    print("=" * 82)
                    print(
                        f"[VAL] Epoch {epoch + 1}/{phase.epochs} ({phase.name}) | Global Epoch {global_epoch + 1}"
                    )
                    print(
                        f"      Val Loss: {v_loss['total']:.4f} (Det: {v_loss['detection']:.3f} | State: {v_loss['state']:.3f} | Rel: {v_loss['relevance']:.3f} | NWD: {v_loss['nwd']:.3f})"
                    )
                    print(
                        f"      Detection: mAP50 = {det['map50']:.4f} | mAP50-95 = {det['map50_95']:.4f} | AP_small = {det['ap_small']:.4f} | AP_med = {det['ap_medium']:.4f}"
                    )
                    print(
                        f"      Relevance: AUPRC = {rel['auprc']:.4f} | F1 = {rel['f1']:.4f} | Prec = {rel['precision']:.4f} | Rec = {rel['recall']:.4f} | RelRedRec = {rel_red_score:.4f}"
                    )
                    print(
                        f"      Attributes: mAP_state = {det['map_state']:.4f} | State Acc = {att['state_accuracy']:.4f} | Maneuver F1 = {att['maneuver_macro_f1']:.4f}"
                    )
                    print(f"      Score = {score:.4f} {best_tag}")
                    print("=" * 82, flush=True)

                    val_event = {
                        "event": "val",
                        "phase": phase.name,
                        "epoch": global_epoch,
                        "global_epoch": global_epoch,
                        "selection_score": score,
                        "is_best": is_best,
                        "best_tl_det_score": best_tl_det_score,
                        "best_relevance_score": best_relevance_score,
                        "best_relevant_red_score": best_relevant_red_score,
                        "mean_losses": v_loss,
                        "detection": det,
                        "relevance": rel,
                        "attributes": att,
                        "samples_evaluated": val_res["samples_evaluated"],
                    }
                    log.write(json.dumps(val_event) + "\n")
                    log.flush()

                if epoch_complete:
                    global_epoch += 1
                    epochs_completed += 1
                if reached_step_cap:
                    stopped_by_step_cap = True
                    break
            if stopped_by_step_cap:
                break
    return {
        "schema": "TLR-YOLO-MTL training run summary v1",
        "optimizer_steps": optimizer_steps,
        "epochs_entered_this_run": epochs_entered,
        "epochs_completed_this_run": epochs_completed,
        "next_global_epoch": global_epoch,
        "best_val_score": best_val_score if math.isfinite(best_val_score) else None,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(output),
        "model_variant": str(config["model_variant"]),
        "micro_batch_size": int(config["micro_batch_size"]),
        "effective_batch_size": int(config["effective_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "optimizer_steps_per_epoch": int(sampler.windows_per_epoch),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "stopped_by_step_cap": stopped_by_step_cap,
        "p2_enabled": bool(config.get("p2_enabled", False)),
    }
