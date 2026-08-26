"""Canonical JSONL dataset, fixed rectangular letterbox, and balanced sampler."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..data.schema import BBox, ImageRecord
from ..data.transforms import horizontal_flip_record
from ..data.zoom_augmentation import context_preserving_zoom
from ..data.scale_matched_augmentation import (
    DEFAULT_SCALE_QUOTAS,
    paired_copy_paste,
    scale_matched_zoom,
)
from ..data.photometric_augmentation import (
    DEFAULT_PHOTOMETRIC_CONFIG,
    PhotometricAugmentationConfig,
    apply_physics_photometric_augmentation,
)
from ..data.counterfactual_sampling import (
    DEFAULT_COUNTERFACTUAL_CONFIG,
    CounterfactualMiningConfig,
    encode_counterfactual_relevance_targets,
)
from ..model.arrows import encode_record_arrows
from ..model.attributes import encode_record_attributes
from ..model.context import encode_record_context_gradient
from ..model.relevance import encode_record_relevance
from ..model.unified import encode_record_unified

DEFAULT_INPUT_SIZE = (800, 1600)  # height, width
DEFAULT_EFFECTIVE_SOURCE_QUOTAS = {
    # Historical default retained for callers that do not load the active
    # config.  The unified DTLD-only YAML passes {DTLD: 32} explicitly.
    "DTLD": 26,
    "AUX_TL": 6,
}


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    byte_offset: int
    source_dataset: str
    split: str
    image_id: str


def source_group(source_dataset: str) -> str:
    if source_dataset == "DTLD":
        return "DTLD"
    if source_dataset in {"ATLAS", "LISA"}:
        return "AUX_TL"
    raise ValueError(f"source is not active in the TLR pipeline: {source_dataset}")


def letterbox_parameters(
    original_size: tuple[int, int],
    target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> tuple[float, int, int, int, int]:
    """Return scale, left/top padding, and resized width/height."""

    original_height, original_width = original_size
    target_height, target_width = target_size
    if min(original_height, original_width, target_height, target_width) <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = min(target_width, max(1, round(original_width * scale)))
    resized_height = min(target_height, max(1, round(original_height * scale)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    return scale, left, top, resized_width, resized_height


def letterbox_box(
    box: BBox,
    *,
    scale: float,
    left: int,
    top: int,
    target_size: tuple[int, int],
) -> BBox:
    target_height, target_width = target_size
    x1, y1, x2, y2 = box
    transformed = (
        max(0.0, min(float(target_width), x1 * scale + left)),
        max(0.0, min(float(target_height), y1 * scale + top)),
        max(0.0, min(float(target_width), x2 * scale + left)),
        max(0.0, min(float(target_height), y2 * scale + top)),
    )
    return transformed


def xyxy_to_normalized_xywh(
    box: BBox, target_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    target_height, target_width = target_size
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2) / target_width,
        ((y1 + y2) / 2) / target_height,
        (x2 - x1) / target_width,
        (y2 - y1) / target_height,
    )


def _boxes_tensor(
    boxes: Sequence[BBox],
    *,
    scale: float,
    left: int,
    top: int,
    target_size: tuple[int, int],
) -> torch.Tensor:
    values = [
        xyxy_to_normalized_xywh(
            letterbox_box(
                box,
                scale=scale,
                left=left,
                top=top,
                target_size=target_size,
            ),
            target_size,
        )
        for box in boxes
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


def _photometric_augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Conservative traffic-light-safe photometric augmentation."""

    value = image.astype(np.float32) / 255.0
    gamma = rng.uniform(0.75, 1.25)
    value = np.power(value, gamma)
    hsv = cv2.cvtColor((value * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue_shift = int(rng.uniform(-0.01, 0.01) * 180)
    saturation_scale = rng.uniform(0.8, 1.2)
    brightness_scale = rng.uniform(0.7, 1.3)
    hsv = hsv.astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_scale, 0, 255)
    augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.15:
        sigma = rng.uniform(1.0, 5.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0.0, sigma, augmented.shape
        )
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )
    if rng.random() < 0.15:
        kernel = rng.choice((3, 5))
        augmented = cv2.GaussianBlur(augmented, (kernel, kernel), 0)
    return augmented


def prepare_training_sample(
    image_rgb: np.ndarray,
    record: ImageRecord,
    *,
    target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    training: bool = False,
    horizontal_flip: bool = False,
    context_zoom: bool = False,
    zoom_prob: float = 0.5,
    scale_matched_zoom_enabled: bool = False,
    scale_quotas: tuple[float, float, float] = DEFAULT_SCALE_QUOTAS,
    paired_copy_paste_enabled: bool = False,
    donor_image_rgb: np.ndarray | None = None,
    donor_record: ImageRecord | None = None,
    copy_paste_prob: float = 0.35,
    photometric_suite_enabled: bool = True,
    photometric_config: PhotometricAugmentationConfig = DEFAULT_PHOTOMETRIC_CONFIG,
    counterfactual_mining_enabled: bool = False,
    counterfactual_config: CounterfactualMiningConfig = DEFAULT_COUNTERFACTUAL_CONFIG,
    rng: random.Random | None = None,
) -> dict[str, torch.Tensor | str]:
    """Transform one canonical record and every task target atomically."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("training image must be RGB H×W×3")
    if image_rgb.shape[:2] != (record.original_height, record.original_width):
        raise ValueError(
            f"image size {image_rgb.shape[:2]} differs from manifest "
            f"{(record.original_height, record.original_width)}"
        )
    resolved_rng = rng or random.Random()
    transformed_record = record
    transformed_image = image_rgb
    if training and scale_matched_zoom_enabled:
        transformed_image, transformed_record = scale_matched_zoom(
            transformed_image,
            transformed_record,
            zoom_prob=zoom_prob,
            scale_quotas=scale_quotas,
            rng=resolved_rng,
        )
    elif training and context_zoom:
        transformed_image, transformed_record = context_preserving_zoom(
            transformed_image,
            transformed_record,
            zoom_prob=zoom_prob,
            rng=resolved_rng,
        )
    if (
        training
        and paired_copy_paste_enabled
        and donor_image_rgb is not None
        and donor_record is not None
    ):
        transformed_image, transformed_record = paired_copy_paste(
            transformed_image,
            transformed_record,
            donor_image_rgb,
            donor_record,
            copy_paste_prob=copy_paste_prob,
            rng=resolved_rng,
        )
    if training and horizontal_flip and resolved_rng.random() < 0.5:
        transformed_image = np.ascontiguousarray(transformed_image[:, ::-1])
        transformed_record = horizontal_flip_record(transformed_record)
    if training and photometric_suite_enabled:
        transformed_image = apply_physics_photometric_augmentation(
            transformed_image,
            transformed_record,
            config=photometric_config,
            rng=resolved_rng,
        )
    elif training:
        transformed_image = _photometric_augment(transformed_image, resolved_rng)

    scale, left, top, resized_width, resized_height = letterbox_parameters(
        transformed_image.shape[:2], target_size
    )
    resized = cv2.resize(
        transformed_image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    target_height, target_width = target_size
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    image_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().div_(255.0)

    traffic_boxes = _boxes_tensor(
        [item.bbox_xyxy for item in transformed_record.traffic_lights],
        scale=scale,
        left=left,
        top=top,
        target_size=target_size,
    )
    arrow_boxes = _boxes_tensor(
        [item.bbox_xyxy for item in transformed_record.road_arrows],
        scale=scale,
        left=left,
        top=top,
        target_size=target_size,
    )
    ignore_boxes = _boxes_tensor(
        [item.bbox_xyxy for item in transformed_record.ignore_regions],
        scale=scale,
        left=left,
        top=top,
        target_size=target_size,
    )
    attributes = encode_record_attributes(transformed_record)
    arrows = encode_record_arrows(transformed_record)
    relevance = encode_record_relevance(transformed_record)
    context = encode_record_context_gradient(transformed_record)
    unified = encode_record_unified(transformed_record)
    object_boxes = torch.cat((traffic_boxes, arrow_boxes), dim=0)

    if (counterfactual_mining_enabled or (training and counterfactual_config.enabled)) and transformed_record.traffic_lights:
        cf_targets = encode_counterfactual_relevance_targets(
            transformed_record,
            config=counterfactual_config,
            rng=resolved_rng,
        )
    else:
        num_tls = len(transformed_record.traffic_lights)
        cf_targets = {
            "counterfactual_weights": torch.ones(num_tls, dtype=torch.float32),
            "counterfactual_confuser_mask": torch.zeros(num_tls, dtype=torch.bool),
            "is_hard_negative": torch.zeros(num_tls, dtype=torch.bool),
        }

    return {
        "image": image_tensor,
        "image_id": transformed_record.image_id,
        "source_dataset": transformed_record.source_dataset,
        "cls": torch.zeros((traffic_boxes.shape[0], 1), dtype=torch.float32),
        "bboxes": traffic_boxes,
        "tl_state": attributes["tl_state"],
        "tl_pictogram": attributes["tl_pictogram"],
        "tl_relevance": relevance["tl_relevance"],
        "tl_relevance_valid": relevance["tl_relevance_valid"],
        "arrow_bboxes": arrow_boxes,
        "arrow_direction": arrows["arrow_direction"],
        "arrow_detection_valid": arrows["arrow_detection_valid"],
        "ignore_bboxes": ignore_boxes,
        "counterfactual_weights": cf_targets["counterfactual_weights"],
        "counterfactual_confuser_mask": cf_targets["counterfactual_confuser_mask"],
        "is_hard_negative": cf_targets["is_hard_negative"],
        # Active architecture: one ordered GT stream for both object types.
        "object_cls": unified["object_cls"],
        "object_bboxes": object_boxes,
        "object_state": unified["object_state"],
        "object_round": unified["object_round"],
        "object_maneuver": unified["object_maneuver"],
        "object_relevance": unified["object_relevance"],
        "object_ego_lane": unified["object_ego_lane"],
        "unified_detection_valid": unified["unified_detection_valid"],
        "traffic_relevance_valid": unified["traffic_relevance_valid"],
        **context,
    }



class CanonicalMultiTaskDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Memory-light random-access view over the canonical JSONL manifest."""

    def __init__(
        self,
        records_path: str | Path,
        *,
        split: str = "train",
        target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        training: bool = False,
        horizontal_flip: bool = False,
        context_zoom: bool = False,
        zoom_prob: float = 0.5,
        scale_matched_zoom: bool = False,
        scale_quotas: tuple[float, float, float] = DEFAULT_SCALE_QUOTAS,
        paired_copy_paste: bool = False,
        copy_paste_prob: float = 0.35,
        photometric_suite: bool = True,
        photometric_config: PhotometricAugmentationConfig = DEFAULT_PHOTOMETRIC_CONFIG,
        counterfactual_mining: bool = False,
        counterfactual_config: CounterfactualMiningConfig = DEFAULT_COUNTERFACTUAL_CONFIG,
        seed: int = 42,
        allowed_sources: Sequence[str] = ("DTLD", "ATLAS", "LISA"),
        require_paired: bool = False,
    ) -> None:
        self.records_path = Path(records_path).resolve()
        self.split = split
        self.target_size = target_size
        self.training = training
        self.horizontal_flip = bool(horizontal_flip)
        self.context_zoom = bool(context_zoom)
        self.zoom_prob = float(zoom_prob)
        self.scale_matched_zoom = bool(scale_matched_zoom)
        self.scale_quotas = scale_quotas
        self.paired_copy_paste = bool(paired_copy_paste)
        self.copy_paste_prob = float(copy_paste_prob)
        self.photometric_suite = bool(photometric_suite)
        self.photometric_config = photometric_config
        self.counterfactual_mining = bool(counterfactual_mining)
        self.counterfactual_config = counterfactual_config
        self.seed = int(seed)
        self.epoch = 0
        self.allowed_sources = frozenset(allowed_sources)
        self.require_paired = bool(require_paired)
        self.entries = self._build_index()
        self._stream: BinaryIO | None = None
        if not self.entries:
            raise ValueError(f"no {split!r} records found in {self.records_path}")

    def _build_index(self) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        with self.records_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                source = str(row["source_dataset"])
                tasks = row.get("task_valid", {})
                paired = bool(
                    tasks.get("traffic_light_detection", False)
                    and tasks.get("traffic_light_relevance", False)
                    and tasks.get("arrow_detection", False)
                )
                if (
                    row["split"] == self.split
                    and source in self.allowed_sources
                    and (not self.require_paired or paired)
                ):
                    entries.append(
                        ManifestEntry(
                            byte_offset=offset,
                            source_dataset=source,
                            split=str(row["split"]),
                            image_id=str(row["image_id"]),
                        )
                    )
        return entries

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_stream"] = None
        return state

    def __del__(self) -> None:
        if self._stream is not None:
            self._stream.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def _record(self, index: int) -> ImageRecord:
        if self._stream is None:
            self._stream = self.records_path.open("rb")
        self._stream.seek(self.entries[index].byte_offset)
        return ImageRecord.from_dict(json.loads(self._stream.readline()))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self._record(index)
        image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read training image: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rng = random.Random(self.seed + self.epoch * len(self.entries) + index)

        donor_image: np.ndarray | None = None
        donor_record: ImageRecord | None = None
        if self.training and self.paired_copy_paste and len(self.entries) > 1:
            donor_idx = rng.randrange(len(self.entries))
            if donor_idx == index:
                donor_idx = (index + 1) % len(self.entries)
            donor_rec = self._record(donor_idx)
            if donor_rec.traffic_lights:
                donor_img = cv2.imread(donor_rec.image_path, cv2.IMREAD_COLOR)
                if donor_img is not None:
                    donor_image = cv2.cvtColor(donor_img, cv2.COLOR_BGR2RGB)
                    donor_record = donor_rec

        return prepare_training_sample(
            image,
            record,
            target_size=self.target_size,
            training=self.training,
            horizontal_flip=self.horizontal_flip,
            context_zoom=self.context_zoom,
            zoom_prob=self.zoom_prob,
            scale_matched_zoom_enabled=self.scale_matched_zoom,
            scale_quotas=self.scale_quotas,
            paired_copy_paste_enabled=self.paired_copy_paste,
            donor_image_rgb=donor_image,
            donor_record=donor_record,
            copy_paste_prob=self.copy_paste_prob,
            photometric_suite_enabled=self.photometric_suite,
            photometric_config=self.photometric_config,
            counterfactual_mining_enabled=self.counterfactual_mining,
            counterfactual_config=self.counterfactual_config,
            rng=rng,
        )


def canonical_multitask_collate(
    samples: Sequence[Mapping[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate separate TL, arrow, and ignore instance streams."""

    if not samples:
        raise ValueError("cannot collate an empty batch")

    def concatenate(name: str, *, width: int | None = None) -> torch.Tensor:
        values = [sample[name] for sample in samples]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"batch field {name!r} must contain tensors")
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if tensors:
            return torch.cat(tensors, dim=0)
        shape = (0,) if width is None else (0, width)
        return torch.empty(shape)

    def instance_indices(name: str) -> torch.Tensor:
        indices = [
            torch.full(
                (int(sample[name].shape[0]),), index, dtype=torch.long
            )
            for index, sample in enumerate(samples)
            if isinstance(sample[name], torch.Tensor)
        ]
        return torch.cat(indices) if indices else torch.empty(0, dtype=torch.long)

    return {
        "img": torch.stack(
            [sample["image"] for sample in samples if isinstance(sample["image"], torch.Tensor)]
        ),
        "image_ids": [str(sample["image_id"]) for sample in samples],
        "source_datasets": [str(sample["source_dataset"]) for sample in samples],
        "batch_idx": instance_indices("bboxes"),
        "cls": concatenate("cls", width=1),
        "bboxes": concatenate("bboxes", width=4),
        "tl_state": concatenate("tl_state"),
        "tl_pictogram": concatenate("tl_pictogram"),
        "tl_relevance": concatenate("tl_relevance"),
        "tl_relevance_valid": torch.stack(
            [sample["tl_relevance_valid"] for sample in samples]
        ),
        "arrow_batch_idx": instance_indices("arrow_bboxes"),
        "arrow_bboxes": concatenate("arrow_bboxes", width=4),
        "arrow_direction": concatenate("arrow_direction", width=3),
        "arrow_detection_valid": torch.stack(
            [sample["arrow_detection_valid"] for sample in samples]
        ),
        "ignore_batch_idx": instance_indices("ignore_bboxes"),
        "ignore_bboxes": concatenate("ignore_bboxes", width=4),
        "counterfactual_weights": concatenate("counterfactual_weights"),
        "counterfactual_confuser_mask": concatenate("counterfactual_confuser_mask"),
        "is_hard_negative": concatenate("is_hard_negative"),
        "object_batch_idx": instance_indices("object_bboxes"),
        "object_cls": concatenate("object_cls", width=1),
        "object_bboxes": concatenate("object_bboxes", width=4),
        "object_state": concatenate("object_state"),
        "object_round": concatenate("object_round"),
        "object_maneuver": concatenate("object_maneuver", width=3),
        "object_relevance": concatenate("object_relevance"),
        "object_ego_lane": concatenate("object_ego_lane"),
        "unified_detection_valid": torch.stack(
            [sample["unified_detection_valid"] for sample in samples]
        ),
        "traffic_relevance_valid": torch.stack(
            [sample["traffic_relevance_valid"] for sample in samples]
        ),
        "relevance_arrow_context_scale": torch.stack(
            [sample["relevance_arrow_context_scale"] for sample in samples]
        ),
        "relevance_arrow_context_paired": torch.stack(
            [sample["relevance_arrow_context_paired"] for sample in samples]
        ),
    }



class BalancedEffectiveBatchSampler(Sampler[list[int]]):
    """Enforce source quotas over each gradient-accumulation window."""

    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        *,
        micro_batch_size: int = 1,
        quotas: Mapping[str, int] = DEFAULT_EFFECTIVE_SOURCE_QUOTAS,
        seed: int = 42,
        windows_per_epoch: int | None = None,
    ) -> None:
        self.entries = entries
        self.micro_batch_size = int(micro_batch_size)
        self.quotas = {name: int(value) for name, value in quotas.items()}
        self.seed = int(seed)
        self.epoch = 0
        self.effective_batch_size = sum(self.quotas.values())
        if self.micro_batch_size <= 0:
            raise ValueError("micro batch size must be positive")
        if self.effective_batch_size <= 0:
            raise ValueError("effective source quotas must be positive")
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective batch size must be divisible by micro batch size")
        self.by_group: dict[str, list[int]] = {name: [] for name in self.quotas}
        for index, entry in enumerate(entries):
            group = source_group(entry.source_dataset)
            if group in self.by_group:
                self.by_group[group].append(index)
        missing = [name for name, values in self.by_group.items() if not values]
        if missing:
            raise ValueError(f"no records available for sampler groups: {missing}")
        self.windows_per_epoch = windows_per_epoch or math.ceil(
            len(entries) / self.effective_batch_size
        )

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.micro_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.windows_per_epoch * self.accumulation_steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = {name: list(values) for name, values in self.by_group.items()}
        positions = {name: len(values) for name, values in pools.items()}

        def take(group: str, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                if positions[group] >= len(pools[group]):
                    rng.shuffle(pools[group])
                    positions[group] = 0
                available = min(count - len(selected), len(pools[group]) - positions[group])
                start = positions[group]
                selected.extend(pools[group][start : start + available])
                positions[group] += available
            return selected

        for _ in range(self.windows_per_epoch):
            window: list[int] = []
            for group, count in self.quotas.items():
                window.extend(take(group, count))
            rng.shuffle(window)
            for offset in range(0, len(window), self.micro_batch_size):
                yield window[offset : offset + self.micro_batch_size]
