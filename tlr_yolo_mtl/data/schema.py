"""Loss-aware schema shared by every TLR-YOLO-MTL dataset.

There are two distinct masking levels:

* :class:`TaskValidity` says whether a task is exhaustively annotated for the
  whole image.  ``False`` means *unknown*, not an all-background image.
* Instance ``valid_*`` fields mask individual attributes while retaining a
  valid detection box.  For example, ``red_yellow`` remains a traffic-light
  target but has ``valid_state=False``.

Coordinates are absolute ``[x1, y1, x2, y2]`` pixels with an exclusive lower
right corner.  This keeps geometry reversible until the dataloader applies its
final letterbox transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "3.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"2.1", SCHEMA_VERSION})

STATES = frozenset({"red", "yellow", "green", "off"})
PICTOGRAMS = frozenset({"round", "left", "straight", "right"})
SPLITS = frozenset({"train", "val", "test"})
OCCLUSIONS = frozenset(
    {"not_occluded", "partially_occluded", "fully_occluded", "unknown"}
)

BBox = tuple[float, float, float, float]
DirectionMultiHot = tuple[int, int, int]


class SchemaError(ValueError):
    """Raised when a unified record violates the training contract."""


def _bbox(value: Sequence[float]) -> BBox:
    if len(value) != 4:
        raise SchemaError(f"bbox must contain four coordinates, got {value!r}")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _multihot(value: Sequence[int]) -> DirectionMultiHot:
    if len(value) != 3:
        raise SchemaError(f"direction_multihot must contain three values, got {value!r}")
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def _migrate_pictogram(
    value: object | None,
) -> tuple[int | None, DirectionMultiHot | None, bool, bool]:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label in {"circle", "round"}:
        return 1, None, True, False
    maneuvers: dict[str, DirectionMultiHot] = {
        "left": (1, 0, 0),
        "arrow_left": (1, 0, 0),
        "straight": (0, 1, 0),
        "arrow_straight": (0, 1, 0),
        "right": (0, 0, 1),
        "arrow_right": (0, 0, 1),
        "straight_left": (1, 1, 0),
        "left_straight": (1, 1, 0),
        "arrow_straight_left": (1, 1, 0),
        "arrow_left_straight": (1, 1, 0),
        "str_left": (1, 1, 0),
        "straight_right": (0, 1, 1),
        "right_straight": (0, 1, 1),
        "arrow_straight_right": (0, 1, 1),
        "arrow_right_straight": (0, 1, 1),
        "str_right": (0, 1, 1),
        "left_right": (1, 0, 1),
        "arrow_left_right": (1, 0, 1),
    }
    maneuver = maneuvers.get(label)
    return (0, maneuver, True, True) if maneuver is not None else (None, None, False, False)


@dataclass(slots=True)
class TaskValidity:
    traffic_light_detection: bool = False
    traffic_light_state: bool = False
    traffic_light_pictogram: bool = False
    traffic_light_relevance: bool = False
    arrow_detection: bool = False
    traffic_light_round: bool = False
    traffic_light_maneuver: bool = False
    arrow_ego_lane: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "traffic_light_detection": self.traffic_light_detection,
            "traffic_light_state": self.traffic_light_state,
            "traffic_light_pictogram": self.traffic_light_pictogram,
            "traffic_light_relevance": self.traffic_light_relevance,
            "arrow_detection": self.arrow_detection,
            "traffic_light_round": self.traffic_light_round,
            "traffic_light_maneuver": self.traffic_light_maneuver,
            "arrow_ego_lane": self.arrow_ego_lane,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskValidity":
        fields = {name: bool(value.get(name, False)) for name in cls.__slots__}
        # Schema 2.1 represented both factorized targets with one categorical
        # pictogram validity bit.  Preserve that information during migration.
        if "traffic_light_round" not in value:
            fields["traffic_light_round"] = fields["traffic_light_pictogram"]
        if "traffic_light_maneuver" not in value:
            fields["traffic_light_maneuver"] = fields["traffic_light_pictogram"]
        return cls(**fields)


@dataclass(slots=True)
class TrafficLightAnnotation:
    bbox_xyxy: BBox
    state: str | None = None
    pictogram: str | None = None
    relevance: int | None = None
    occlusion: str = "unknown"
    valid_state: bool = False
    valid_pictogram: bool = False
    valid_relevance: bool = False
    round_target: int | None = None
    maneuver_multihot: DirectionMultiHot | None = None
    valid_round: bool = False
    valid_maneuver: bool = False
    source_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "bbox_xyxy": list(self.bbox_xyxy),
            "state": self.state,
            "pictogram": self.pictogram,
            "relevance": self.relevance,
            "occlusion": self.occlusion,
            "valid_state": self.valid_state,
            "valid_pictogram": self.valid_pictogram,
            "valid_relevance": self.valid_relevance,
            "round_target": self.round_target,
            "maneuver_multihot": (
                list(self.maneuver_multihot)
                if self.maneuver_multihot is not None
                else None
            ),
            "valid_round": self.valid_round,
            "valid_maneuver": self.valid_maneuver,
        }
        if self.source_attributes:
            result["source_attributes"] = self.source_attributes
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrafficLightAnnotation":
        round_target = value.get("round_target")
        maneuver_value = value.get("maneuver_multihot")
        valid_round = bool(value.get("valid_round", False))
        valid_maneuver = bool(value.get("valid_maneuver", False))
        pictogram = value.get("pictogram")
        valid_pictogram = bool(value.get("valid_pictogram", False))
        if "round_target" not in value or "maneuver_multihot" not in value:
            source_attributes = dict(value.get("source_attributes", {}))
            migrated = _migrate_pictogram(
                pictogram if pictogram is not None else source_attributes.get("pictogram")
            )
            if "round_target" not in value:
                round_target, valid_round = migrated[0], migrated[2]
            if "maneuver_multihot" not in value:
                maneuver_value, valid_maneuver = migrated[1], migrated[3]
        return cls(
            bbox_xyxy=_bbox(value["bbox_xyxy"]),
            state=value.get("state"),
            pictogram=pictogram,
            relevance=value.get("relevance"),
            occlusion=str(value.get("occlusion", "unknown")),
            valid_state=bool(value.get("valid_state", False)),
            valid_pictogram=valid_pictogram,
            valid_relevance=bool(value.get("valid_relevance", False)),
            round_target=(None if round_target is None else int(round_target)),
            maneuver_multihot=(
                None if maneuver_value is None else _multihot(maneuver_value)
            ),
            valid_round=valid_round,
            valid_maneuver=valid_maneuver,
            source_attributes=dict(value.get("source_attributes", {})),
        )


@dataclass(slots=True)
class RoadArrowAnnotation:
    bbox_xyxy: BBox
    direction_multihot: DirectionMultiHot
    segmentation_xy: tuple[tuple[float, float], ...] = ()
    is_ego_lane: int | None = None
    valid_ego_lane: bool = False
    source_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "bbox_xyxy": list(self.bbox_xyxy),
            "direction_multihot": list(self.direction_multihot),
        }
        if self.segmentation_xy:
            result["segmentation_xy"] = [list(point) for point in self.segmentation_xy]
        result["is_ego_lane"] = self.is_ego_lane
        result["valid_ego_lane"] = self.valid_ego_lane
        if self.source_attributes:
            result["source_attributes"] = self.source_attributes
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoadArrowAnnotation":
        return cls(
            bbox_xyxy=_bbox(value["bbox_xyxy"]),
            direction_multihot=_multihot(value["direction_multihot"]),
            segmentation_xy=tuple(
                (float(point[0]), float(point[1]))
                for point in value.get("segmentation_xy", ())
            ),
            is_ego_lane=(
                None
                if value.get("is_ego_lane") is None
                else int(value["is_ego_lane"])
            ),
            valid_ego_lane=bool(value.get("valid_ego_lane", False)),
            source_attributes=dict(value.get("source_attributes", {})),
        )


@dataclass(slots=True)
class IgnoreRegion:
    bbox_xyxy: BBox
    reason: str
    source_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "bbox_xyxy": list(self.bbox_xyxy),
            "reason": self.reason,
        }
        if self.source_label is not None:
            result["source_label"] = self.source_label
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IgnoreRegion":
        return cls(
            bbox_xyxy=_bbox(value["bbox_xyxy"]),
            reason=str(value["reason"]),
            source_label=value.get("source_label"),
        )


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    image_path: str
    source_dataset: str
    original_width: int
    original_height: int
    split: str
    sequence_id: str | None
    task_valid: TaskValidity
    traffic_lights: list[TrafficLightAnnotation] = field(default_factory=list)
    road_arrows: list[RoadArrowAnnotation] = field(default_factory=list)
    ignore_regions: list[IgnoreRegion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "source_dataset": self.source_dataset,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "split": self.split,
            "sequence_id": self.sequence_id,
            "task_valid": self.task_valid.to_dict(),
            "traffic_lights": [item.to_dict() for item in self.traffic_lights],
            "road_arrows": [item.to_dict() for item in self.road_arrows],
            "ignore_regions": [item.to_dict() for item in self.ignore_regions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageRecord":
        version = str(value.get("schema_version", SCHEMA_VERSION))
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaError(
                f"unsupported schema_version={version!r}; expected one of "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)!r}"
            )
        return cls(
            image_id=str(value["image_id"]),
            image_path=str(value["image_path"]),
            source_dataset=str(value["source_dataset"]),
            original_width=int(value["original_width"]),
            original_height=int(value["original_height"]),
            split=str(value["split"]),
            sequence_id=(
                None if value.get("sequence_id") is None else str(value["sequence_id"])
            ),
            task_valid=TaskValidity.from_dict(value.get("task_valid", {})),
            traffic_lights=[
                TrafficLightAnnotation.from_dict(item)
                for item in value.get("traffic_lights", ())
            ],
            road_arrows=[
                RoadArrowAnnotation.from_dict(item)
                for item in value.get("road_arrows", ())
            ],
            ignore_regions=[
                IgnoreRegion.from_dict(item) for item in value.get("ignore_regions", ())
            ],
            metadata=dict(value.get("metadata", {})),
        )

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        prefix = self.image_id or "<missing-image-id>"
        if not self.image_id:
            errors.append("image_id is empty")
        if not self.image_path:
            errors.append(f"{prefix}: image_path is empty")
        if self.original_width <= 0 or self.original_height <= 0:
            errors.append(
                f"{prefix}: invalid image size {self.original_width}x{self.original_height}"
            )
        if self.split not in SPLITS:
            errors.append(f"{prefix}: unsupported split {self.split!r}")

        tasks = self.task_valid
        if (
            tasks.traffic_light_state
            or tasks.traffic_light_pictogram
            or tasks.traffic_light_relevance
            or tasks.traffic_light_round
            or tasks.traffic_light_maneuver
        ) and not tasks.traffic_light_detection:
            errors.append(f"{prefix}: TL attribute task enabled while detection is masked")
        if self.traffic_lights and not tasks.traffic_light_detection:
            errors.append(f"{prefix}: traffic-light instances exist while task is masked")
        if self.road_arrows and not tasks.arrow_detection:
            errors.append(f"{prefix}: arrow instances exist while task is masked")

        for index, item in enumerate(self.traffic_lights):
            label = f"{prefix}: traffic_lights[{index}]"
            errors.extend(self._box_errors(item.bbox_xyxy, label))
            if item.valid_state:
                if not tasks.traffic_light_state:
                    errors.append(f"{label}: valid_state=True while image task is masked")
                if item.state not in STATES:
                    errors.append(f"{label}: invalid state target {item.state!r}")
            elif item.state is not None and item.state in STATES:
                errors.append(f"{label}: canonical state present but valid_state=False")
            if item.valid_pictogram:
                if not tasks.traffic_light_pictogram:
                    errors.append(
                        f"{label}: valid_pictogram=True while image task is masked"
                    )
                if item.pictogram not in PICTOGRAMS:
                    errors.append(f"{label}: invalid pictogram target {item.pictogram!r}")
            elif item.pictogram is not None and item.pictogram in PICTOGRAMS:
                errors.append(
                    f"{label}: canonical pictogram present but valid_pictogram=False"
                )
            if item.valid_relevance:
                if not tasks.traffic_light_relevance:
                    errors.append(
                        f"{label}: valid_relevance=True while image task is masked"
                    )
                if item.relevance not in (0, 1):
                    errors.append(f"{label}: relevance must be binary")
            elif item.relevance is not None:
                errors.append(f"{label}: relevance present but valid_relevance=False")
            if item.valid_round:
                if not (tasks.traffic_light_round or tasks.traffic_light_pictogram):
                    errors.append(f"{label}: valid_round=True while image task is masked")
                if item.round_target not in (0, 1):
                    errors.append(f"{label}: round_target must be binary")
            elif item.round_target is not None:
                errors.append(f"{label}: round_target present but valid_round=False")
            if item.valid_maneuver:
                if not (
                    tasks.traffic_light_maneuver
                    or tasks.traffic_light_pictogram
                ):
                    errors.append(
                        f"{label}: valid_maneuver=True while image task is masked"
                    )
                if item.maneuver_multihot is None:
                    errors.append(f"{label}: valid maneuver has no target")
                elif any(value not in (0, 1) for value in item.maneuver_multihot):
                    errors.append(f"{label}: maneuver_multihot is not binary")
                elif not any(item.maneuver_multihot):
                    errors.append(f"{label}: maneuver_multihot has no direction")
            elif item.maneuver_multihot is not None:
                errors.append(
                    f"{label}: maneuver_multihot present but valid_maneuver=False"
                )
            if item.occlusion not in OCCLUSIONS:
                errors.append(f"{label}: unknown occlusion {item.occlusion!r}")
            if item.occlusion == "fully_occluded":
                errors.append(f"{label}: fully occluded object must be an ignore region")

        for index, item in enumerate(self.road_arrows):
            label = f"{prefix}: road_arrows[{index}]"
            errors.extend(self._box_errors(item.bbox_xyxy, label))
            if any(value not in (0, 1) for value in item.direction_multihot):
                errors.append(f"{label}: direction_multihot is not binary")
            if not any(item.direction_multihot):
                errors.append(f"{label}: direction_multihot has no positive direction")
            if item.valid_ego_lane:
                if not tasks.arrow_ego_lane:
                    errors.append(
                        f"{label}: valid_ego_lane=True while image task is masked"
                    )
                if item.is_ego_lane not in (0, 1):
                    errors.append(f"{label}: is_ego_lane must be binary")
            elif item.is_ego_lane is not None:
                errors.append(f"{label}: is_ego_lane present but valid_ego_lane=False")

        for index, item in enumerate(self.ignore_regions):
            label = f"{prefix}: ignore_regions[{index}]"
            errors.extend(self._box_errors(item.bbox_xyxy, label))
            if not item.reason:
                errors.append(f"{label}: reason is empty")
        return errors

    def _box_errors(self, bbox: BBox, label: str) -> list[str]:
        x1, y1, x2, y2 = bbox
        errors: list[str] = []
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{label}: zero/negative-area bbox {bbox}")
        if x1 < 0 or y1 < 0 or x2 > self.original_width or y2 > self.original_height:
            errors.append(
                f"{label}: bbox {bbox} outside "
                f"{self.original_width}x{self.original_height}"
            )
        return errors

    def validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise SchemaError("\n".join(errors))


def validate_records(records: Iterable[ImageRecord]) -> None:
    """Validate all records and report every failing image in one exception."""

    errors: list[str] = []
    for record in records:
        errors.extend(record.validation_errors())
    if errors:
        preview = "\n".join(errors[:100])
        suffix = "" if len(errors) <= 100 else f"\n... {len(errors) - 100} more"
        raise SchemaError(preview + suffix)
