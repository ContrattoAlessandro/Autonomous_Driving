"""Canonical class mappings and augmentation-safe left/right transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def normalize_label(value: object | None) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True, slots=True)
class AttributeMapping:
    target: str | None
    valid: bool
    ignore_object: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FactorizedPictogram:
    """Round/maneuver targets used by the unified detector.

    ``round`` and ``maneuver`` deliberately have separate validity flags.  A
    round signal is not encoded as the artificial multi-hot target
    ``[1, 1, 1]``: its compatibility with directional arrows is learned by the
    relevance module instead.
    """

    round: int | None
    maneuver: tuple[int, int, int] | None
    valid_round: bool
    valid_maneuver: bool
    ignore_object: bool = False
    reason: str | None = None


STATE_ALIASES = {
    "red": "red",
    "yellow": "yellow",
    "amber": "yellow",
    "green": "green",
    "off": "off",
    "unlit": "off",
    "none": "off",
}
MASKED_STATES = {
    "unknown",
    "flashing",
    "blinking",
    "red_yellow",
    "yellow_red",
}

PICTOGRAM_ALIASES = {
    "circle": "round",
    "round": "round",
    "arrow_left": "left",
    "left": "left",
    "arrow_straight": "straight",
    "straight": "straight",
    "arrow_right": "right",
    "right": "right",
}
MASKED_PICTOGRAMS = {
    "unknown",
    "straight_left",
    "left_straight",
    "arrow_straight_left",
    "arrow_left_straight",
    "straight_right",
    "right_straight",
    "arrow_straight_right",
    "arrow_right_straight",
    "left_right",
    "arrow_left_right",
}
IGNORE_PICTOGRAMS = {
    "pedestrian",
    "cyclist",
    "bicycle",
    "tram",
    "pedestrian_bicycle",
    "pedestrian_cyclist",
}

ARROW_DIRECTIONS = {
    "left": (1, 0, 0),
    "arrow_left": (1, 0, 0),
    "straight": (0, 1, 0),
    "arrow_straight": (0, 1, 0),
    "right": (0, 0, 1),
    "arrow_right": (0, 0, 1),
    "straight_left": (1, 1, 0),
    "left_straight": (1, 1, 0),
    "arrow_straight_left": (1, 1, 0),
    "str_left": (1, 1, 0),
    "straight_right": (0, 1, 1),
    "right_straight": (0, 1, 1),
    "arrow_straight_right": (0, 1, 1),
    "str_right": (0, 1, 1),
    "left_right": (1, 0, 1),
    "arrow_left_right": (1, 0, 1),
}


def map_state(value: object | None) -> AttributeMapping:
    label = normalize_label(value)
    target = STATE_ALIASES.get(label)
    if target is not None:
        return AttributeMapping(target=target, valid=True)
    reason = "masked_state" if label in MASKED_STATES else "unknown_state"
    return AttributeMapping(target=None, valid=False, reason=reason)


def map_pictogram(value: object | None) -> AttributeMapping:
    label = normalize_label(value)
    target = PICTOGRAM_ALIASES.get(label)
    if target is not None:
        return AttributeMapping(target=target, valid=True)
    if label in IGNORE_PICTOGRAMS:
        return AttributeMapping(
            target=None,
            valid=False,
            ignore_object=True,
            reason="non_vehicle_pictogram",
        )
    reason = "masked_pictogram" if label in MASKED_PICTOGRAMS else "unknown_pictogram"
    return AttributeMapping(target=None, valid=False, reason=reason)


def factor_pictogram(value: object | None) -> FactorizedPictogram:
    """Map a TL pictogram to independent round and maneuver supervision.

    This is intentionally a new mapping rather than a semantic change to
    :func:`map_pictogram`, whose four-way categorical contract is retained for
    reading historical manifests and reports.
    """

    label = normalize_label(value)
    if label in IGNORE_PICTOGRAMS:
        return FactorizedPictogram(
            round=None,
            maneuver=None,
            valid_round=False,
            valid_maneuver=False,
            ignore_object=True,
            reason="non_vehicle_pictogram",
        )
    canonical = PICTOGRAM_ALIASES.get(label)
    if canonical == "round":
        return FactorizedPictogram(
            round=1,
            maneuver=None,
            valid_round=True,
            valid_maneuver=False,
        )
    maneuver = ARROW_DIRECTIONS.get(label)
    if maneuver is not None:
        return FactorizedPictogram(
            round=0,
            maneuver=maneuver,
            valid_round=True,
            valid_maneuver=True,
        )
    return FactorizedPictogram(
        round=None,
        maneuver=None,
        valid_round=False,
        valid_maneuver=False,
        reason="unknown_pictogram",
    )


def map_arrow_direction(value: object) -> tuple[int, int, int] | None:
    return ARROW_DIRECTIONS.get(normalize_label(value))


def flip_pictogram(value: str | None) -> str | None:
    if value is None:
        return None
    val_norm = normalize_label(value)
    flip_map = {
        "left": "right",
        "right": "left",
        "arrow_left": "arrow_right",
        "arrow_right": "arrow_left",
        "straight_left": "straight_right",
        "straight_right": "straight_left",
        "str_left": "str_right",
        "str_right": "str_left",
        "left_straight": "right_straight",
        "right_straight": "left_straight",
    }
    return flip_map.get(val_norm, value)


def flip_direction_multihot(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("direction vector must have [left, straight, right]")
    return int(values[2]), int(values[1]), int(values[0])


def split_atlas_class(name: str) -> tuple[AttributeMapping, AttributeMapping]:
    """Factor an ATLAS composite label into independent state/pictogram masks."""

    label = normalize_label(name)
    if label == "off":
        return map_state("off"), map_pictogram("unknown")
    for suffix in ("red_yellow", "yellow", "green", "red"):
        marker = "_" + suffix
        if label.endswith(marker):
            pictogram = label[: -len(marker)]
            return map_state(suffix), map_pictogram(pictogram)
    return map_state("unknown"), map_pictogram("unknown")


def normalize_occlusion(value: object | None) -> str:
    label = normalize_label(value)
    if label in {"not_occluded", "visible", "none", "0"}:
        return "not_occluded"
    if label in {"occluded", "partially_occluded", "partial", "1"}:
        return "partially_occluded"
    if label in {"fully_occluded", "totally_occluded", "full", "total"}:
        return "fully_occluded"
    return "unknown"


def map_binary_relevance(value: object | None) -> tuple[int | None, bool]:
    label = normalize_label(value)
    if label in {"relevant", "true", "yes", "1"}:
        return 1, True
    if label in {"not_relevant", "irrelevant", "false", "no", "0"}:
        return 0, True
    return None, False
