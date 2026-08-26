"""Counterfactual Hard-Negative Mining and Balanced Sampling for Ego-Lane Relevance (Ticket E43).

Provides scene-coherent counterfactual pair mining to eliminate relevance false positives
and cross-lane false alarms without introducing auxiliary loss gradient interference:
1. Cross-Lane Confusers: True road arrows paired with non-governing TLs in the same intersection
   with conflicting or adjacent-lane directional semantics.
2. Spatial Neighbor Confusers: Pairs of TLs mounted on the same overhead mast-arm (small lateral
   and vertical offset) where one governs ego-lane and the other governs adjacent lanes.
3. Easy / Standard Negatives: Distant, non-governing lights or background light-arrow pairs.
4. Quota-Balanced Sampler (40% Positive : 30% Easy Neg : 15% Cross-Lane Hard Neg : 15% Spatial Hard Neg).
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .schema import BBox, ImageRecord, RoadArrowAnnotation, TrafficLightAnnotation
from .taxonomy import factor_pictogram, normalize_label


class CounterfactualPairType(str, enum.Enum):
    """Taxonomy of pair relationships in intersection scene relevance."""

    POSITIVE = "positive"
    EASY_NEGATIVE = "easy_negative"
    CROSS_LANE_CONFUSER = "cross_lane_confuser"
    SPATIAL_NEIGHBOR_CONFUSER = "spatial_neighbor_confuser"
    OPPOSING_MANEUVER_CONFUSER = "opposing_maneuver_confuser"


@dataclass(frozen=True, slots=True)
class CounterfactualRelevancePair:
    """A mined traffic light and road arrow (or TL neighbor) relevance association pair."""

    tl_index: int
    arrow_index: int | None
    pair_type: CounterfactualPairType
    relevance_label: int  # 1 = relevant to ego-path, 0 = non-relevant / distractor
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)



@dataclass(frozen=True, slots=True)
class CounterfactualMiningConfig:
    """Configuration for counterfactual hard-negative mining and quota balancing."""

    enabled: bool = True
    target_pos_ratio: float = 0.40
    target_easy_neg_ratio: float = 0.30
    target_cross_lane_hard_ratio: float = 0.15
    target_spatial_neighbor_hard_ratio: float = 0.15
    spatial_dx_threshold_px: float = 100.0
    spatial_dy_threshold_px: float = 40.0
    max_pairs_per_image: int = 32
    min_pairs_per_image: int = 1
    seed: int = 42

    def validate(self) -> None:
        """Validate probability distribution ratios and spatial thresholds."""
        total_ratio = (
            self.target_pos_ratio
            + self.target_easy_neg_ratio
            + self.target_cross_lane_hard_ratio
            + self.target_spatial_neighbor_hard_ratio
        )
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"target ratios must sum to 1.0, got {total_ratio:.4f}")
        for name, val in (
            ("target_pos_ratio", self.target_pos_ratio),
            ("target_easy_neg_ratio", self.target_easy_neg_ratio),
            ("target_cross_lane_hard_ratio", self.target_cross_lane_hard_ratio),
            ("target_spatial_neighbor_hard_ratio", self.target_spatial_neighbor_hard_ratio),
        ):
            if val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.spatial_dx_threshold_px <= 0 or self.spatial_dy_threshold_px <= 0:
            raise ValueError("spatial thresholds must be positive")
        if self.max_pairs_per_image <= 0:
            raise ValueError("max_pairs_per_image must be positive")


DEFAULT_COUNTERFACTUAL_CONFIG = CounterfactualMiningConfig()


def _get_tl_maneuvers(tl: TrafficLightAnnotation) -> tuple[bool, bool, bool, bool]:
    """Extract (is_round, left, straight, right) maneuver booleans from a TL annotation."""
    if tl.round_target is not None and tl.maneuver_multihot is not None:
        is_round = bool(tl.round_target == 1)
        m = tl.maneuver_multihot
        return is_round, bool(m[0]), bool(m[1]), bool(m[2])

    pictogram = factor_pictogram(normalize_label(tl.pictogram or "circle"))
    is_round = bool(pictogram.round == 1)
    if pictogram.maneuver is not None:
        m = pictogram.maneuver
        left, straight, right = bool(m[0]), bool(m[1]), bool(m[2])
    else:
        left, straight, right = False, False, False
    return is_round, left, straight, right



def _get_arrow_maneuvers(arrow: RoadArrowAnnotation) -> tuple[bool, bool, bool]:
    """Extract (left, straight, right) maneuver booleans from an arrow annotation."""
    if hasattr(arrow, "direction_multihot") and arrow.direction_multihot is not None:
        dm = arrow.direction_multihot
        return bool(dm[0]), bool(dm[1]), bool(dm[2])
    direction = str(getattr(arrow, "direction", "straight") or "straight").lower()
    left = "left" in direction
    straight = "straight" in direction or "through" in direction
    right = "right" in direction
    return left, straight, right



def _is_maneuver_compatible(
    tl: TrafficLightAnnotation,
    arrow: RoadArrowAnnotation,
) -> bool:
    """Determine if a traffic light and road arrow have compatible maneuver semantics."""
    is_round, tl_l, tl_s, tl_r = _get_tl_maneuvers(tl)
    ar_l, ar_s, ar_r = _get_arrow_maneuvers(arrow)

    # Round traffic lights are directional wildcards compatible with any road arrow in lane
    if is_round:
        return True

    # Arrow indicates specific maneuver(s)
    has_overlap = (tl_l and ar_l) or (tl_s and ar_s) or (tl_r and ar_r)
    return bool(has_overlap)


def mine_scene_counterfactual_pairs(
    record: ImageRecord,
    config: CounterfactualMiningConfig = DEFAULT_COUNTERFACTUAL_CONFIG,
) -> list[CounterfactualRelevancePair]:
    """Mine all valid positive, easy negative, cross-lane hard negative, and spatial neighbor pairs."""
    tls = record.traffic_lights
    arrows = record.road_arrows
    if not tls:
        return []

    mined_pairs: list[CounterfactualRelevancePair] = []

    # Map relevant vs distractor TL indices
    pos_tl_indices = [i for i, tl in enumerate(tls) if tl.relevance == 1]
    neg_tl_indices = [i for i, tl in enumerate(tls) if tl.relevance == 0]

    # 1. Mine Positive Pairs
    for i in pos_tl_indices:
        tl = tls[i]
        if arrows:
            # Pair with compatible arrows or best matching arrow
            for j, arrow in enumerate(arrows):
                compat = _is_maneuver_compatible(tl, arrow)
                mined_pairs.append(
                    CounterfactualRelevancePair(
                        tl_index=i,
                        arrow_index=j,
                        pair_type=CounterfactualPairType.POSITIVE,
                        relevance_label=1,
                        weight=1.0,
                        metadata={"compatible": compat, "has_arrow": True},
                    )
                )
        else:
            # Positive TL without visible arrows
            mined_pairs.append(
                CounterfactualRelevancePair(
                    tl_index=i,
                    arrow_index=None,
                    pair_type=CounterfactualPairType.POSITIVE,
                    relevance_label=1,
                    weight=1.0,
                    metadata={"compatible": True, "has_arrow": False},
                )
            )

    # 2. Mine Cross-Lane Confusers (Hard Negatives)
    # A true arrow in the scene paired with a distractor TL, or paired with a TL of incompatible maneuver
    if arrows:
        for i in neg_tl_indices:
            tl = tls[i]
            for j, arrow in enumerate(arrows):
                compat = _is_maneuver_compatible(tl, arrow)
                pair_type = (
                    CounterfactualPairType.CROSS_LANE_CONFUSER
                    if compat
                    else CounterfactualPairType.OPPOSING_MANEUVER_CONFUSER
                )
                mined_pairs.append(
                    CounterfactualRelevancePair(
                        tl_index=i,
                        arrow_index=j,
                        pair_type=pair_type,
                        relevance_label=0,
                        weight=1.2 if compat else 1.0,  # Cross-lane confusers get high attention
                        metadata={"compatible": compat, "cross_lane": True},
                    )
                )

    # 3. Mine Spatial Neighbor Confusers (Hard Negatives on the Same Mast-Arm)
    # When a relevant TL and a distractor TL are in close spatial proximity (same overhead structure)
    for pos_idx in pos_tl_indices:
        pos_box = tls[pos_idx].bbox_xyxy
        pos_cx = (pos_box[0] + pos_box[2]) / 2.0
        pos_cy = (pos_box[1] + pos_box[3]) / 2.0

        for neg_idx in neg_tl_indices:
            neg_box = tls[neg_idx].bbox_xyxy
            neg_cx = (neg_box[0] + neg_box[2]) / 2.0
            neg_cy = (neg_box[1] + neg_box[3]) / 2.0

            dx = abs(pos_cx - neg_cx)
            dy = abs(pos_cy - neg_cy)

            if dx <= config.spatial_dx_threshold_px and dy <= config.spatial_dy_threshold_px:
                # Spatial mast-arm neighbor confuser
                mined_pairs.append(
                    CounterfactualRelevancePair(
                        tl_index=neg_idx,
                        arrow_index=None,
                        pair_type=CounterfactualPairType.SPATIAL_NEIGHBOR_CONFUSER,
                        relevance_label=0,
                        weight=1.5,  # High weighting for subtle mast-arm confusers
                        metadata={
                            "anchor_pos_tl": pos_idx,
                            "dx_px": round(dx, 1),
                            "dy_px": round(dy, 1),
                            "same_mast_arm": True,
                        },
                    )
                )

    # 4. Mine Easy / Background Negatives
    for neg_idx in neg_tl_indices:
        mined_pairs.append(
            CounterfactualRelevancePair(
                tl_index=neg_idx,
                arrow_index=None,
                pair_type=CounterfactualPairType.EASY_NEGATIVE,
                relevance_label=0,
                weight=0.8,
                metadata={"easy_background": True},
            )
        )

    return mined_pairs


class CounterfactualRelevanceSampler:
    """Balanced quota sampler enforcing a 40:30:15:15 pair distribution."""

    def __init__(
        self,
        config: CounterfactualMiningConfig = DEFAULT_COUNTERFACTUAL_CONFIG,
    ) -> None:
        config.validate()
        self.config = config

    def sample_pairs(
        self,
        record: ImageRecord,
        *,
        max_pairs: int | None = None,
        rng: random.Random | None = None,
    ) -> list[CounterfactualRelevancePair]:
        """Sample a balanced set of pairs according to the target ratio distribution."""
        resolved_rng = rng or random.Random(self.config.seed)
        budget = max_pairs or self.config.max_pairs_per_image

        all_pairs = mine_scene_counterfactual_pairs(record, self.config)
        if not all_pairs:
            return []

        # Partition mined pairs by taxonomy category
        buckets: dict[str, list[CounterfactualRelevancePair]] = {
            "pos": [],
            "easy_neg": [],
            "cross_lane": [],
            "spatial": [],
        }

        for pair in all_pairs:
            if pair.pair_type == CounterfactualPairType.POSITIVE:
                buckets["pos"].append(pair)
            elif pair.pair_type == CounterfactualPairType.EASY_NEGATIVE:
                buckets["easy_neg"].append(pair)
            elif pair.pair_type in (
                CounterfactualPairType.CROSS_LANE_CONFUSER,
                CounterfactualPairType.OPPOSING_MANEUVER_CONFUSER,
            ):
                buckets["cross_lane"].append(pair)
            elif pair.pair_type == CounterfactualPairType.SPATIAL_NEIGHBOR_CONFUSER:
                buckets["spatial"].append(pair)

        # Compute target counts per bucket
        target_pos = max(1 if buckets["pos"] else 0, round(budget * self.config.target_pos_ratio))
        target_easy = round(budget * self.config.target_easy_neg_ratio)
        target_cross = round(budget * self.config.target_cross_lane_hard_ratio)
        target_spatial = round(budget * self.config.target_spatial_neighbor_hard_ratio)

        selected: list[CounterfactualRelevancePair] = []

        def take_from_bucket(
            items: list[CounterfactualRelevancePair], count: int
        ) -> list[CounterfactualRelevancePair]:
            if not items or count <= 0:
                return []
            if len(items) <= count:
                return list(items)
            return resolved_rng.sample(items, count)

        sel_pos = take_from_bucket(buckets["pos"], target_pos)
        sel_easy = take_from_bucket(buckets["easy_neg"], target_easy)
        sel_cross = take_from_bucket(buckets["cross_lane"], target_cross)
        sel_spatial = take_from_bucket(buckets["spatial"], target_spatial)

        selected.extend(sel_pos)
        selected.extend(sel_easy)
        selected.extend(sel_cross)
        selected.extend(sel_spatial)

        # Backfill if shortfall exists
        shortfall = budget - len(selected)
        if shortfall > 0:
            remaining_negatives: list[CounterfactualRelevancePair] = []
            for b_name in ("cross_lane", "spatial", "easy_neg"):
                already_chosen = set(selected)
                unpicked = [p for p in buckets[b_name] if p not in already_chosen]
                remaining_negatives.extend(unpicked)

            if remaining_negatives:
                selected.extend(
                    take_from_bucket(remaining_negatives, min(shortfall, len(remaining_negatives)))
                )

        # Shuffle selected pairs
        resolved_rng.shuffle(selected)
        return selected[:budget]


def encode_counterfactual_relevance_targets(
    record: ImageRecord,
    config: CounterfactualMiningConfig = DEFAULT_COUNTERFACTUAL_CONFIG,
    rng: random.Random | None = None,
) -> dict[str, torch.Tensor]:
    """Encode per-traffic-light counterfactual relevance training targets with instance weights."""
    sampler = CounterfactualRelevanceSampler(config=config)
    sampled_pairs = sampler.sample_pairs(record, rng=rng)

    num_tls = len(record.traffic_lights)
    weights = torch.ones(num_tls, dtype=torch.float32)
    confuser_mask = torch.zeros(num_tls, dtype=torch.bool)
    is_hard_negative = torch.zeros(num_tls, dtype=torch.bool)

    for pair in sampled_pairs:
        idx = pair.tl_index
        if 0 <= idx < num_tls:
            weights[idx] = max(weights[idx].item(), pair.weight)
            if pair.pair_type in (
                CounterfactualPairType.CROSS_LANE_CONFUSER,
                CounterfactualPairType.SPATIAL_NEIGHBOR_CONFUSER,
                CounterfactualPairType.OPPOSING_MANEUVER_CONFUSER,
            ):
                confuser_mask[idx] = True
                is_hard_negative[idx] = True

    return {
        "counterfactual_weights": weights,
        "counterfactual_confuser_mask": confuser_mask,
        "is_hard_negative": is_hard_negative,
    }
