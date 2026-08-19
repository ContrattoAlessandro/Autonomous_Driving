"""Unified Evaluation Contract specification and verification utilities (Ticket E29).

Enforces standardized evaluation protocol, checkpoint selection criteria, matching oracle,
calibration splits, thresholding methods, and safety waterfall analysis across Phase 4.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class EvaluationContractConfig:
    """Canonical Unified Evaluation Contract (E29 Standard)."""

    primary_checkpoint: str = "best_composite.pt"
    diagnostic_checkpoints: tuple[str, ...] = (
        "best_composite.pt",
        "best_relevance.pt",
        "best_tl_detection.pt",
        "best_relevant_red_recall.pt",
        "last.pt",
    )
    matching_protocol: str = "iou"  # "iou" or "hungarian"
    iou_threshold: float = 0.50
    population_split: str = "val"
    calibration_split_ratio: float = 0.50
    calibration_salt: str = "e29_unified_contract"
    standard_threshold: float = 0.50
    safety_target_recalls: tuple[float, ...] = (0.90, 0.95, 0.975)
    resolution: tuple[int, int] = (800, 1600)  # [H, W]
    k_tl: int = 32
    k_arrow: int = 32
    ema: bool = True
    precision: str = "fp16"

    def validate(self) -> list[str]:
        """Validate configuration conformity against E29 standard."""
        violations: list[str] = []
        if self.primary_checkpoint != "best_composite.pt":
            violations.append(
                f"Primary checkpoint must be 'best_composite.pt', got '{self.primary_checkpoint}'"
            )
        if self.iou_threshold != 0.50:
            violations.append(
                f"Matching IoU threshold must be 0.50, got {self.iou_threshold}"
            )
        if self.resolution != (800, 1600):
            violations.append(
                f"Canonical base resolution must be (800, 1600), got {self.resolution}"
            )
        if self.k_tl != 32 or self.k_arrow != 32:
            violations.append(
                f"Candidate token pools must be K_TL=32, K_Arrow=32, got K_TL={self.k_tl}, K_Arrow={self.k_arrow}"
            )
        if self.calibration_split_ratio != 0.50:
            violations.append(
                f"Calibration holdout split must be 50/50, got {self.calibration_split_ratio}"
            )
        return violations


def deterministic_contract_split(
    image_id: str, salt: str = "e29_unified_contract"
) -> bool:
    """Returns True if image belongs to Calibration Split (50%), False for Evaluation Holdout (50%)."""
    key = f"{salt}_{image_id}".encode("utf-8")
    h = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return (h % 2) == 0


@dataclass(slots=True)
class SafetyWaterfallBreakdown:
    """4-Stage Safety Waterfall Failure Decomposition for Relevant Red Traffic Lights."""

    gt_relevant_red_total: int = 0
    # Stage 1: Perception (Object Detection @ IoU=0.50)
    perception_detected: int = 0
    perception_missed: int = 0
    # Stage 2: Candidate Selection (Top-K pool inclusion)
    candidate_selected: int = 0
    candidate_missed: int = 0
    # Stage 3: State Classification (Predicted as RED)
    state_classified_red: int = 0
    state_misclassified: int = 0
    # Stage 4: Relevance Gate (P(rel) >= tau)
    relevance_accepted: int = 0
    relevance_rejected: int = 0

    @property
    def end_to_end_recalled(self) -> int:
        return self.relevance_accepted

    @property
    def end_to_end_recall(self) -> float:
        if self.gt_relevant_red_total == 0:
            return 0.0
        return self.relevance_accepted / self.gt_relevant_red_total

    @property
    def perception_recall(self) -> float:
        if self.gt_relevant_red_total == 0:
            return 0.0
        return self.perception_detected / self.gt_relevant_red_total

    @property
    def candidate_selection_rate(self) -> float:
        if self.perception_detected == 0:
            return 0.0
        return self.candidate_selected / self.perception_detected

    @property
    def state_classification_rate(self) -> float:
        if self.candidate_selected == 0:
            return 0.0
        return self.state_classified_red / self.candidate_selected

    @property
    def relevance_acceptance_rate(self) -> float:
        if self.state_classified_red == 0:
            return 0.0
        return self.relevance_accepted / self.state_classified_red

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_relevant_red_total": self.gt_relevant_red_total,
            "perception_detected": self.perception_detected,
            "perception_missed": self.perception_missed,
            "perception_recall": self.perception_recall,
            "candidate_selected": self.candidate_selected,
            "candidate_missed": self.candidate_missed,
            "candidate_selection_rate": self.candidate_selection_rate,
            "state_classified_red": self.state_classified_red,
            "state_misclassified": self.state_misclassified,
            "state_classification_rate": self.state_classification_rate,
            "relevance_accepted": self.relevance_accepted,
            "relevance_rejected": self.relevance_rejected,
            "relevance_acceptance_rate": self.relevance_acceptance_rate,
            "end_to_end_recalled": self.end_to_end_recalled,
            "end_to_end_recall": self.end_to_end_recall,
        }
