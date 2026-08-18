"""Resource-aware multi-dataset training utilities for TLR-YOLO-MTL."""

from .losses import MultiTaskLossWeights, TLRMultiTaskCriterion

__all__ = ["MultiTaskLossWeights", "TLRMultiTaskCriterion"]
