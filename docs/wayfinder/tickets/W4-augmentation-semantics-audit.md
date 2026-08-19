---
title: "W4: Augmentation Semantics & Label Invariance Audit"
type: task
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

Do active or candidate data augmentations preserve semantic integrity and cross-attention relationships, or do they introduce systematic label noise (e.g. unflipped directional vectors, HSV hue corruptions of traffic states, or synthetic Mosaic cross-image pairings)?

## Context & Requirements

1. **Horizontal Flip Audit**:
   - Verified that when horizontal flip is active, directional maneuver targets are strictly inverted:
     $$\text{maneuver} = [left, straight, right] \longrightarrow [right, straight, left]$$
     for both traffic lights and road arrows.
   - Verified bounding box coordinate inversion: $x_1, x_2 \to W - x_2, W - x_1$ and arrow polygon segmentations $(x, y) \to (W - x, y)$.
   - Extended `flip_pictogram` in `tlr_yolo_mtl/data/taxonomy.py` to handle compound and arrow pictograms (`straight_left` $\leftrightarrow$ `straight_right`, `left` $\leftrightarrow$ `right`, etc.).

2. **Photometric / Color Augmentation Stability**:
   - Tested conservative HSV perturbations in `_photometric_augment` across 500 randomized trials.
   - Retained 100% color polarity across Red, Yellow, and Green states without state label corruption.

3. **Contextual Augmentation Isolation**:
   - Confirmed Mosaic, MixUp, and CutMix remain strictly disabled (`0.0`), preventing synthetic cross-image pairings that would corrupt cross-attention learning.

## Empirical Resolution & Diagnostic Artifacts

- **Unit Tests**: `tests/test_augmentation_semantics.py` (6/6 tests passing)
- **Audit Script**: `scripts/audit_augmentation_semantics.py`
- **Diagnostic Report**: `results/audit_augmentation_semantics.md`
