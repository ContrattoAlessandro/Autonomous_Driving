"""Unit tests for Continuous Scale-Conditioned Quality Fusion (Ticket E70)."""

import pytest
import torch

from tlr_yolo_mtl.model.quality import (
    compute_scale_conditioned_alpha,
    compute_scale_conditioned_quality_scores,
    QualityScoringConfig,
)


def test_scale_conditioned_alpha_bounds():
    # Sub-4px (side <= 4px, area <= 16 px^2)
    tiny_areas = torch.tensor([1.0, 4.0, 9.0, 16.0])
    alpha_tiny = compute_scale_conditioned_alpha(
        tiny_areas, alpha_min=0.38, alpha_max=0.90, side_min=2.0, side_max=16.0
    )
    # Side=2 => alpha=0.38, Side=4 => alpha ~ 0.454
    assert alpha_tiny[0].item() == pytest.approx(0.38, abs=1e-3)
    assert (alpha_tiny >= 0.38).all()
    assert (alpha_tiny <= 0.50).all()

    # Macro (>16px, area >= 256 px^2)
    macro_areas = torch.tensor([256.0, 400.0, 1024.0])
    alpha_macro = compute_scale_conditioned_alpha(
        macro_areas, alpha_min=0.38, alpha_max=0.90, side_min=2.0, side_max=16.0
    )
    assert torch.allclose(alpha_macro, torch.full_like(alpha_macro, 0.90), atol=1e-3)


def test_scale_conditioned_alpha_from_boxes():
    # Boxes: [x1, y1, x2, y2]
    # Box 1: 2x2 => side=2, area=4
    # Box 2: 16x16 => side=16, area=256
    boxes = torch.tensor([
        [10.0, 10.0, 12.0, 12.0],
        [10.0, 10.0, 26.0, 26.0],
    ])
    alpha = compute_scale_conditioned_alpha(boxes, alpha_min=0.38, alpha_max=0.90)
    assert alpha[0].item() == pytest.approx(0.38, abs=1e-3)
    assert alpha[1].item() == pytest.approx(0.90, abs=1e-3)


def test_scale_conditioned_scoring_properties():
    # Candidate 1: Tiny (sub-4px), high quality (0.9), moderate class prob (0.6)
    # Candidate 2: Macro (>16px), moderate quality (0.6), high class prob (0.9)
    probs = torch.tensor([0.6, 0.9])
    quals = torch.tensor([0.9, 0.6])
    areas = torch.tensor([4.0, 400.0])

    scores = compute_scale_conditioned_quality_scores(
        probs, quals, areas, alpha_min=0.38, alpha_max=0.90
    )

    # For tiny: alpha=0.38, 1-alpha=0.62 -> s1 = (0.6)^0.38 * (0.9)^0.62 = 0.822 * 0.937 = 0.770
    # For macro: alpha=0.90, 1-alpha=0.10 -> s2 = (0.9)^0.90 * (0.6)^0.10 = 0.909 * 0.950 = 0.864
    assert scores[0] > 0.75  # Quality lifts tiny proposal
    assert scores[1] > 0.85  # Class prob dominates macro proposal
    assert not torch.isnan(scores).any()
