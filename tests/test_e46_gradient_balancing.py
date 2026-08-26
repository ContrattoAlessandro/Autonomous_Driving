"""Unit tests for Ticket E46: Multi-Task Gradient Balancing & Conflict Diagnostics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import numpy as np
import pytest
import torch
import torch.nn as nn

from tlr_yolo_mtl.training.gradient_balancing import (
    TASK_NAMES,
    GradNormBalancer,
    PCGradProjector,
    compute_gradient_cosine_similarity,
    flatten_gradients,
    partition_model_parameters,
    unflatten_gradients,
)


def test_gradient_cosine_similarity_edge_cases():
    # Identical vectors -> 1.0
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([1.0, 2.0, 3.0])
    assert pytest.approx(compute_gradient_cosine_similarity(v1, v2), 1e-5) == 1.0

    # Opposite vectors -> -1.0
    v3 = torch.tensor([-1.0, -2.0, -3.0])
    assert pytest.approx(compute_gradient_cosine_similarity(v1, v3), 1e-5) == -1.0

    # Orthogonal vectors -> 0.0
    v4 = torch.tensor([1.0, 0.0, 0.0])
    v5 = torch.tensor([0.0, 1.0, 0.0])
    assert pytest.approx(compute_gradient_cosine_similarity(v4, v5), 1e-5) == 0.0

    # Zero vector -> 0.0
    v0 = torch.tensor([0.0, 0.0, 0.0])
    assert pytest.approx(compute_gradient_cosine_similarity(v1, v0), 1e-5) == 0.0


def test_flatten_and_unflatten_gradients():
    p1 = nn.Parameter(torch.randn(2, 3))
    p2 = nn.Parameter(torch.randn(4))
    params = [p1, p2]

    g1 = torch.randn(2, 3)
    g2 = torch.randn(4)
    grads = [g1, g2]

    flat = flatten_gradients(grads, params, torch.device("cpu"))
    assert flat.shape == (10,)

    unflat = unflatten_gradients(flat, params)
    assert len(unflat) == 2
    assert torch.allclose(unflat[0], g1)
    assert torch.allclose(unflat[1], g2)


def test_gradnorm_balancer_weight_normalization():
    balancer = GradNormBalancer(
        task_names=TASK_NAMES,
        initial_weights={"Detection": 1.0, "NWD": 0.5, "State": 0.75, "Round": 0.5, "Maneuver": 1.0, "Relevance": 1.0},
        alpha=1.5,
        update_rate=0.05,
    )

    initial_w = balancer.get_weights_dict()
    assert len(initial_w) == 6
    assert pytest.approx(sum(initial_w.values()), 1e-4) == 6.0

    # Step 1: Initialize losses
    initial_losses = [2.5, 0.8, 1.2, 0.4, 0.9, 0.6]
    initial_norms = [1.0, 0.3, 0.5, 0.2, 0.4, 0.3]

    balancer.set_initial_losses(initial_losses)

    # Step 2: Simulate epoch update where Detection loss hasn't dropped much (slow learning)
    # while Relevance dropped significantly (fast learning)
    current_losses = [2.4, 0.4, 0.6, 0.2, 0.45, 0.1]
    current_norms = [0.8, 0.3, 0.5, 0.2, 0.4, 0.2]

    updated_w = balancer.update_weights(current_losses, current_norms)
    assert pytest.approx(sum(updated_w.values()), 1e-4) == 6.0

    # The weight for Detection (slower relative learner) should increase relative to fast learners
    assert updated_w["Detection"] > initial_w["Detection"]


def test_pcgrad_projection_orthogonality():
    # Create two conflicting gradients: g1 = [1, 0], g2 = [-1, 0.5]
    # Dot product = -1.0 < 0
    p = nn.Parameter(torch.zeros(2))
    projector = PCGradProjector([p])

    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 0.5])

    assert torch.dot(g1, g2).item() < 0.0

    fused, telemetry = projector.project_conflicting_gradients([g1, g2], shuffle=False)
    assert telemetry["conflicts_detected"] > 0
    assert fused.shape == (2,)

    # Verify that the projected g1 is orthogonal to g2:
    # g1_proj = g1 - (dot/|g2|^2)*g2
    # dot(g1_proj, g2) should be approximately 0
    dot_val = torch.dot(g1, g2)
    norm_sq = torch.dot(g2, g2)
    g1_proj = g1 - (dot_val / norm_sq) * g2
    assert pytest.approx(torch.dot(g1_proj, g2).item(), abs=1e-6) == 0.0


def test_model_partitioning_toy_model():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3),  # layer 0 (backbone)
                nn.Conv2d(16, 32, 3),  # layer 1 (backbone)
                nn.Conv2d(32, 64, 3),  # layer 2 (neck)
            )

    toy = ToyModel()
    partitions = partition_model_parameters(toy)
    assert "backbone" in partitions
    assert "neck" in partitions
    assert "shared_all" in partitions
    assert partitions["shared_all"].param_count > 0
