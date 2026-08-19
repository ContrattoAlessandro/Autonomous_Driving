"""Unit tests for Ticket E26: TL <-> Road Arrow Semantic Contrastive Alignment."""

import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.training.contrastive_loss import (
    TLArrowContrastiveProjector,
    TLArrowContrastiveLoss,
)


def test_contrastive_projector_shapes_and_norm():
    """Verify projector produces unit-norm embeddings."""
    projector = TLArrowContrastiveProjector(token_dim=128, embed_dim=64)

    B, K_TL, K_Arrow, D = 2, 8, 16, 128
    tl_tokens = torch.randn(B, K_TL, D)
    ar_tokens = torch.randn(B, K_Arrow, D)

    tl_embeds, ar_embeds = projector(tl_tokens, ar_tokens)
    assert tl_embeds.shape == (B, K_TL, 64)
    assert ar_embeds.shape == (B, K_Arrow, 64)

    # L2 norm must be 1.0
    assert torch.allclose(torch.norm(tl_embeds, p=2, dim=-1), torch.ones(B, K_TL), atol=1e-5)
    assert torch.allclose(torch.norm(ar_embeds, p=2, dim=-1), torch.ones(B, K_Arrow), atol=1e-5)


def test_contrastive_loss_computation_and_gradients():
    """Verify InfoNCE loss calculation, positive margin, and backward gradient flow."""
    criterion = TLArrowContrastiveLoss(token_dim=128, embed_dim=64, temperature=0.1)

    B, K_TL, K_Arrow, D = 2, 4, 8, 128
    tl_tokens = torch.randn(B, K_TL, D, requires_grad=True)
    ar_tokens = torch.randn(B, K_Arrow, D, requires_grad=True)

    # Define maneuvers:
    # TL 0: Left [1, 0, 0], TL 1: Straight [0, 1, 0], TL 2: Right [0, 0, 1], TL 3: Left [1, 0, 0]
    tl_man = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]]).expand(B, -1, -1)
    # Arrows 0..3: Left [1, 0, 0], Arrows 4..7: Right [0, 0, 1]
    ar_man = torch.tensor([[[1., 0., 0.]] * 4 + [[0., 0., 1.]] * 4]).expand(B, -1, -1)

    tl_round = torch.zeros(B, K_TL)  # all directional
    tl_valid = torch.ones(B, K_TL, dtype=torch.bool)
    ar_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    loss, metrics = criterion(
        tl_tokens,
        ar_tokens,
        traffic_maneuver=tl_man,
        arrow_maneuver=ar_man,
        traffic_round=tl_round,
        traffic_valid=tl_valid,
        arrow_valid=ar_valid,
    )

    assert loss >= 0.0
    assert not torch.isnan(loss)
    assert "contrastive_loss" in metrics
    assert "alignment_margin" in metrics
    assert metrics["valid_queries"] > 0

    # Backward pass
    loss.backward()
    assert tl_tokens.grad is not None
    assert ar_tokens.grad is not None


def test_contrastive_loss_zero_positive_pairs_fallback():
    """Verify zero loss when no valid pairs exist."""
    criterion = TLArrowContrastiveLoss(token_dim=128, embed_dim=64)

    B, K_TL, K_Arrow, D = 1, 2, 2, 128
    tl_tokens = torch.randn(B, K_TL, D)
    ar_tokens = torch.randn(B, K_Arrow, D)
    tl_man = torch.zeros(B, K_TL, 3)
    ar_man = torch.zeros(B, K_Arrow, 3)
    tl_round = torch.ones(B, K_TL)  # all round lights
    tl_valid = torch.ones(B, K_TL, dtype=torch.bool)
    ar_valid = torch.ones(B, K_Arrow, dtype=torch.bool)

    loss, metrics = criterion(
        tl_tokens,
        ar_tokens,
        traffic_maneuver=tl_man,
        arrow_maneuver=ar_man,
        traffic_round=tl_round,
        traffic_valid=tl_valid,
        arrow_valid=ar_valid,
    )

    assert float(loss.item()) == 0.0
    assert metrics["valid_queries"] == 0
