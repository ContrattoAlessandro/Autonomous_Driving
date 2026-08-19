from __future__ import annotations

import math
import unittest
import torch
import numpy as np

from tlr_yolo_mtl.model.unified import (
    GatedLaneAwareCrossAttention,
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
)
from tlr_yolo_mtl.model.milestone2 import build_detection_model


def compute_attention_entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Shannon entropy H = -sum(p * log(p)) along the key dimension."""
    p = weights.clamp_min(eps)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    return entropy


class CrossAttentionDynamicsInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 2
        self.k_tl = 4
        self.k_arrow = 3
        self.dim = 128
        self.heads = 4

        self.block = GatedLaneAwareCrossAttention(dimension=self.dim, heads=self.heads)
        self.block.eval()

        self.traffic_tokens = torch.randn(self.batch_size, self.k_tl, self.dim)
        self.arrow_tokens = torch.randn(self.batch_size, self.k_arrow, self.dim)
        self.traffic_boxes = torch.rand(self.batch_size, self.k_tl, 4)
        self.arrow_boxes = torch.rand(self.batch_size, self.k_arrow, 4)
        self.traffic_round = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        self.traffic_maneuver = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ])
        self.arrow_maneuver = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ])
        self.arrow_ego_lane = torch.full((self.batch_size, self.k_arrow), 0.5)
        self.arrow_valid = torch.tensor([[True, True, False], [True, False, False]])

    def test_alpha_zero_exact_identity_property(self) -> None:
        """When gate is zero, conditioned representation must equal unconditioned normalized tokens."""
        with torch.no_grad():
            self.block.gate.fill_(0.0)
            conditioned, weights, bias = self.block(
                self.traffic_tokens,
                self.arrow_tokens,
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes,
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver,
                arrow_ego_lane=self.arrow_ego_lane,
                arrow_valid=self.arrow_valid,
                enabled=True,
            )
            local_conditioned = self.block.normalization(self.traffic_tokens)
            self.assertTrue(torch.allclose(conditioned, local_conditioned, atol=1e-6))

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights over (K_arrow + 1 null token) must sum to 1.0."""
        with torch.no_grad():
            _, weights, _ = self.block(
                self.traffic_tokens,
                self.arrow_tokens,
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes,
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver,
                arrow_ego_lane=self.arrow_ego_lane,
                arrow_valid=self.arrow_valid,
            )
            # weights shape: [B, Heads, K_tl, K_arrow + 1]
            self.assertEqual(weights.shape, (self.batch_size, self.heads, self.k_tl, self.k_arrow + 1))
            sums = weights.sum(dim=-1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_invalid_arrow_masked_to_zero_weight(self) -> None:
        """Invalid arrow slots must receive 0 attention weight."""
        with torch.no_grad():
            _, weights, _ = self.block(
                self.traffic_tokens,
                self.arrow_tokens,
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes,
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver,
                arrow_ego_lane=self.arrow_ego_lane,
                arrow_valid=self.arrow_valid,
            )
            # In batch 0, arrow index 2 is invalid (False)
            invalid_weight_b0 = weights[0, :, :, 2]
            self.assertTrue(torch.all(invalid_weight_b0 < 1e-6))
            # In batch 1, arrow index 1 and 2 are invalid (False)
            self.assertTrue(torch.all(weights[1, :, :, 1] < 1e-6))
            self.assertTrue(torch.all(weights[1, :, :, 2] < 1e-6))

    def test_attention_entropy_theoretical_bounds(self) -> None:
        """Attention entropy H must be between 0 (one-hot) and log(K_arrow + 1) (uniform)."""
        with torch.no_grad():
            _, weights, _ = self.block(
                self.traffic_tokens,
                self.arrow_tokens,
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes,
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver,
                arrow_ego_lane=self.arrow_ego_lane,
                arrow_valid=self.arrow_valid,
            )
            entropy = compute_attention_entropy(weights)
            max_possible_entropy = math.log(self.k_arrow + 1)
            self.assertTrue(torch.all(entropy >= 0.0))
            self.assertTrue(torch.all(entropy <= max_possible_entropy + 1e-5))

    def test_intervention_shuffled_arrows_changes_cross_attention(self) -> None:
        """Shuffling arrows across batch items alters attended features."""
        with torch.no_grad():
            self.block.gate.fill_(1.0)
            cond_orig, _, _ = self.block(
                self.traffic_tokens,
                self.arrow_tokens,
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes,
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver,
                arrow_ego_lane=self.arrow_ego_lane,
                arrow_valid=self.arrow_valid,
            )
            # Permute arrows across batch items: [0, 1] -> [1, 0]
            perm = torch.tensor([1, 0])
            cond_shuffled, _, _ = self.block(
                self.traffic_tokens,
                self.arrow_tokens[perm],
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes[perm],
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver[perm],
                arrow_ego_lane=self.arrow_ego_lane[perm],
                arrow_valid=self.arrow_valid[perm],
            )
            self.assertFalse(torch.allclose(cond_orig, cond_shuffled, atol=1e-4))

    def test_intervention_null_token_forcing(self) -> None:
        """When all arrow valid flags are set to False, 100% attention goes to null token."""
        with torch.no_grad():
            all_false_valid = torch.zeros_like(self.arrow_valid)
            _, weights, _ = self.block(
                self.traffic_tokens,
                self.arrow_tokens,
                traffic_boxes=self.traffic_boxes,
                arrow_boxes=self.arrow_boxes,
                traffic_round=self.traffic_round,
                traffic_maneuver=self.traffic_maneuver,
                arrow_maneuver=self.arrow_maneuver,
                arrow_ego_lane=self.arrow_ego_lane,
                arrow_valid=all_false_valid,
            )
            null_weights = weights[..., -1]
            self.assertTrue(torch.allclose(null_weights, torch.ones_like(null_weights), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
