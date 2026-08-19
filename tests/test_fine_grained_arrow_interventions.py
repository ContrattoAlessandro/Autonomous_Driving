from __future__ import annotations

import math
import unittest
import torch
import numpy as np

from scripts.audit_fine_grained_arrow_interventions import (
    compute_attention_entropy,
    permute_geometry,
    permute_maneuver,
    permute_appearance,
)
from tlr_yolo_mtl.model.unified import (
    GatedLaneAwareCrossAttention,
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
)


class FineGrainedArrowInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cpu")
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
        self.arrow_feats = torch.randn(self.batch_size, self.k_arrow, 64)

    def test_geometry_permutation_preserves_shape_and_modifies_boxes(self) -> None:
        """Geometry perturbation must preserve tensor shape while modifying coordinates."""
        perm_boxes = permute_geometry(self.arrow_boxes, self.arrow_valid, self.device)
        self.assertEqual(perm_boxes.shape, self.arrow_boxes.shape)
        # Check that valid boxes have been modified or reordered
        self.assertFalse(torch.allclose(perm_boxes, self.arrow_boxes, atol=1e-5))

    def test_maneuver_permutation_cycles_classes(self) -> None:
        """Maneuver perturbation must preserve shape and cycle/permute classes."""
        perm_man = permute_maneuver(self.arrow_maneuver, self.arrow_valid, self.device)
        self.assertEqual(perm_man.shape, self.arrow_maneuver.shape)
        # In batch 1, single valid arrow has maneuver [0, 1, 0] (Straight) -> cycled to [0, 0, 1] (Right)
        orig_m1 = self.arrow_maneuver[1, 0]
        perm_m1 = perm_man[1, 0]
        self.assertEqual(float(perm_m1[0]), float(orig_m1[2]))
        self.assertEqual(float(perm_m1[1]), float(orig_m1[0]))
        self.assertEqual(float(perm_m1[2]), float(orig_m1[1]))

    def test_appearance_permutation_generates_noise(self) -> None:
        """Appearance perturbation must generate random noise of same shape."""
        noise = permute_appearance(self.arrow_feats, self.arrow_valid, self.device)
        self.assertEqual(noise.shape, self.arrow_feats.shape)
        self.assertFalse(torch.allclose(noise, self.arrow_feats, atol=1e-3))

    def test_attention_entropy_mathematical_bounds(self) -> None:
        """Entropy must be non-negative and bounded by log(K_arrow + 1)."""
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
            max_entropy = math.log(self.k_arrow + 1)
            self.assertTrue(torch.all(entropy >= 0.0))
            self.assertTrue(torch.all(entropy <= max_entropy + 1e-5))

    def test_null_forcing_100_percent_mass(self) -> None:
        """Forcing all arrow_valid to False must allocate 100% attention mass to null token."""
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
                arrow_valid=torch.zeros_like(self.arrow_valid),
            )
            null_weights = weights[..., -1]
            self.assertTrue(torch.allclose(null_weights, torch.ones_like(null_weights), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
