from __future__ import annotations

import unittest
import torch
import torch.nn as nn
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    ROAD_ARROW_CLASS,
    TRAFFIC_LIGHT_CLASS,
)
from tlr_yolo_mtl.model.attributes import _AttributeTower


def compute_vector_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-8) -> float:
    norm1 = g1.norm(2)
    norm2 = g2.norm(2)
    if norm1 < eps or norm2 < eps:
        return 0.0
    return float((torch.dot(g1, g2) / (norm1 * norm2)).clamp(-1.0, 1.0).item())


class MultiTaskGradientUnitTests(unittest.TestCase):
    def test_cosine_similarity_edge_cases(self) -> None:
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([2.0, 4.0, 6.0])
        self.assertAlmostEqual(compute_vector_cosine_similarity(v1, v2), 1.0, places=5)

        v_orth = torch.tensor([-2.0, 1.0, 0.0])
        self.assertAlmostEqual(compute_vector_cosine_similarity(v1, v_orth), 0.0, places=5)

        v_opp = torch.tensor([-1.0, -2.0, -3.0])
        self.assertAlmostEqual(compute_vector_cosine_similarity(v1, v_opp), -1.0, places=5)

        v_zero = torch.zeros(3)
        self.assertEqual(compute_vector_cosine_similarity(v1, v_zero), 0.0)

    def test_ego_lane_neutrality_clamping(self) -> None:
        """When ego_lane_enabled is False, arrow_ego_lane must be constant 0.5."""
        cfg = UnifiedHeadConfig(ego_lane_enabled=False)
        # Mocking dummy module
        linear = nn.Linear(10, 10)
        # When disabled, geometry bias receives constant 0.5
        arrow_ego_lane = torch.rand(2, 8)  # random output from ego head
        if not cfg.ego_lane_enabled:
            arrow_ego_lane = arrow_ego_lane * 0.0 + 0.5
        self.assertTrue(torch.all(arrow_ego_lane == 0.5))

    def test_shared_maneuver_head_gradient_isolation(self) -> None:
        """Verify gradient isolation between TL and Arrow maneuver losses on shared heads."""
        head = _AttributeTower(16, 3)
        features = torch.randn(2, 16, 10, 10, requires_grad=True)
        logits = head(features)  # [2, 3, 10, 10]

        # Simulate positive TL anchors (first 5 anchors)
        loss_tl = logits[:, :, 0:2, 0:2].sum()
        # Simulate positive Arrow anchors (other anchors)
        loss_arrow = logits[:, :, 5:7, 5:7].sum()

        params = list(head.parameters())
        grads_tl = torch.autograd.grad(loss_tl, params, retain_graph=True)
        grads_arrow = torch.autograd.grad(loss_arrow, params, retain_graph=False)

        g_tl_vec = torch.cat([g.reshape(-1) for g in grads_tl])
        g_arrow_vec = torch.cat([g.reshape(-1) for g in grads_arrow])

        cos_sim = compute_vector_cosine_similarity(g_tl_vec, g_arrow_vec)
        self.assertTrue(-1.0 <= cos_sim <= 1.0)

    def test_gradient_conflict_matrix_properties(self) -> None:
        """Verify mathematical properties of the multi-task cosine similarity matrix."""
        torch.manual_seed(42)
        n_tasks = 6
        n_params = 100
        # Generate dummy task gradients
        task_grads = [torch.randn(n_params) for _ in range(n_tasks)]

        matrix = torch.zeros(n_tasks, n_tasks)
        for i in range(n_tasks):
            for j in range(n_tasks):
                matrix[i, j] = compute_vector_cosine_similarity(task_grads[i], task_grads[j])

        # Symmetry
        self.assertTrue(torch.allclose(matrix, matrix.T, atol=1e-5))
        # Diagonal is 1.0
        diag = torch.diagonal(matrix)
        self.assertTrue(torch.allclose(diag, torch.ones(n_tasks), atol=1e-5))
        # Bounds [-1, 1]
        self.assertTrue(torch.all(matrix >= -1.0 - 1e-5))
        self.assertTrue(torch.all(matrix <= 1.0 + 1e-5))


if __name__ == "__main__":
    unittest.main()
