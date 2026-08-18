from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.deployment.postprocess import (
    postprocess_multitask_outputs,
    retained_nms_indices,
)


class DeploymentTests(unittest.TestCase):
    def test_nms_returns_original_dense_indices(self) -> None:
        decoded = torch.tensor(
            [
                [
                    [10.0, 10.5, 30.0],
                    [10.0, 10.5, 30.0],
                    [8.0, 8.0, 4.0],
                    [8.0, 8.0, 4.0],
                    [0.9, 0.8, 0.7],
                ]
            ]
        )
        indices = retained_nms_indices(
            decoded,
            confidence_threshold=0.1,
            iou_threshold=0.5,
            max_detections=10,
        )
        self.assertEqual(indices[0].tolist(), [0, 2])

    def test_postprocess_keeps_all_outputs_aligned_and_scores_separate(self) -> None:
        detection = torch.tensor(
            [[[10.0, 10.5, 30.0], [10.0, 10.5, 30.0], [8.0, 8.0, 4.0], [8.0, 8.0, 4.0], [0.9, 0.8, 0.7]]]
        )
        states = torch.zeros((1, 4, 3))
        states[0, 2, 0] = 5.0
        states[0, 1, 2] = 5.0
        pictograms = torch.zeros((1, 4, 3))
        pictograms[0, 3, 0] = 5.0
        pictograms[0, 0, 2] = 5.0
        relevance = torch.tensor([[[2.0, -3.0, 1.0]]])
        directions = torch.tensor(
            [[[4.0, 0.0, -4.0], [-4.0, 0.0, 4.0], [-4.0, 0.0, -4.0]]]
        )
        result = postprocess_multitask_outputs(
            (detection, states, pictograms, detection, directions, relevance),
            traffic_confidence=0.1,
            arrow_confidence=0.1,
            iou_threshold=0.5,
        )
        traffic = result["traffic_lights"]
        arrows = result["road_arrows"]
        self.assertEqual(traffic["dense_indices"].tolist(), [[0, 2]])
        self.assertEqual(traffic["state_indices"].tolist(), [[2, 1]])
        self.assertEqual(traffic["pictogram_indices"].tolist(), [[3, 0]])
        self.assertTrue(torch.allclose(traffic["detection_scores"], torch.tensor([[0.9, 0.7]])))
        self.assertTrue(
            torch.allclose(
                traffic["joint_scores"],
                traffic["detection_scores"] * traffic["relevance_probabilities"],
            )
        )
        self.assertEqual(arrows["direction_multihot"].permute(0, 2, 1).tolist(), [[[1, 0, 0], [0, 1, 0]]])


if __name__ == "__main__":
    unittest.main()
