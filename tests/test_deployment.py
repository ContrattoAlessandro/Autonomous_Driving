from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.deployment.postprocess import (
    compute_pairwise_iou,
    compute_pairwise_nwd,
    nwd_nms,
    postprocess_multitask_outputs,
    retained_nms_indices,
    size_adaptive_nms,
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

    def test_pairwise_metrics_computation(self) -> None:
        boxes1 = torch.tensor([[10.0, 10.0, 14.0, 14.0]])  # 4x4 box
        boxes2 = torch.tensor([
            [10.0, 10.0, 14.0, 14.0],  # identical
            [11.5, 11.5, 15.5, 15.5],  # 1.5px diagonal shift
            [50.0, 50.0, 54.0, 54.0],  # far away
        ])
        iou = compute_pairwise_iou(boxes1, boxes2)
        nwd = compute_pairwise_nwd(boxes1, boxes2, constant=12.0)
        self.assertEqual(iou.shape, (1, 3))
        self.assertEqual(nwd.shape, (1, 3))
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(nwd[0, 0]), 1.0, places=4)
        # 1.5px diagonal shift on 4x4 box: IoU drops below 0.35, but NWD remains high (> 0.80)
        self.assertLess(float(iou[0, 1]), 0.35)
        self.assertGreater(float(nwd[0, 1]), 0.80)
        # Far box
        self.assertEqual(float(iou[0, 2]), 0.0)
        self.assertLess(float(nwd[0, 2]), 0.05)

    def test_nwd_nms_suppresses_tiny_jitter_cluster(self) -> None:
        # Two tiny 4x4 boxes shifted by 1.5px:
        # Standard IoU is ~0.26 (< 0.50), so standard IoU-NMS does NOT suppress the duplicate.
        # NWD is ~0.83 (>= 0.50), so NWD-NMS correctly suppresses the duplicate.
        boxes = torch.tensor([
            [10.0, 10.0, 14.0, 14.0],
            [11.5, 11.5, 15.5, 15.5],
        ])
        scores = torch.tensor([0.9, 0.8])
        kept_nwd = nwd_nms(boxes, scores, nwd_threshold=0.5, nwd_constant=12.0)
        self.assertEqual(kept_nwd.tolist(), [0])

    def test_size_adaptive_nms_scale_branching(self) -> None:
        # 4 boxes:
        # 0: Tiny high-score box (4x4, area=16) at (10, 10, 14, 14), score=0.95
        # 1: Tiny duplicate shifted by 1.5px (area=16) at (11.5, 11.5, 15.5, 15.5), score=0.85 (suppressed by NWD)
        # 2: Large box (20x20, area=400) at (100, 100, 120, 120), score=0.90
        # 3: Large duplicate shifted by 2px (area=400) at (102, 102, 122, 122), score=0.80 (IoU=0.68 >= 0.65, suppressed by IoU)
        boxes = torch.tensor([
            [10.0, 10.0, 14.0, 14.0],
            [11.5, 11.5, 15.5, 15.5],
            [100.0, 100.0, 120.0, 120.0],
            [102.0, 102.0, 122.0, 122.0],
        ])
        scores = torch.tensor([0.95, 0.85, 0.90, 0.80])
        kept = size_adaptive_nms(
            boxes,
            scores,
            iou_threshold=0.65,
            nwd_threshold=0.50,
            nwd_constant=12.0,
            area_threshold=64.0,
        )
        self.assertEqual(sorted(kept.tolist()), [0, 2])

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
            nms_type="size_adaptive",
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
