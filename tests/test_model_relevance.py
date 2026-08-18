from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.model.relevance import (
    assigned_relevance_focal_bce,
    combine_detection_relevance_scores,
    encode_record_relevance,
    gather_candidate_relevance,
)


class LocalRelevanceHeadTests(unittest.TestCase):
    def test_multiple_relevant_lights_are_preserved(self) -> None:
        record = ImageRecord(
            image_id="DTLD/train/multiple",
            image_path="multiple.jpg",
            source_dataset="DTLD",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="multiple",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_relevance=True,
            ),
            traffic_lights=[
                TrafficLightAnnotation(
                    bbox_xyxy=(10, 10, 20, 30),
                    relevance=1,
                    valid_relevance=True,
                ),
                TrafficLightAnnotation(
                    bbox_xyxy=(40, 10, 50, 30),
                    relevance=1,
                    valid_relevance=True,
                ),
            ],
        )
        encoded = encode_record_relevance(record)
        self.assertEqual(encoded["tl_relevance"].tolist(), [1, 1])
        self.assertTrue(bool(encoded["tl_relevance_valid"]))

    def test_missing_image_task_masks_all_instances(self) -> None:
        record = ImageRecord(
            image_id="LISA/train/masked",
            image_path="masked.jpg",
            source_dataset="LISA",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="masked",
            task_valid=TaskValidity(traffic_light_detection=True),
            traffic_lights=[TrafficLightAnnotation(bbox_xyxy=(1, 2, 3, 4))],
        )
        encoded = encode_record_relevance(record)
        self.assertEqual(encoded["tl_relevance"].tolist(), [-1])
        self.assertFalse(bool(encoded["tl_relevance_valid"]))

    def test_loss_only_touches_valid_positive_matches(self) -> None:
        logits = torch.zeros((1, 1, 4), requires_grad=True)
        targets = torch.tensor([[1, -1]])
        foreground = torch.tensor([[False, True, True, False]])
        target_indices = torch.tensor([[0, 0, 1, 0]])
        loss, count = assigned_relevance_focal_bce(
            logits, targets, foreground, target_indices
        )
        loss.backward()
        gradient = logits.grad.abs().squeeze(0).squeeze(0)
        self.assertEqual(count, 1)
        self.assertGreater(float(gradient[1]), 0)
        self.assertEqual(float(gradient[[0, 2, 3]].sum()), 0.0)

    def test_image_mask_produces_connected_zero_gradient(self) -> None:
        logits = torch.randn((1, 1, 3), requires_grad=True)
        loss, count = assigned_relevance_focal_bce(
            logits,
            torch.tensor([[1]]),
            torch.tensor([[True, False, False]]),
            torch.zeros((1, 3), dtype=torch.long),
            image_valid=torch.tensor([False]),
        )
        loss.backward()
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_candidate_selection_keeps_relevance_aligned(self) -> None:
        logits = torch.tensor([[[1.0, -2.0, 3.0]]])
        selected = gather_candidate_relevance(logits, torch.tensor([[2, 0]]))
        self.assertEqual(selected["relevance_logits"].tolist(), [[[3.0, 1.0]]])

    def test_detection_and_relevance_scores_remain_separate(self) -> None:
        detection = torch.tensor([[0.8, 0.6]])
        logits = torch.tensor([[[2.0, 1.0]]])
        scores = combine_detection_relevance_scores(detection, logits)
        self.assertTrue(torch.equal(scores["detection_scores"], detection))
        self.assertTrue(torch.all(scores["relevance_probabilities"] > 0.5))
        self.assertGreater(float(scores["relevance_probabilities"].sum()), 1.0)
        self.assertTrue(
            torch.allclose(
                scores["joint_scores"],
                detection * scores["relevance_probabilities"],
            )
        )


if __name__ == "__main__":
    unittest.main()
