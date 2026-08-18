from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.data.schema import ImageRecord, TaskValidity
from tlr_yolo_mtl.model.context import (
    build_soft_arrow_heatmaps,
    encode_record_context_gradient,
    scale_arrow_context_gradient,
)


class ArrowContextTests(unittest.TestCase):
    def test_soft_heatmaps_are_pre_nms_products(self) -> None:
        scores = torch.zeros((1, 1, 5))
        directions = torch.zeros((1, 3, 5))
        maps = build_soft_arrow_heatmaps(scores, directions, [(2, 2), (1, 1)])
        self.assertEqual([tuple(value.shape) for value in maps], [(1, 3, 2, 2), (1, 3, 1, 1)])
        self.assertTrue(torch.allclose(maps[0], torch.full_like(maps[0], 0.25)))
        self.assertTrue(torch.allclose(maps[1], torch.full_like(maps[1], 0.25)))

    def test_gradient_scaling_preserves_forward_values(self) -> None:
        values = torch.tensor([[1.0], [2.0]], requires_grad=True)
        scaled = scale_arrow_context_gradient(values, torch.tensor([0.0, 0.25]))
        self.assertTrue(torch.equal(scaled.detach(), values.detach()))
        scaled.sum().backward()
        self.assertTrue(
            torch.allclose(values.grad, torch.tensor([[0.0], [0.25]]))
        )

    def test_unpaired_record_stops_relevance_to_arrow_gradient(self) -> None:
        record = ImageRecord(
            image_id="DTLD/train/unpaired",
            image_path="unpaired.jpg",
            source_dataset="DTLD",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="unpaired",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_relevance=True,
                arrow_detection=False,
            ),
        )
        encoded = encode_record_context_gradient(record)
        self.assertEqual(float(encoded["relevance_arrow_context_scale"]), 0.0)
        self.assertFalse(bool(encoded["relevance_arrow_context_paired"]))

    def test_only_genuinely_paired_record_gets_quarter_gradient(self) -> None:
        record = ImageRecord(
            image_id="future/paired",
            image_path="paired.jpg",
            source_dataset="future_paired",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="paired",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_relevance=True,
                arrow_detection=True,
            ),
        )
        encoded = encode_record_context_gradient(record)
        self.assertEqual(float(encoded["relevance_arrow_context_scale"]), 0.25)
        self.assertTrue(bool(encoded["relevance_arrow_context_paired"]))

    def test_invalid_gradient_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scale_arrow_context_gradient(torch.zeros((1, 2)), 1.5)


if __name__ == "__main__":
    unittest.main()
