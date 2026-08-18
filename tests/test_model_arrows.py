from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
)
from tlr_yolo_mtl.model.arrows import (
    _pad_multihot_targets,
    assigned_direction_bce,
    encode_record_arrows,
    gather_arrow_directions,
)


class ArrowHeadTests(unittest.TestCase):
    def test_compound_direction_is_preserved(self) -> None:
        record = ImageRecord(
            image_id="CeyMo/train/arrow",
            image_path="arrow.jpg",
            source_dataset="CeyMo",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="arrow",
            task_valid=TaskValidity(arrow_detection=True),
            road_arrows=[
                RoadArrowAnnotation(
                    bbox_xyxy=(10, 10, 40, 40),
                    direction_multihot=(1, 1, 0),
                )
            ],
        )
        encoded = encode_record_arrows(record)
        self.assertEqual(encoded["arrow_direction"].tolist(), [[1.0, 1.0, 0.0]])
        self.assertTrue(bool(encoded["arrow_detection_valid"]))

    def test_direction_loss_only_touches_foreground(self) -> None:
        logits = torch.zeros((1, 3, 4), requires_grad=True)
        targets = torch.tensor([[[1.0, 0.0, 1.0]]])
        foreground = torch.tensor([[False, True, False, False]])
        target_indices = torch.zeros((1, 4), dtype=torch.long)
        loss, count = assigned_direction_bce(
            logits, targets, foreground, target_indices
        )
        loss.backward()
        gradient = logits.grad.abs().sum(1).squeeze(0)
        self.assertEqual(count, 1)
        self.assertGreater(float(gradient[1]), 0)
        self.assertEqual(float(gradient[[0, 2, 3]].sum()), 0.0)

    def test_no_arrow_targets_produce_connected_zero_direction_loss(self) -> None:
        logits = torch.randn((2, 3, 5), requires_grad=True)
        loss, count = assigned_direction_bce(
            logits,
            torch.empty((2, 0, 3)),
            torch.zeros((2, 5), dtype=torch.bool),
            torch.zeros((2, 5), dtype=torch.long),
        )
        loss.backward()
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_multihot_padding_preserves_per_image_order(self) -> None:
        values = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        padded = _pad_multihot_targets(values, torch.tensor([0, 0, 1]), 2)
        self.assertEqual(
            padded.tolist(),
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
        )

    def test_candidate_selection_keeps_direction_logits_aligned(self) -> None:
        logits = torch.tensor(
            [[[1.0, -1.0, 3.0], [2.0, 4.0, -2.0], [-3.0, 5.0, 6.0]]]
        )
        selected = gather_arrow_directions(logits, torch.tensor([[2, 0]]))
        self.assertEqual(
            selected["direction_logits"].tolist(),
            [[[3.0, 1.0], [-2.0, 2.0], [6.0, -3.0]]],
        )


if __name__ == "__main__":
    unittest.main()
