from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.model.attributes import (
    PICTOGRAM_TO_INDEX,
    STATE_TO_INDEX,
    _pad_targets,
    assigned_attribute_cross_entropy,
    encode_record_attributes,
    gather_candidate_attributes,
)


class AttributeHeadTests(unittest.TestCase):
    def test_canonical_class_order(self) -> None:
        self.assertEqual(STATE_TO_INDEX, {"red": 0, "yellow": 1, "green": 2, "off": 3})
        self.assertEqual(
            PICTOGRAM_TO_INDEX,
            {"round": 0, "left": 1, "straight": 2, "right": 3},
        )

    def test_record_encoding_preserves_box_order_and_masks_unknown(self) -> None:
        record = ImageRecord(
            image_id="DTLD/train/attributes",
            image_path="attributes.jpg",
            source_dataset="DTLD",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="sequence",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_state=True,
                traffic_light_pictogram=True,
            ),
            traffic_lights=[
                TrafficLightAnnotation(
                    bbox_xyxy=(10, 5, 20, 25),
                    state="green",
                    pictogram="left",
                    valid_state=True,
                    valid_pictogram=True,
                ),
                TrafficLightAnnotation(bbox_xyxy=(30, 5, 40, 25)),
            ],
        )
        encoded = encode_record_attributes(record)
        self.assertEqual(encoded["tl_state"].tolist(), [2, -1])
        self.assertEqual(encoded["tl_pictogram"].tolist(), [1, -1])

    def test_loss_backpropagates_only_to_valid_positive_matches(self) -> None:
        logits = torch.zeros((1, 4, 6), requires_grad=True)
        targets = torch.tensor([[2, -1]])
        foreground = torch.tensor([[False, True, True, False, False, False]])
        target_indices = torch.tensor([[0, 0, 1, 0, 0, 0]])
        loss, count = assigned_attribute_cross_entropy(
            logits, targets, foreground, target_indices
        )
        loss.backward()
        gradient = logits.grad.abs().sum(1).squeeze(0)
        self.assertEqual(count, 1)
        self.assertGreater(float(gradient[1]), 0)
        self.assertEqual(float(gradient[[0, 2, 3, 4, 5]].sum()), 0.0)

    def test_all_unknown_attributes_have_connected_zero_loss(self) -> None:
        logits = torch.randn((1, 4, 3), requires_grad=True)
        loss, count = assigned_attribute_cross_entropy(
            logits,
            torch.tensor([[-1]]),
            torch.tensor([[True, False, False]]),
            torch.zeros((1, 3), dtype=torch.long),
        )
        loss.backward()
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_padding_matches_yolo_per_image_instance_order(self) -> None:
        values = torch.tensor([2, -1, 0])
        batch_indices = torch.tensor([0, 0, 1])
        padded = _pad_targets(values, batch_indices, batch_size=2)
        self.assertEqual(padded.tolist(), [[2, -1], [0, -1]])

    def test_candidate_selection_keeps_attributes_aligned(self) -> None:
        state = torch.tensor(
            [[[1.0, 2.0, 9.0], [8.0, 3.0, 0.0], [0.0, 7.0, 1.0], [0.0, 0.0, 2.0]]]
        )
        pictogram = state.flip(1)
        selected = gather_candidate_attributes(
            state, pictogram, torch.tensor([[2, 0]])
        )
        self.assertEqual(selected["state_indices"].tolist(), [[0, 1]])
        self.assertEqual(selected["pictogram_indices"].tolist(), [[3, 2]])


if __name__ == "__main__":
    unittest.main()
