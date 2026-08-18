from __future__ import annotations

import unittest

import torch

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.data.taxonomy import factor_pictogram
from tlr_yolo_mtl.deployment.postprocess import postprocess_multitask_outputs
from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import (
    GatedLaneAwareCrossAttention,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
    encode_record_unified,
    fixed_topk_candidates,
)
from tlr_yolo_mtl.training.losses import (
    assigned_binary_focal_bce,
    assigned_multilabel_focal_bce,
)


class FactorizedTargetTests(unittest.TestCase):
    def test_schema_21_compound_pictogram_migrates_to_factorized_targets(self) -> None:
        record = ImageRecord.from_dict(
            {
                "schema_version": "2.1",
                "image_id": "DTLD/train/legacy",
                "image_path": "unused.jpg",
                "source_dataset": "DTLD",
                "original_width": 100,
                "original_height": 50,
                "split": "train",
                "sequence_id": "legacy",
                "task_valid": {
                    "traffic_light_detection": True,
                    "traffic_light_pictogram": True,
                },
                "traffic_lights": [
                    {
                        "bbox_xyxy": [10, 5, 20, 25],
                        "pictogram": None,
                        "valid_pictogram": False,
                        "source_attributes": {
                            "pictogram": "arrow_straight_left"
                        },
                    }
                ],
            }
        )
        light = record.traffic_lights[0]
        self.assertEqual(light.round_target, 0)
        self.assertTrue(light.valid_round)
        self.assertEqual(light.maneuver_multihot, (1, 1, 0))
        self.assertTrue(light.valid_maneuver)
        self.assertTrue(record.task_valid.traffic_light_round)
        self.assertTrue(record.task_valid.traffic_light_maneuver)
        self.assertEqual(record.validation_errors(), [])

    def test_compound_tl_pictogram_becomes_multihot(self) -> None:
        mapped = factor_pictogram("arrow_straight_left")
        self.assertEqual(mapped.round, 0)
        self.assertTrue(mapped.valid_round)
        self.assertEqual(mapped.maneuver, (1, 1, 0))
        self.assertTrue(mapped.valid_maneuver)

    def test_round_is_not_encoded_as_three_directions(self) -> None:
        mapped = factor_pictogram("circle")
        self.assertEqual(mapped.round, 1)
        self.assertTrue(mapped.valid_round)
        self.assertIsNone(mapped.maneuver)
        self.assertFalse(mapped.valid_maneuver)

    def test_unified_encoding_keeps_one_object_order(self) -> None:
        record = ImageRecord(
            image_id="DTLD/train/factorized",
            image_path="unused.jpg",
            source_dataset="DTLD",
            original_width=100,
            original_height=50,
            split="train",
            sequence_id="factorized",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_state=True,
                traffic_light_relevance=True,
                arrow_detection=True,
                traffic_light_round=True,
                traffic_light_maneuver=True,
                arrow_ego_lane=True,
            ),
            traffic_lights=[
                TrafficLightAnnotation(
                    bbox_xyxy=(10, 5, 20, 25),
                    state="green",
                    relevance=1,
                    valid_state=True,
                    valid_relevance=True,
                    source_attributes={"pictogram": "arrow_straight_left"},
                )
            ],
            road_arrows=[
                RoadArrowAnnotation(
                    bbox_xyxy=(20, 30, 60, 48),
                    direction_multihot=(0, 1, 1),
                    is_ego_lane=1,
                    valid_ego_lane=True,
                )
            ],
        )
        encoded = encode_record_unified(record)
        self.assertEqual(encoded["object_cls"].reshape(-1).tolist(), [0.0, 1.0])
        self.assertEqual(encoded["object_state"].tolist(), [2, -1])
        self.assertEqual(encoded["object_round"].tolist(), [0.0, -1.0])
        self.assertEqual(
            encoded["object_maneuver"].tolist(),
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        )
        self.assertEqual(encoded["object_relevance"].tolist(), [1, -1])
        self.assertEqual(encoded["object_ego_lane"].tolist(), [-1.0, 1.0])


class CandidateAndAttentionTests(unittest.TestCase):
    def test_topk_is_fixed_width_and_marks_padding(self) -> None:
        indices, scores, valid = fixed_topk_candidates(
            torch.tensor([[0.2, 0.9, 0.1]]), 5, threshold=0.15
        )
        self.assertEqual(indices.shape, (1, 5))
        self.assertEqual(indices[0, :3].tolist(), [1, 0, 2])
        self.assertEqual(valid.tolist(), [[True, True, False, False, False]])
        self.assertEqual(scores[0, 3:].tolist(), [0.0, 0.0])

    def test_gate_zero_preserves_local_path_and_null_is_always_valid(self) -> None:
        module = GatedLaneAwareCrossAttention(dimension=8, heads=2)
        traffic = torch.randn((1, 2, 8))
        arrows = torch.randn((1, 3, 8))
        boxes_tl = torch.rand((1, 2, 4))
        boxes_arrow = torch.rand((1, 3, 4))
        conditioned, weights, bias = module(
            traffic,
            arrows,
            traffic_boxes=boxes_tl,
            arrow_boxes=boxes_arrow,
            traffic_round=torch.ones((1, 2)),
            traffic_maneuver=torch.zeros((1, 2, 3)),
            arrow_maneuver=torch.rand((1, 3, 3)),
            arrow_ego_lane=torch.rand((1, 3)),
            arrow_valid=torch.zeros((1, 3), dtype=torch.bool),
        )
        self.assertTrue(torch.allclose(conditioned, module.normalization(traffic)))
        self.assertTrue(torch.equal(weights[..., :3], torch.zeros_like(weights[..., :3])))
        self.assertTrue(torch.allclose(weights[..., -1], torch.ones_like(weights[..., -1])))
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))
        self.assertEqual(float(module.gate.detach()), 0.0)


class ConditionalLossTests(unittest.TestCase):
    def test_binary_and_multilabel_targets_mask_other_object_type(self) -> None:
        foreground = torch.tensor([[True, True]])
        target_indices = torch.tensor([[0, 1]])
        binary_logits = torch.zeros((1, 1, 2), requires_grad=True)
        binary, binary_count = assigned_binary_focal_bce(
            binary_logits,
            torch.tensor([[[1.0], [-1.0]]]),
            foreground,
            target_indices,
        )
        multi_logits = torch.zeros((1, 3, 2), requires_grad=True)
        multi, multi_count = assigned_multilabel_focal_bce(
            multi_logits,
            torch.tensor([[[1.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]]),
            foreground,
            target_indices,
        )
        (binary + multi).backward()
        self.assertEqual((binary_count, multi_count), (1, 1))
        self.assertGreater(float(binary_logits.grad[0, 0, 0].abs()), 0.0)
        self.assertEqual(float(binary_logits.grad[0, 0, 1]), 0.0)
        self.assertGreater(float(multi_logits.grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(multi_logits.grad[:, :, 1].abs().sum()), 0.0)


class UnifiedModelIntegrationTests(unittest.TestCase):
    def test_real_yolo_forward_has_two_types_and_padded_sets(self) -> None:
        wrapper = build_detection_model()
        attach_unified_relevance_head(wrapper)
        model = wrapper.model.eval()
        with torch.inference_mode():
            decoded, raw = model(torch.zeros((1, 3, 64, 64)))
        head = model.model[-1]
        self.assertIsInstance(head, UnifiedTrafficControlDetect)
        self.assertEqual(decoded.shape, (1, 6, 84))
        self.assertEqual(raw["state_logits"].shape, (1, 4, 84))
        self.assertEqual(raw["round_logits"].shape, (1, 1, 84))
        self.assertEqual(raw["maneuver_logits"].shape, (1, 3, 84))
        self.assertEqual(raw["traffic_candidate_indices"].shape, (1, 32))
        self.assertEqual(raw["arrow_candidate_indices"].shape, (1, 16))
        self.assertEqual(raw["attention_weights"].shape, (1, 4, 32, 17))
        self.assertTrue(
            torch.allclose(raw["relevance_logits"], raw["local_relevance_logits"])
        )

    def test_unified_postprocess_keeps_relevance_slot_alignment(self) -> None:
        detection = torch.zeros((1, 6, 4))
        detection[0, :4] = torch.tensor(
            [[10.0, 30.0, 50.0, 70.0], [10.0, 30.0, 10.0, 30.0], [4.0] * 4, [4.0] * 4]
        )
        detection[0, 4] = torch.tensor([0.9, 0.1, 0.8, 0.1])
        detection[0, 5] = torch.tensor([0.1, 0.95, 0.1, 0.85])
        states = torch.zeros((1, 4, 4))
        states[0, 2, 0] = 5.0
        rounds = torch.zeros((1, 1, 4))
        maneuvers = torch.zeros((1, 3, 4))
        maneuvers[0, 1, 1] = 5.0
        ego = torch.zeros((1, 1, 4))
        output = postprocess_multitask_outputs(
            (
                detection,
                states,
                rounds,
                maneuvers,
                ego,
                torch.tensor([[0, 2]]),
                torch.tensor([[True, True]]),
                torch.tensor([[1, 3]]),
                torch.tensor([[True, True]]),
                torch.tensor([[[2.0, -2.0]]]),
                torch.full((1, 2, 2, 3), 1 / 3),
            ),
            traffic_confidence=0.2,
            arrow_confidence=0.2,
        )
        traffic = output["traffic_lights"]
        arrows = output["road_arrows"]
        self.assertEqual(traffic["dense_indices"].tolist(), [[0, 2]])
        self.assertEqual(traffic["candidate_slots"].tolist(), [[0, 1]])
        self.assertEqual(traffic["state_indices"].tolist(), [[2, 0]])
        self.assertGreater(
            float(traffic["relevance_probabilities"][0, 0]),
            float(traffic["relevance_probabilities"][0, 1]),
        )
        self.assertEqual(arrows["dense_indices"].tolist(), [[1, 3]])


if __name__ == "__main__":
    unittest.main()
