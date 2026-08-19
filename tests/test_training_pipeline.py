from __future__ import annotations

import unittest

import numpy as np
import torch

from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    RoadArrowAnnotation,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.training.data import (
    BalancedEffectiveBatchSampler,
    ManifestEntry,
    canonical_multitask_collate,
    letterbox_box,
    letterbox_parameters,
    prepare_training_sample,
    source_group,
)
from tlr_yolo_mtl.training.diagnostics import validate_smoke_context_batch
from tlr_yolo_mtl.training.engine import (
    ExponentialMovingAverage,
    apply_training_overrides,
    load_training_config,
    parse_phases,
)
from tlr_yolo_mtl.training.losses import normalized_wasserstein_loss


class TrainingDataTests(unittest.TestCase):
    def test_letterbox_geometry_for_two_to_one_image(self) -> None:
        scale, left, top, width, height = letterbox_parameters((4, 8), (8, 8))
        self.assertEqual((scale, left, top, width, height), (1.0, 0, 2, 8, 4))
        box = letterbox_box(
            (0, 0, 4, 4),
            scale=scale,
            left=left,
            top=top,
            target_size=(8, 8),
        )
        self.assertEqual(box, (0.0, 2.0, 4.0, 6.0))

    def test_training_sample_keeps_task_streams_separate(self) -> None:
        record = ImageRecord(
            image_id="synthetic/sample",
            image_path="unused.png",
            source_dataset="DTLD",
            original_width=8,
            original_height=4,
            split="train",
            sequence_id="sample",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_state=True,
                traffic_light_pictogram=True,
                traffic_light_relevance=True,
            ),
            traffic_lights=[
                TrafficLightAnnotation(
                    bbox_xyxy=(0, 0, 4, 4),
                    state="red",
                    pictogram="round",
                    relevance=1,
                    valid_state=True,
                    valid_pictogram=True,
                    valid_relevance=True,
                )
            ],
        )
        sample = prepare_training_sample(
            np.zeros((4, 8, 3), dtype=np.uint8),
            record,
            target_size=(8, 8),
        )
        self.assertEqual(tuple(sample["image"].shape), (3, 8, 8))
        self.assertTrue(
            torch.allclose(
                sample["bboxes"], torch.tensor([[0.25, 0.5, 0.5, 0.5]])
            )
        )
        self.assertEqual(sample["tl_state"].tolist(), [0])
        self.assertEqual(sample["tl_pictogram"].tolist(), [0])
        self.assertEqual(sample["tl_relevance"].tolist(), [1])
        self.assertFalse(bool(sample["arrow_detection_valid"]))

    def test_horizontal_flip_flag_controls_flip(self) -> None:
        record = ImageRecord(
            image_id="synthetic/flip_test",
            image_path="unused.png",
            source_dataset="DTLD",
            original_width=8,
            original_height=4,
            split="train",
            sequence_id="sample",
            task_valid=TaskValidity(traffic_light_detection=True),
            traffic_lights=[
                TrafficLightAnnotation(
                    bbox_xyxy=(0, 0, 2, 4),
                    state="red",
                    pictogram="round",
                    relevance=1,
                )
            ],
        )
        sample_no_flip = prepare_training_sample(
            np.zeros((4, 8, 3), dtype=np.uint8),
            record,
            target_size=(8, 8),
            training=True,
            horizontal_flip=False,
        )
        # Without horizontal flip, box center x is (0+2)/2 / 8 = 0.125 with letterbox left=0
        # In letterbox 8x4 into 8x8 (scale 1.0, left 0, top 2): transformed box is (0, 2, 2, 6)
        # Normalized center x is 1.0 / 8 = 0.125
        self.assertAlmostEqual(float(sample_no_flip["bboxes"][0, 0]), 0.125, places=3)

    def test_collate_enables_context_gradient_only_for_paired_dtld(self) -> None:
        dtld = ImageRecord(
            image_id="DTLD/train/one",
            image_path="unused.png",
            source_dataset="DTLD",
            original_width=8,
            original_height=4,
            split="train",
            sequence_id="one",
            task_valid=TaskValidity(
                traffic_light_detection=True,
                traffic_light_relevance=True,
                arrow_detection=True,
            ),
            traffic_lights=[
                TrafficLightAnnotation(
                    bbox_xyxy=(1, 1, 2, 3),
                    relevance=1,
                    valid_relevance=True,
                )
            ],
            road_arrows=[
                RoadArrowAnnotation(
                    bbox_xyxy=(2, 1, 6, 4), direction_multihot=(1, 1, 0)
                )
            ],
        )
        lisa = ImageRecord(
            image_id="LISA/train/two",
            image_path="unused.png",
            source_dataset="LISA",
            original_width=8,
            original_height=4,
            split="train",
            sequence_id="two",
            task_valid=TaskValidity(traffic_light_detection=True),
        )
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        batch = canonical_multitask_collate(
            [
                prepare_training_sample(image, dtld, target_size=(8, 8)),
                prepare_training_sample(image, lisa, target_size=(8, 8)),
            ]
        )
        self.assertEqual(batch["arrow_detection_valid"].tolist(), [True, False])
        self.assertEqual(batch["arrow_batch_idx"].tolist(), [0])
        self.assertEqual(batch["relevance_arrow_context_scale"].tolist(), [0.25, 0.0])

    def test_effective_window_has_exact_active_source_quota(self) -> None:
        entries = [
            *[
                ManifestEntry(i, "DTLD", "train", f"d{i}")
                for i in range(4)
            ],
            *[
                ManifestEntry(10 + i, "ATLAS", "train", f"a{i}")
                for i in range(2)
            ],
            *[
                ManifestEntry(20 + i, "LISA", "train", f"l{i}")
                for i in range(2)
            ],
        ]
        sampler = BalancedEffectiveBatchSampler(
            entries, micro_batch_size=2, windows_per_epoch=1
        )
        indices = [index for batch in sampler for index in batch]
        groups = [source_group(entries[index].source_dataset) for index in indices]
        self.assertEqual(len(indices), 32)
        self.assertEqual(groups.count("DTLD"), 26)
        self.assertEqual(groups.count("AUX_TL"), 6)
        self.assertEqual(sampler.accumulation_steps, 16)

    def test_ceymo_is_not_an_active_training_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "not active"):
            source_group("CeyMo")


class TrainingRuntimeTests(unittest.TestCase):
    @staticmethod
    def config() -> dict[str, object]:
        return {
            "input_size": [800, 1600],
            "p2_enabled": False,
            "model_variant": "yolo11n",
            "model_config": "configs/model/tlr_yolo11n.yaml",
            "warmstart_weights": "yolo11n.pt",
            "micro_batch_size": 1,
            "effective_batch_size": 32,
            "gradient_accumulation_steps": 32,
            "source_quotas_per_effective_batch": {
                "DTLD": 26,
                "AUX_TL": 6,
            },
            "workers": 0,
            "device": "cuda",
        }

    def test_batch_override_preserves_effective_batch(self) -> None:
        original = self.config()
        resolved = apply_training_overrides(original, micro_batch_size=4)
        self.assertEqual(resolved["micro_batch_size"], 4)
        self.assertEqual(resolved["gradient_accumulation_steps"], 8)
        self.assertEqual(resolved["effective_batch_size"], 32)
        self.assertEqual(original["micro_batch_size"], 1)

    def test_batch_override_must_divide_effective_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            apply_training_overrides(self.config(), micro_batch_size=3)

    def test_ceymo_quota_is_rejected_by_training_contract(self) -> None:
        invalid = self.config()
        invalid["source_quotas_per_effective_batch"] = {
            "DTLD": 20,
            "AUX_TL": 6,
            "CeyMo": 6,
        }
        with self.assertRaisesRegex(ValueError, "active source quota groups"):
            apply_training_overrides(invalid)

    def test_smoke_context_contract_accepts_one_paired_dtld_image(self) -> None:
        summary = validate_smoke_context_batch(
            {
                "relevance_arrow_context_paired": torch.tensor([True, False]),
                "relevance_arrow_context_scale": torch.tensor([0.25, 0.0]),
            }
        )
        self.assertEqual(summary["context_paired_images"], 1)
        self.assertEqual(summary["context_gradient_scales"], [0.25, 0.0])

    def test_smoke_context_contract_rejects_old_unpaired_batch(self) -> None:
        with self.assertRaisesRegex(AssertionError, "exactly one paired"):
            validate_smoke_context_batch(
                {
                    "relevance_arrow_context_paired": torch.tensor([False, False]),
                    "relevance_arrow_context_scale": torch.tensor([0.0, 0.0]),
                }
            )

    def test_ema_state_round_trip(self) -> None:
        source = torch.nn.Linear(3, 2)
        ema = ExponentialMovingAverage(source, decay=0.9)
        ema.update(source)
        restored = ExponentialMovingAverage(torch.nn.Linear(3, 2), decay=0.5)
        restored.load_state_dict(ema.state_dict())
        self.assertEqual(restored.decay, 0.9)
        self.assertEqual(restored.updates, 1)
        for name in ema.shadow:
            self.assertTrue(torch.equal(restored.shadow[name], ema.shadow[name]))

    def test_official_configs_are_single_phase_and_valid(self) -> None:
        for config_rel in ("configs/tlr_yolo_mtl_train.yaml", "configs/tlr_yolov8s_train.yaml"):
            config = load_training_config(config_rel)
            phases = parse_phases(config)
            self.assertEqual(len(phases), 1, f"{config_rel} should have exactly 1 phase")
            self.assertEqual(phases[0].name, "joint_training_single_phase")
            self.assertEqual(phases[0].epochs, 130)
            self.assertTrue(phases[0].context_enabled)
            self.assertFalse(phases[0].freeze_backbone)
            self.assertFalse(phases[0].freeze_perception)
            self.assertEqual(phases[0].relevance_perception_gradient_scale, 0.0)
            self.assertEqual(phases[0].relevance_perception_gradient_scale_end, 1.0)


class TrainingLossTests(unittest.TestCase):
    def test_nwd_is_zero_for_identical_boxes(self) -> None:
        boxes = torch.tensor([[1.0, 2.0, 5.0, 8.0]])
        loss = normalized_wasserstein_loss(boxes, boxes)
        self.assertLess(float(loss), 1e-4)

    def test_gradient_cosine_similarity(self) -> None:
        # Verify gradient cosine computation between two loss objectives on a shared parameter
        param = torch.nn.Parameter(torch.tensor([10.0, 10.0, 20.0, 20.0], requires_grad=True))
        target = torch.tensor([12.0, 12.0, 22.0, 22.0])

        loss1 = (param - target).square().sum()
        loss2 = normalized_wasserstein_loss(param.unsqueeze(0), target.unsqueeze(0))

        grad1 = torch.autograd.grad(loss1, [param], retain_graph=True)[0]
        grad2 = torch.autograd.grad(loss2, [param], retain_graph=False)[0]

        cos = (torch.dot(grad1, grad2) / (grad1.norm(2) * grad2.norm(2))).item()
        self.assertGreater(cos, 0.9)  # Both losses pull in the same direction


if __name__ == "__main__":
    unittest.main()
