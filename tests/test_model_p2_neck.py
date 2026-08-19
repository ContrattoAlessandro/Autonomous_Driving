"""Unit and regression tests for P2 (stride-4) high-resolution neck integration."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from tlr_yolo_mtl.model.milestone2 import (
    ALLOWED_STRIDES,
    build_detection_model,
    load_coco_warmstart,
)
from tlr_yolo_mtl.model.unified import (
    UnifiedHeadConfig,
    UnifiedTrafficControlDetect,
    attach_unified_relevance_head,
)
from tlr_yolo_mtl.training.engine import (
    assert_active_pyramid,
    load_training_config,
)


class P2NeckIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.p2_model_config = self.project_root / "configs" / "model" / "tlr_yolo11n_p2.yaml"
        self.p2_train_config = self.project_root / "configs" / "b2_p2_neck.yaml"
        self.weights_path = self.project_root / "yolo11n.pt"

    def test_p2_model_configuration_and_strides(self) -> None:
        self.assertTrue(self.p2_model_config.is_file())
        wrapper = build_detection_model(self.p2_model_config)
        detect = wrapper.model.model[-1]
        strides = tuple(int(value) for value in detect.stride.tolist())
        self.assertEqual(strides, (4, 8, 16, 32))
        self.assertIn(strides, ALLOWED_STRIDES)
        self.assertEqual(int(detect.nc), 2)
        self.assertEqual(int(detect.reg_max), 16)

    def test_p2_coco_warmstart_loading(self) -> None:
        if not self.weights_path.is_file():
            self.skipTest("yolo11n.pt weights not found locally")
        wrapper = build_detection_model(self.p2_model_config)
        report = load_coco_warmstart(wrapper, self.weights_path)
        self.assertTrue(report["p2_enabled"])
        self.assertEqual(report["pyramid_levels"], ["P2", "P3", "P4", "P5"])
        self.assertGreater(report["loaded_state_items"], 100)
        self.assertGreater(report["loaded_state_items_by_region"]["backbone"], 0)

    def test_p2_unified_head_attachment(self) -> None:
        wrapper = build_detection_model(self.p2_model_config)
        config = UnifiedHeadConfig(max_traffic_lights=32, max_arrows=16)
        attach_unified_relevance_head(wrapper, config=config)
        model = wrapper.model
        head = model.model[-1]

        self.assertIsInstance(head, UnifiedTrafficControlDetect)
        self.assertEqual(tuple(int(s) for s in head.stride.tolist()), (4, 8, 16, 32))
        # 4 pyramid levels -> 4 towers in each module list
        self.assertEqual(len(head.state_heads), 4)
        self.assertEqual(len(head.round_heads), 4)
        self.assertEqual(len(head.maneuver_heads), 4)
        self.assertEqual(len(head.ego_lane_heads), 4)
        self.assertEqual(len(head.local_relevance_heads), 4)
        self.assertEqual(len(head.token_feature_heads), 4)
        self.assertEqual(len(head.attribute_channels), 4)

        # Verify pyramid assertion passes with p2_enabled=True
        assert_active_pyramid(model, p2_enabled=True)

    def test_p2_forward_inference_shapes(self) -> None:
        wrapper = build_detection_model(self.p2_model_config)
        config = UnifiedHeadConfig(max_traffic_lights=32, max_arrows=16)
        attach_unified_relevance_head(wrapper, config=config)
        wrapper.model.eval()

        h, w = 384, 640
        sample = torch.zeros((1, 3, h, w))
        expected_dense = sum((h // s) * (w // s) for s in (4, 8, 16, 32))
        self.assertEqual(expected_dense, (384 // 4) * (640 // 4) + (384 // 8) * (640 // 8) + (384 // 16) * (640 // 16) + (384 // 32) * (640 // 32))

        with torch.no_grad():
            decoded, raw = wrapper.model(sample)

        self.assertEqual(decoded.shape, (1, 6, expected_dense))
        self.assertEqual(raw["state_logits"].shape, (1, 4, expected_dense))
        self.assertEqual(raw["round_logits"].shape, (1, 1, expected_dense))
        self.assertEqual(raw["maneuver_logits"].shape, (1, 3, expected_dense))
        self.assertEqual(raw["traffic_candidate_boxes"].shape, (1, 32, 4))
        self.assertEqual(raw["arrow_candidate_boxes"].shape, (1, 16, 4))
        self.assertEqual(raw["relevance_logits"].shape, (1, 1, 32))
        self.assertEqual(raw["attention_weights"].shape, (1, 4, 32, 17))

    def test_p2_forward_training_backward_flow(self) -> None:
        wrapper = build_detection_model(self.p2_model_config)
        config = UnifiedHeadConfig(max_traffic_lights=32, max_arrows=16)
        attach_unified_relevance_head(wrapper, config=config)
        wrapper.model.train()

        h, w = 128, 256
        sample = torch.randn((1, 3, h, w), requires_grad=True)
        raw = wrapper.model(sample)
        self.assertIsInstance(raw, dict)

        loss = (
            raw["relevance_logits"].sum()
            + raw["state_logits"].sum()
            + raw["token_features"].sum()
        )
        loss.backward()

        self.assertIsNotNone(sample.grad)
        head = wrapper.model.model[-1]
        for tower in head.state_heads:
            for p in tower.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)

    def test_p2_training_config_load(self) -> None:
        self.assertTrue(self.p2_train_config.is_file())
        cfg = load_training_config(self.p2_train_config)
        self.assertTrue(cfg["p2_enabled"])
        self.assertEqual(cfg["model_variant"], "yolo11n")
        self.assertEqual(cfg["architecture"]["max_traffic_lights"], 32)
        self.assertEqual(cfg["architecture"]["max_arrows"], 16)


if __name__ == "__main__":
    unittest.main()
