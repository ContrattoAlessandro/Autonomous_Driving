from __future__ import annotations

import unittest

import yaml

from tlr_yolo_mtl.model.milestone2 import (
    DEFAULT_CONFIG,
    EXPECTED_STRIDES,
    INPUT_SIZE,
    TARGET_TO_SOURCE_LAYER,
    _source_key_for_target,
)


class Milestone2SpecTests(unittest.TestCase):
    def test_active_prototype_uses_nano_without_p2(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.name, "tlr_yolo11n.yaml")
        config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["head"][-1][0], [16, 19, 22])
        self.assertEqual(config["head"][-1][2], "Detect")

    def test_deployment_geometry(self) -> None:
        height, width = INPUT_SIZE
        self.assertEqual(EXPECTED_STRIDES, (8, 16, 32))
        self.assertEqual(
            [(height // stride, width // stride) for stride in EXPECTED_STRIDES],
            [(100, 200), (50, 100), (25, 50)],
        )

    def test_standard_layers_keep_official_indices(self) -> None:
        self.assertEqual(TARGET_TO_SOURCE_LAYER[17], 17)
        self.assertEqual(TARGET_TO_SOURCE_LAYER[19], 19)
        self.assertEqual(TARGET_TO_SOURCE_LAYER[22], 22)
        self.assertEqual(TARGET_TO_SOURCE_LAYER[23], 23)

    def test_detect_p3_p5_towers_keep_official_indices(self) -> None:
        self.assertEqual(
            _source_key_for_target("model.23.cv2.0.0.conv.weight"),
            "model.23.cv2.0.0.conv.weight",
        )
        self.assertEqual(
            _source_key_for_target("model.23.cv3.2.2.bias"),
            "model.23.cv3.2.2.bias",
        )
        self.assertEqual(
            _source_key_for_target("model.23.dfl.conv.weight"),
            "model.23.dfl.conv.weight",
        )


if __name__ == "__main__":
    unittest.main()
