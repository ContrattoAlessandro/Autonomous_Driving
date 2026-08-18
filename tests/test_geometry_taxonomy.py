from __future__ import annotations

import unittest

from tlr_yolo_mtl.data.geometry import (
    horizontal_flip_box,
    letterbox_box,
    letterbox_parameters,
    xywh_to_xyxy,
    xyxy_to_xywh,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from tlr_yolo_mtl.data.taxonomy import (
    flip_direction_multihot,
    flip_pictogram,
    map_arrow_direction,
    map_pictogram,
    map_state,
    split_atlas_class,
)


class GeometryTests(unittest.TestCase):
    def test_xywh_xyxy_round_trip(self) -> None:
        box = xywh_to_xyxy(10, 20, 30, 40)
        self.assertEqual(box, (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(xyxy_to_xywh(box), (10.0, 20.0, 30.0, 40.0))

    def test_yolo_round_trip(self) -> None:
        box = yolo_to_xyxy(0.5, 0.25, 0.2, 0.1, 1000, 500)
        result = xyxy_to_yolo(box, 1000, 500)
        for actual, expected in zip(result, (0.5, 0.25, 0.2, 0.1)):
            self.assertAlmostEqual(actual, expected)

    def test_letterbox_rectangular_input(self) -> None:
        scale, pad_x, pad_y = letterbox_parameters((1920, 1200), (1600, 800))
        self.assertAlmostEqual(scale, 2 / 3)
        self.assertAlmostEqual(pad_x, 160)
        self.assertAlmostEqual(pad_y, 0)
        self.assertEqual(letterbox_box((0, 0, 1920, 1200), (1920, 1200), (1600, 800)),
                         (160.0, 0.0, 1440.0, 800.0))

    def test_horizontal_flip_box(self) -> None:
        self.assertEqual(horizontal_flip_box((10, 20, 30, 40), 100),
                         (70.0, 20.0, 90.0, 40.0))


class TaxonomyTests(unittest.TestCase):
    def test_state_mapping_masks_red_yellow_and_unknown(self) -> None:
        self.assertEqual(map_state("amber").target, "yellow")
        self.assertTrue(map_state("off").valid)
        self.assertFalse(map_state("red-yellow").valid)
        self.assertFalse(map_state("flashing").valid)
        self.assertFalse(map_state("unknown").valid)

    def test_pictogram_mapping_masks_composites_and_ignores_non_vehicle(self) -> None:
        self.assertEqual(map_pictogram("circle").target, "round")
        self.assertEqual(map_pictogram("arrow_left").target, "left")
        self.assertFalse(map_pictogram("arrow_straight_left").valid)
        pedestrian = map_pictogram("pedestrian")
        self.assertFalse(pedestrian.valid)
        self.assertTrue(pedestrian.ignore_object)

    def test_arrow_multi_label_mapping(self) -> None:
        self.assertEqual(map_arrow_direction("left"), (1, 0, 0))
        self.assertEqual(map_arrow_direction("straight-right"), (0, 1, 1))
        self.assertEqual(map_arrow_direction("left_right"), (1, 0, 1))

    def test_flip_swaps_all_left_right_semantics(self) -> None:
        self.assertEqual(flip_pictogram("left"), "right")
        self.assertEqual(flip_pictogram("right"), "left")
        self.assertEqual(flip_pictogram("round"), "round")
        self.assertEqual(flip_direction_multihot((1, 1, 0)), (0, 1, 1))

    def test_atlas_factorization_masks_independent_attributes(self) -> None:
        state, pictogram = split_atlas_class("circle_red_yellow")
        self.assertFalse(state.valid)
        self.assertTrue(pictogram.valid)
        self.assertEqual(pictogram.target, "round")
        state, pictogram = split_atlas_class("arrow_straight_left_green")
        self.assertTrue(state.valid)
        self.assertFalse(pictogram.valid)


if __name__ == "__main__":
    unittest.main()

