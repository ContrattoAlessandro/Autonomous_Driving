from __future__ import annotations

import unittest

from tlr_yolo_mtl.data.schema import (
    IgnoreRegion,
    ImageRecord,
    RoadArrowAnnotation,
    SchemaError,
    TaskValidity,
    TrafficLightAnnotation,
)
from tlr_yolo_mtl.data.transforms import horizontal_flip_record


def make_record() -> ImageRecord:
    return ImageRecord(
        image_id="fixture/1",
        image_path="fixture.jpg",
        source_dataset="fixture",
        original_width=100,
        original_height=50,
        split="train",
        sequence_id="sequence/1",
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_state=True,
            traffic_light_pictogram=True,
            traffic_light_relevance=True,
            arrow_detection=True,
        ),
        traffic_lights=[
            TrafficLightAnnotation(
                bbox_xyxy=(10, 5, 30, 20),
                state="green",
                pictogram="left",
                relevance=1,
                occlusion="not_occluded",
                valid_state=True,
                valid_pictogram=True,
                valid_relevance=True,
            ),
            TrafficLightAnnotation(
                bbox_xyxy=(40, 5, 50, 20),
                state="red",
                pictogram="round",
                relevance=1,
                occlusion="partially_occluded",
                valid_state=True,
                valid_pictogram=True,
                valid_relevance=True,
            ),
        ],
        road_arrows=[
            RoadArrowAnnotation(
                bbox_xyxy=(5, 25, 25, 45),
                direction_multihot=(1, 1, 0),
                segmentation_xy=((5, 25), (25, 25), (15, 45)),
            )
        ],
        ignore_regions=[IgnoreRegion((80, 5, 90, 15), "non_vehicle_pictogram")],
    )


class SchemaTests(unittest.TestCase):
    def test_multiple_relevant_lights_are_valid(self) -> None:
        record = make_record()
        record.validate()
        self.assertEqual(sum(item.relevance == 1 for item in record.traffic_lights), 2)

    def test_round_trip(self) -> None:
        record = make_record()
        restored = ImageRecord.from_dict(record.to_dict())
        self.assertEqual(restored.to_dict(), record.to_dict())
        restored.validate()

    def test_image_task_mask_cannot_hide_instances(self) -> None:
        record = make_record()
        record.task_valid.arrow_detection = False
        with self.assertRaises(SchemaError):
            record.validate()

    def test_missing_instance_attribute_is_legal_when_masked(self) -> None:
        record = make_record()
        record.traffic_lights[0].state = None
        record.traffic_lights[0].valid_state = False
        record.validate()

    def test_fully_occluded_positive_is_rejected(self) -> None:
        record = make_record()
        record.traffic_lights[0].occlusion = "fully_occluded"
        with self.assertRaises(SchemaError):
            record.validate()

    def test_horizontal_flip_is_semantically_atomic(self) -> None:
        flipped = horizontal_flip_record(make_record())
        self.assertEqual(flipped.traffic_lights[0].bbox_xyxy, (70.0, 5.0, 90.0, 20.0))
        self.assertEqual(flipped.traffic_lights[0].pictogram, "right")
        self.assertEqual(flipped.road_arrows[0].direction_multihot, (0, 1, 1))
        self.assertEqual(flipped.road_arrows[0].bbox_xyxy, (75.0, 25.0, 95.0, 45.0))
        self.assertEqual(flipped.ignore_regions[0].bbox_xyxy, (10.0, 5.0, 20.0, 15.0))


if __name__ == "__main__":
    unittest.main()

