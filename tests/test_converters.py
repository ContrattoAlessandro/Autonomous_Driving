from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tlr_yolo_mtl.data.converters.atlas import convert_atlas
from tlr_yolo_mtl.data.converters.dtld import convert_dtld_file
from tlr_yolo_mtl.data.converters.lisa import convert_lisa


class ConverterTests(unittest.TestCase):
    def test_dtld_rejects_annotated_preview_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview = root / "DTLD_jpg" / "train"
            preview.mkdir(parents=True)
            labels = root / "labels.json"
            labels.write_text('{"images": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "DTLD_jpg_plain"):
                convert_dtld_file(labels, preview, "train")

    def test_dtld_preserves_boxes_and_masks_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "frame.jpg").write_bytes(b"fixture")

            def label(state: str, pictogram: str, **attributes: str) -> dict:
                return {
                    "x": 10,
                    "y": 20,
                    "w": 5,
                    "h": 10,
                    "attributes": {
                        "state": state,
                        "pictogram": pictogram,
                        "direction": attributes.get("direction", "front"),
                        "occlusion": attributes.get("occlusion", "not_occluded"),
                        "relevance": attributes.get("relevance", "relevant"),
                    },
                }

            payload = {
                "images": [
                    {
                        "image_path": "./City/Route/Sequence/frame.tiff",
                        "labels": [
                            label("green", "circle"),
                            label("red_yellow", "arrow_straight_left"),
                            label("unknown", "unknown"),
                            label("red", "pedestrian"),
                            label("red", "circle", direction="back"),
                            label("red", "circle", occlusion="fully_occluded"),
                        ],
                    }
                ]
            }
            labels = root / "labels.json"
            labels.write_text(json.dumps(payload), encoding="utf-8")
            result = convert_dtld_file(labels, images, "train")
            self.assertEqual(len(result.records), 1)
            record = result.records[0]
            self.assertEqual(len(record.traffic_lights), 3)
            self.assertEqual(len(record.ignore_regions), 3)
            self.assertTrue(record.traffic_lights[0].valid_state)
            self.assertFalse(record.traffic_lights[1].valid_state)
            self.assertFalse(record.traffic_lights[1].valid_pictogram)
            self.assertFalse(record.traffic_lights[2].valid_state)
            self.assertTrue(record.task_valid.traffic_light_relevance)
            record.validate()

    def test_atlas_factorizes_composite_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ATLAS_classes.yaml").write_text(
                "names:\n  0: circle_green\n  1: circle_red_yellow\n"
                "  2: arrow_straight_left_green\n",
                encoding="utf-8",
            )
            images = root / "train" / "front_medium" / "images"
            labels = root / "train" / "front_medium" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "front_medium_1-1.jpg").write_bytes(b"fixture")
            (labels / "front_medium_1-1.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.1 0.1\n2 0.6 0.6 0.1 0.1\n",
                encoding="utf-8",
            )
            result = convert_atlas(root)
            record = result.records[0]
            self.assertEqual(len(record.traffic_lights), 3)
            self.assertTrue(record.traffic_lights[0].valid_state)
            self.assertFalse(record.traffic_lights[1].valid_state)
            self.assertTrue(record.traffic_lights[1].valid_pictogram)
            self.assertTrue(record.traffic_lights[2].valid_state)
            self.assertFalse(record.traffic_lights[2].valid_pictogram)

    def test_atlas_uses_actual_image_dimensions_for_normalized_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ATLAS_classes.yaml").write_text(
                "names:\n  0: circle_green\n", encoding="utf-8"
            )
            images = root / "train" / "front_medium" / "images"
            labels = root / "train" / "front_medium" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (100, 50)).save(images / "front_medium_1-1.jpg")
            (labels / "front_medium_1-1.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
            result = convert_atlas(root)
            record = result.records[0]
            self.assertEqual((record.original_width, record.original_height), (100, 50))
            self.assertEqual(record.traffic_lights[0].bbox_xyxy, (40.0, 15.0, 60.0, 35.0))
            self.assertEqual(result.stats["non_nominal_image_dimensions"], 1)

    def test_lisa_keeps_negative_frames_and_factorizes_tags(self) -> None:
        header = (
            "Filename;Annotation tag;Upper left corner X;Upper left corner Y;"
            "Lower right corner X;Lower right corner Y;Origin file;"
            "Origin frame number;Origin track;Origin track frame number\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_annotations = (
                root
                / "Annotations"
                / "Annotations"
                / "dayTrain"
                / "dayClip1"
                / "frameAnnotationsBOX.csv"
            )
            test_annotations = (
                root
                / "Annotations"
                / "Annotations"
                / "nightSequence1"
                / "frameAnnotationsBOX.csv"
            )
            train_frames = root / "dayTrain" / "dayTrain" / "dayClip1" / "frames"
            test_frames = root / "nightSequence1" / "nightSequence1" / "frames"
            train_annotations.parent.mkdir(parents=True)
            test_annotations.parent.mkdir(parents=True)
            train_frames.mkdir(parents=True)
            test_frames.mkdir(parents=True)
            (train_frames / "dayClip1--00000.jpg").write_bytes(b"fixture")
            (train_frames / "dayClip1--00001.jpg").write_bytes(b"fixture")
            (test_frames / "nightSequence1--00000.jpg").write_bytes(b"fixture")
            train_annotations.write_text(
                header
                + "dayTraining/dayClip1--00000.jpg;go;10;20;30;50;x;0;x;0\n"
                + "dayTraining/dayClip1--00000.jpg;goLeft;10;20;30;50;x;0;x;0\n",
                encoding="utf-8",
            )
            test_annotations.write_text(
                header
                + "nightTest/nightSequence1--00000.jpg;warning;40;30;60;70;x;0;x;0\n",
                encoding="utf-8",
            )

            result = convert_lisa(root)
            self.assertEqual(len(result.records), 3)
            train = [record for record in result.records if record.split == "train"]
            test = [record for record in result.records if record.split == "test"]
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 1)
            self.assertEqual(train[0].traffic_lights[0].state, "green")
            self.assertIsNone(train[0].traffic_lights[0].pictogram)
            self.assertFalse(train[0].traffic_lights[0].valid_pictogram)
            self.assertEqual(result.stats["merged_same_box_rows"], 1)
            self.assertEqual(result.stats["conflicting_pictogram_boxes"], 1)
            self.assertEqual(train[1].traffic_lights, [])
            self.assertTrue(train[1].task_valid.traffic_light_detection)
            self.assertEqual(test[0].traffic_lights[0].state, "yellow")
            self.assertEqual(result.stats["negative_images"], 1)


if __name__ == "__main__":
    unittest.main()
