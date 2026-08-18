from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tlr_yolo_mtl.data.converters.dtld_arrows import fuse_dtld_arrow_annotations
from tlr_yolo_mtl.data.schema import (
    ImageRecord,
    TaskValidity,
    TrafficLightAnnotation,
)


def _dtld_record(stem: str, *, official_split: str = "train") -> ImageRecord:
    return ImageRecord(
        image_id=f"DTLD/{official_split}/{stem}",
        image_path=f"canonical/{official_split}/{stem}.jpg",
        source_dataset="DTLD",
        original_width=100,
        original_height=50,
        split=official_split,
        sequence_id=f"DTLD/sequence-{stem}",
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_state=True,
            traffic_light_pictogram=True,
            traffic_light_relevance=True,
        ),
        traffic_lights=[
            TrafficLightAnnotation(
                bbox_xyxy=(40, 5, 45, 15),
                state="green",
                pictogram="round",
                relevance=1,
                valid_state=True,
                valid_pictogram=True,
                valid_relevance=True,
            )
        ],
        metadata={"official_split": official_split},
    )


def _annotation_tree(root: Path, labels: dict[str, str]) -> Path:
    images = root / "images"
    label_dir = root / "labels"
    images.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for stem, text in labels.items():
        Image.new("RGB", (100, 50)).save(images / f"dtld_{stem}.jpg")
        (label_dir / f"dtld_{stem}.txt").write_text(text, encoding="utf-8")
    return root


class DtldArrowFusionTests(unittest.TestCase):
    def test_fuses_compound_arrow_and_preserves_all_dtld_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _annotation_tree(
                Path(directory),
                {
                    # Extends five pixels beyond the left edge and must be
                    # clipped deterministically rather than rejected silently.
                    "positive": "3 0.05 0.5 0.2 0.4\n",
                    "negative": "",
                },
            )
            positive = _dtld_record("positive")
            negative = _dtld_record("negative")

            result = fuse_dtld_arrow_annotations([positive, negative], root)

            fused_positive, fused_negative = result.records
            self.assertEqual(fused_positive.traffic_lights, positive.traffic_lights)
            self.assertTrue(fused_positive.task_valid.arrow_detection)
            self.assertEqual(len(fused_positive.road_arrows), 1)
            arrow = fused_positive.road_arrows[0]
            self.assertEqual(arrow.direction_multihot, (1, 1, 0))
            self.assertEqual(arrow.bbox_xyxy, (0.0, 15.0, 15.0, 35.0))
            self.assertTrue(arrow.source_attributes["box_clipped_to_image"])
            self.assertTrue(fused_positive.metadata["paired_relevance_arrow"])
            self.assertEqual(
                fused_positive.metadata["arrow_annotation_status"],
                "human_verified_exhaustive",
            )
            self.assertTrue(fused_negative.task_valid.arrow_detection)
            self.assertEqual(fused_negative.road_arrows, [])
            self.assertEqual(result.stats["annotated_images"], 2)
            self.assertEqual(result.stats["positive_images"], 1)
            self.assertEqual(result.stats["negative_images"], 1)
            self.assertEqual(result.stats["clipped_boxes"], 1)
            fused_positive.validate()
            fused_negative.validate()

    def test_official_dtld_test_remains_arrow_masked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _annotation_tree(Path(directory), {"train-frame": "0 0.5 0.5 0.2 0.2\n"})
            train = _dtld_record("train-frame")
            test = _dtld_record("test-frame", official_split="test")

            result = fuse_dtld_arrow_annotations([train, test], root)

            self.assertTrue(result.records[0].task_valid.arrow_detection)
            self.assertFalse(result.records[1].task_valid.arrow_detection)
            self.assertEqual(result.records[1].road_arrows, [])

    def test_missing_or_extra_annotation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _annotation_tree(Path(directory), {"unexpected": ""})
            with self.assertRaisesRegex(ValueError, "annotation coverage mismatch"):
                fuse_dtld_arrow_annotations([_dtld_record("expected")], root)

    def test_missing_image_copy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "labels" / "dtld_frame.txt").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image/label coverage mismatch"):
                fuse_dtld_arrow_annotations([_dtld_record("frame")], root)

    def test_invalid_rows_fail_with_file_and_line_number(self) -> None:
        invalid_rows = (
            "5 0.5 0.5 0.2 0.2\n",
            "0 0.5 0.5 0.2\n",
            "0 nan 0.5 0.2 0.2\n",
            "0 0.5 0.5 -0.2 0.2\n",
        )
        for row in invalid_rows:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                root = _annotation_tree(Path(directory), {"frame": row})
                with self.assertRaisesRegex(ValueError, r"dtld_frame\.txt:1"):
                    fuse_dtld_arrow_annotations([_dtld_record("frame")], root)

    def test_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _annotation_tree(Path(directory), {"frame": ""})
            Image.new("RGB", (101, 50)).save(root / "images" / "dtld_frame.jpg")
            with self.assertRaisesRegex(ValueError, "dimension mismatch"):
                fuse_dtld_arrow_annotations([_dtld_record("frame")], root)


if __name__ == "__main__":
    unittest.main()
