from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tlr_yolo_mtl.data.io import read_records, write_records
from tlr_yolo_mtl.data.qa import build_qa_report
from tlr_yolo_mtl.data.schema import ImageRecord, TaskValidity, TrafficLightAnnotation
from tlr_yolo_mtl.data.splits import (
    assign_grouped_validation,
    audit_split_leakage,
    coalesce_content_split_groups,
    protect_official_test_groups,
)


def record(
    image_id: str,
    image_path: str,
    sequence: str,
    split: str = "train",
    source: str = "DTLD",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        image_path=image_path,
        source_dataset=source,
        original_width=100,
        original_height=50,
        split=split,
        sequence_id=sequence,
        task_valid=TaskValidity(
            traffic_light_detection=True,
            traffic_light_state=True,
            traffic_light_pictogram=True,
            traffic_light_relevance=True,
        ),
        traffic_lights=[
            TrafficLightAnnotation(
                bbox_xyxy=(10, 5, 12, 10),
                state="red",
                pictogram="round",
                relevance=1,
                occlusion="not_occluded",
                valid_state=True,
                valid_pictogram=True,
                valid_relevance=True,
            )
        ],
    )


class SplitTests(unittest.TestCase):
    def test_group_never_crosses_train_validation(self) -> None:
        rows = []
        for group_index in range(5):
            for frame_index in range(2):
                rows.append(
                    record(
                        f"id-{group_index}-{frame_index}",
                        f"image-{group_index}-{frame_index}.jpg",
                        f"sequence-{group_index}",
                    )
                )
        split = assign_grouped_validation(rows, "DTLD", 0.2, seed=42)
        by_sequence: dict[str, set[str]] = {}
        for item in split:
            by_sequence.setdefault(item.sequence_id or "", set()).add(item.split)
        self.assertTrue(any(item.split == "val" for item in split))
        self.assertTrue(all(len(splits) == 1 for splits in by_sequence.values()))
        self.assertTrue(audit_split_leakage(split)["ok"])

    def test_sequence_leakage_is_detected(self) -> None:
        rows = [
            record("a", "a.jpg", "same", "train"),
            record("b", "b.jpg", "same", "test"),
        ]
        report = audit_split_leakage(rows)
        self.assertFalse(report["ok"])
        self.assertIn("sequence_id", {issue["kind"] for issue in report["issues"]})

    def test_content_hash_leakage_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.jpg"
            second = Path(directory) / "b.jpg"
            first.write_bytes(b"identical")
            second.write_bytes(b"identical")
            rows = [
                record("a", str(first), "one", "train"),
                record("b", str(second), "two", "test"),
            ]
            report = audit_split_leakage(rows, hash_images=True)
            self.assertFalse(report["ok"])
            self.assertIn("sha256", {issue["kind"] for issue in report["issues"]})

    def test_official_test_group_quarantines_train_variants(self) -> None:
        rows = [
            record("base", "base.jpg", "scene-419", "train", "CeyMo"),
            record("variant", "variant.jpg", "scene-419", "test", "CeyMo"),
            record("other", "other.jpg", "scene-420", "train", "CeyMo"),
        ]
        corrected = protect_official_test_groups(rows, "CeyMo")
        self.assertEqual([item.split for item in corrected], ["test", "test", "train"])
        self.assertEqual(
            corrected[0].metadata["split_adjustment"],
            "moved_to_protected_test_group",
        )
        self.assertTrue(audit_split_leakage(corrected)["ok"])

    def test_identical_content_coalesces_distinct_source_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jpg"
            second = Path(directory) / "second.jpg"
            third = Path(directory) / "third.jpg"
            first.write_bytes(b"same encoded image")
            second.write_bytes(b"same encoded image")
            third.write_bytes(b"different image")
            rows = [
                record("a", str(first), "scene-602", source="CeyMo"),
                record("b", str(second), "scene-615", source="CeyMo"),
                record("c", str(third), "scene-700", source="CeyMo"),
            ]
            coalesced = coalesce_content_split_groups(rows, "CeyMo")
            self.assertEqual(
                coalesced[0].metadata["split_group"],
                coalesced[1].metadata["split_group"],
            )
            split = assign_grouped_validation(
                coalesced, "CeyMo", 0.5, seed=42, stratify_arrows=True
            )
            self.assertEqual(split[0].split, split[1].split)
            self.assertTrue(audit_split_leakage(split, hash_images=True)["ok"])

    def test_identical_content_connected_to_test_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "train.jpg"
            second = Path(directory) / "test.jpg"
            first.write_bytes(b"same encoded image")
            second.write_bytes(b"same encoded image")
            rows = [
                record("train", str(first), "scene-train", "train", "CeyMo"),
                record("test", str(second), "scene-test", "test", "CeyMo"),
            ]
            corrected = coalesce_content_split_groups(rows, "CeyMo")
            self.assertEqual([item.split for item in corrected], ["test", "test"])
            self.assertEqual(
                corrected[0].metadata["split_adjustment"],
                "moved_to_protected_test_content_group",
            )
            self.assertTrue(audit_split_leakage(corrected, hash_images=True)["ok"])


class IoAndQaTests(unittest.TestCase):
    def test_qa_uses_network_resolution_size_bins(self) -> None:
        report = build_qa_report([record("a", "a.jpg", "one")])
        bins = report["traffic_lights"]["network_width_bins"]
        self.assertEqual(bins["normal_w_gt_16"], 1)
        self.assertTrue(report["split_leakage"]["ok"])

    def test_jsonl_io_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            source = [record("a", "a.jpg", "one")]
            self.assertEqual(write_records(path, source), 1)
            restored = list(read_records(path))
            self.assertEqual(restored[0].to_dict(), source[0].to_dict())


if __name__ == "__main__":
    unittest.main()
