from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dtld_images import (
    prepare_dtld_images,
    prepare_output_tree,
    validate_source_tree,
)


class PrepareDtldImagesTests(unittest.TestCase):
    def test_output_tree_refuses_to_delete_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                prepare_output_tree(target, ("train", "test"))

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_output_tree_creates_requested_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output"

            prepared = prepare_output_tree(target, ("train", "test"))

            self.assertEqual(prepared, target.resolve())
            self.assertTrue((prepared / "train").is_dir())
            self.assertTrue((prepared / "test").is_dir())

    def test_source_validation_requires_each_label_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "DTLD"
            labels = root / "labels"
            data.mkdir()
            labels.mkdir()
            (labels / "DTLD_train.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "DTLD_test.json"):
                validate_source_tree(data, labels, ("train", "test"))

    def test_source_root_is_rejected_as_output_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "DTLD"
            labels = root / "labels"
            data.mkdir()
            labels.mkdir()
            for split in ("train", "test"):
                (labels / f"DTLD_{split}.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must differ"):
                prepare_dtld_images(data, data, labels, workers=1)

            self.assertEqual(list(data.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
