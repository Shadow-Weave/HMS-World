import tempfile
import unittest
from pathlib import Path

from continuity.image_records import ImageRecord, ImageRecordStore


def record(image_id: str) -> ImageRecord:
    return ImageRecord(image_id=image_id, kind="shot_sample", asset_path=f"/private/{image_id}.png", character_id=None, shot_idx=1, sample_time_seconds=1.0, view=None, summary="local metadata")


class ImageRecordStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ImageRecordStore(Path(self.tempdir.name) / "records.jsonl")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_append_get_and_all_records_are_stable(self):
        self.store.append(record("sample-1"))
        self.store.append(record("sample-1"))

        self.assertEqual(self.store.get("sample-1").image_id, "sample-1")
        self.assertEqual([item.image_id for item in self.store.all_records()], ["sample-1"])
        with self.assertRaisesRegex(KeyError, "Unknown stable"):
            self.store.get("unknown")

    def test_conflicting_stable_id_is_rejected(self):
        self.store.append(record("sample-1"))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.store.append(record("sample-1").model_copy(update={"summary": "changed"}))


if __name__ == "__main__":
    unittest.main()
