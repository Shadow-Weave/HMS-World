import tempfile
import unittest
from pathlib import Path

from continuity.history_materials import historical_sample_groups, state_field_lineage
from continuity.image_records import ImageRecord, ImageRecordStore
from continuity.shot_sampling import SampledFrame, write_observation_samples
from continuity.timeline import (
    CharacterContent,
    CharacterEntity,
    EntityRegistry,
    ObservedShotSource,
    SceneEntity,
    Timeline,
    WorldState,
)
from interfaces import ShotDescription


def make_shot(idx: int) -> ShotDescription:
    return ShotDescription(
        idx=idx,
        is_last=idx == 1,
        cam_idx=0,
        visual_desc="Mira waits.",
        variation_type="small",
        variation_reason="test",
        ff_desc="Mira waits.",
        ff_vis_char_idxs=[0],
        lf_desc="",
        lf_vis_char_idxs=[],
        motion_desc="waits",
        audio_desc="",
    )


class HistoryMaterialsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.registry = EntityRegistry(
            characters=[CharacterEntity(
                entity_id="character-0",
                name="Mira",
                identity="",
                persona="",
                session="",
                stable_appearance="coat",
            )],
            scenes=[SceneEntity(entity_id="scene-0", name="room")],
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _observed_timeline(self) -> Timeline:
        timeline = Timeline.initialize(self.registry, [make_shot(0), make_shot(1)])
        timeline.mark_observed(
            0,
            WorldState(characters={"character-0": CharacterContent(position="desk")}),
        )
        return timeline

    def _lineage(self) -> list:
        return state_field_lineage(
            "character-0",
            CharacterContent(position="desk"),
            {"position": ObservedShotSource(shot_idx=0)},
        )

    def _formal_samples(self, store: ImageRecordStore) -> tuple[Path, list[SampledFrame]]:
        samples: list[SampledFrame] = []
        for index in range(2):
            asset_path = self.root / f"sample-{index}.png"
            asset_path.write_bytes(b"sample")
            sample = SampledFrame(
                shot_idx=0,
                image_id=f"sample-{index}",
                asset_path=str(asset_path),
                time_seconds=float(index),
            )
            samples.append(sample)
            store.append(ImageRecord(
                image_id=sample.image_id,
                kind="shot_sample",
                asset_path=sample.asset_path,
                character_id=None,
                shot_idx=0,
                sample_time_seconds=sample.time_seconds,
                view=None,
                summary="actual sampled frame",
            ))
        manifest = self.root / "observation_samples.json"
        write_observation_samples(manifest, samples)
        return manifest, samples

    def test_source_shot_exposes_all_actual_samples_without_binding_a_field_to_one(self):
        timeline = self._observed_timeline()
        store = ImageRecordStore(self.root / "records.jsonl")
        manifest, _ = self._formal_samples(store)

        groups = historical_sample_groups(self._lineage(), timeline, {0: manifest}, store)

        self.assertEqual(groups[0].source.shot_idx, 0)
        self.assertEqual(
            [item.image_id for item in groups[0].samples],
            ["sample-0", "sample-1"],
        )
        self.assertEqual(groups[0].lineage[0].source, ObservedShotSource(shot_idx=0))

    def test_field_sources_must_exactly_cover_non_null_content(self):
        content = CharacterContent(position="desk", outfit="coat")
        source = ObservedShotSource(shot_idx=0)

        with self.assertRaisesRegex(ValueError, "missing=.*outfit"):
            state_field_lineage("character-0", content, {"position": source})
        with self.assertRaisesRegex(ValueError, "extra=.*pose"):
            state_field_lineage(
                "character-0",
                CharacterContent(position="desk"),
                {"position": source, "pose": source},
            )

    def test_observed_source_requires_an_observed_timeline_position(self):
        timeline = Timeline.initialize(self.registry, [make_shot(0), make_shot(1)])
        store = ImageRecordStore(self.root / "records.jsonl")
        manifest, _ = self._formal_samples(store)

        with self.assertRaisesRegex(ValueError, "is not observed"):
            historical_sample_groups(self._lineage(), timeline, {0: manifest}, store)

    def test_observed_source_requires_a_valid_manifest(self):
        timeline = self._observed_timeline()
        store = ImageRecordStore(self.root / "records.jsonl")
        manifest, _ = self._formal_samples(store)

        with self.assertRaisesRegex(ValueError, "no observation sample manifest"):
            historical_sample_groups(self._lineage(), timeline, {}, store)
        manifest.write_text("{malformed")
        with self.assertRaisesRegex(ValueError, "Cannot read valid observation samples"):
            historical_sample_groups(self._lineage(), timeline, {0: manifest}, store)

    def test_sample_records_must_match_the_source_shot_manifest_and_asset(self):
        for label, replacement in (
            ("kind", {"kind": "camera"}),
            ("shot", {"shot_idx": 1}),
            ("time", {"sample_time_seconds": 9.0}),
            ("path", {"asset_path": str(self.root / "missing.png")}),
        ):
            with self.subTest(label=label):
                timeline = self._observed_timeline()
                baseline = ImageRecordStore(self.root / f"{label}.jsonl")
                manifest, samples = self._formal_samples(baseline)
                store = ImageRecordStore(self.root / f"{label}-replacement.jsonl")
                first = samples[0]
                store.append(ImageRecord(
                    image_id=first.image_id,
                    kind=replacement.get("kind", "shot_sample"),
                    asset_path=replacement.get("asset_path", first.asset_path),
                    character_id=None,
                    shot_idx=replacement.get("shot_idx", 0),
                    sample_time_seconds=replacement.get("sample_time_seconds", first.time_seconds),
                    view=None,
                    summary="actual sampled frame",
                ))
                second = samples[1]
                store.append(ImageRecord(
                    image_id=second.image_id,
                    kind="shot_sample",
                    asset_path=second.asset_path,
                    character_id=None,
                    shot_idx=0,
                    sample_time_seconds=second.time_seconds,
                    view=None,
                    summary="actual sampled frame",
                ))

                with self.assertRaisesRegex(ValueError, "does not match"):
                    historical_sample_groups(self._lineage(), timeline, {0: manifest}, store)


if __name__ == "__main__":
    unittest.main()
