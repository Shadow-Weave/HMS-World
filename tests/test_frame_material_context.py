import tempfile
import unittest
from pathlib import Path

from continuity.frame_material_context import (
    FrameMaterialCapacityError,
    build_frame_material_context,
)
from continuity.image_records import ImageRecord, ImageRecordStore
from continuity.shot_sampling import (
    SampledFrame,
    image_records_for_observed_samples,
    write_observation_samples,
)
from continuity.timeline import (
    CharacterContent,
    CharacterEntity,
    EntityRegistry,
    PropContent,
    PropEntity,
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
        lf_desc="Mira waits.",
        lf_vis_char_idxs=[0],
        motion_desc="waits",
        audio_desc="",
    )


class FrameMaterialContextTests(unittest.TestCase):
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
                identity_image_ids=["front", "side", "back"],
            )],
            scenes=[SceneEntity(entity_id="scene-0", name="room")],
        )
        self.timeline = Timeline.initialize(self.registry, [make_shot(0), make_shot(1)])
        self.timeline.mark_observed(
            0,
            WorldState(characters={"character-0": CharacterContent(position="desk")}),
        )
        self.position = self.timeline.refresh_for_execution(1, self.registry)
        self.store = ImageRecordStore(self.root / "records.jsonl")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _history_manifest(self) -> Path:
        samples: list[SampledFrame] = []
        for index in range(2):
            asset_path = self.root / f"sample-{index}.png"
            asset_path.write_bytes(b"sample")
            samples.append(SampledFrame(
                shot_idx=0,
                image_id=f"sample-{index}",
                asset_path=str(asset_path),
                time_seconds=float(index),
            ))
        for record in image_records_for_observed_samples(samples, 0):
            self.store.append(record)
        manifest = self.root / "observation_samples.json"
        write_observation_samples(manifest, samples)
        return manifest

    def _portraits(self) -> None:
        for view in ("front", "side", "back"):
            asset_path = self.root / f"{view}.png"
            asset_path.write_bytes(b"portrait")
            self.store.append(ImageRecord(
                image_id=view,
                kind="portrait",
                asset_path=str(asset_path),
                character_id="character-0",
                shot_idx=None,
                sample_time_seconds=None,
                view=view,
                summary="portrait",
            ))

    def test_unrelated_character_history_is_not_sent_as_visual_reference(self):
        manifest = self._history_manifest()
        self._portraits()

        context = build_frame_material_context(
            self.position,
            "first",
            self.timeline,
            self.registry,
            {0: manifest},
            self.store,
            2,
        )

        self.assertEqual(context.required_slots, 1)
        self.assertEqual(context.history_sample_groups, [])
        self.assertEqual(context.top_k - context.required_slots, 1)

    def test_current_prop_history_is_kept_for_visual_reference(self):
        self.registry.props = [PropEntity(entity_id="prop-0", name="glass chamber")]
        shots = [make_shot(0), make_shot(1)]
        shots[1].ff_desc = "Mira waits beside the glass chamber."
        timeline = Timeline.initialize(
            self.registry, shots, prop_bindings={0: ["prop-0"], 1: ["prop-0"]},
        )
        timeline.mark_observed(0, WorldState(
            characters={"character-0": CharacterContent(position="desk")},
            props={"prop-0": PropContent(position="on the table")},
        ))
        position = timeline.refresh_for_execution(1, self.registry)
        manifest = self._history_manifest()
        self._portraits()

        context = build_frame_material_context(
            position, "first", timeline, self.registry, {0: manifest}, self.store, 2,
        )

        self.assertEqual([group.source.shot_idx for group in context.history_sample_groups], [0])
        self.assertEqual(len(context.history_sample_groups[0].samples), 2)
        self.assertEqual(position.plan.bound_prop_entity_ids, ["prop-0"])

    def test_legacy_plan_separates_bound_props_from_previous_holding(self):
        self.registry.props = [
            PropEntity(entity_id="prop-0", name="glass chamber"),
            PropEntity(entity_id="prop-1", name="calibration card"),
        ]
        timeline = Timeline.initialize(
            self.registry, [make_shot(0), make_shot(1)],
            prop_bindings={0: ["prop-0", "prop-1"], 1: ["prop-0"]},
        )
        timeline.mark_observed(0, WorldState(
            characters={"character-0": CharacterContent(holding=["prop-1"])},
            props={"prop-0": PropContent(position="table"), "prop-1": PropContent(position="held")},
        ))
        timeline.refresh_for_execution(1, self.registry)
        timeline.position(1).plan.bound_prop_entity_ids = None

        refreshed = timeline.refresh_for_execution(1, self.registry)

        self.assertEqual(refreshed.plan.bound_prop_entity_ids, ["prop-0"])
        self.assertEqual(
            [prop.entity_id for prop in refreshed.plan.props],
            ["prop-0", "prop-1"],
        )

    def test_legacy_initial_plan_preserves_a_bound_prop_that_is_also_held(self):
        self.registry.props = [PropEntity(entity_id="prop-0", name="calibration card")]
        timeline = Timeline.initialize(
            self.registry, [make_shot(0), make_shot(1)],
            prop_bindings={0: ["prop-0"], 1: ["prop-0"]},
        )
        timeline.mark_observed(0, WorldState(
            characters={"character-0": CharacterContent(holding=["prop-0"])},
            props={"prop-0": PropContent(position="held")},
        ))
        timeline.position(1).plan.bound_prop_entity_ids = None

        refreshed = timeline.refresh_for_execution(1, self.registry)

        self.assertEqual(refreshed.plan.bound_prop_entity_ids, ["prop-0"])

    def test_last_frame_capacity_counts_only_required_first_frame_and_portrait(self):
        manifest = self._history_manifest()
        self._portraits()
        generated_path = self.root / "current-first.png"
        generated_path.write_bytes(b"generated")
        self.store.append(ImageRecord(
            image_id="current-first",
            kind="generated_first_frame",
            asset_path=str(generated_path),
            character_id=None,
            shot_idx=1,
            sample_time_seconds=None,
            view=None,
            summary="current first frame",
        ))

        with self.assertRaisesRegex(FrameMaterialCapacityError, "fixed=1, portraits=1, top_k=1"):
            build_frame_material_context(
                self.position,
                "last",
                self.timeline,
                self.registry,
                {0: manifest},
                self.store,
                1,
                current_first_frame_id="current-first",
            )


if __name__ == "__main__":
    unittest.main()
