"""Contract tests for §20.3 single-track Timeline transitions."""

import unittest

from pydantic import ValidationError

from continuity.timeline import (
    CharacterContent,
    CharacterEntity,
    EntityRegistry,
    ObservedPosition,
    PlannedPosition,
    SceneEntity,
    Timeline,
    WorldState,
)
from interfaces import ShotDescription


def registry() -> EntityRegistry:
    return EntityRegistry(
        characters=[
            CharacterEntity(
                entity_id="character-0",
                name="Mira",
                identity="detective",
                persona="methodical",
                session="investigating",
                stable_appearance="dark coat",
                initial_content=CharacterContent(outfit="blue coat", position="chair"),
            )
        ],
        scenes=[SceneEntity(entity_id="scene-0", name="room", initial_content={"lighting": "warm"})],
    )


def shot(idx: int, *, visual: str | None = None) -> ShotDescription:
    return ShotDescription(
        idx=idx,
        is_last=idx == 2,
        cam_idx=0,
        visual_desc=visual or f"shot {idx}",
        variation_type="small",
        variation_reason="test",
        ff_desc="Mira is visible.",
        ff_vis_char_idxs=[0],
        lf_desc="unused tail",
        lf_vis_char_idxs=[],
        motion_desc="Mira moves.",
        audio_desc="",
    )


def observed(position: str) -> WorldState:
    return WorldState(characters={"character-0": CharacterContent(position=position)})


class SingleTrackTimelineTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()
        self.timeline = Timeline.initialize(self.registry, [shot(0), shot(1), shot(2)])

    def test_positions_are_discriminated_and_observed_requires_one_state_and_sources(self):
        self.assertTrue(all(isinstance(position, PlannedPosition) for position in self.timeline.positions))
        restored = Timeline.model_validate(self.timeline.model_dump())
        self.assertTrue(all(isinstance(position, PlannedPosition) for position in restored.positions))
        with self.assertRaises(ValidationError):
            ObservedPosition(shot_idx=0)
        self.timeline.mark_observed(0, observed("door"))
        restored_observed = Timeline.model_validate(self.timeline.model_dump())
        self.assertIsInstance(restored_observed.positions[0], ObservedPosition)
        dumped = restored_observed.positions[0].model_dump()
        self.assertIn("state", dumped)
        self.assertIn("sources", dumped)
        self.assertNotIn("samples", dumped)

    def test_mark_observed_replaces_same_position_without_old_plan(self):
        old_planned = self.timeline.positions[0]
        result = self.timeline.mark_observed(0, observed("door"))

        self.assertIsInstance(self.timeline.positions[0], ObservedPosition)
        self.assertNotIn("plan", self.timeline.positions[0].model_dump())
        self.assertEqual(self.timeline.positions[0].final_state.characters["character-0"].position, "door")
        self.assertNotEqual(type(old_planned), type(self.timeline.positions[0]))
        self.assertIsNot(result, self.timeline.positions[0])

    def test_observed_position_is_rejected_by_all_writing_operations(self):
        self.timeline.mark_observed(0, observed("door"))

        with self.assertRaisesRegex(ValueError, "already observed"):
            self.timeline.refresh_for_execution(0, self.registry)
        with self.assertRaisesRegex(ValueError, "already observed"):
            self.timeline.mark_observed(0, observed("other"))

    def test_cannot_skip_a_future_position(self):
        with self.assertRaisesRegex(ValueError, "cannot be observed before shot 0"):
            self.timeline.mark_observed(1, observed("door"))
        with self.assertRaisesRegex(ValueError, "cannot refresh until shot 0 is observed"):
            self.timeline.refresh_for_execution(1, self.registry)

    def test_refresh_uses_previous_final_state_for_nonfirst_shot(self):
        self.timeline.mark_observed(0, observed("door"))

        refreshed = self.timeline.refresh_for_execution(1, self.registry)

        self.assertEqual(refreshed.plan.characters[0].current_content.position, "door")
        self.assertIsNot(refreshed, self.timeline.positions[1])

    def test_first_shot_refresh_uses_initial_projection(self):
        refreshed = self.timeline.refresh_for_execution(0, self.registry)

        self.assertEqual(refreshed.plan.characters[0].current_content.position, "chair")

    def test_empty_entity_observation_does_not_overwrite_initial_content(self):
        result = self.timeline.mark_observed(
            0,
            WorldState(characters={"character-0": CharacterContent()}),
        )

        self.assertEqual(result.state.characters["character-0"].outfit, "blue coat")
        self.assertEqual(result.state.characters["character-0"].position, "chair")

    def test_refresh_does_not_search_older_observations_when_latest_snapshot_lacks_entity(self):
        self.timeline.mark_observed(0, observed("door"))
        self.timeline.refresh_for_execution(1, self.registry)
        self.timeline.mark_observed(1, WorldState())

        refreshed = self.timeline.refresh_for_execution(2, self.registry)

        character = refreshed.plan.characters[0]
        self.assertEqual(character.current_content.position, "door")
        self.assertEqual(character.field_sources["position"].model_dump(), {
            "source_kind": "observed", "shot_idx": 0,
        })

    def test_latest_previous_observation_wins_over_older_observed_value(self):
        self.timeline.mark_observed(0, observed("door"))
        self.timeline.refresh_for_execution(1, self.registry)
        self.timeline.mark_observed(1, observed("desk"))

        refreshed = self.timeline.refresh_for_execution(2, self.registry)

        self.assertEqual(refreshed.plan.characters[0].current_content.position, "desk")
        self.assertEqual(refreshed.plan.characters[0].field_sources["position"].model_dump(), {
            "source_kind": "observed", "shot_idx": 1,
        })

    def test_explicit_absent_scene_fields_are_not_resurrected_from_history(self):
        self.timeline.mark_observed(0, WorldState(scenes={"scene-0": {"weather": "fog"}}))
        self.timeline.refresh_for_execution(1, self.registry)
        self.timeline.mark_observed(1, WorldState(scenes={
            "scene-0": {"lighting": None, "weather": None},
        }))

        refreshed = self.timeline.refresh_for_execution(2, self.registry)

        self.assertEqual(refreshed.plan.scene.current_content, {"lighting": None, "weather": None})
        self.assertEqual(refreshed.plan.scene.field_sources, {})

    def test_replacement_uses_latest_explicit_absent_without_history_backfill(self):
        self.timeline.mark_observed(0, WorldState(characters={
            "character-0": CharacterContent(outfit=None, position=None),
        }))

        replacement = self.timeline.apply_replacement(0, shot(1, visual="replacement"), self.registry)

        character = replacement.plan.characters[0]
        self.assertIsNone(character.current_content.outfit)
        self.assertIsNone(character.current_content.position)
        self.assertEqual(character.field_sources, {})

    def test_unchanged_observation_retains_planned_field_source(self):
        self.timeline.mark_observed(0, observed("door"))
        self.timeline.refresh_for_execution(1, self.registry)

        result = self.timeline.mark_observed(1, observed("door"))

        self.assertEqual(result.sources.characters["character-0"]["position"].model_dump(), {
            "source_kind": "observed", "shot_idx": 0,
        })

    def test_changed_observation_records_the_current_shot_as_its_source(self):
        self.timeline.mark_observed(0, observed("door"))
        self.timeline.refresh_for_execution(1, self.registry)

        result = self.timeline.mark_observed(1, observed("window"))

        self.assertEqual(result.sources.characters["character-0"]["position"].model_dump(), {
            "source_kind": "observed", "shot_idx": 1,
        })

    def test_no_change_preserves_next_position_identity_and_serialization(self):
        self.timeline.mark_observed(0, observed("door"))
        next_position = self.timeline.positions[1]
        before = next_position.model_dump_json()

        self.timeline.apply_no_change(0)

        self.assertIs(self.timeline.positions[1], next_position)
        self.assertEqual(self.timeline.positions[1].model_dump_json(), before)

    def test_replacement_changes_only_next_and_next_refresh_keeps_replacement_shot(self):
        self.timeline.mark_observed(0, observed("door"))
        future_position = self.timeline.positions[2]
        future_dump = future_position.model_dump_json()
        replacement = shot(1, visual="REPLACED after actual door state")

        self.timeline.apply_replacement(0, replacement, self.registry)

        self.assertEqual(self.timeline.positions[1].plan.shot.visual_desc, replacement.visual_desc)
        self.assertIs(self.timeline.positions[2], future_position)
        self.assertEqual(self.timeline.positions[2].model_dump_json(), future_dump)
        replacement.visual_desc = "mutated caller object"
        self.assertEqual(self.timeline.positions[1].plan.shot.visual_desc, "REPLACED after actual door state")
        refreshed = self.timeline.refresh_for_execution(1, self.registry)
        self.assertEqual(refreshed.plan.shot.visual_desc, "REPLACED after actual door state")
        self.assertEqual(refreshed.plan.characters[0].current_content.position, "door")


if __name__ == "__main__":
    unittest.main()
