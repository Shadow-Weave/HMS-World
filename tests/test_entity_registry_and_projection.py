"""Contract tests for §20.2 global entity registration and Entity@time."""

import inspect
import hashlib
import unittest
from unittest.mock import AsyncMock

from agents.entity_extractor import (
    CharacterContentDraft,
    CharacterEntityDraft,
    EntityExtractor,
    EntityRegistryDraft,
    PropEntityDraft,
    ReplacementCharacterDraft,
    ReplacementPropDraft,
    ReplacementRegistryExtension,
    RelationDraft,
    SceneEntityDraft,
    ShotPropBinding,
    ShotSceneBinding,
    materialize_entity_registry,
)
from continuity.timeline import (
    CharacterContent,
    EntityRegistry,
    ObservedShotSource,
    ObservedStateSources,
    SceneEntity,
    WorldState,
    project_shot_plan,
)
from interfaces import CharacterInScene, ShotDescription


def character(idx: int, name: str) -> CharacterInScene:
    return CharacterInScene(
        idx=idx,
        identifier_in_scene=name,
        is_visible=True,
        static_features=f"{name} static appearance",
        dynamic_features=None,
    )


def shot(*, variation_type: str, ff: list[int], lf: list[int]) -> ShotDescription:
    return ShotDescription(
        idx=4,
        is_last=False,
        cam_idx=0,
        visual_desc="At the desk, the letter is visible.",
        variation_type=variation_type,
        variation_reason="test",
        ff_desc="Mira stands by the desk.",
        ff_vis_char_idxs=ff,
        lf_desc="Noah enters the room.",
        lf_vis_char_idxs=lf,
        motion_desc="Mira keeps the key in hand.",
        audio_desc="room tone",
    )


def registry_draft() -> EntityRegistryDraft:
    return EntityRegistryDraft(
        characters=[
            CharacterEntityDraft(
                character_idx=7,
                identity="detective",
                persona="methodical",
                session="investigating",
                stable_appearance="dark coat",
                initial_content=CharacterContentDraft(outfit="blue coat", position="chair"),
                relations=[RelationDraft(target_kind="character", target_index=2, relation="questions")],
            ),
            CharacterEntityDraft(
                character_idx=2,
                identity="witness",
                persona="guarded",
                session="waiting",
                stable_appearance="gray sweater",
            ),
        ],
        scenes=[SceneEntityDraft(name="interview room", initial_content={"lighting": "warm", "layout": "desk"})],
        props=[
            PropEntityDraft(name="key", initial_content={"position": "table", "condition": "old"}),
            PropEntityDraft(name="letter", initial_content={"position": "desk", "condition": "sealed"}),
        ],
    )


class RecordingChatModel:
    def __init__(self, response: EntityRegistryDraft):
        self.response = response
        self.messages = None
        self.calls = 0

    def bind(self, **kwargs):
        self.bound_kwargs = kwargs
        return self

    def __or__(self, parser):
        self.parser = parser
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self.response


class EntityExtractorContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_final_retry_keeps_registry_without_duplicate_additions(self):
        characters = [character(2, "Noah"), character(7, "Mira")]
        registry = materialize_entity_registry(registry_draft(), characters)
        current = shot(variation_type="small", ff=[7], lf=[])
        model = RecordingChatModel(ReplacementRegistryExtension(
            props=[ReplacementPropDraft(name=registry.props[0].name)],
        ))
        extractor = EntityExtractor(model)
        extractor.keep_registry_on_extension_failure = [{
            "replacement_sha256": hashlib.sha256(current.model_dump_json().encode()).hexdigest(),
            "reason": "Explicit final retry approval",
        }]
        receipts = []
        extractor.extension_failure_receipt = receipts.append
        result, added = await extractor.extend_for_replacement(current, set(), registry, characters)
        self.assertEqual(result, registry)
        self.assertIsNot(result, registry)
        self.assertEqual(added, [])
        self.assertEqual(model.calls, 1)
        self.assertEqual(receipts[0]["action"], "keep_existing_registry")
        extractor.keep_registry_on_extension_failure = []
        with self.assertRaisesRegex(ValueError, "already registered"):
            await extractor.extend_for_replacement(current, set(), registry, characters)

    async def test_one_model_call_receives_only_global_inputs_and_binds_deterministic_ids(self):
        model = RecordingChatModel(registry_draft())
        extractor = EntityExtractor(model)
        production_characters = [character(2, "Noah"), character(7, "Mira")]

        registry = await extractor.extract_entities(
            screenplay="Mira gives Noah a letter.",
            characters=production_characters,
            user_requirements="naturalistic",
        )

        self.assertEqual(model.calls, 1)
        human_content = str(model.messages[1].content).casefold()
        self.assertEqual(model.bound_kwargs["response_format"], {
            "type": "json_schema",
            "json_schema": {
                "name": "EntityRegistryDraft",
                "schema": EntityRegistryDraft.model_json_schema(),
                "strict": True,
            },
        })
        self.assertIn("<screenplay>", human_content)
        self.assertIn("<hms_world_characters>", human_content)
        self.assertIn("<user_requirements>", human_content)
        self.assertNotIn("shotdescription", human_content)
        self.assertNotIn("storyboard", human_content)
        self.assertNotIn("camera", human_content)
        self.assertEqual([item.entity_id for item in registry.characters], ["character-2", "character-7"])
        self.assertEqual([item.name for item in registry.characters], ["Noah", "Mira"])
        self.assertEqual(registry.characters[0].identity_image_ids, [])
        self.assertEqual(registry.characters[1].relations[0].target_entity_id, "character-2")
        self.assertEqual([item.entity_id for item in registry.scenes], ["scene-0"])
        self.assertEqual([item.entity_id for item in registry.props], ["prop-0", "prop-1"])

    async def test_replacement_extension_registers_only_unknown_characters_and_append_only_props(self):
        initial = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        replacement = shot(variation_type="medium", ff=[7, 9], lf=[9])
        replacement.visual_desc = "Mira gives newcomer Ivo a flashlight."
        model = RecordingChatModel(ReplacementRegistryExtension(
            characters=[ReplacementCharacterDraft(
                character_idx=9, name="Ivo", static_features="green coat",
                identity="courier", persona="urgent", session="arriving", stable_appearance="green coat",
            )],
            props=[ReplacementPropDraft(name="flashlight", initial_content={"position": "hand", "condition": "lit"})],
        ))

        registry, added_characters = await EntityExtractor(model).extend_for_replacement(
            replacement, {9}, initial, [character(2, "Noah"), character(7, "Mira")]
        )

        self.assertEqual(model.calls, 1)
        prompt = str(model.messages[1].content).casefold()
        self.assertIn("<replacement_entity_text>", prompt)
        self.assertIn("<unknown_character_idxs>", prompt)
        self.assertIn("character-7", prompt)
        self.assertIn("newcomer ivo", prompt)
        self.assertIn("flashlight", prompt)
        self.assertNotIn("timeline", prompt)
        self.assertNotIn("camera", prompt)
        self.assertNotIn("image_records", prompt)
        self.assertNotIn("cam_idx", prompt)
        self.assertNotIn("is_last", prompt)
        self.assertNotIn("variation_reason", prompt)
        self.assertNotIn("audio_desc", prompt)
        self.assertNotIn("existing_characters", prompt)
        self.assertEqual([item.idx for item in added_characters], [9])
        self.assertEqual([item.entity_id for item in registry.characters], ["character-2", "character-7", "character-9"])
        self.assertEqual([item.entity_id for item in registry.props], ["prop-0", "prop-1", "prop-2"])
        self.assertEqual(registry.props[-1].name, "flashlight")
        self.assertEqual(
            [
                item.entity_id
                for item in project_shot_plan(
                    registry,
                    None,
                    replacement,
                    prop_entity_ids=["prop-0", "prop-2"],
                ).props
            ],
            ["prop-0", "prop-2"],
        )

    async def test_scoped_extension_feedback_keeps_original_inputs_and_registry(self):
        characters = [character(2, "Noah"), character(7, "Mira")]
        initial = materialize_entity_registry(registry_draft(), characters)
        before = initial.model_dump_json()
        replacement = shot(variation_type="small", ff=[7], lf=[])
        bad = ReplacementRegistryExtension(props=[ReplacementPropDraft(name="key")])
        model = RecordingChatModel(bad)
        model.ainvoke = AsyncMock(side_effect=[bad, ReplacementRegistryExtension()])
        extractor = EntityExtractor(model)
        extractor.replacement_validation_feedback_approvals = [{
            "replacement_sha256": hashlib.sha256(replacement.model_dump_json().encode()).hexdigest(),
            "reason": "Explicit approval for this replacement only",
        }]
        receipts = []
        extractor.replacement_validation_feedback_receipt = receipts.append
        result, added = await extractor.extend_for_replacement(replacement, set(), initial, characters)
        self.assertEqual(model.ainvoke.await_count, 2)
        original = model.ainvoke.await_args_list[0].args[0]
        corrected = model.ainvoke.await_args_list[1].args[0]
        self.assertEqual(corrected[:2], original)
        self.assertIn("key", corrected[-1].content)
        self.assertEqual(result.model_dump_json(), before)
        self.assertEqual(initial.model_dump_json(), before)
        self.assertEqual(added, [])
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["shot_idx"], replacement.idx)

    async def test_extension_feedback_requires_exact_scope_and_still_rejects_bad_retry(self):
        characters = [character(2, "Noah"), character(7, "Mira")]
        initial = materialize_entity_registry(registry_draft(), characters)
        replacement = shot(variation_type="small", ff=[7], lf=[])
        bad = ReplacementRegistryExtension(props=[ReplacementPropDraft(name="key")])
        digest = hashlib.sha256(replacement.model_dump_json().encode()).hexdigest()
        for scope, expected_calls in [("another-replacement", 1), (digest, 2)]:
            with self.subTest(scope=scope):
                model = RecordingChatModel(bad)
                model.ainvoke = AsyncMock(side_effect=[bad, bad])
                extractor = EntityExtractor(model)
                extractor.replacement_validation_feedback_approvals = [{
                    "replacement_sha256": scope, "reason": "Scoped approval",
                }]
                with self.assertRaisesRegex(ValueError, "already registered"):
                    await extractor.extend_for_replacement(replacement, set(), initial, characters)
                self.assertEqual(model.ainvoke.await_count, expected_calls)

    async def test_prop_binding_resolves_paraphrases_once_to_stable_registry_ids(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        current_shot = shot(variation_type="medium", ff=[7], lf=[2])
        current_shot.visual_desc = "Mira studies a crumpled note at the desk."
        current_shot.ff_desc = "The folded paper rests under her palm."
        current_shot.motion_desc = "She turns the paper toward Noah."
        model = RecordingChatModel(ShotPropBinding(shot_idx=4, prop_entity_ids=["prop-1"]))

        prop_ids = await EntityExtractor(model).bind_props_for_shot(current_shot, registry)

        self.assertEqual(prop_ids, ["prop-1"])
        self.assertEqual(model.calls, 1)
        prompt = str(model.messages[1].content)
        self.assertIn('"entity_id": "prop-1"', prompt)
        self.assertIn("crumpled note", prompt)
        self.assertNotIn("audio_desc", prompt)
        plan = project_shot_plan(registry, None, current_shot, prop_entity_ids=prop_ids)
        self.assertEqual([item.entity_id for item in plan.props], ["prop-1"])

    async def test_prop_binding_rejects_unregistered_ids(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        model = RecordingChatModel(ShotPropBinding(shot_idx=4, prop_entity_ids=["prop-99"]))

        with self.assertRaisesRegex(ValueError, "unregistered IDs"):
            await EntityExtractor(model).bind_props_for_shot(
                shot(variation_type="small", ff=[7], lf=[]),
                registry,
            )

    async def test_scene_binding_single_scene_needs_no_model_call(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        model = RecordingChatModel(None)

        scene_id = await EntityExtractor(model).bind_scene_for_shot(
            shot(variation_type="small", ff=[7], lf=[]), registry,
        )

        self.assertEqual(scene_id, "scene-0")
        self.assertEqual(model.calls, 0)

        with self.assertRaisesRegex(ValueError, "at least one registered scene"):
            await EntityExtractor(model).bind_scene_for_shot(
                shot(variation_type="small", ff=[7], lf=[]), EntityRegistry(),
            )
        self.assertEqual(model.calls, 0)

    async def test_scene_binding_returns_registered_scene_for_multi_scene_shot(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        registry.scenes.append(SceneEntity(
            entity_id="scene-1", name="courtyard", initial_content={"lighting": "daylight"},
        ))
        current_shot = shot(variation_type="medium", ff=[7], lf=[2])
        current_shot.visual_desc = "In the daylight courtyard, Mira meets Noah by the fountain."
        model = RecordingChatModel(ShotSceneBinding(shot_idx=4, scene_entity_id="scene-1"))

        scene_id = await EntityExtractor(model).bind_scene_for_shot(current_shot, registry)

        self.assertEqual(scene_id, "scene-1")
        self.assertEqual(model.calls, 1)
        prompt = str(model.messages[1].content)
        self.assertIn('"entity_id": "scene-0"', prompt)
        self.assertIn('"entity_id": "scene-1"', prompt)
        self.assertIn("daylight courtyard", prompt)
        self.assertNotIn("audio_desc", prompt)

    async def test_scene_binding_rejects_wrong_shot_or_unknown_scene(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        registry.scenes.append(SceneEntity(entity_id="scene-1", name="courtyard"))
        current_shot = shot(variation_type="small", ff=[7], lf=[])

        with self.subTest("wrong shot"):
            with self.assertRaisesRegex(ValueError, "supplied shot"):
                await EntityExtractor(RecordingChatModel(
                    ShotSceneBinding(shot_idx=3, scene_entity_id="scene-1"),
                )).bind_scene_for_shot(current_shot, registry)
        with self.subTest("unknown scene"):
            with self.assertRaisesRegex(ValueError, "unregistered ID"):
                await EntityExtractor(RecordingChatModel(
                    ShotSceneBinding(shot_idx=4, scene_entity_id="scene-99"),
                )).bind_scene_for_shot(current_shot, registry)

    async def test_replacement_extension_rejects_casefold_or_whitespace_duplicate_names(self):
        initial = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        model = RecordingChatModel(ReplacementRegistryExtension(
            props=[ReplacementPropDraft(name=" Letter ")],
        ))
        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            await EntityExtractor(model).extend_for_replacement(
                shot(variation_type="small", ff=[7], lf=[]), set(), initial,
                [character(2, "Noah"), character(7, "Mira")],
            )

    def test_projection_is_pure_idx_based_and_refreshes_known_previous_fields(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        previous_state = WorldState(
            characters={
                "character-7": CharacterContent(expression="worried", position="door", holding=["prop-0"]),
            },
            props={"prop-0": {"position": "hand", "condition": None}},
            scenes={"scene-0": {"lighting": "blue", "layout": None}},
        )
        current_shot = shot(variation_type="medium", ff=[7], lf=[2])
        before = current_shot.model_dump()

        plan = project_shot_plan(
            registry,
            previous_state,
            current_shot,
            ObservedStateSources(
                characters={"character-7": {
                    "expression": ObservedShotSource(shot_idx=3),
                    "position": ObservedShotSource(shot_idx=3),
                    "holding": ObservedShotSource(shot_idx=3),
                }},
                props={"prop-0": {"position": ObservedShotSource(shot_idx=3)}},
                scenes={"scene-0": {"lighting": ObservedShotSource(shot_idx=3)}},
            ),
            prop_entity_ids=["prop-0", "prop-1"],
        )

        self.assertEqual([item.entity_id for item in plan.characters], ["character-7", "character-2"])
        mira_at_time = plan.characters[0]
        mira = mira_at_time.current_content
        self.assertIsNone(mira.outfit)
        self.assertEqual(mira.expression, "worried")
        self.assertEqual(mira.position, "door")
        self.assertIsNone(mira.orientation)
        self.assertIsNone(mira.pose, "motion text must not become an Entity@time action")
        self.assertNotIn("outfit", mira_at_time.field_sources)
        self.assertEqual(mira_at_time.field_sources["position"].model_dump(), {
            "source_kind": "observed", "shot_idx": 3,
        })
        self.assertNotIn("orientation", mira_at_time.field_sources)
        self.assertEqual(plan.scene.current_content, {"lighting": "blue", "layout": None})
        self.assertEqual([item.entity_id for item in plan.props], ["prop-0", "prop-1"])
        self.assertEqual(plan.props[0].current_content.position, "hand")
        self.assertEqual(current_shot.model_dump(), before)
        self.assertIsNot(plan.shot, current_shot)
        current_shot.motion_desc = "mutated after projection"
        self.assertNotEqual(plan.shot.motion_desc, current_shot.motion_desc)
        source = inspect.getsource(project_shot_plan)
        self.assertNotIn("chat_model", source)
        self.assertNotIn("await ", source)

    def test_small_projection_uses_only_native_first_frame_visible_indices(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])

        plan = project_shot_plan(
            registry,
            None,
            shot(variation_type="small", ff=[7], lf=[2]),
            prop_entity_ids=[],
        )

        self.assertEqual([item.entity_id for item in plan.characters], ["character-7"])
        self.assertEqual(plan.characters[0].current_content.outfit, "blue coat")

    def test_projection_refuses_to_guess_a_scene_when_current_shot_has_no_scene_id(self):
        draft = registry_draft()
        draft.scenes.append(SceneEntityDraft(name="hallway"))
        registry = materialize_entity_registry(draft, [character(2, "Noah"), character(7, "Mira")])

        with self.assertRaisesRegex(ValueError, "exactly one registered scene"):
            project_shot_plan(
                registry,
                None,
                shot(variation_type="small", ff=[7], lf=[]),
                prop_entity_ids=[],
            )

    def test_projection_does_not_infer_prop_ids_from_shot_text(self):
        registry = materialize_entity_registry(registry_draft(), [character(2, "Noah"), character(7, "Mira")])
        current_shot = shot(variation_type="small", ff=[7], lf=[2])
        current_shot.visual_desc = "A monkey watches silently."
        current_shot.ff_desc = "Mira waits."
        current_shot.motion_desc = "She does not touch anything."
        current_shot.lf_desc = "The key appears only in this non-executed last-frame text."

        plan = project_shot_plan(registry, None, current_shot, prop_entity_ids=[])

        self.assertEqual(plan.props, [])


if __name__ == "__main__":
    unittest.main()
