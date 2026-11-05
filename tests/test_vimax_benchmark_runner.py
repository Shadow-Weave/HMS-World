"""Contract tests for the native ViMax-Bench serial runner."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from continuity.timeline import CharacterEntity, EntityRegistry, PropEntity, SceneEntity, Timeline, WorldState
from interfaces import CharacterInScene, ShotDescription
from evals.vimax import runner as run_vimax_benchmark
from evals.vimax.catalog import (
    BENCHMARK_DATA_ROOT,
    BenchmarkSceneRegistrationDraft,
    BenchmarkShotBindingDraft,
    benchmark_character_source,
    load_catalog,
    prepare_story,
    select_stories,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _character(idx: int, name: str) -> CharacterInScene:
    return CharacterInScene(
        idx=idx,
        identifier_in_scene=name,
        is_visible=True,
        static_features=f"{name} stable appearance",
        dynamic_features=f"{name} benchmark clothing",
    )


def _shot(idx: int) -> ShotDescription:
    return ShotDescription(
        idx=idx,
        is_last=idx == 1,
        cam_idx=idx,
        visual_desc=f"frame-{idx}",
        variation_type="small",
        variation_reason="benchmark keyframe protocol",
        ff_desc=f"frame-{idx}",
        ff_vis_char_idxs=[],
        lf_desc="",
        lf_vis_char_idxs=[],
        motion_desc=f"motion-{idx}",
        audio_desc="",
    )


class _SceneAnalyzer:
    def __init__(self):
        self.calls = []

    async def analyze(self, scene, characters):
        self.calls.append(scene.scene_num)
        return BenchmarkSceneRegistrationDraft(
            scene_num=scene.scene_num,
            shots=[
                BenchmarkShotBindingDraft(
                    shot_id=shot.shot_id,
                    visible_character_idxs=list(range(len(characters))),
                    camera_group=0,
                )
                for shot in scene.shots
            ],
        )


class _EntityExtractor:
    def __init__(self, characters):
        self.characters = characters
        self.extract_calls = []
        self.prop_calls = []
        self.scene_calls = []

    async def extract_entities(self, screenplay, characters, user_requirements):
        self.extract_calls.append((screenplay, characters, user_requirements))
        return EntityRegistry(
            characters=[
                CharacterEntity(
                    entity_id=f"character-{character.idx}",
                    name=character.identifier_in_scene,
                    identity=character.identifier_in_scene,
                    persona="persona",
                    session="session",
                    stable_appearance=character.static_features,
                )
                for character in characters
            ],
            scenes=[
                SceneEntity(entity_id="scene-0", name="plaza"),
                SceneEntity(entity_id="scene-1", name="rehearsal room"),
                SceneEntity(entity_id="scene-2", name="airport"),
            ],
            props=[PropEntity(entity_id="prop-0", name="instrument")],
        )

    async def bind_props_for_shot(self, shot, registry):
        self.prop_calls.append((shot.idx, registry))
        return ["prop-0"] if shot.idx == 0 else []

    async def bind_scene_for_shot(self, shot, registry):
        self.scene_calls.append((shot.idx, registry))
        if shot.idx < 2:
            return "scene-0"
        if shot.idx == 2:
            return "scene-1"
        return "scene-2"


class ViMaxBenchmarkDatasetTests(unittest.TestCase):
    def test_official_folder_is_one_index_plus_35_strict_stories(self):
        catalog = load_catalog(PROJECT_ROOT / "evals/vimax/data")

        self.assertEqual(catalog.index.total_stories, 35)
        self.assertEqual(len(catalog.stories), 35)
        self.assertEqual(len(list(BENCHMARK_DATA_ROOT.glob("*.json"))), 36)
        self.assertEqual(catalog.stories[26].ref.theme, "three_friends_surprise_planning")
        self.assertEqual(catalog.stories[26].story.metadata.requested_shots, 15)

    def test_selector_rejects_duplicate_id_and_theme_for_same_story(self):
        catalog = load_catalog(PROJECT_ROOT / "evals/vimax/data")

        with self.assertRaisesRegex(ValueError, "selected more than once"):
            select_stories(
                catalog,
                ids=[27],
                themes=["three_friends_surprise_planning"],
            )

    def test_validate_only_does_not_construct_any_provider(self):
        with patch.object(run_vimax_benchmark, "load_serial_demo") as load:
            result = asyncio.run(run_vimax_benchmark.amain([
                "27",
                "--validate-only",
                "--benchmark-root",
                str(PROJECT_ROOT / "evals/vimax/data"),
            ]))

        load.assert_not_called()
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["selected"][0]["shots"], 15)


class ViMaxBenchmarkAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_source_does_not_mislabel_benchmark_groups_as_scenes(self):
        story = load_catalog(PROJECT_ROOT / "evals/vimax/data").stories[2].story

        source = benchmark_character_source(story)

        self.assertIn("BENCHMARK GROUP 1", source)
        self.assertIn("not a physical scene", source)
        self.assertNotIn("\nSCENE 1\n", source)

    async def test_adapter_preserves_every_official_prompt_and_binds_all_scenes(self):
        item = load_catalog(PROJECT_ROOT / "evals/vimax/data").stories[26]
        analyzer = _SceneAnalyzer()
        characters = [
            _character(0, "Character A"),
            _character(1, "Character B"),
            _character(2, "Character C"),
        ]
        entity_extractor = _EntityExtractor(characters)

        plan = await prepare_story(item.story, characters, analyzer, entity_extractor)

        official = [shot for scene in item.story.scenes for shot in scene.shots]
        self.assertEqual([shot.ff_desc for shot in plan.shots], [shot.first_frame for shot in official])
        self.assertEqual([shot.motion_desc for shot in plan.shots], [shot.video_prompt for shot in official])
        for prepared, source in zip(plan.shots, official, strict=True):
            self.assertIn(source.first_frame, prepared.visual_desc)
            self.assertIn(source.video_prompt, prepared.visual_desc)
            self.assertNotEqual(prepared.visual_desc, prepared.ff_desc)
        self.assertTrue(all(shot.variation_type == "small" for shot in plan.shots))
        self.assertTrue(all(shot.lf_desc == "" for shot in plan.shots))
        self.assertEqual([scene.name for scene in plan.registry.scenes], ["plaza", "rehearsal room", "airport"])
        self.assertEqual(plan.scene_bindings[0], "scene-0")
        self.assertEqual(plan.scene_bindings[1], "scene-0")
        self.assertEqual(plan.scene_bindings[2], "scene-1")
        self.assertEqual(plan.scene_bindings[3], "scene-2")
        self.assertEqual(plan.prop_bindings[0], ["prop-0"])
        self.assertTrue(all(plan.prop_bindings[index] == [] for index in range(1, len(plan.shots))))
        self.assertEqual(analyzer.calls, [1, 2, 3, 4])
        self.assertEqual(len(entity_extractor.extract_calls), 1)
        self.assertEqual(entity_extractor.extract_calls[0][0], benchmark_character_source(item.story))
        self.assertEqual(entity_extractor.extract_calls[0][1], characters)
        self.assertEqual(entity_extractor.extract_calls[0][2], "")
        self.assertEqual([idx for idx, _ in entity_extractor.prop_calls], list(range(len(plan.shots))))
        self.assertEqual([idx for idx, _ in entity_extractor.scene_calls], list(range(len(plan.shots))))


class MultiSceneTimelineTests(unittest.TestCase):
    def test_explicit_scene_bindings_are_required_and_preserved(self):
        registry = EntityRegistry(scenes=[
            SceneEntity(entity_id="scene-0", name="inside"),
            SceneEntity(entity_id="scene-1", name="outside"),
        ])
        shots = [_shot(0), _shot(1)]

        with self.assertRaisesRegex(ValueError, "explicit scene bindings"):
            Timeline.initialize(registry, shots)

        timeline = Timeline.initialize(
            registry,
            shots,
            prop_bindings={0: [], 1: []},
            scene_bindings={0: "scene-0", 1: "scene-1"},
        )

        self.assertEqual(timeline.position(0).plan.scene.entity_id, "scene-0")
        self.assertEqual(timeline.position(1).plan.scene.entity_id, "scene-1")

    def test_scene_binding_prevents_cross_scene_projection_and_replacement_keeps_binding(self):
        registry = EntityRegistry(scenes=[
            SceneEntity(entity_id="scene-0", name="inside", initial_content={"lighting": "warm"}),
            SceneEntity(entity_id="scene-1", name="outside", initial_content={"lighting": "cool"}),
        ])
        shots = [_shot(0), _shot(1)]
        timeline = Timeline.initialize(
            registry,
            shots,
            prop_bindings={0: [], 1: []},
            scene_bindings={0: "scene-0", 1: "scene-1"},
        )

        timeline.mark_observed(0, WorldState(scenes={"scene-0": {"lighting": "red"}}))
        refreshed = timeline.refresh_for_execution(1, registry)
        self.assertEqual(refreshed.plan.scene.entity_id, "scene-1")
        self.assertEqual(refreshed.plan.scene.current_content, {"lighting": "cool"})

        replacement = _shot(1)
        replacement.visual_desc = "outside replacement"
        replaced = timeline.apply_replacement(0, replacement, registry, prop_entity_ids=[])
        self.assertEqual(replaced.plan.scene.entity_id, "scene-1")

    def test_catalog_rejects_an_unindexed_json_file(self):
        source = BENCHMARK_DATA_ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in source.glob("*.json"):
                (root / path.name).write_bytes(path.read_bytes())
            (root / "unexpected.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "file set differs"):
                load_catalog(root)
