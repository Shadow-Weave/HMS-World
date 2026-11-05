"""§20.9 three-shot loop over real accepted roles and single-shot seams."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
from PIL import Image
from langchain_core.messages import AIMessage

from agents.continuity_image_retriever import (
    ContinuityImageRetriever,
    ImageRetrievalResult,
    PortraitSelection,
    RetrievedImage,
    FrameSelections,
)
from agents.entity_extractor import (
    CharacterEntityDraft,
    EntityExtractor,
    EntityRegistryDraft,
    ReplacementCharacterDraft,
    ReplacementPropDraft,
    ReplacementRegistryExtension,
    SceneEntityDraft,
    ShotPropBinding,
)
from agents.frame_prompt_writer import FinalFramePrompt, FramePromptWriter
from agents.shot_observer import ShotObserver
from agents.shot_replanner import NoChange, Replacement, ShotReplanner
from continuity.timeline import (
    CameraState, CharacterContent, CharacterEntity, EntityRegistry, ObservedPosition,
    ObservedShotSource, PlannedPosition, PropContent, SceneEntity, Timeline, WorldState,
)
from continuity.image_records import ImageRecord, ImageRecordStore
from continuity.shot_sampling import SampledFrame, write_observation_samples
from continuity.frame_material_context import (
    FixedFrameMaterial,
    FrameMaterialContext,
    IdentityPortraitCandidate,
    VisibleCharacterPortraitCandidates,
    history_sample_purpose,
)
from continuity.history_materials import HistoricalSampleCandidate, HistoricalSampleGroup, StateFieldLineage
from interfaces import Camera, CharacterInScene, ImageOutput, ShotBriefDescription, ShotDescription, VideoOutput
from pipelines.script2video_pipeline import Script2VideoPipeline
from pipelines.serial_demo_pipeline import SerialDemoPipeline


def _shot(idx: int, camera_idx: int, variation: str, *, is_last: bool = False) -> ShotDescription:
    return ShotDescription(
        idx=idx, is_last=is_last, cam_idx=camera_idx,
        visual_desc=f"visual-{idx}", variation_type=variation, variation_reason="test",
        ff_desc=f"first-target-{idx}", ff_vis_char_idxs=[0],
        lf_desc=f"last-target-{idx}", lf_vis_char_idxs=[0],
        motion_desc=f"motion-{idx}", audio_desc="",
    )


class _SequencedChatModel:
    def __init__(self, responses, *, events=None, label=None):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []
        self.events = events
        self.label = label
        self.raw_messages = False

    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools, *, strict):
        self.raw_messages = True
        return self

    def __or__(self, parser):
        self.parser = parser
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.events is not None and self.label is not None:
            self.events.append(self.label)
        response = self.responses.pop(0)
        if self.raw_messages:
            return AIMessage(content=response.model_dump_json(exclude_unset=True))
        return response


class _ImageProvider:
    def __init__(self, events):
        self.calls = []
        self.events = events

    async def generate_single_image(self, prompt, reference_image_paths=None, size=None, progress=None):
        self.calls.append({"prompt": prompt, "references": list(reference_image_paths or [])})
        self.events.append("frame-provider:" + prompt)
        return ImageOutput("pil", "png", Image.fromarray(np.full((24, 32, 3), len(self.calls) * 10, dtype=np.uint8)))


class _VideoProvider:
    def __init__(self, root: Path, events):
        self.root = root
        self.calls = []
        self.events = events

    async def generate_single_video(self, prompt, reference_image_paths, task_receipt_path=None, progress=None):
        self.calls.append({"prompt": prompt, "references": list(reference_image_paths)})
        self.events.append("video-provider:" + prompt)
        path = self.root / f"provider-{len(self.calls)}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 24))
        for value in range(len(self.calls) * 30, len(self.calls) * 30 + 120, 10):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        writer.release()
        return VideoOutput("bytes", "mp4", path.read_bytes())


def _state(*, include_new: bool) -> WorldState:
    characters = {
        "character-0": CharacterContent(outfit="coat", expression=None, pose=None, position="state-0", orientation=None, holding=None, emotion=None),
    }
    props = {}
    if include_new:
        characters["character-1"] = CharacterContent(outfit="green", expression=None, pose=None, position="door", orientation=None, holding=None, emotion=None)
        props["prop-0"] = PropContent(position="hand", condition="lit")
    return WorldState(
        characters=characters,
        props=props,
        scenes={"scene-0": {"place": "room"}},
        camera=CameraState(shot_scale="medium", angle=None, composition=None),
    )


def _observation(*, include_new: bool) -> WorldState:
    characters = {
        "character-0": CharacterContent(outfit="coat", position="state-0"),
    }
    props = {}
    if include_new:
        characters["character-1"] = CharacterContent(outfit="green", position="door")
        props["prop-0"] = PropContent(position="hand", condition="lit")
    return WorldState(
        characters=characters,
        props=props,
        scenes={"scene-0": {"place": "room"}},
        camera=CameraState(shot_scale="medium", angle=None, composition=None),
    )


def _write_resume_fixture(root: Path, *, observed_count: int = 1) -> tuple[EntityRegistry, Timeline, list[ShotDescription]]:
    """Write the formal, single-track checkpoint artifacts used by resume tests."""
    registry = EntityRegistry(
        characters=[CharacterEntity(
            entity_id="character-0", name="Mira", identity="Mira", persona="", session="",
            stable_appearance="dark hair",
        )],
        scenes=[SceneEntity(entity_id="scene-0", name="room", initial_content={"place": "room"})],
    )
    shots = [_shot(0, 0, "small"), _shot(1, 0, "small", is_last=True)]
    timeline = Timeline.initialize(registry, shots)
    for shot_idx in range(observed_count):
        timeline.mark_observed(shot_idx, _observation(include_new=False))
    (root / "continuity").mkdir(parents=True, exist_ok=True)
    (root / "characters.json").write_text(json.dumps([CharacterInScene(
        idx=0, identifier_in_scene="Mira", is_visible=True, static_features="dark hair",
    ).model_dump()]), encoding="utf-8")
    (root / "character_portraits_registry.json").write_text(json.dumps({}), encoding="utf-8")
    (root / "camera_tree.json").write_text(json.dumps([Camera(idx=0, active_shot_idxs=[0, 1]).model_dump()]), encoding="utf-8")
    (root / "continuity" / "entity_registry.json").write_text(registry.model_dump_json(), encoding="utf-8")
    (root / "continuity" / "timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")
    store = ImageRecordStore(root / "continuity" / "image_records.jsonl")
    for shot in shots:
        shot_dir = root / "shots" / str(shot.idx)
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "shot_description.json").write_text(shot.model_dump_json(), encoding="utf-8")
    for shot_idx in range(observed_count):
        shot_dir = root / "shots" / str(shot_idx)
        (shot_dir / "video.mp4").write_bytes(b"checkpoint video")
        Image.new("RGB", (2, 2)).save(shot_dir / "first_frame.png")
        sample_path = shot_dir / "samples" / "frame-0.png"
        sample_path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2)).save(sample_path)
        image_id = f"shot-{shot_idx}-sample-frame-0"
        write_observation_samples(shot_dir / "observation_samples.json", [SampledFrame(
            shot_idx=shot_idx,
            image_id=image_id,
            asset_path=str(sample_path),
            time_seconds=0.0,
        )])
        store.append(ImageRecord(
            image_id=image_id, kind="shot_sample", asset_path=str(sample_path), character_id=None,
            shot_idx=shot_idx, sample_time_seconds=0.0, view=None, summary="actual sample",
        ))
    return registry, timeline, shots


class SerialDemoPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_promoted_shot_prepares_last_event_before_native_tail_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            images = _ImageProvider(events)
            videos = _VideoProvider(root, events)
            production = Script2VideoPipeline("unused", images, videos, str(root))
            shot = _shot(0, 0, "medium", is_last=True).model_copy(update={
                "ff_vis_char_idxs": [], "lf_vis_char_idxs": [],
            })
            registry = EntityRegistry(scenes=[SceneEntity(entity_id="scene-0", name="room")])
            timeline = Timeline.initialize(registry, [shot], prop_bindings={0: []},
                                           scene_bindings={0: "scene-0"})
            position = timeline.refresh_for_execution(0, registry)
            frame = root / "shots" / "0" / "first_frame.png"
            frame.parent.mkdir(parents=True)
            Image.new("RGB", (32, 24)).save(frame)
            original_first = frame.read_bytes()
            # A replanned shot now needs a tail, while native events still reflect its original small variation.
            production.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline = SerialDemoPipeline(production, MagicMock(), MagicMock(), MagicMock(),
                                          MagicMock(), MagicMock(), max_reference_images=4)

            async def select_tail(_available, _description):
                return [(str(frame), "current first frame")], shot.lf_desc

            pipeline._continuity_selector = MagicMock(return_value=select_tail)
            pipeline.observe_position = AsyncMock()
            await pipeline.produce_position(position, timeline, registry, [], {},
                                            [Camera(idx=0, active_shot_idxs=[0])])

            self.assertEqual(frame.read_bytes(), original_first)
            self.assertTrue(production.frame_events[0]["first_frame"].is_set())
            self.assertTrue(production.frame_events[0]["last_frame"].is_set())
            self.assertEqual(len(images.calls), 1)
            self.assertEqual(images.calls[0]["prompt"], shot.lf_desc)
            self.assertEqual(videos.calls[0]["references"], [str(frame), str(frame.with_name("last_frame.png"))])

    async def test_fresh_serial_plan_binds_every_shot_to_an_explicit_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            characters = [CharacterInScene(
                idx=0, identifier_in_scene="Mira", is_visible=True, static_features="coat",
            )]
            registry = EntityRegistry(
                characters=[CharacterEntity(
                    entity_id="character-0", name="Mira", identity="detective", persona="", session="",
                    stable_appearance="coat",
                )],
                scenes=[
                    SceneEntity(entity_id="scene-0", name="office", initial_content={"lighting": "warm"}),
                    SceneEntity(entity_id="scene-1", name="courtyard", initial_content={"lighting": "daylight"}),
                ],
            )
            shots = [_shot(0, 0, "small"), _shot(1, 0, "small", is_last=True)]
            briefs = [
                ShotBriefDescription(idx=item.idx, is_last=item.is_last, cam_idx=item.cam_idx,
                                     visual_desc=item.visual_desc, audio_desc="")
                for item in shots
            ]
            production = MagicMock(working_dir=str(root))
            production.extract_characters = AsyncMock(return_value=characters)
            production.design_storyboard = AsyncMock(return_value=briefs)
            production.decompose_visual_description_for_single_shot_brief_description = AsyncMock(side_effect=shots)
            production.construct_camera_tree = AsyncMock(return_value=[Camera(idx=0, active_shot_idxs=[0, 1])])
            extractor = MagicMock()
            extractor.extract_entities = AsyncMock(return_value=registry)
            extractor.bind_props_for_shot = AsyncMock(side_effect=[[], []])
            extractor.bind_scene_for_shot = AsyncMock(side_effect=["scene-0", "scene-1"])
            pipeline = SerialDemoPipeline(
                production, extractor, MagicMock(), MagicMock(), MagicMock(), MagicMock(), max_reference_images=4,
            )
            pipeline._generate_and_register_portraits = AsyncMock(return_value={})

            _characters, _registry, _portraits, _cameras, timeline = await pipeline._start_fresh(
                "script", "requirements", "style",
            )

            self.assertEqual(extractor.bind_scene_for_shot.await_count, 2)
            self.assertEqual(
                [timeline.position(index).plan.scene.entity_id for index in range(2)],
                ["scene-0", "scene-1"],
            )

    async def test_resume_all_observed_loads_real_checkpoint_and_only_concatenates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = MagicMock(working_dir=str(root))
            production.extract_characters = AsyncMock()
            production.concatenate_rendered_shots.return_value = str(root / "final_video.mp4")
            pipeline = SerialDemoPipeline(
                production, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            _write_resume_fixture(root, observed_count=2)

            result = await pipeline.run("", "", "style", resume=True)

            self.assertEqual(result, str(root / "final_video.mp4"))
            production.extract_characters.assert_not_awaited()
            production.concatenate_rendered_shots.assert_called_once_with([0, 1])

    async def test_resume_rejects_planned_artifact_that_conflicts_with_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_resume_fixture(root)
            stale = _shot(1, 0, "medium", is_last=True)
            (root / "shots" / "1" / "shot_description.json").write_text(stale.model_dump_json(), encoding="utf-8")
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            with self.assertRaisesRegex(ValueError, "planned shot description conflicts"):
                await pipeline.run("", "", "style", resume=True)

    async def test_resume_rejects_nonempty_inputs_before_loading_artifacts(self):
        pipeline = SerialDemoPipeline(
            MagicMock(working_dir="unused"), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            max_reference_images=4,
        )
        with self.assertRaisesRegex(ValueError, "does not accept"):
            await pipeline.run("screenplay", "", "style", resume=True)

    async def test_resume_rejects_empty_timeline_and_registry_without_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_resume_fixture(root)
            (root / "continuity" / "timeline.json").write_text(Timeline(positions=[]).model_dump_json(), encoding="utf-8")
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            with self.assertRaisesRegex(ValueError, "must contain at least one position"):
                await pipeline.run("", "", "style", resume=True)
            _write_resume_fixture(root)
            registry = EntityRegistry()
            (root / "continuity" / "entity_registry.json").write_text(registry.model_dump_json(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain at least one scene"):
                await pipeline.run("", "", "style", resume=True)

    async def test_resume_applies_saved_unapplied_replacement_from_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, timeline, shots = _write_resume_fixture(root)
            replacement_shot = shots[1].model_copy(deep=True)
            replacement_shot.ff_desc = "replacement first endpoint"
            decision = Replacement(outcome="replacement", replacement_shot=replacement_shot, reason="checkpoint")
            (root / "shots" / "0" / "replan.json").write_text(decision.model_dump_json(), encoding="utf-8")
            extractor = MagicMock()
            extractor.extend_for_replacement = AsyncMock(return_value=(registry, []))
            extractor.bind_props_for_shot = AsyncMock(return_value=[])
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), extractor, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            pipeline._generate_and_register_portraits = AsyncMock(return_value={})
            characters, loaded_registry, portraits, cameras, loaded_timeline = pipeline._load_resume_artifacts()
            await pipeline._ensure_replan_after_observed(
                0, loaded_timeline, loaded_registry, characters, portraits, cameras, "style",
            )
            extractor.extend_for_replacement.assert_awaited_once()
            self.assertEqual(loaded_timeline.position(1).plan.shot.ff_desc, "replacement first endpoint")
            self.assertEqual(
                ShotDescription.model_validate_json((root / "shots" / "1" / "shot_description.json").read_text()).ff_desc,
                "replacement first endpoint",
            )

    async def test_replacement_with_only_new_prop_still_runs_registry_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, timeline, shots = _write_resume_fixture(root)
            replacement_shot = shots[1].model_copy(deep=True)
            extractor = MagicMock()
            extractor.extend_for_replacement = AsyncMock(return_value=(registry, []))
            extractor.bind_props_for_shot = AsyncMock(return_value=[])
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), extractor, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            pipeline._generate_and_register_portraits = AsyncMock(return_value={})
            await pipeline._apply_replacement_decision(
                0, Replacement(outcome="replacement", replacement_shot=replacement_shot, reason="new prop"),
                timeline, registry, [], {}, "style",
            )
            extractor.extend_for_replacement.assert_awaited_once()

    async def test_resume_restores_native_frame_events_from_existing_formal_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_resume_fixture(root)
            Image.new("RGB", (2, 2)).save(root / "shots" / "1" / "first_frame.png")
            production = MagicMock(working_dir=str(root))
            pipeline = SerialDemoPipeline(
                production, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            _characters, _registry, _portraits, cameras, timeline = pipeline._load_resume_artifacts()
            pipeline._validate_resume_timeline(timeline, cameras)
            pipeline._restore_native_frame_events(timeline)
            self.assertTrue(production.frame_events[0]["first_frame"].is_set())
            self.assertTrue(production.frame_events[1]["first_frame"].is_set())

    async def test_resume_observed_artifact_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            _write_resume_fixture(root)
            (root / "shots" / "0" / "video.mp4").unlink()
            with self.assertRaisesRegex(ValueError, "video or description mismatch"):
                await pipeline.run("", "", "style", resume=True)
            _write_resume_fixture(root)
            (root / "shots" / "0" / "observation_samples.json").write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifacts are missing or malformed"):
                await pipeline.run("", "", "style", resume=True)
            _write_resume_fixture(root)
            (root / "continuity" / "image_records.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing image record"):
                await pipeline.run("", "", "style", resume=True)

    async def test_resume_rejects_character_index_gaps_before_provider_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_resume_fixture(root)
            characters = json.loads((root / "characters.json").read_text(encoding="utf-8"))
            characters[0]["idx"] = 1
            (root / "characters.json").write_text(json.dumps(characters), encoding="utf-8")
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            with self.assertRaisesRegex(ValueError, "contiguous stable indices"):
                await pipeline.run("", "", "style", resume=True)

    async def test_resume_rejects_observed_position_after_planned_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _registry, timeline, _shots = _write_resume_fixture(root, observed_count=2)
            invalid = Timeline(positions=[
                PlannedPosition(shot_idx=0, plan=Timeline.initialize(_registry, _shots).position(0).plan),
                timeline.position(1),
            ])
            (root / "continuity" / "timeline.json").write_text(invalid.model_dump_json(), encoding="utf-8")
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            with self.assertRaisesRegex(ValueError, "observed prefix"):
                await pipeline.run("", "", "style", resume=True)

    async def test_resume_missing_replan_calls_replanner_once_from_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_resume_fixture(root)
            replanner = MagicMock()
            replanner.replan = AsyncMock(return_value=NoChange(outcome="no-change", reason="continue"))
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), replanner,
                max_reference_images=4,
            )
            characters, registry, portraits, cameras, timeline = pipeline._load_resume_artifacts()
            await pipeline._ensure_replan_after_observed(0, timeline, registry, characters, portraits, cameras, "style")
            replanner.replan.assert_awaited_once()
            self.assertIsInstance(replanner.replan.await_args.args[3], SampledFrame)
            self.assertTrue((root / "shots" / "0" / "replan.json").is_file())

    async def test_resume_rejects_saved_replan_three_way_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _registry, _timeline, shots = _write_resume_fixture(root)
            replacement = shots[1].model_copy(deep=True)
            replacement.ff_desc = "replacement"
            (root / "shots" / "0" / "replan.json").write_text(
                Replacement(outcome="replacement", replacement_shot=replacement, reason="saved").model_dump_json(),
                encoding="utf-8",
            )
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                max_reference_images=4,
            )
            characters, registry, portraits, cameras, timeline = pipeline._load_resume_artifacts()
            inconsistent = shots[1].model_copy(deep=True)
            inconsistent.ff_desc = "neither timeline nor replacement"
            (root / "shots" / "1" / "shot_description.json").write_text(inconsistent.model_dump_json(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no coherent pre-apply checkpoint"):
                await pipeline._ensure_replan_after_observed(0, timeline, registry, characters, portraits, cameras, "style")

    async def test_resume_reuses_existing_planned_frames_and_video_then_observes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_resume_fixture(root)
            shot_dir = root / "shots" / "1"
            Image.new("RGB", (2, 2)).save(shot_dir / "first_frame.png")
            (shot_dir / "video.mp4").write_bytes(b"existing planned video")
            portrait_paths = {}
            for view in ("front", "side", "back"):
                path = shot_dir / f"resume-portrait-{view}.png"
                Image.new("RGB", (2, 2)).save(path)
                portrait_paths[view] = str(path)
            context = FrameMaterialContext(
                shot_idx=1,
                frame_type="first",
                fixed_materials=[],
                field_lineage=[],
                history_sample_groups=[],
                portrait_candidates=[VisibleCharacterPortraitCandidates(
                    character_id="character-0",
                    candidates=[
                        IdentityPortraitCandidate(
                            image_id=f"resume-portrait-{view}",
                            asset_path=portrait_paths[view],
                            view=view,
                        )
                        for view in ("front", "side", "back")
                    ],
                )],
                required_slots=1,
                top_k=4,
            )
            (shot_dir / "selected_first_frame_images.json").write_text(
                ImageRetrievalResult(images=[RetrievedImage(
                    candidate_id="resume-portrait-front",
                    purpose="character identity portrait",
                )]).model_dump_json(),
                encoding="utf-8",
            )
            image_provider = MagicMock()
            image_provider.generate_single_image = AsyncMock()
            video_provider = MagicMock()
            video_provider.generate_single_video = AsyncMock()
            production = Script2VideoPipeline("unused", image_provider, video_provider, str(root))
            production.concatenate_rendered_shots = MagicMock(return_value=str(root / "final.mp4"))
            observer = MagicMock()
            observer.observe = AsyncMock(return_value=_observation(include_new=False))
            replanner = MagicMock()
            replanner.replan = AsyncMock(return_value=NoChange(outcome="no-change", reason="continue"))
            retriever = MagicMock()
            retriever.retrieve = AsyncMock()
            prompt_writer = MagicMock()
            prompt_writer.write = AsyncMock()
            pipeline = SerialDemoPipeline(
                production, MagicMock(), retriever, prompt_writer, observer, replanner,
                max_reference_images=4,
            )
            sample_path = shot_dir / "samples" / "resume.png"
            sample_path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (2, 2)).save(sample_path)
            samples = [
                SampledFrame(
                    shot_idx=1,
                    image_id=f"shot-1-sample-frame-{index}",
                    asset_path=str(sample_path),
                    time_seconds=float(index),
                )
                for index in range(11)
            ]
            with patch("pipelines.serial_demo_pipeline.build_frame_material_context", return_value=context), patch(
                "pipelines.serial_demo_pipeline.sample_video", return_value=samples,
            ):
                await pipeline.run("", "", "style", resume=True)
            image_provider.generate_single_image.assert_not_awaited()
            video_provider.generate_single_video.assert_not_awaited()
            retriever.retrieve.assert_not_awaited()
            prompt_writer.write.assert_not_awaited()
            observer.observe.assert_awaited_once()
            self.assertEqual(observer.observe.await_args.args[4], [{
                "board_slot": 0,
                "kind": "CHARACTER_IDENTITY",
                "entity_id": "character-0",
                "image_id": "resume-portrait-front",
            }])
            self.assertIsInstance(
                Timeline.model_validate_json((root / "continuity" / "timeline.json").read_text()).position(1),
                ObservedPosition,
            )


    async def test_continuity_selector_excludes_deterministic_paths_and_deduplicates_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("first", "observed", "portrait-front", "portrait-side", "portrait-back", "native"):
                path = root / f"{name}.png"
                path.write_bytes(name.encode())
                paths[name] = str(path)
            source = ObservedShotSource(shot_idx=1)
            lineage = [StateFieldLineage(
                entity_id="character-0",
                field="position",
                value="desk",
                source=source,
            )]
            history = HistoricalSampleGroup(
                source=source,
                lineage=lineage,
                samples=[HistoricalSampleCandidate(
                    image_id="shot-1-sample-frame-0",
                    asset_path=paths["observed"],
                    time_seconds=0.0,
                )],
            )
            context = FrameMaterialContext(
                shot_idx=2, frame_type="last",
                fixed_materials=[FixedFrameMaterial(
                    image_id="shot-2-first-frame", asset_path=paths["first"],
                    role="current_first_frame_starting_state",
                )],
                field_lineage=lineage,
                history_sample_groups=[history],
                portrait_candidates=[VisibleCharacterPortraitCandidates(character_id="character-0", candidates=[
                    IdentityPortraitCandidate(image_id="portrait-front", asset_path=paths["portrait-front"], view="front"),
                    IdentityPortraitCandidate(image_id="portrait-side", asset_path=paths["portrait-side"], view="side"),
                    IdentityPortraitCandidate(image_id="portrait-back", asset_path=paths["portrait-back"], view="back"),
                ])],
                required_slots=2, top_k=4,
            )
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(return_value=ImageRetrievalResult(images=[
                RetrievedImage(candidate_id="native-2-last-3", purpose="native ViMax frame candidate"),
            ]))
            writer = MagicMock()
            writer.write = AsyncMock(return_value="prompt")
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), retriever, writer,
                MagicMock(), MagicMock(), max_reference_images=4,
            )
            timeline_shot = _shot(2, 0, "medium")
            callback = pipeline._continuity_selector(
                timeline_shot, "last", context,
                "shots/2/selected_last_frame_images.json", "shots/2/last_frame_prompt.json",
            )

            references, prompt = await callback([
                (paths["first"], "current first duplicate"),
                (paths["observed"], "observed duplicate"),
                (paths["portrait-front"], "portrait duplicate"),
                (paths["native"], "first native description"),
                (paths["native"], "second native description"),
            ], timeline_shot.lf_desc)

            prepared_context = retriever.retrieve.await_args.args[0]
            self.assertEqual([(item.candidate_id, item.description) for item in prepared_context.native_candidates], [
                ("native-2-last-3", "first native description"),
            ])
            self.assertEqual(references, [(paths["native"], "native ViMax frame candidate")])
            # The writer receives the native descriptions; its final provider
            # prompt is persisted and returned without appending raw labels.
            writer.write.assert_awaited_once_with(
                timeline_shot.lf_desc, prepared_context, retriever.retrieve.return_value,
            )
            self.assertEqual(prompt, "prompt")
            self.assertEqual(
                json.loads((root / "shots/2/last_frame_prompt.json").read_text()),
                {"prompt": prompt},
            )

            with self.assertRaisesRegex(ValueError, "does not match Timeline shot 2"):
                await callback([(paths["native"], "native")], "stale disk description")
            self.assertEqual(retriever.retrieve.await_count, 1)
            self.assertEqual(writer.write.await_count, 1)

            wrong_context = context.model_copy(update={"frame_type": "first"})
            wrong_callback = pipeline._continuity_selector(
                timeline_shot, "last", wrong_context,
                "shots/2/ignored.json", "shots/2/ignored_prompt.json",
            )
            with self.assertRaisesRegex(ValueError, "Continuity context does not match"):
                await wrong_callback([(paths["native"], "native")], timeline_shot.lf_desc)
            self.assertEqual(retriever.retrieve.await_count, 1)
            self.assertEqual(writer.write.await_count, 1)

    def test_observation_reference_receipts_keep_portraits_and_only_first_selected_scene_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("fixed", "character-history", "scene-first", "scene-last", "front", "side", "back", "native"):
                path = root / f"{name}.png"
                path.write_bytes(name.encode())
                paths[name] = str(path)
            source_character = ObservedShotSource(shot_idx=0)
            source_scene = ObservedShotSource(shot_idx=1)
            character_lineage = StateFieldLineage(
                entity_id="character-0",
                field="position",
                value="desk",
                source=source_character,
            )
            scene_lineage = StateFieldLineage(
                entity_id="scene-0",
                field="lighting",
                value="warm",
                source=source_scene,
            )
            character_history = HistoricalSampleGroup(
                source=source_character,
                lineage=[character_lineage],
                samples=[HistoricalSampleCandidate(
                    image_id="character-history",
                    asset_path=paths["character-history"],
                    time_seconds=0.0,
                )],
            )
            scene_history = HistoricalSampleGroup(
                source=source_scene,
                lineage=[scene_lineage],
                samples=[
                    HistoricalSampleCandidate(
                        image_id="scene-first",
                        asset_path=paths["scene-first"],
                        time_seconds=0.0,
                    ),
                    HistoricalSampleCandidate(
                        image_id="scene-last",
                        asset_path=paths["scene-last"],
                        time_seconds=1.0,
                    ),
                ],
            )
            portraits = VisibleCharacterPortraitCandidates(character_id="character-0", candidates=[
                IdentityPortraitCandidate(image_id="portrait-front", asset_path=paths["front"], view="front"),
                IdentityPortraitCandidate(image_id="portrait-side", asset_path=paths["side"], view="side"),
                IdentityPortraitCandidate(image_id="portrait-back", asset_path=paths["back"], view="back"),
            ])
            fixed = FixedFrameMaterial(
                image_id="fixed", asset_path=paths["fixed"], role="current_first_frame_starting_state",
            )
            first_context = FrameMaterialContext(
                shot_idx=2,
                frame_type="first",
                fixed_materials=[fixed],
                field_lineage=[character_lineage, scene_lineage],
                history_sample_groups=[character_history, scene_history],
                portrait_candidates=[portraits],
                required_slots=2,
                top_k=8,
            )
            last_context = first_context.model_copy(update={"frame_type": "last"})
            pipeline = SerialDemoPipeline(
                MagicMock(working_dir=str(root)), MagicMock(), MagicMock(), MagicMock(),
                MagicMock(), MagicMock(), max_reference_images=8,
            )
            shot_description = _shot(2, 0, "medium")
            pipeline._save("shots/2/selected_first_frame_images.json", ImageRetrievalResult(images=[
                RetrievedImage(candidate_id="fixed", purpose="current shot first-frame starting state"),
                RetrievedImage(
                    candidate_id="character-history",
                    purpose=history_sample_purpose(
                        character_history,
                        character_history.samples[0],
                    ),
                ),
                RetrievedImage(
                    candidate_id="scene-first",
                    purpose=history_sample_purpose(scene_history, scene_history.samples[0]),
                ),
                RetrievedImage(candidate_id="portrait-front", purpose="character identity portrait"),
                RetrievedImage(candidate_id="native-2-first-0", purpose="native ViMax frame candidate"),
            ]).model_dump())
            pipeline._save("shots/2/selected_last_frame_images.json", ImageRetrievalResult(images=[
                RetrievedImage(
                    candidate_id="scene-last",
                    purpose=history_sample_purpose(scene_history, scene_history.samples[1]),
                ),
                RetrievedImage(candidate_id="portrait-side", purpose="character identity portrait"),
            ]).model_dump())

            slots = pipeline._observation_reference_slots(
                shot_description, "scene-0", first_context, last_context,
            )

            self.assertEqual(slots, [
                ("CHARACTER_IDENTITY", "character-0", "portrait-front", paths["front"]),
                ("CHARACTER_IDENTITY", "character-0", "portrait-side", paths["side"]),
                ("SCENE_HISTORY", "scene-0", "scene-first", paths["scene-first"]),
            ])

    async def test_real_roles_and_seams_close_a_three_shot_serial_loop(self):
        await self._exercise_three_shot_loop(preplanned=False)

    async def test_reviewed_plan_resumes_into_the_same_three_shot_serial_loop(self):
        await self._exercise_three_shot_loop(preplanned=True)

    async def _exercise_three_shot_loop(self, *, preplanned):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            image_provider = _ImageProvider(events)
            video_provider = _VideoProvider(root, events)
            production = Script2VideoPipeline("unused", image_provider, video_provider, str(root))
            shots = [_shot(0, 0, "small"), _shot(1, 1, "medium"), _shot(2, 1, "large", is_last=True)]
            replacement = shots[2].model_copy(deep=True)
            replacement.visual_desc = "replacement sentinel visual: Ivo lifts the lantern."
            replacement.ff_desc = "replacement-sentinel-first-target with Ivo and lantern"
            replacement.ff_vis_char_idxs = [0, 1]
            replacement.lf_desc = "replacement-sentinel-last-target with Ivo and lantern"
            replacement.lf_vis_char_idxs = [0, 1]
            replacement.motion_desc = "replacement-sentinel-motion"
            characters = [CharacterInScene(idx=0, identifier_in_scene="Mira", is_visible=True, static_features="coat")]
            briefs = [ShotBriefDescription(idx=item.idx, is_last=item.is_last, cam_idx=item.cam_idx, visual_desc=item.visual_desc, audio_desc="") for item in shots]
            cameras = [
                Camera(idx=0, active_shot_idxs=[0]),
                Camera(idx=1, active_shot_idxs=[1, 2], parent_cam_idx=0, parent_shot_idx=0),
            ]
            production.extract_characters = AsyncMock(return_value=characters)
            production.design_storyboard = AsyncMock(return_value=briefs)

            async def decompose_with_frame_events(brief, *_args, **_kwargs):
                shot = shots[brief.idx].model_copy(deep=True)
                production.frame_events[shot.idx] = {"first_frame": asyncio.Event()}
                if shot.variation_type in {"medium", "large"}:
                    production.frame_events[shot.idx]["last_frame"] = asyncio.Event()
                return shot

            production.decompose_visual_description_for_single_shot_brief_description = AsyncMock(
                side_effect=decompose_with_frame_events
            )
            production.construct_camera_tree = AsyncMock(return_value=cameras)
            async def create_child_camera_material(*_args, **_kwargs):
                path = root / "shots" / "1" / "new_camera_1.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"native-new-camera")
                return str(path)

            production._generate_child_camera_material = AsyncMock(
                side_effect=create_child_camera_material
            )
            native_frame_inputs = []
            original_generate_frames = production.generate_frames_for_single_camera

            async def capture_native_frame_inputs(*args, **kwargs):
                descriptions = kwargs["shot_descriptions"]
                current_idx = kwargs["serial_shot_idx"]
                native_frame_inputs.append((kwargs["serial_frame_type"], descriptions[current_idx]))
                return await original_generate_frames(*args, **kwargs)

            production.generate_frames_for_single_camera = AsyncMock(side_effect=capture_native_frame_inputs)

            entity_model = _SequencedChatModel([
                EntityRegistryDraft(
                    characters=[CharacterEntityDraft(character_idx=0, identity="detective", persona="careful", session="room", stable_appearance="coat")],
                    scenes=[SceneEntityDraft(name="room")],
                ),
                ReplacementRegistryExtension(
                    characters=[ReplacementCharacterDraft(
                        character_idx=1, name="Ivo", static_features="green coat",
                        identity="courier", persona="urgent", session="arriving", stable_appearance="green coat",
                    )],
                    props=[ReplacementPropDraft(name="lantern", initial_content={"position": "hand", "condition": "lit"})],
                ),
                ShotPropBinding(shot_idx=2, prop_entity_ids=["prop-0"]),
            ])
            retriever_model = _SequencedChatModel([
                FrameSelections(selections=[PortraitSelection(character_id="character-0", image_id="portrait-character-0-front")]),
                FrameSelections(selections=[PortraitSelection(character_id="character-0", image_id="portrait-character-0-front")]),
                FrameSelections(selections=[
                    PortraitSelection(character_id="character-0", image_id="portrait-character-0-front"),
                    PortraitSelection(character_id="character-1", image_id="portrait-character-1-front"),
                ]),
                FrameSelections(selections=[
                    PortraitSelection(character_id="character-0", image_id="portrait-character-0-front"),
                    PortraitSelection(character_id="character-1", image_id="portrait-character-1-front"),
                ]),
            ])
            prompt_model = _SequencedChatModel([
                FinalFramePrompt(prompt="static-0"),
                FinalFramePrompt(prompt="static-1"),
                FinalFramePrompt(prompt="replacement-sentinel-first-target"),
                FinalFramePrompt(prompt="replacement-sentinel-last-target"),
            ])
            observer_model = _SequencedChatModel([
                _observation(include_new=False),
                _observation(include_new=False),
                _observation(include_new=True),
            ], events=events, label="observer")
            replanner_model = _SequencedChatModel([
                {"outcome": "no-change", "reason": "shot one remains valid"},
                {"outcome": "replacement", "replacement_shot": replacement.model_dump(), "reason": "new arrival needs a rewrite"},
            ], events=events, label="replan")
            pipeline = SerialDemoPipeline(
                production,
                EntityExtractor(entity_model),
                ContinuityImageRetriever(retriever_model),
                FramePromptWriter(prompt_model),
                ShotObserver(observer_model),
                ShotReplanner(replanner_model),
                max_reference_images=4,
            )

            original_mark_observed = Timeline.mark_observed
            original_append = ImageRecordStore.append

            def mark_observed_logged(timeline, shot_idx, state):
                observed = original_mark_observed(timeline, shot_idx, state)
                events.append(f"mark-observed-{shot_idx}")
                return observed

            def append_logged(store, record):
                original_append(store, record)
                if record.kind == "shot_sample":
                    events.append(f"sample-record-{record.shot_idx}")

            with patch.object(Timeline, "mark_observed", mark_observed_logged), patch.object(ImageRecordStore, "append", append_logged):
                progress = []
                pipeline.progress = lambda stage, message, metadata: progress.append(stage)
                if preplanned:
                    await pipeline.plan("screenplay", "requirements")
                    self.assertEqual(image_provider.calls, [])
                    self.assertEqual(video_provider.calls, [])
                    production.frame_events.clear()  # Resume must restore readiness from disk.
                    final_video = await pipeline.resume_preplanned("style")
                else:
                    final_video = await pipeline.run("screenplay", "requirements", "style")
                self.assertEqual(progress.count("observing"), 3)
                self.assertEqual(progress.count("timeline_updated"), 3)
                self.assertEqual(progress.count("replanning"), 2)
                self.assertEqual(progress[-1], "render_done")

            self.assertTrue(Path(final_video).is_file())
            capture = cv2.VideoCapture(final_video)
            self.assertTrue(capture.isOpened())
            capture.release()
            self.assertEqual(entity_model.calls, 3, "replacement registers then binds its prop identity")
            self.assertEqual(
                [call.kwargs["user_requirement"] for call in production.decompose_visual_description_for_single_shot_brief_description.await_args_list],
                ["requirements", "requirements", "requirements"],
            )
            self.assertEqual(replanner_model.calls, 2)
            self.assertEqual(prompt_model.calls, 4)
            self.assertEqual(observer_model.calls, 3)
            self.assertFalse((root / "shots" / "0" / "last_frame.png").exists())
            self.assertTrue((root / "shots" / "1" / "last_frame.png").is_file())
            self.assertIn("replacement-sentinel-first-target", (root / "shots" / "2" / "first_frame_prompt.json").read_text())
            self.assertIn("replacement-sentinel-motion", video_provider.calls[-1]["prompt"])
            replacement_native_inputs = [
                description for _, description in native_frame_inputs
                if description.idx == 2
            ]
            self.assertEqual(len(replacement_native_inputs), 2)
            self.assertTrue(all(item.ff_desc == replacement.ff_desc for item in replacement_native_inputs))
            self.assertTrue(all(item.ff_vis_char_idxs == [0, 1] for item in replacement_native_inputs))
            selected_shot_two = (root / "shots" / "2" / "selected_first_frame_images.json").read_text()
            self.assertNotIn("shot-1-sample-frame-11", selected_shot_two)
            self.assertEqual([item["references"] for item in video_provider.calls[-3:]][0], [str(root / "shots" / "0" / "first_frame.png")])
            self.assertEqual(video_provider.calls[-2]["references"][0], str(root / "shots" / "1" / "first_frame.png"))
            self.assertEqual(video_provider.calls[-1]["references"][0], str(root / "shots" / "2" / "first_frame.png"))
            static_one_call = next(item for item in image_provider.calls if "static-1" in item["prompt"])
            replacement_first_call = next(item for item in image_provider.calls if "replacement-sentinel-first-target" in item["prompt"])
            self.assertNotIn(str(root / "shots" / "1" / "first_frame.png"), replacement_first_call["references"])
            self.assertEqual(replacement_first_call["references"], [
                pipeline.image_store.get("portrait-character-0-front").asset_path,
                pipeline.image_store.get("portrait-character-1-front").asset_path,
            ])
            prompt_text = "\n".join(item.get("text", "") for item in prompt_model.messages[2][1].content)
            self.assertIn("replacement-sentinel-first-target", prompt_text)
            observer_events = [index for index, event in enumerate(events) if event == "observer"]
            replan_events = [index for index, event in enumerate(events) if event == "replan"]
            first_shot_one_frame = next(index for index, event in enumerate(events) if "static-1" in event)
            first_shot_two_frame = next(index for index, event in enumerate(events) if "replacement-sentinel-first-target" in event)
            self.assertLess(observer_events[0], events.index("sample-record-0"))
            self.assertLess(events.index("sample-record-0"), events.index("mark-observed-0"))
            self.assertLess(events.index("mark-observed-0"), replan_events[0])
            self.assertLess(replan_events[0], first_shot_one_frame)
            self.assertLess(observer_events[1], events.index("sample-record-1"))
            self.assertLess(events.index("sample-record-1"), events.index("mark-observed-1"))
            self.assertLess(events.index("mark-observed-1"), replan_events[1])
            self.assertLess(replan_events[1], first_shot_two_frame)
            self.assertTrue((root / "continuity" / "timeline.json").is_file())
            self.assertTrue((root / "continuity" / "image_records.jsonl").is_file())
            self.assertTrue((root / "shots" / "0" / "observation_samples.json").is_file())
            self.assertTrue((root / "shots" / "1" / "observation_samples.json").is_file())
            self.assertTrue((root / "shots" / "2" / "observation_samples.json").is_file())
            for shot_idx in range(3):
                self.assertFalse((root / "shots" / str(shot_idx) / "observation_report.json").exists())
            first_replan_text = "\n".join(
                item.get("text", "") for item in replanner_model.messages[0][1].content
            )
            second_replan_text = "\n".join(
                item.get("text", "") for item in replanner_model.messages[1][1].content
            )
            self.assertNotIn("SCRIPT_DEVIATIONS", first_replan_text)
            self.assertNotIn("VISUAL_INTEGRITY_ISSUES", first_replan_text)
            self.assertNotIn("SCRIPT_DEVIATIONS", second_replan_text)
            self.assertNotIn("VISUAL_INTEGRITY_ISSUES", second_replan_text)
            self.assertTrue((root / "continuity" / "entity_registry.json").is_file())
            self.assertTrue((root / "characters.json").is_file())
            self.assertTrue((root / "character_portraits_registry.json").is_file())
            self.assertTrue((root / "storyboard.json").is_file())
            self.assertTrue((root / "camera_tree.json").is_file())
            self.assertIn("replacement-sentinel-first-target", (root / "shots" / "2" / "shot_description.json").read_text())
            registry_text = (root / "continuity" / "entity_registry.json").read_text()
            self.assertIn("character-1", registry_text)
            self.assertIn("prop-0", registry_text)
            self.assertIn("portrait-character-1-front", (root / "continuity" / "image_records.jsonl").read_text())
            timeline = Timeline.model_validate_json((root / "continuity" / "timeline.json").read_text())
            self.assertEqual([position.phase for position in timeline.positions], ["observed", "observed", "observed"])
            self.assertEqual(len(timeline.positions), 3)
            self.assertNotIn("__call__", Script2VideoPipeline.__dict__)


if __name__ == "__main__":
    unittest.main()
