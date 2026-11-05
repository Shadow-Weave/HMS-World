"""Contract tests for sampling and scoped observation with optional detail views."""

import base64
import copy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from pydantic import ValidationError
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.shot_observer import (
    ShotObserver,
    _OBSERVER_OUTPUT_CONTRACT,
    _SYSTEM_PROMPT,
    _validated_world_state,
)
from continuity.image_records import ImageRecord, ImageRecordStore
from continuity.shot_sampling import (
    SampledFrame,
    build_observation_board,
    image_records_for_observed_samples,
    last_observation_sample_from_manifest,
    observation_sample_metadata,
    read_observation_samples,
    sample_video,
    write_observation_samples,
)
from continuity.timeline import (
    CameraState, CharacterContent, CharacterEntity, EntityRegistry, PropContent,
    PropEntity, SceneEntity, WorldState, project_shot_plan,
)
from interfaces import ShotDescription


def shot() -> ShotDescription:
    return ShotDescription(
        idx=3,
        is_last=False,
        cam_idx=0,
        visual_desc="Mira studies the letter.",
        variation_type="medium",
        variation_reason="test",
        ff_desc="Mira at the desk.",
        ff_vis_char_idxs=[0],
        lf_desc="Mira raises the letter.",
        lf_vis_char_idxs=[],
        motion_desc="MOTION_SENTINEL",
        audio_desc="AUDIO_SENTINEL",
    )


def registry() -> EntityRegistry:
    return EntityRegistry(
        characters=[
            CharacterEntity(entity_id="character-0", name="Mira", identity="IDENTITY_SECRET", persona="PERSONA_SECRET", session="SESSION_SECRET", stable_appearance="coat", identity_image_ids=["mira-front", "mira-side", "mira-back"]),
            CharacterEntity(entity_id="character-1", name="Noah", identity="other", persona="other", session="other", stable_appearance="sweater"),
        ],
        props=[PropEntity(entity_id="prop-0", name="letter")],
        scenes=[SceneEntity(entity_id="scene-0", name="room")],
    )


def observation(*, mira_position: str) -> WorldState:
    return WorldState(
        characters={"character-0": CharacterContent(position=mira_position)},
        camera={"shot_scale": "medium", "angle": "eye level", "composition": "desk"},
    )


def plan_for(current_shot: ShotDescription):
    return project_shot_plan(registry(), None, current_shot, prop_entity_ids=[])


class SequencedChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []
        self.calls = 0

    def bind_tools(self, tools, *, strict):
        self.tools = tools
        self.strict = strict
        return self

    def __or__(self, parser):
        self.parser = parser
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        response = self.responses.pop(0)
        return response if isinstance(response, AIMessage) else AIMessage(content=response.model_dump_json(exclude_unset=True))


class SamplingAndObserverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_observer_prompt_defines_sparse_transition_logic_and_observation_domains(self):
        prompt = " ".join(_SYSTEM_PROMPT.split())
        for section in ("ROLE", "INPUTS", "STATE TRANSITION LOGIC", "OBSERVATION DOMAINS", "OUTPUT"):
            self.assertIn(section, prompt)
        self.assertIn("Read one completed take as it unfolds", prompt)
        self.assertIn("The Timeline, not you, owns accumulated state", prompt)
        self.assertIn("Omit a field when evidence is insufficient", prompt)
        self.assertIn("Use null only when", prompt)
        self.assertIn("Do not output IDs outside PARTICIPANT_SCOPE", prompt)
        self.assertNotIn("PLANNED_MOTION_DESC", prompt)
        self.assertIn("TEMPORAL_BOARD: Observed sequence of actions and state transitions", prompt)
        for scope in (
            "Performance, body pose, facing vector, and gaze direction",
            "Contact, holding state, physical support, displacement, and quantity",
            "Changes in costume, hair, or physical condition",
            "Set dressing modifications, scene state, and visible marks",
            "Framing, angle, and screen-space relations",
        ):
            self.assertIn(scope, prompt)
        self.assertIn("strict sparse WorldState JSON", prompt)
        self.assertNotIn("Use `expression`", prompt)
        self.assertNotIn("script_deviations", prompt)
        self.assertNotIn("visual_integrity_issues", prompt)

    def test_observer_output_contract_is_direct_and_describes_sparse_fields(self):
        self.assertNotIn('"title"', _OBSERVER_OUTPUT_CONTRACT)
        self.assertNotIn('"default"', _OBSERVER_OUTPUT_CONTRACT)
        self.assertNotIn("anyOf", _OBSERVER_OUTPUT_CONTRACT)
        self.assertNotIn('"foo"', _OBSERVER_OUTPUT_CONTRACT)
        for field in ("characters", "props", "scenes", "camera"):
            self.assertIn(f"- {field}:", _OBSERVER_OUTPUT_CONTRACT)
        for field in CharacterContent.model_fields:
            self.assertIn(field, _OBSERVER_OUTPUT_CONTRACT)
        for field in PropContent.model_fields:
            self.assertIn(field, _OBSERVER_OUTPUT_CONTRACT)
        self.assertIn("Omit unknown fields", _OBSERVER_OUTPUT_CONTRACT)
        self.assertIn("explicit null field value", _OBSERVER_OUTPUT_CONTRACT)

    def test_board_sampler_includes_metadata_edges_and_only_metadata_carries_media(self):
        video_path = self.root / "video.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (16, 16))
        for index in range(6):
            writer.write(np.full((16, 16, 3), index * 30, dtype=np.uint8))
        writer.release()

        samples = sample_video(video_path, 3, self.root / "shots", 8)

        self.assertEqual([sample.image_id for sample in samples], [
            "shot-3-sample-frame-0", "shot-3-sample-frame-1", "shot-3-sample-frame-3", "shot-3-sample-frame-5",
        ])
        self.assertEqual(samples[0].time_seconds, 0.0)
        self.assertAlmostEqual(samples[-1].time_seconds, 2.5)
        self.assertTrue(all(Path(sample.asset_path).is_file() for sample in samples))
        metadata = observation_sample_metadata(samples)
        self.assertEqual(metadata[-1].model_dump(), {"index": 3, "time_seconds": 2.5, "image_id": "shot-3-sample-frame-5"})
        observer_path = self.root / "shots" / "3" / "observation_samples.json"
        write_observation_samples(observer_path, samples)
        self.assertEqual(json.loads(observer_path.read_text())[-1]["image_id"], "shot-3-sample-frame-5")
        records = image_records_for_observed_samples(samples, 3)
        self.assertEqual(records[0].kind, "shot_sample")
        self.assertEqual(records[0].summary, "actual sampled frame from shot 3 at 0s")
        with self.assertRaisesRegex(ValueError, "leave at least two"):
            sample_video(video_path, 3, self.root / "shots", 11)

    def test_sampler_fails_closed_when_metadata_target_cannot_be_decoded(self):
        class EarlyEofCapture:
            def __init__(self):
                self.frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)]

            def isOpened(self):
                return True

            def get(self, property_id):
                if property_id == cv2.CAP_PROP_FPS:
                    return 2.0
                if property_id == cv2.CAP_PROP_FRAME_COUNT:
                    return 9999
                return 0

            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)

            def release(self):
                pass

        with patch("continuity.shot_sampling.cv2.VideoCapture", return_value=EarlyEofCapture()):
            with self.assertRaisesRegex(ValueError, "metadata-planned"):
                sample_video(self.root / "lied.avi", 9, self.root / "shots", 10)

    def test_observation_manifest_rejects_invalid_identity_and_time_order(self):
        manifest = self.root / "observation_samples.json"
        valid = [
            {"index": 0, "time_seconds": 0.0, "image_id": "sample-0"},
            {"index": 1, "time_seconds": 1.0, "image_id": "sample-1"},
        ]
        for payload, message in (
            ([{**valid[0], "index": -1}], "Cannot read valid observation samples"),
            ([{**valid[0], "time_seconds": -1.0}], "Cannot read valid observation samples"),
            ([valid[0], {**valid[1], "image_id": "sample-0"}], "unique image IDs"),
            ([valid[0], {**valid[1], "time_seconds": 0.0}], "strictly increasing"),
        ):
            with self.subTest(payload=payload):
                manifest.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    read_observation_samples(manifest)

    def test_replan_tail_is_recovered_from_observation_samples_and_formal_sample_record(self):
        sample_path = self.root / "tail.png"
        sample_path.write_bytes(b"tail")
        sample = SampledFrame(shot_idx=4, image_id="shot-4-sample-frame-8", asset_path=str(sample_path), time_seconds=2.0)
        store = ImageRecordStore(self.root / "records.jsonl")
        for record in image_records_for_observed_samples([sample], 4):
            store.append(record)
        metadata_path = self.root / "observation_samples.json"
        write_observation_samples(metadata_path, [sample])

        recovered = last_observation_sample_from_manifest(metadata_path, store, 4)

        self.assertEqual(recovered, sample)
        with self.assertRaisesRegex(ValueError, "formal image record"):
            store = ImageRecordStore(self.root / "empty-records.jsonl")
            last_observation_sample_from_manifest(metadata_path, store, 4)

    def test_tail_helper_rejects_reordered_or_malformed_observation_manifest(self):
        paths = [self.root / "sample-first.png", self.root / "sample-last.png"]
        for path in paths:
            path.write_bytes(b"sample")
        samples = [
            SampledFrame(shot_idx=6, image_id="shot-6-sample-frame-0", asset_path=str(paths[0]), time_seconds=0.0),
            SampledFrame(shot_idx=6, image_id="shot-6-sample-frame-2", asset_path=str(paths[1]), time_seconds=1.0),
        ]
        store = ImageRecordStore(self.root / "two-sample-records.jsonl")
        for record in image_records_for_observed_samples(samples, 6):
            store.append(record)
        metadata_path = self.root / "two-sample-observation-samples.json"
        write_observation_samples(metadata_path, samples)

        reordered = json.loads(metadata_path.read_text())
        reordered[0], reordered[1] = reordered[1], reordered[0]
        metadata_path.write_text(json.dumps(reordered))
        with self.assertRaisesRegex(ValueError, "non-empty contiguous order"):
            last_observation_sample_from_manifest(metadata_path, store, 6)

        reordered[0], reordered[1] = reordered[1], reordered[0]
        reordered[1]["unexpected"] = "not allowed"
        metadata_path.write_text(json.dumps(reordered))
        with self.assertRaisesRegex(ValueError, "Cannot read valid observation samples"):
            last_observation_sample_from_manifest(metadata_path, store, 6)

    async def test_observer_calls_once_with_one_board_and_returns_scoped_update(self):
        reference = self.root / "mira-front.png"
        reference.write_bytes(b"reference")
        samples = []
        for index in range(11):
            path = self.root / f"sample-{index}.png"
            path.write_bytes(f"sample-{index}".encode())
            samples.append(SampledFrame(shot_idx=3, image_id=f"sample-{index}", asset_path=str(path), time_seconds=float(index)))
        board = self.root / "observation-board.png"
        with patch("continuity.shot_sampling.cv2.imread", return_value=np.zeros((16, 16, 3), dtype=np.uint8)):
            build_observation_board([("CHAR character-0", reference)], samples, board)
        model = SequencedChatModel([observation(mira_position="table")])
        reference_map = [{"board_slot": 0, "kind": "CHARACTER_IDENTITY", "entity_id": "character-0", "image_id": "mira-front"}]

        result = await ShotObserver(model).observe(plan_for(shot()), board, samples, registry(), reference_map)

        self.assertEqual(model.calls, 1)
        self.assertIs(model.strict, True)
        self.assertEqual(result.characters["character-0"].position, "table")
        self.assertIsInstance(result, WorldState)
        content = model.messages[0][1].content
        images = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        text = "\n".join(item.get("text", "") for item in content)
        self.assertNotIn("character-1", text)
        self.assertNotIn("prop-0", text)
        self.assertIn("scene-0", text)
        self.assertNotIn("PLANNED_MOTION_DESC", text)
        self.assertNotIn("MOTION_SENTINEL", text)
        self.assertNotIn("Mira studies the letter.", text)
        self.assertNotIn("AUDIO_SENTINEL", text)
        self.assertIn("T0 through T10", text)
        self.assertNotIn("REFERENCE_SLOT_MAP", text)
        self.assertNotIn("TEMPORAL_SLOT_MAP", text)
        self.assertNotIn("board_slot", text)
        self.assertNotIn("temporal_index", text)
        self.assertNotIn("image_id", text)
        self.assertNotIn("sample-10", text)
        self.assertNotIn("mira-front", text)
        self.assertNotIn("TERMINAL_SAMPLE", text)
        self.assertNotIn("IDENTITY_SECRET", text)
        self.assertNotIn("PERSONA_SECRET", text)
        self.assertNotIn("SESSION_SECRET", text)
        self.assertNotIn(str(self.root), text)

    async def test_observer_can_inspect_requested_original_frames_before_returning_one_update(self):
        board = self.root / "board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        samples = []
        for index in range(11):
            path = self.root / f"detail-{index}.png"
            cv2.imwrite(str(path), np.full((72, 128, 3), index * 10, dtype=np.uint8))
            samples.append(SampledFrame(shot_idx=3, image_id=f"sample-{index}", asset_path=str(path), time_seconds=float(index)))
        request = AIMessage(content="", tool_calls=[{
            "name": "inspect_samples", "args": {"frames": [10, 9]}, "id": "detail-call",
        }])
        model = SequencedChatModel([request, observation(mira_position="desk")])
        reference_map = [{"board_slot": 0, "kind": "CHARACTER_IDENTITY", "entity_id": "character-0", "image_id": "mira-front"}]

        result = await ShotObserver(model).observe(plan_for(shot()), board, samples, registry(), reference_map)

        self.assertEqual(model.calls, 2)
        self.assertEqual(result.characters["character-0"].position, "desk")
        messages = model.messages[1]
        self.assertIsInstance(messages[2], AIMessage)
        self.assertIsInstance(messages[3], ToolMessage)
        self.assertEqual(messages[3].tool_call_id, "detail-call")
        self.assertIsInstance(messages[4], HumanMessage)
        detail = messages[4].content
        self.assertIn("T9 at 9s", detail[0]["text"])
        self.assertIn("T10 at 10s", detail[2]["text"])
        self.assertEqual(detail[3]["image_url"]["detail"], "high")
        self.assertEqual(base64.b64decode(detail[3]["image_url"]["url"].split(",", 1)[1]), Path(samples[-1].asset_path).read_bytes())

    async def test_observer_limits_inspection_to_two_rounds_and_then_requests_terminal_state(self):
        board = self.root / "board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        samples = [SampledFrame(shot_idx=3, image_id=f"sample-{index}", asset_path=str(board), time_seconds=float(index)) for index in range(12)]
        requests = [AIMessage(content="", tool_calls=[{
            "name": "inspect_samples", "args": {"frames": [11]}, "id": f"detail-{index}",
        }]) for index in range(2)]
        model = SequencedChatModel([*requests, WorldState()])

        result = await ShotObserver(model).observe(plan_for(shot()), board, samples, registry(), [])

        self.assertEqual(model.calls, 3)
        self.assertEqual(result, WorldState())
        self.assertIn("Return the sparse terminal WorldState", model.messages[2][-1].content)

    async def test_authorized_validation_feedback_is_bound_to_one_shot_and_video(self):
        board = self.root / "board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        (self.root / "video.mp4").write_bytes(b"approved-take")
        digest = hashlib.sha256(b"approved-take").hexdigest()
        samples = [SampledFrame(shot_idx=3, image_id=f"sample-{i}", asset_path=str(board),
                                time_seconds=float(i)) for i in range(12)]
        invalid = WorldState(characters={"Mira": CharacterContent(position="desk")})
        for approved_shot, approved_video in [(2, digest), (3, "different-video"), (3, digest)]:
            with self.subTest(shot=approved_shot, video=approved_video):
                model = SequencedChatModel([invalid, WorldState()])
                observer = ShotObserver(model)
                observer.validation_feedback_approvals = [{"shot_idx": approved_shot,
                    "video_sha256": approved_video, "reason": "explicit approval"}]
                receipts = []
                observer.validation_feedback_receipt = receipts.append
                if approved_shot != 3 or approved_video != digest:
                    with self.assertRaisesRegex(ValueError, "outside participant scope"):
                        await observer.observe(plan_for(shot()), board, samples, registry(), [])
                    self.assertEqual(model.calls, 1)
                    self.assertEqual(receipts, [])
                    continue
                await observer.observe(plan_for(shot()), board, samples, registry(), [])
                self.assertEqual(model.calls, 2)
                self.assertEqual(len(receipts), 1)
                self.assertEqual(observer.validation_feedback_approvals, [])
                self.assertEqual(model.messages[0], model.messages[1][:2])
                self.assertIn("outside participant scope", model.messages[1][-1].content)
                model.responses.append(invalid)
                with self.assertRaisesRegex(ValueError, "outside participant scope"):
                    await observer.observe(plan_for(shot()), board, samples, registry(), [])
                self.assertEqual(len(receipts), 1)

    async def test_authorized_feedback_still_rejects_invalid_correction(self):
        board = self.root / "board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        (self.root / "video.mp4").write_bytes(b"approved-take")
        samples = [SampledFrame(shot_idx=3, image_id=f"sample-{i}", asset_path=str(board),
                                time_seconds=float(i)) for i in range(12)]
        invalid = WorldState(characters={"Mira": CharacterContent(position="desk")})
        model = SequencedChatModel([invalid, invalid])
        observer = ShotObserver(model)
        observer.validation_feedback_approvals = [{"shot_idx": 3,
            "video_sha256": hashlib.sha256(b"approved-take").hexdigest(), "reason": "explicit approval"}]
        receipts = []
        observer.validation_feedback_receipt = receipts.append
        with self.assertRaisesRegex(ValueError, "outside participant scope"):
            await observer.observe(plan_for(shot()), board, samples, registry(), [])
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(receipts), 1)

    async def test_malformed_textual_tool_call_gets_one_format_retry_without_repairing_its_state(self):
        board = self.root / "board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        samples = [SampledFrame(shot_idx=3, image_id=f"sample-{index}", asset_path=str(board),
                                time_seconds=float(index)) for index in range(12)]
        invalid = AIMessage(content='<tool_call>{"name":"inspect_samples","arguments":{"frames":[T10]}}</tool_call>')
        model = SequencedChatModel([invalid, observation(mira_position="desk")])
        result = await ShotObserver(model).observe(plan_for(shot()), board, samples, registry(), [])
        self.assertEqual(result.characters["character-0"].position, "desk")
        self.assertEqual(model.calls, 2)
        self.assertIn('integer frame numbers', model.messages[1][-1].content)

        from langchain_core.exceptions import OutputParserException
        repeated = SequencedChatModel([invalid, invalid])
        with self.assertRaises(OutputParserException):
            await ShotObserver(repeated).observe(plan_for(shot()), board, samples, registry(), [])
        self.assertEqual(repeated.calls, 2)

    async def test_observer_inspection_cannot_read_an_image_outside_this_take(self):
        board = self.root / "board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        samples = [SampledFrame(shot_idx=3, image_id=f"sample-{index}", asset_path=str(board), time_seconds=float(index)) for index in range(12)]
        model = SequencedChatModel([AIMessage(content="", tool_calls=[{
            "name": "inspect_samples", "args": {"frames": [12]}, "id": "outside",
        }])])

        with self.assertRaisesRegex(ValueError, "T-frame numbers from this take"):
            await ShotObserver(model).observe(plan_for(shot()), board, samples, registry(), [])
        self.assertEqual(model.calls, 1)

    async def test_sparse_output_accepts_omitted_fields_but_forbids_unknown_schema_fields(self):
        board = self.root / "observation-board.png"
        cv2.imwrite(str(board), np.zeros((32, 32, 3), dtype=np.uint8))
        samples = [
            SampledFrame(shot_idx=3, image_id=f"sample-{index}", asset_path=str(board), time_seconds=float(index))
            for index in range(12)
        ]
        model = SequencedChatModel([WorldState()])
        # No portrait is required when the target shot exposes no character.
        hidden = shot()
        hidden.ff_vis_char_idxs = []
        hidden.lf_vis_char_idxs = []
        result = await ShotObserver(model).observe(plan_for(hidden), board, samples, registry(), [])
        self.assertEqual(result.characters, {})
        with self.assertRaises(ValidationError):
            WorldState.model_validate({"characters": {}, "props": {}, "scenes": {}, "camera": None, "changed": True})

    def test_observer_accepts_explicit_absence_in_sparse_state(self):
        current = WorldState(characters={"character-0": CharacterContent(position=None)})

        validated = _validated_world_state(current, plan_for(shot()))

        self.assertEqual(validated, current)

    def test_observer_rejects_held_props_outside_scope_or_duplicates(self):
        prop_plan = project_shot_plan(registry(), None, shot(), prop_entity_ids=["prop-0"])
        unregistered = WorldState(characters={
            "character-0": CharacterContent(holding=["prop-99"]),
        })
        with self.assertRaisesRegex(ValueError, "outside this shot's participant scope"):
            _validated_world_state(unregistered, prop_plan)

        duplicated = WorldState(characters={
            "character-0": CharacterContent(holding=["prop-0", "prop-0"]),
        })
        with self.assertRaisesRegex(ValueError, "duplicate held prop IDs"):
            _validated_world_state(duplicated, prop_plan)

    async def test_observer_rejects_wrong_shot_or_duplicate_sample_ids_before_model_call(self):
        sample_path = self.root / "sample-check.png"
        sample_path.write_bytes(b"sample")
        model = SequencedChatModel([])
        observer = ShotObserver(model)
        wrong_shot = SampledFrame(shot_idx=99, image_id="stable-id", asset_path=str(sample_path), time_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "belong to shot 3"):
            await observer.observe(plan_for(shot()), self.root / "missing-board.png", [wrong_shot], registry(), [])
        duplicate_a = SampledFrame(shot_idx=3, image_id="stable-id", asset_path=str(sample_path), time_seconds=0.0)
        duplicate_b = SampledFrame(shot_idx=3, image_id="stable-id", asset_path=str(sample_path), time_seconds=1.0)
        with self.assertRaisesRegex(ValueError, "unique stable image IDs"):
            await observer.observe(plan_for(shot()), self.root / "missing-board.png", [duplicate_a, duplicate_b], registry(), [])
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
