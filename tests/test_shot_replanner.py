"""Contract tests for §20.8 single-next-shot replanning."""

import base64
import inspect
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from agents.shot_replanner import NoChange, Replacement, ShotReplanner, _SYSTEM_PROMPT
from continuity.shot_sampling import SampledFrame
from continuity.timeline import CharacterContent, CharacterEntity, EntityRegistry, PropEntity, WorldState
from interfaces import Camera, ShotDescription


def next_shot() -> ShotDescription:
    return ShotDescription(
        idx=1,
        is_last=False,
        cam_idx=4,
        visual_desc="Mira enters the room.",
        variation_type="small",
        variation_reason="test",
        ff_desc="Mira opens the door.",
        ff_vis_char_idxs=[0],
        lf_desc="Mira steps inside.",
        lf_vis_char_idxs=[0],
        motion_desc="Mira crosses the threshold.",
        audio_desc="door creaks",
    )


def registry() -> EntityRegistry:
    return EntityRegistry(
        characters=[CharacterEntity(entity_id="character-0", name="Mira", identity="IDENTITY_SECRET", persona="PERSONA_SECRET", session="SESSION_SECRET", stable_appearance="coat")],
        props=[PropEntity(entity_id="prop-0", name="letter")],
    )


class RecordingChatModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.messages = None

    def __or__(self, parser):
        self.parser = parser
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self.response


class ShotReplannerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.sample_path = self.root / "tail.png"
        self.sample_path.write_bytes(b"actual-tail")
        self.sample = SampledFrame(shot_idx=0, image_id="sample-secret", asset_path=str(self.sample_path), time_seconds=1.5)
        self.camera = Camera(idx=4, active_shot_idxs=[1])
        self.state = WorldState(characters={"character-0": CharacterContent(position="hall")})

    def tearDown(self):
        self.tempdir.cleanup()

    async def test_no_change_uses_one_tail_image_and_whitelisted_text_only(self):
        model = RecordingChatModel({"outcome": "no-change", "reason": "The hall state still supports the entrance."})
        source_shot = next_shot()
        before = source_shot.model_dump()

        result = await ShotReplanner(model).replan(
            "room", "Wide shot. Mira leaves the hall after setting down the letter.",
            self.state, self.sample, source_shot, self.camera, registry(),
        )

        self.assertIsInstance(result, NoChange)
        self.assertEqual(model.calls, 1)
        self.assertEqual(source_shot.model_dump(), before)
        content = model.messages[1].content
        images = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        self.assertEqual(images, ["data:image/png;base64," + base64.b64encode(b"actual-tail").decode()])
        text = "\n".join(item.get("text", "") for item in content)
        self.assertIn("Previous shot's actual terminal sampled frame", text)
        self.assertNotIn("tail-state evidence", text)
        self.assertIn(
            "<PREVIOUS_SHOT_VISUAL_DESC>\nWide shot. Mira leaves the hall after setting down the letter.\n</PREVIOUS_SHOT_VISUAL_DESC>",
            text,
        )
        self.assertNotIn("PREVIOUS_SHOT_INTENT", text)
        self.assertNotIn("NEXT_SHOT_INTENT", text)
        self.assertIn("Mira enters the room.", text)
        self.assertNotIn("SCRIPT_DEVIATIONS", text)
        self.assertNotIn("VISUAL_INTEGRITY_ISSUES", text)
        self.assertIn("character-0", text)
        self.assertIn("prop-0", text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("sample-secret", text)
        self.assertNotIn("1.5", text)
        self.assertNotIn("IDENTITY_SECRET", text)
        self.assertNotIn("PERSONA_SECRET", text)
        self.assertNotIn("SESSION_SECRET", text)

    def test_replanner_prompt_uses_existing_evidence_for_directorial_cut_decision(self):
        prompt = " ".join(_SYSTEM_PROMPT.split())
        self.assertIn("PREVIOUS_SHOT_VISUAL_DESC", prompt)
        self.assertIn("PREVIOUS_FINAL_STATE", prompt)
        self.assertIn("Judge the cut, not whether the take reproduced its plan exactly", prompt)
        self.assertIn("Eyeline match and match on action apply only when the planned cut claims", prompt)
        self.assertIn("Off-screen action, ellipsis, reframing, and montage may bridge", prompt)
        self.assertIn("Return replacement only when concrete evidence shows", prompt)
        self.assertIn("make the smallest coherent adjustment", prompt)
        self.assertIn("within the supplied Camera", prompt)
        self.assertIn("generation defects rather than story events to canonize", prompt)

    async def test_complete_replacement_allows_new_character_indices_without_timeline_write(self):
        replacement = next_shot().model_dump()
        replacement.update({
            "visual_desc": "A new witness joins Mira in the room.",
            "ff_vis_char_idxs": [0, 9],
            "lf_vis_char_idxs": [0, 9],
            "motion_desc": "Mira and the witness exchange the letter before the cut.",
            "audio_desc": "quiet room tone",
        })
        model = RecordingChatModel({"outcome": "replacement", "replacement_shot": replacement, "reason": "The observed letter requires the witness to be present."})

        result = await ShotReplanner(model).replan(
            "room", "Mira leaves the hall.", self.state, self.sample,
            next_shot(), self.camera, registry(),
        )

        self.assertIsInstance(result, Replacement)
        self.assertEqual(result.replacement_shot.ff_vis_char_idxs, [0, 9])
        self.assertEqual(model.calls, 1)
        source = inspect.getsource(ShotReplanner)
        self.assertNotIn("Timeline", source)
        self.assertNotIn("apply_replacement", source)

    async def test_identical_replacement_is_no_change_without_rewriting_the_shot(self):
        shot = next_shot()
        before = shot.model_dump()
        model = RecordingChatModel({"outcome": "replacement", "replacement_shot": before,
                                    "reason": "The cut needs a replacement."})
        result = await ShotReplanner(model).replan(
            "room", "Mira leaves the hall.", self.state, self.sample,
            shot, self.camera, registry(),
        )
        self.assertIsInstance(result, NoChange)
        self.assertEqual(shot.model_dump(), before)
        self.assertEqual(model.calls, 1)

    async def test_unused_last_frame_fields_do_not_replace_a_small_variation(self):
        shot = next_shot()
        proposed = shot.model_dump()
        proposed.update(lf_desc="an unused tail description", lf_vis_char_idxs=[0, 9])
        model = RecordingChatModel({"outcome": "replacement", "replacement_shot": proposed,
                                    "reason": "The cut needs a replacement."})
        result = await ShotReplanner(model).replan(
            "room", "Mira leaves the hall.", self.state, self.sample,
            shot, self.camera, registry(),
        )
        self.assertIsInstance(result, NoChange)
        self.assertEqual(shot.lf_desc, "Mira steps inside.")

    async def test_invalid_replacement_boundaries_and_extra_duration_fail(self):
        base = next_shot().model_dump()
        wrong_idx = dict(base, idx=2)
        model = RecordingChatModel({"outcome": "replacement", "replacement_shot": wrong_idx, "reason": "needed"})
        with self.assertRaisesRegex(ValueError, "idx must match"):
            await ShotReplanner(model).replan("room", "visual desc", self.state, self.sample, next_shot(), self.camera, registry())

        with_duration = dict(base, duration=3)
        duration_model = RecordingChatModel({"outcome": "replacement", "replacement_shot": with_duration, "reason": "needed"})
        with self.assertRaises(ValidationError):
            await ShotReplanner(duration_model).replan("room", "visual desc", self.state, self.sample, next_shot(), self.camera, registry())

        no_change_extra_model = RecordingChatModel({"outcome": "no-change", "reason": "needed", "patch": {}})
        with self.assertRaises(ValidationError):
            await ShotReplanner(no_change_extra_model).replan("room", "visual desc", self.state, self.sample, next_shot(), self.camera, registry())
        replacement_extra_model = RecordingChatModel({"outcome": "replacement", "replacement_shot": base, "reason": "needed", "patch": {}})
        with self.assertRaises(ValidationError):
            await ShotReplanner(replacement_extra_model).replan("room", "visual desc", self.state, self.sample, next_shot(), self.camera, registry())

        incomplete = dict(base)
        incomplete.pop("lf_vis_char_idxs")
        incomplete_model = RecordingChatModel({"outcome": "replacement", "replacement_shot": incomplete, "reason": "needed"})
        with self.assertRaisesRegex(ValueError, "complete ShotDescription"):
            await ShotReplanner(incomplete_model).replan("room", "visual desc", self.state, self.sample, next_shot(), self.camera, registry())

        wrong_sample = SampledFrame(shot_idx=4, image_id="wrong", asset_path=str(self.sample_path), time_seconds=0.0)
        no_call_model = RecordingChatModel({"outcome": "no-change", "reason": "unused"})
        with self.assertRaisesRegex(ValueError, "does not precede"):
            await ShotReplanner(no_call_model).replan("room", "visual desc", self.state, wrong_sample, next_shot(), self.camera, registry())
        self.assertEqual(no_call_model.calls, 0)

    async def test_camera_mismatches_fail_before_model_call(self):
        wrong_idx_camera = Camera(idx=99, active_shot_idxs=[1])
        wrong_active_camera = Camera(idx=4, active_shot_idxs=[8])
        for camera, message in ((wrong_idx_camera, "idx must match"), (wrong_active_camera, "must list")):
            model = RecordingChatModel({"outcome": "no-change", "reason": "unused"})
            with self.assertRaisesRegex(ValueError, message):
                await ShotReplanner(model).replan("room", "visual desc", self.state, self.sample, next_shot(), camera, registry())
            self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
