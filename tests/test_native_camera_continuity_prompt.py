"""The selected camera candidate's limitations reach the image provider."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from agents.continuity_image_retriever import ImageRetrievalResult, RetrievedImage
from agents.frame_prompt_writer import FinalFramePrompt, FramePromptWriter
from continuity.frame_material_context import FrameMaterialContext
from interfaces import Camera, ImageOutput, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline
from pipelines.serial_demo_pipeline import SerialDemoPipeline


class _PromptModel:
    def __or__(self, parser):
        return self

    async def ainvoke(self, messages):
        # Exercise the gap: the writer can omit a candidate's known limitations.
        return FinalFramePrompt(prompt="Render the target frame.")


class _ImageProvider:
    def __init__(self):
        self.calls = []

    async def generate_single_image(self, prompt, reference_image_paths, size):
        self.calls.append((prompt, reference_image_paths))
        return ImageOutput("pil", "png", Image.new("RGB", (2, 2)))


class NativeCameraContinuityPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_selected_camera_candidate_adds_missing_detail_to_provider_prompt(self):
        for selected in (True, False):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "shots" / "0").mkdir(parents=True)
                (root / "shots" / "1").mkdir()
                parent = root / "shots" / "0" / "first_frame.png"
                candidate = root / "shots" / "1" / "new_camera_1.png"
                Image.new("RGB", (2, 2)).save(parent)
                Image.new("RGB", (2, 2)).save(candidate)
                provider = _ImageProvider()
                production = Script2VideoPipeline(MagicMock(), provider, MagicMock(), directory)
                production.frame_events[0] = {"first_frame": asyncio.Event()}
                production.frame_events[0]["first_frame"].set()
                production.frame_events[1] = {"first_frame": asyncio.Event()}
                production._generate_child_camera_material = AsyncMock(return_value=str(candidate))
                shot0 = ShotDescription(
                    idx=0, is_last=False, cam_idx=0, visual_desc="room", variation_type="small",
                    variation_reason="test", ff_desc="room", ff_vis_char_idxs=[], lf_desc="room",
                    lf_vis_char_idxs=[], motion_desc="", audio_desc="",
                )
                shot1 = shot0.model_copy(update={
                    "idx": 1, "cam_idx": 1,
                    "ff_desc": "A new flush-mounted wall safe appears beside the bookcase, away from the exit.",
                })
                retriever = MagicMock()
                retriever.retrieve = AsyncMock(return_value=ImageRetrievalResult(images=[
                    RetrievedImage(candidate_id="native-1-first-0", purpose="native ViMax frame candidate")
                ] if selected else []))
                serial = SerialDemoPipeline(
                    production, MagicMock(), retriever, FramePromptWriter(_PromptModel()),
                    MagicMock(), MagicMock(), max_reference_images=1,
                )
                context = FrameMaterialContext(
                    shot_idx=1, frame_type="first", fixed_materials=[], field_lineage=[],
                    history_sample_groups=[], portrait_candidates=[], required_slots=0, top_k=1,
                )
                callback = serial._continuity_selector(
                    shot1, "first", context,
                    "shots/1/selected_first_frame_images.json", "shots/1/first_frame_prompt.json",
                )

                await production.generate_frames_for_single_camera(
                    Camera(
                        idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=0,
                        missing_info="The flush-mounted wall safe beside the bookcase.",
                    ),
                    [shot0, shot1], [], {}, serial_shot_idx=1,
                    serial_frame_type="first_frame", continuity_selector=callback,
                )

                prompt, references = provider.calls[0]
                if selected:
                    self.assertEqual(references, [str(candidate)])
                    self.assertEqual(prompt, "Render the target frame.")
                    for leaked in ("native-1-first-0", "shot 1", "sample 0", "role=", "ViMax", "selector"):
                        self.assertNotIn(leaked, prompt)
                else:
                    self.assertEqual(references, [])
                    self.assertEqual(prompt, "Render the target frame.")


if __name__ == "__main__":
    unittest.main()
