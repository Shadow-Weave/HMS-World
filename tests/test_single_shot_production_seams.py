"""Contracts for native serial frame production without a Camera resolver."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from interfaces import Camera, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline


class Output:
    def save(self, path):
        Path(path).write_bytes(b"frame")


def shot(index, camera=0, variation="small"):
    return ShotDescription(
        idx=index, is_last=False, cam_idx=camera, visual_desc=f"visual {index}",
        variation_type=variation, variation_reason="test", ff_desc=f"first {index}",
        ff_vis_char_idxs=[], lf_desc=f"last {index}", lf_vis_char_idxs=[],
        motion_desc="motion", audio_desc="",
    )


class NativeSerialFrameTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, directory):
        pipeline = Script2VideoPipeline(MagicMock(), MagicMock(), MagicMock(), directory)
        return pipeline

    async def test_serial_selector_receives_same_camera_anchor_as_ordinary_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory)
            shots = [shot(0), shot(1)]
            first = Path(directory) / "shots" / "0" / "first_frame.png"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"anchor")
            (Path(directory) / "shots" / "1").mkdir()
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            pipeline.frame_events[1] = {"first_frame": asyncio.Event()}
            pipeline.image_generator.generate_single_image = AsyncMock(return_value=Output())
            received = []

            async def selector(candidates, description):
                received.extend(candidates)
                return [], "continuity prompt"

            await pipeline.generate_frames_for_single_camera(
                Camera(idx=0, active_shot_idxs=[0, 1]), shots, [], {},
                serial_shot_idx=1, serial_frame_type="first_frame", continuity_selector=selector,
            )

            self.assertEqual(received, [(str(first), "first 0")])
            self.assertEqual(
                pipeline.image_generator.generate_single_image.await_args.kwargs["reference_image_paths"], []
            )

    async def test_invalid_serial_combinations_fail_before_any_production_action(self):
        invalid_calls = [
            {"serial_shot_idx": 0},
            {"serial_frame_type": "first_frame"},
            {"continuity_selector": AsyncMock()},
        ]
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                pipeline = self.pipeline(directory)
                pipeline._generate_child_camera_material = AsyncMock()
                pipeline.image_generator.generate_single_image = AsyncMock(return_value=Output())
                selector = kwargs.get("continuity_selector")

                with self.assertRaisesRegex(ValueError, "serial"):
                    await pipeline.generate_frames_for_single_camera(
                        Camera(idx=0, active_shot_idxs=[0]), [shot(0)], [], {}, **kwargs,
                    )

                pipeline._generate_child_camera_material.assert_not_awaited()
                pipeline.image_generator.generate_single_image.assert_not_awaited()
                if selector is not None:
                    selector.assert_not_awaited()

    async def test_serial_later_camera_shot_requires_existing_camera_first_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory)
            selector = AsyncMock()
            pipeline.image_generator.generate_single_image = AsyncMock(return_value=Output())

            with self.assertRaisesRegex(FileNotFoundError, "first shot 0 has a first frame"):
                await pipeline.generate_frames_for_single_camera(
                    Camera(idx=0, active_shot_idxs=[0, 1]), [shot(0), shot(1)], [], {},
                    serial_shot_idx=1, serial_frame_type="first_frame", continuity_selector=selector,
                )

            selector.assert_not_awaited()
            pipeline.image_generator.generate_single_image.assert_not_awaited()

    async def test_native_frame_path_uses_continuity_output_without_reference_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory)
            pipeline.frame_events[2] = {"first_frame": asyncio.Event()}
            (Path(directory) / "shots" / "2").mkdir(parents=True)
            pipeline.image_generator.generate_single_image = AsyncMock(return_value=Output())
            asset = Path(directory) / "history.png"
            asset.write_bytes(b"history")

            async def selector(candidates, description):
                return [(str(asset), "observed state evidence")], "clean continuity prompt"

            await pipeline.generate_frame_for_single_shot(
                2, "first_frame", None, "target", [], {},
                available_image_path_and_text_pairs=[], continuity_selector=selector,
            )

            pipeline.image_generator.generate_single_image.assert_awaited_once_with(
                prompt="clean continuity prompt", reference_image_paths=[str(asset)], size="1600x900",
            )

    async def test_child_without_missing_info_keeps_native_copy_branch_even_with_continuity(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory)
            shots = [shot(0), shot(1, camera=1)]
            parent = Path(directory) / "shots" / "0" / "first_frame.png"
            parent.parent.mkdir(parents=True)
            parent.write_bytes(b"parent")
            new_camera = Path(directory) / "shots" / "1" / "new_camera_1.png"
            new_camera.parent.mkdir(parents=True)
            new_camera.write_bytes(b"native-camera")
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            pipeline.frame_events[1] = {"first_frame": asyncio.Event()}
            pipeline._generate_child_camera_material = AsyncMock(return_value=str(new_camera))
            selector = AsyncMock()

            await pipeline.generate_frames_for_single_camera(
                Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=0, missing_info=None),
                shots, [], {}, serial_shot_idx=1, serial_frame_type="first_frame", continuity_selector=selector,
            )

            self.assertEqual((Path(directory) / "shots" / "1" / "first_frame.png").read_bytes(), b"native-camera")
            # The source image is the parent's first frame, not its later action.
            self.assertEqual(pipeline._generate_child_camera_material.await_args.args[2], "first 0")
            self.assertEqual(pipeline._generate_child_camera_material.await_args.args[3], str(parent))
            selector.assert_not_awaited()
            pipeline.image_generator.generate_single_image.assert_not_called()


    async def test_serial_child_callback_receives_factual_new_camera_candidate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory)
            shots = [shot(0), shot(1, camera=1)]
            parent = Path(directory) / "shots" / "0" / "first_frame.png"
            parent.parent.mkdir(parents=True)
            parent.write_bytes(b"parent")
            new_camera = Path(directory) / "shots" / "1" / "new_camera_1.png"
            new_camera.parent.mkdir(parents=True)
            new_camera.write_bytes(b"native-camera")
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            pipeline.frame_events[1] = {"first_frame": asyncio.Event()}
            pipeline._generate_child_camera_material = AsyncMock(return_value=str(new_camera))
            pipeline.image_generator.generate_single_image = AsyncMock(return_value=Output())
            received = []

            async def selector(candidates, _description):
                received.extend(candidates)
                return [], "serial prompt"

            await pipeline.generate_frames_for_single_camera(
                Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=0, missing_info="new prop"),
                shots, [], {}, serial_shot_idx=1, serial_frame_type="first_frame", continuity_selector=selector,
            )

            description = received[-1][1]
            self.assertEqual(description, "native ViMax frame candidate")
            self.assertNotIn("must select", description.casefold())
            self.assertNotIn("main reference", description.casefold())
            pipeline._generate_child_camera_material.assert_awaited_once()

    async def test_serial_endpoint_directly_awaits_without_native_gather(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.pipeline(directory)
            shots = [shot(0), shot(1)]
            first = Path(directory) / "shots" / "0" / "first_frame.png"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"anchor")
            (Path(directory) / "shots" / "1").mkdir()
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            pipeline.frame_events[0]["first_frame"].set()
            pipeline.frame_events[1] = {"first_frame": asyncio.Event()}
            pipeline.image_generator.generate_single_image = AsyncMock(return_value=Output())

            async def selector(_candidates, _description):
                return [], "serial prompt"

            with patch("pipelines.script2video_pipeline.asyncio.gather", side_effect=AssertionError("gather")):
                await pipeline.generate_frames_for_single_camera(
                    Camera(idx=0, active_shot_idxs=[0, 1]), shots, [], {},
                    serial_shot_idx=1, serial_frame_type="first_frame", continuity_selector=selector,
                )

            self.assertTrue((Path(directory) / "shots" / "1" / "first_frame.png").is_file())
