"""Wiring and fail-fast tests for the real serial-demo CLI entrypoint."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

import main_serial_demo
from agents.continuity_image_retriever import ContinuityImageRetriever
from agents.entity_extractor import EntityExtractor
from agents.frame_prompt_writer import FramePromptWriter
from agents.shot_observer import ShotObserver
from agents.shot_replanner import ShotReplanner


def config(*, continuity=None, chat_key="chat", image_key="image", video_key="video"):
    return {
        "chat_model": {"init_args": {"model": "test-model", "model_provider": "openai", "api_key": chat_key, "base_url": "http://localhost:9000/v1"}},
        "image_generator": {"class_path": "tools.image_generator_comfy_flux.ImageGeneratorComfyFlux", "init_args": {"api_key": image_key, "base_url": "http://localhost:8188"}},
        "video_generator": {"class_path": "tools.video_generator_comfy_h3.VideoGeneratorComfyH3", "init_args": {"api_key": video_key, "base_url": "http://localhost:8189"}},
        "working_dir": ".working_dir/test-serial",
        "continuity": continuity or {
            "enabled": True,
            "serial": True,
            "max_reference_images": 6,
        },
    }


class _RecordingSerial:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.run_args = None
        self.__class__.instances.append(self)

    async def run(self, screenplay, user_requirements, style, *, resume=False):
        self.run_args = (screenplay, user_requirements, style, resume)
        return "/tmp/real-serial-result.mp4"


class MainSerialDemoTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.config_path = self.root / "serial.yaml"
        _RecordingSerial.instances = []

    def tearDown(self):
        self.tempdir.cleanup()

    def write_config(self, value):
        self.config_path.write_text(yaml.safe_dump(value), encoding="utf-8")


    def test_wires_one_chat_model_into_all_five_continuity_roles_and_forwards_run_inputs(self):
        self.write_config(config())
        chat_model = object()
        production = MagicMock(chat_model=chat_model, working_dir=str(self.root / "configured-output"))
        with patch.object(main_serial_demo.Script2VideoPipeline, "init_from_config", return_value=production) as init, patch.object(main_serial_demo, "SerialDemoPipeline", _RecordingSerial):
            result = asyncio.run(main_serial_demo.amain([
                "--config", str(self.config_path),
                "--script", "script input",
                "--user-requirement", "requirement input",
                "--style", "realistic",
                "--output", str(self.root / "override-output"),
            ]))

        self.assertEqual(result, "/tmp/real-serial-result.mp4")
        init.assert_called_once_with(str(self.config_path))
        instance = _RecordingSerial.instances[0]
        self.assertIs(instance.args[0], production)
        self.assertIsInstance(instance.args[1], EntityExtractor)
        self.assertIsInstance(instance.args[2], ContinuityImageRetriever)
        self.assertIsInstance(instance.args[3], FramePromptWriter)
        self.assertIsInstance(instance.args[4], ShotObserver)
        self.assertIsInstance(instance.args[5], ShotReplanner)
        self.assertTrue(all(role.chat_model is chat_model for role in instance.args[1:6]))
        self.assertEqual(instance.kwargs, {"max_reference_images": 6})
        self.assertEqual(instance.run_args, ("script input", "requirement input", "realistic", False))
        self.assertEqual(production.working_dir, str(self.root / "override-output"))

    def test_resume_forwards_empty_inputs_and_explicit_flag(self):
        self.write_config(config())
        production = MagicMock(chat_model=object(), working_dir=str(self.root / "configured-output"))
        with patch.object(main_serial_demo.Script2VideoPipeline, "init_from_config", return_value=production), patch.object(
            main_serial_demo, "SerialDemoPipeline", _RecordingSerial,
        ):
            result = asyncio.run(main_serial_demo.amain([
                "--config", str(self.config_path), "--style", "flat", "--resume",
                "--output", str(self.root / "resume-output"),
            ]))
        self.assertEqual(result, "/tmp/real-serial-result.mp4")
        self.assertEqual(_RecordingSerial.instances[0].run_args, ("", "", "flat", True))

    def test_optional_vision_model_receives_roles_that_see_images(self):
        value = config()
        value["vision_chat_model"] = {"init_args": {"model": "local-vlm", "api_key": "local", "base_url": "http://localhost:9000/v1"}}
        self.write_config(value)
        text_model, vision_model = object(), object()
        production = MagicMock(chat_model=text_model)
        with patch.object(main_serial_demo.Script2VideoPipeline, "init_from_config", return_value=production), patch.object(
            main_serial_demo, "init_chat_model", return_value=vision_model,
        ), patch.object(main_serial_demo, "SerialDemoPipeline", _RecordingSerial):
            main_serial_demo.load_serial_demo(str(self.config_path))
        instance = _RecordingSerial.instances[0]
        self.assertIs(instance.args[1].chat_model, text_model)
        self.assertTrue(all(role.chat_model is vision_model for role in instance.args[2:6]))

    def test_provider_overrides_apply_in_memory(self):
        self.write_config(config())
        local_chat = {"init_args": {"model": "local-llm", "api_key": "local", "base_url": "http://localhost:9000/v1"}}
        production = MagicMock(chat_model=object())
        with patch.object(main_serial_demo.Script2VideoPipeline, "init_from_config_data", return_value=production) as initialize:
            main_serial_demo.load_serial_demo(str(self.config_path), config_overrides={"chat_model": local_chat})
        self.assertEqual(initialize.call_args.args[0]["chat_model"], local_chat)
        self.assertEqual(yaml.safe_load(self.config_path.read_text()), config())

    def test_resume_without_output_stops_before_loading_pipeline(self):
        with patch.object(main_serial_demo, "load_serial_demo") as load:
            with self.assertRaisesRegex(ValueError, "requires an explicit non-empty --output"):
                asyncio.run(main_serial_demo.amain(["--resume", "--style", "flat"]))
        load.assert_not_called()

    def test_resume_input_conflict_stops_before_loading_pipeline(self):
        with patch.object(main_serial_demo, "load_serial_demo") as load:
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                asyncio.run(main_serial_demo.amain([
                    "--resume", "--output", str(self.root / "resume-output"),
                    "--script", "not allowed", "--style", "flat",
                ]))
        load.assert_not_called()

    def test_fresh_missing_required_input_stops_before_loading_pipeline(self):
        with patch.object(main_serial_demo, "load_serial_demo") as load:
            with self.assertRaisesRegex(ValueError, "are required"):
                asyncio.run(main_serial_demo.amain([
                    "--script", "only screenplay", "--style", "flat",
                ]))
        load.assert_not_called()

    def test_missing_credential_stops_before_provider_initialization(self):
        self.write_config(config(image_key=""))
        with patch.object(main_serial_demo.Script2VideoPipeline, "init_from_config") as init:
            with self.assertRaisesRegex(ValueError, "image_generator.init_args.api_key"):
                main_serial_demo.load_serial_demo(str(self.config_path))
        init.assert_not_called()

    def test_disabled_continuity_stops_before_provider_initialization(self):
        self.write_config(config(continuity={
            "enabled": False,
            "serial": True,
            "max_reference_images": 6,
        }))
        with patch.object(main_serial_demo.Script2VideoPipeline, "init_from_config") as init:
            with self.assertRaisesRegex(ValueError, "continuity.enabled"):
                main_serial_demo.load_serial_demo(str(self.config_path))
        init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
