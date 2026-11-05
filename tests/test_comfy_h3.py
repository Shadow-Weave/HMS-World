import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.video_generator_comfy_h3 import build_workflow, normalize_frame


class H3WorkflowTests(unittest.TestCase):
    def test_keeps_entire_wide_source_and_pads_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"wide.png"
            image = Image.new("RGB", (1920, 1080), "red")
            image.paste("blue", (0, 0, 30, 1080))
            image.paste("green", (1890, 0, 1920, 1080))
            image.save(path)
            normalized = Image.open(BytesIO(normalize_frame(str(path))))
            self.assertEqual(normalized.size, (1280, 736))
            self.assertEqual(normalized.getpixel((640, 0)), (0, 0, 0))
            self.assertEqual(normalized.getpixel((0, 360)), (0, 0, 255))
            self.assertEqual(normalized.getpixel((1279, 360)), (0, 128, 0))

    def test_binds_first_and_last_frames_without_mutating_template(self):
        template = {str(i): {"inputs": {}} for i in (7, 8, 10, 17)}
        before = json.dumps(template)
        workflow = build_workflow(template, "original motion", ["first.png", "last.png"], 7)
        self.assertEqual(workflow["7"]["inputs"]["image"], "first.png")
        self.assertEqual(workflow["8"]["inputs"]["last_frame"], ["18", 0])
        self.assertEqual(workflow["18"]["inputs"]["image"], "last.png")
        self.assertEqual(workflow["8"]["inputs"]["prompt"], "original motion")
        self.assertEqual(json.dumps(template), before)

    def test_single_frame_does_not_reuse_stale_last_frame(self):
        template = {str(i): {"inputs": {}} for i in (7, 8, 10, 17)}
        template["8"]["inputs"]["last_frame"] = ["18", 0]
        self.assertNotIn("last_frame", build_workflow(template, "motion", ["first.png"], 1)["8"]["inputs"])
