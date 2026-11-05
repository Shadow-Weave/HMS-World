from pathlib import Path
from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock

import numpy as np

from agents import camera_image_generator as camera_module


def test_native_reference_target_excludes_the_subsequent_shot_action(tmp_path):
    from PIL import Image
    from interfaces.image_output import ImageOutput
    from production.cameras import _generate_child_camera_material

    render = AsyncMock(return_value=SimpleNamespace(save=lambda path: Path(path).write_bytes(b"video")))
    camera_generator = SimpleNamespace(
        generate_transition_video=render,
        get_new_camera_image=lambda path: ImageOutput("pil", "png", Image.new("RGB", (4, 4))),
    )
    pipeline = SimpleNamespace(working_dir=str(tmp_path), camera_image_generator=camera_generator)
    shot = SimpleNamespace(idx=1, ff_desc="Mara stands alone beside three cups.",
                           visual_desc="Mara stands alone. Later the camera moves behind her shoulder and she pours.")
    asyncio.run(_generate_child_camera_material(
        pipeline, SimpleNamespace(idx=1, parent_shot_idx=0), shot,
        "Original parent description", str(tmp_path / "parent.png"),
    ))
    arguments = render.await_args.kwargs
    assert arguments["second_shot_visual_desc"] == shot.ff_desc
    assert "Later" not in arguments["second_shot_visual_desc"]
    assert arguments["first_shot_visual_desc"] == "Original parent description"
    assert arguments["first_shot_ff_path"] == str(tmp_path / "parent.png")


def test_replaced_transition_cannot_read_an_old_second_scene(tmp_path, monkeypatch):
    video = tmp_path / "transition.mp4"
    video.write_bytes(b"old transition with two scenes")
    monkeypatch.setattr(camera_module, "open_video", lambda _: object())
    monkeypatch.setattr(camera_module, "SceneManager", lambda: SimpleNamespace(
        add_detector=lambda _: None, detect_scenes=lambda *a, **k: None,
        get_scene_list=lambda: []))
    directories = []

    def split(_video, _scenes, output_dir, **kwargs):
        directories.append(output_dir)
        if len(directories) == 1:
            (Path(output_dir) / "transition-Scene-002.mp4").write_bytes(b"old second scene")

    monkeypatch.setattr(camera_module, "split_video_ffmpeg", split)
    monkeypatch.setattr(camera_module, "VideoFileClip", lambda path: SimpleNamespace(
        duration=5, fps=24,
        get_frame=lambda _: np.full((2, 2, 3), 10 if "Scene-002" in str(path) else 20,
                                    dtype=np.uint8)))
    generator = object.__new__(camera_module.CameraImageGenerator)
    first = generator.get_new_camera_image(str(video))
    video.write_bytes(b"new transition with only one scene")
    second = generator.get_new_camera_image(str(video))
    assert directories[0] != directories[1]
    assert first.data.getpixel((0, 0)) == (10, 10, 10)
    assert second.data.getpixel((0, 0)) == (20, 20, 20)
