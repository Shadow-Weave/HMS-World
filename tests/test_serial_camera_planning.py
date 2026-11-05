import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.camera_image_generator import CameraImageGenerator
from interfaces import Camera, ShotDescription
from interfaces.camera import validate_serial_camera_parents
from pipelines.script2video_pipeline import Script2VideoPipeline


def shots():
    return [ShotDescription(
        idx=i, is_last=i == 4, cam_idx=i, visual_desc=f"Official shot {i}",
        variation_type="small", variation_reason="official", ff_desc=f"Frame {i}",
        ff_vis_char_idxs=[], lf_desc="", lf_vis_char_idxs=[], motion_desc=f"Motion {i}",
        audio_desc="",
    ) for i in range(5)]


def planner(future_parent, captured):
    items = [{"parent_cam_idx": None if i == 0 else 0,
              "parent_shot_idx": None if i == 0 else 0,
              "reason": "model selection"} for i in range(5)]
    if future_parent:
        items[3].update(parent_cam_idx=4, parent_shot_idx=4)

    def reply(messages):
        captured.append(messages)
        return AIMessage(content=json.dumps({"camera_parent_items": items}))

    return CameraImageGenerator(RunnableLambda(reply), None, None)


def test_serial_planner_tells_model_candidates_and_rejects_future_before_mutation():
    captured = []
    cameras = [Camera(idx=i, active_shot_idxs=[i]) for i in range(5)]
    with pytest.raises(ValueError, match="strictly earlier"):
        asyncio.run(planner(True, captured).construct_camera_tree(
            cameras, shots()))
    assert all(camera.parent_shot_idx is None for camera in cameras)
    assert "strictly smaller" in captured[0][0].content
    candidates = captured[0][1].content.split("CAMERA_3: eligible", 1)[1].splitlines()[0]
    assert "(2, 2)" in candidates and "(4, 4)" not in candidates


def test_serial_parent_must_belong_to_its_camera():
    with pytest.raises(ValueError, match="not on its parent camera"):
        validate_serial_camera_parents([
            Camera(idx=0, active_shot_idxs=[0]),
            Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=0),
            Camera(idx=2, active_shot_idxs=[2], parent_cam_idx=1, parent_shot_idx=0),
        ])


def test_invalid_serial_graph_is_never_persisted_or_reused(tmp_path):
    pipeline = object.__new__(Script2VideoPipeline)
    pipeline.working_dir = str(tmp_path)
    captured = []
    pipeline.camera_image_generator = planner(True, captured)
    cache = tmp_path / "camera_tree.json"
    with pytest.raises(ValueError, match="strictly earlier"):
        asyncio.run(pipeline.construct_camera_tree(shots(), quiet=True))
    assert not cache.exists()

    pipeline.camera_image_generator = planner(False, captured)
    valid = asyncio.run(pipeline.construct_camera_tree(shots(), quiet=True))
    assert valid[3].parent_shot_idx == 0
    assert cache.exists()
    poisoned = json.loads(cache.read_text())
    poisoned[3].update(parent_cam_idx=4, parent_shot_idx=4)
    cache.write_text(json.dumps(poisoned))
    calls = len(captured)
    with pytest.raises(ValueError, match="strictly earlier"):
        asyncio.run(pipeline.construct_camera_tree(shots(), quiet=True))
    assert len(captured) == calls  # Invalid cache is reported, never blindly regenerated.
