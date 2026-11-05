"""Interactive tools must use the serial plan, checkpoints and resume contract."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime import hms_world_adapters as adapters
from agent_runtime.session_index import SessionIndex
from agent_runtime.tools import ToolRuntimeContext
from continuity.models import EntityRegistry, SceneEntity
from interfaces import Camera, ShotBriefDescription, ShotDescription
from pipelines.serial_demo_pipeline import SerialDemoPipeline


def make_pipeline(root):
    shot = ShotDescription(idx=0, is_last=True, cam_idx=0, visual_desc="An empty room",
                           variation_type="small", variation_reason="Static", ff_desc="An empty room",
                           lf_desc="", motion_desc="Light shifts", audio_desc="")
    native = MagicMock(working_dir=str(root))
    native.extract_characters = AsyncMock(return_value=[])
    native.design_storyboard = AsyncMock(return_value=[ShotBriefDescription(
        idx=0, is_last=True, cam_idx=0, visual_desc=shot.visual_desc, audio_desc="")])
    native.decompose_visual_description_for_single_shot_brief_description = AsyncMock(return_value=shot)
    native.construct_camera_tree = AsyncMock(return_value=[Camera(idx=0, active_shot_idxs=[0])])
    extractor = SimpleNamespace(
        extract_entities=AsyncMock(return_value=EntityRegistry(scenes=[SceneEntity(entity_id="scene_0", name="Room")])),
        bind_props_for_shot=AsyncMock(return_value=[]), bind_scene_for_shot=AsyncMock(return_value="scene_0"))
    return SerialDemoPipeline(native, extractor, MagicMock(), MagicMock(), MagicMock(), MagicMock(), max_reference_images=4)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    index = SessionIndex(tmp_path)
    adapter = adapters.HmsWorldAdapters(tmp_path, index)
    made = []
    def factory(session_id, runtime=None):
        pipeline = make_pipeline(index.working_dir(session_id) / "serial")
        pipeline.progress = adapters._pipeline_progress(runtime, session_id)
        made.append(pipeline)
        return pipeline
    monkeypatch.setattr(adapter, "_pipeline", factory)
    return adapter, index, made


def plan(adapter, **kwargs):
    return asyncio.run(adapter.hms_world_narrative_planning({"script": "An empty room", "style": "animation", **kwargs}))


def test_text_plan_uses_real_serial_planning_without_media(workspace):
    adapter, index, made = workspace
    selected = index.create(project_name="Room")
    events = []
    runtime = ToolRuntimeContext("hms_world_narrative_planning", "hms_world_narrative_planning", progress_callback=events.append)
    result = asyncio.run(adapter.hms_world_narrative_planning({"session_id": selected["session_id"], "script": "An empty room"}, runtime))
    assert result.ok, result
    assert result.metadata["session_id"] == selected["session_id"]
    root = index.working_dir(selected["session_id"]) / "serial"
    assert (root / "script.txt").read_text() == "An empty room"
    assert (root / "continuity/timeline.json").is_file()
    assert (root / "continuity/entity_registry.json").is_file()
    assert index.get(selected["session_id"])["stage"] == "narrative_planned"
    made[0].production.generate_portraits_for_single_character.assert_not_called()
    made[0].production.image_generator.generate_single_image.assert_not_called()
    made[0].production.generate_video_for_single_shot.assert_not_called()
    assert "narrative_planned" in [event["progress"]["stage"] for event in events]


def test_same_plan_reuses_timeline_but_new_source_starts_session(workspace):
    adapter, index, made = workspace
    first = plan(adapter)
    assert first.ok, first
    session_id = first.metadata["session_id"]
    checkpoint = index.working_dir(session_id) / "serial/continuity/timeline.json"
    original = checkpoint.read_bytes()
    reused = plan(adapter, session_id=session_id)
    assert reused.ok and reused.metadata["reused"]
    assert len(made) == 1
    changed = plan(adapter, session_id=session_id, script="A different room")
    assert changed.ok and changed.metadata["session_id"] != session_id
    assert checkpoint.read_bytes() == original


def test_render_resumes_prepared_plan_and_preserves_checkpoint_on_error(workspace, monkeypatch):
    adapter, index, _ = workspace
    result = plan(adapter)
    assert result.ok, result
    session_id = result.metadata["session_id"]
    root = index.working_dir(session_id) / "serial"
    original = (root / "continuity/timeline.json").read_bytes()
    resume = AsyncMock(side_effect=[RuntimeError("interrupted"), str(root / "final_video.mp4")])
    monkeypatch.setattr(adapter, "_pipeline", lambda *args: SimpleNamespace(resume_preplanned=resume))
    failed = asyncio.run(adapter.hms_world_render_video({"session_id": session_id}))
    assert not failed.ok
    assert (root / "continuity/timeline.json").read_bytes() == original
    completed = asyncio.run(adapter.hms_world_render_video({"session_id": session_id}))
    assert completed.ok and completed.metadata["render_mode"] == "serial"
    assert resume.await_count == 2
    resume.assert_awaited_with("animation")
    assert index.get(session_id)["stage"] == "rendered"


def test_legacy_session_is_readable_and_not_automatically_converted(workspace, monkeypatch):
    adapter, index, _ = workspace
    legacy = index.create(idea="Old story")
    root = index.working_dir(legacy["session_id"])
    old = root / "script2video/shots/0/shot_description.json"
    old.parent.mkdir(parents=True)
    old.write_text("original")
    monkeypatch.setattr(adapter, "_pipeline", MagicMock(side_effect=AssertionError("must not run")))
    assert not asyncio.run(adapter.hms_world_render_video({"session_id": legacy["session_id"]})).ok
    assert not asyncio.run(adapter.hms_world_narrative_planning({"session_id": legacy["session_id"]})).ok
    assert old.read_text() == "original"
    assert not (root / "serial/continuity/timeline.json").exists()


def test_setup_failure_is_redacted_and_reported(workspace, monkeypatch):
    adapter, index, _ = workspace
    monkeypatch.setattr(adapter, "_pipeline", MagicMock(side_effect=RuntimeError("HTTP 403 sk-example-private-token")))
    result = plan(adapter)
    assert not result.ok
    assert not result.metadata["retryable"]
    assert "sk-example-private-token" not in result.content
    assert index.get(result.metadata["session_id"])["stage"] == "error"


def test_factory_uses_serial_config_and_session_directory(tmp_path, monkeypatch):
    index = SessionIndex(tmp_path)
    session_id = index.create()["session_id"]
    config = tmp_path / "custom.yaml"
    monkeypatch.setattr(adapters, "serial_config_path", lambda root: config)
    loader = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr(adapters, "load_serial_demo", loader)
    adapters.HmsWorldAdapters(tmp_path, index)._pipeline(session_id)
    root = str(index.working_dir(session_id) / "serial")
    loader.assert_called_once_with(str(config), output=root, config_overrides={"working_dir": root})


def test_revision_forks_text_without_copying_media(workspace, monkeypatch):
    adapter, index, _ = workspace
    first = plan(adapter)
    assert first.ok, first
    session_id = first.metadata["session_id"]
    root = index.working_dir(session_id) / "serial"
    shot = make_pipeline(root).production.decompose_visual_description_for_single_shot_brief_description.return_value
    original = root / "shots/0/shot_description.json"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(shot.model_dump_json())
    (root / "shots/0/first_frame.png").write_bytes(b"old media")
    revised = shot.model_copy(update={"ff_desc": "A blue chair"}).model_dump_json()
    monkeypatch.setattr(adapters, "_build_chat_model", lambda *args: None)
    monkeypatch.setattr(adapters, "_revise_artifact_with_llm", AsyncMock(return_value=revised))
    planner = AsyncMock()
    monkeypatch.setattr(adapter, "_pipeline", lambda *args: SimpleNamespace(plan=planner))
    result = asyncio.run(adapter.hms_world_narrative_planning({"session_id": session_id,
        "revision_target": "serial/shots/0/shot_description.json", "revision_instruction": "Add a blue chair"}))
    assert result.ok, result
    new_id = result.metadata["session_id"]
    assert new_id != session_id
    destination = index.working_dir(new_id) / "serial"
    assert "blue chair" in planner.await_args.args[0]
    assert (destination / "shots/0/shot_description.json").read_text() == revised
    assert not (destination / "shots/0/first_frame.png").exists()
    assert original.read_text() == shot.model_dump_json()
    assert (root / "shots/0/first_frame.png").read_bytes() == b"old media"


def test_invalid_revision_does_not_create_session(workspace, monkeypatch):
    adapter, index, _ = workspace
    first = plan(adapter)
    assert first.ok, first
    session_id = first.metadata["session_id"]
    root = index.working_dir(session_id) / "serial"
    shot_path = root / "shots/0/shot_description.json"
    shot_path.parent.mkdir(parents=True, exist_ok=True)
    shot_path.write_text("{}")
    monkeypatch.setattr(adapters, "_build_chat_model", lambda *args: None)
    monkeypatch.setattr(adapters, "_revise_artifact_with_llm", AsyncMock(return_value="{}"))
    before = len(index.load()["sessions"])
    result = asyncio.run(adapter.hms_world_narrative_planning({"session_id": session_id,
        "revision_target": "serial/shots/0/shot_description.json", "revision_instruction": "Revise"}))
    assert not result.ok
    assert len(index.load()["sessions"]) == before


def test_tools_expose_one_planning_and_rendering_path(tmp_path):
    specs = adapters.build_hms_world_adapter_specs(tmp_path, SessionIndex(tmp_path))
    assert [spec.name for spec in specs] == ["hms_world_narrative_planning", "hms_world_render_video"]


@pytest.mark.parametrize("changes", [{"style": "watercolor"}, {"user_requirement": "Two shots"}])
def test_changed_settings_fork_plan_instead_of_reusing_stale_artifacts(workspace, changes):
    adapter, index, _ = workspace
    first = plan(adapter)
    assert first.ok, first
    session_id = first.metadata["session_id"]
    original = index.get(session_id)
    second = plan(adapter, session_id=session_id, **changes)
    assert second.ok and second.metadata["session_id"] != session_id
    assert index.get(session_id)["style"] == original["style"]
    assert index.get(session_id)["user_requirement"] == original["user_requirement"]


def test_storyboard_revision_rebuilds_entity_context_from_reviewed_text(workspace, monkeypatch):
    adapter, index, _ = workspace
    first = plan(adapter)
    assert first.ok, first
    session_id = first.metadata["session_id"]
    original = index.working_dir(session_id) / "serial/storyboard.json"
    before = original.read_bytes()
    revised = '[{"idx":0,"is_last":true,"cam_idx":0,"visual_desc":"A blue chair in a garden","audio_desc":""}]'
    monkeypatch.setattr(adapters, "_build_chat_model", lambda *args: None)
    monkeypatch.setattr(adapters, "_revise_artifact_with_llm", AsyncMock(return_value=revised))
    planner = AsyncMock()
    monkeypatch.setattr(adapter, "_pipeline", lambda *args: SimpleNamespace(plan=planner))
    result = asyncio.run(adapter.hms_world_narrative_planning({"session_id": session_id,
        "revision_target": "serial/storyboard.json", "revision_instruction": "Move to a garden"}))
    assert result.ok, result
    destination = index.working_dir(result.metadata["session_id"]) / "serial"
    assert (destination / "storyboard.json").read_text() == revised
    assert "blue chair in a garden" in planner.await_args.args[0]
    assert not (destination / "camera_tree.json").exists()
    assert adapters._planning_context(destination, "An empty room") == planner.await_args.args[0]
    assert original.read_bytes() == before
