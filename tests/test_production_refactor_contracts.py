"""Compatibility and complete native production across the extracted modules."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import continuity
from continuity import models, timeline
from pipelines.script2video_pipeline import Script2VideoPipeline
from production import videos


@pytest.mark.parametrize("failure_stage", ["open", "concatenate", "write"])
def test_native_concatenation_releases_opened_clips_on_failure(tmp_path, monkeypatch, failure_stage):
    clips = [MagicMock(name="first_clip"), MagicMock(name="second_clip")]
    assembly = MagicMock(name="assembly")
    failure = OSError(f"{failure_stage} failed")
    open_clip = MagicMock(side_effect=[clips[0], failure] if failure_stage == "open" else clips)
    concatenate = MagicMock(return_value=assembly)
    if failure_stage == "concatenate":
        concatenate.side_effect = failure
    elif failure_stage == "write":
        assembly.write_videofile.side_effect = failure
    monkeypatch.setattr(videos, "VideoFileClip", open_clip)
    monkeypatch.setattr(videos, "concatenate_videoclips", concatenate)
    progress = MagicMock()

    with pytest.raises(OSError) as caught:
        Script2VideoPipeline.concatenate_rendered_shots(
            SimpleNamespace(working_dir=str(tmp_path)), [8, 3], progress=progress,
        )

    assert caught.value is failure
    assert open_clip.call_args_list == [
        call(str(tmp_path / "shots/8/video.mp4")),
        call(str(tmp_path / "shots/3/video.mp4")),
    ]
    clips[0].close.assert_called_once_with()
    if failure_stage == "open":
        clips[1].close.assert_not_called()
        concatenate.assert_not_called()
    else:
        clips[1].close.assert_called_once_with()
        concatenate.assert_called_once_with(clips)
    if failure_stage == "write":
        assembly.write_videofile.assert_called_once_with(
            str(tmp_path / "final_video.mp4"), codec="libx264", preset="medium",
        )
        assembly.close.assert_called_once_with()
    else:
        assembly.close.assert_not_called()
    assert [args[0] for args, _ in progress.call_args_list] == ["concat_start"]


def test_legacy_model_paths_keep_the_same_types_and_validation():
    for name in (
        "CharacterContent", "PropContent", "EntityRegistry", "ShotPlan", "WorldState",
        "ObservedPosition", "PlannedPosition", "StateFieldSource", "TimelinePosition",
    ):
        assert getattr(timeline, name) is getattr(models, name)
        assert getattr(continuity, name) is getattr(models, name)
    update = timeline.WorldState.model_validate_json(
        '{"characters":{"character-0":{"holding":[],"pose":null}},"props":{"prop-0":null}}'
    )
    assert isinstance(update, models.WorldState)
    assert update.characters["character-0"].model_fields_set == {"holding", "pose"}
    assert update.model_dump(exclude_unset=True) == {
        "characters": {"character-0": {"holding": [], "pose": None}}, "props": {"prop-0": None},
    }
