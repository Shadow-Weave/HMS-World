"""Produce one serial endpoint while retaining native camera transition materials."""

import os
import shutil
from typing import Any, Callable, Dict, List, Literal

from interfaces import Camera, CharacterInScene, ShotDescription
from production.progress import _emit_render_progress, _scoped_progress


async def generate_frames_for_single_camera(
    self,
    camera: Camera,
    shot_descriptions: List[ShotDescription],
    characters: List[CharacterInScene],
    character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
    progress=None,
    serial_shot_idx: int | None = None,
    serial_frame_type: Literal["first_frame", "last_frame"] | None = None,
    continuity_selector=None,
):
    if serial_shot_idx is None or serial_frame_type not in {"first_frame", "last_frame"} or continuity_selector is None:
        raise ValueError("A serial shot, frame type and continuity selector are required")
    if serial_shot_idx not in camera.active_shot_idxs:
        raise ValueError(f"Shot {serial_shot_idx} is not active on camera {camera.idx}")
    first_idx = camera.active_shot_idxs[0]
    anchor = os.path.join(self.working_dir, "shots", str(first_idx), "first_frame.png")
    shot = shot_descriptions[serial_shot_idx]
    if serial_shot_idx != first_idx and not os.path.exists(anchor):
        raise FileNotFoundError(f"Shot {serial_shot_idx} cannot execute before Camera {camera.idx} first shot {first_idx} has a first frame")
    if serial_frame_type == "last_frame":
        current_first = os.path.join(self.working_dir, "shots", str(serial_shot_idx), "first_frame.png")
        if not os.path.isfile(current_first):
            raise FileNotFoundError(f"Shot {serial_shot_idx} cannot generate a last frame before its first frame")
        if shot.variation_type not in {"medium", "large"}:
            raise ValueError(f"Shot {serial_shot_idx} has no planned last frame")

    scope = {"camera_idx": camera.idx, "shot_idx": serial_shot_idx, "frame_type": serial_frame_type}
    _emit_render_progress(progress, "camera_frames_start", f"Generating {serial_frame_type} for shot {serial_shot_idx}", scope)
    if os.path.isfile(anchor):
        self.frame_events[first_idx]["first_frame"].set()
    candidates = None
    anchor_pair = (anchor, shot_descriptions[first_idx].ff_desc)
    if serial_shot_idx == first_idx and serial_frame_type == "first_frame" and not os.path.isfile(anchor):
        candidates = []
        for idx in shot.ff_vis_char_idxs:
            for item in character_portraits_registry[characters[idx].identifier_in_scene].values():
                candidates.append((item["path"], item["description"]))
        if camera.parent_shot_idx is not None:
            parent_idx = camera.parent_shot_idx
            parent_path = os.path.join(self.working_dir, "shots", str(parent_idx), "first_frame.png")
            if parent_idx >= serial_shot_idx or not os.path.isfile(parent_path):
                raise ValueError("Serial camera material requires an already produced earlier parent frame")
            material = await self._generate_child_camera_material(
                camera, shot, shot_descriptions[parent_idx].ff_desc, parent_path, progress=progress,
            )
            # This native copy path is used by the serial pipeline too.
            if camera.missing_info is None:
                shutil.copy(material, anchor)
                self.frame_events[first_idx]["first_frame"].set()
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_idx}", {**scope, "path": anchor})
                return
            candidates.append((material, "native ViMax frame candidate"))
        anchor_pair = None
    visible = shot.ff_vis_char_idxs if serial_frame_type == "first_frame" else shot.lf_vis_char_idxs
    description = shot.ff_desc if serial_frame_type == "first_frame" else shot.lf_desc
    await self.generate_frame_for_single_shot(
        shot_idx=serial_shot_idx, frame_type=serial_frame_type,
        first_shot_ff_path_and_text_pair=anchor_pair, frame_desc=description,
        visible_characters=[characters[idx] for idx in visible],
        character_portraits_registry=character_portraits_registry,
        available_image_path_and_text_pairs=candidates,
        continuity_selector=continuity_selector, progress=progress,
    )
    _emit_render_progress(progress, "camera_frames_done", f"Frame for shot {serial_shot_idx} ready", scope)


async def _generate_child_camera_material(
    self,
    camera: Camera,
    shot_description: ShotDescription,
    parent_shot_visual_desc: str,
    parent_first_frame_path: str,
    *,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
) -> str:
    """Create the transition-derived first-frame reference for a child camera."""
    if camera.parent_shot_idx is None:
        raise ValueError(f"Camera {camera.idx} has no parent shot for child camera material")

    shot_dir = os.path.join(self.working_dir, "shots", str(shot_description.idx))
    os.makedirs(shot_dir, exist_ok=True)
    parent_shot_idx = camera.parent_shot_idx
    transition_video_path = os.path.join(shot_dir, f"transition_video_from_shot_{parent_shot_idx}.mp4")
    if os.path.exists(transition_video_path):
        print(f"🚀 Skipped generating transition video for shot {shot_description.idx} from shot {parent_shot_idx}, already exists.")
        _emit_render_progress(progress, "transition_video_exists", f"Transition video for shot {shot_description.idx} already exists", {"camera_idx": camera.idx, "shot_idx": shot_description.idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})
    else:
        print(f"🖼️ Starting transition video generation for shot {shot_description.idx} from shot {parent_shot_idx}...")
        _emit_render_progress(progress, "transition_video_start", f"Generating transition video for shot {shot_description.idx}", {"camera_idx": camera.idx, "shot_idx": shot_description.idx, "parent_shot_idx": parent_shot_idx})
        transition_video_output = await self.camera_image_generator.generate_transition_video(
            first_shot_visual_desc=parent_shot_visual_desc,
            # This material establishes the opening view, before the shot's action.
            second_shot_visual_desc=shot_description.ff_desc,
            first_shot_ff_path=parent_first_frame_path,
            task_receipt_path=transition_video_path + ".task.json",
            progress=_scoped_progress(progress, camera_idx=camera.idx, shot_idx=shot_description.idx, parent_shot_idx=parent_shot_idx, artifact="transition_video"),
        )
        transition_video_output.save(transition_video_path)
        print(f"☑️ Generated transition video for shot {shot_description.idx} from shot {parent_shot_idx}, saved to {transition_video_path}.")
        _emit_render_progress(progress, "transition_video_done", f"Transition video for shot {shot_description.idx} generated", {"camera_idx": camera.idx, "shot_idx": shot_description.idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})

    new_camera_image_path = os.path.join(shot_dir, f"new_camera_{camera.idx}.png")
    if os.path.exists(new_camera_image_path):
        print(f"🚀 Skipped generating new camera image for shot {shot_description.idx}, already exists.")
        _emit_render_progress(progress, "new_camera_image_exists", f"New camera image for shot {shot_description.idx} already exists", {"camera_idx": camera.idx, "shot_idx": shot_description.idx, "path": new_camera_image_path})
    else:
        print(f"🖼️ Starting new camera image generation for shot {shot_description.idx}...")
        _emit_render_progress(progress, "new_camera_image_start", f"Extracting new camera image for shot {shot_description.idx}", {"camera_idx": camera.idx, "shot_idx": shot_description.idx})
        new_camera_image = self.camera_image_generator.get_new_camera_image(transition_video_path)
        new_camera_image.save(new_camera_image_path)
        print(f"☑️ Generated new camera image for shot {shot_description.idx} (not completed), saved to {new_camera_image_path}.")
        _emit_render_progress(progress, "new_camera_image_done", f"New camera image for shot {shot_description.idx} extracted", {"camera_idx": camera.idx, "shot_idx": shot_description.idx, "path": new_camera_image_path})
    return new_camera_image_path
