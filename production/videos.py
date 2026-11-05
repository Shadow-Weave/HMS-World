"""Single-shot video production and story-order assembly of existing clips."""

import os
from typing import Any, Callable, Dict, List

from moviepy import VideoFileClip, concatenate_videoclips

from interfaces import ShotDescription
from production.progress import _emit_render_progress, _scoped_progress


def concatenate_rendered_shots(self, shot_idxs: List[int], progress=None) -> str:
    """Concatenate already rendered clips in explicit story order."""
    final_video_path = os.path.join(self.working_dir, "final_video.mp4")
    if os.path.exists(final_video_path):
        _emit_render_progress(progress, "final_video_exists", "Final video already exists", {"path": final_video_path})
        return final_video_path
    _emit_render_progress(progress, "concat_start", "Concatenating video clips", {"shot_count": len(shot_idxs)})
    clips = []
    final = None
    try:
        for idx in shot_idxs:
            clips.append(VideoFileClip(os.path.join(self.working_dir, "shots", str(idx), "video.mp4")))
        final = concatenate_videoclips(clips)
        final.write_videofile(final_video_path, codec="libx264", preset="medium")
    finally:
        if final is not None:
            final.close()
        for clip in clips:
            clip.close()
    _emit_render_progress(progress, "concat_done", "Final video concatenated", {"path": final_video_path})
    return final_video_path


async def generate_video_for_single_shot(
    self,
    shot_description: ShotDescription,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
) -> str:
    video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
    if os.path.exists(video_path):
        print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
        _emit_render_progress(progress, "video_clip_exists", f"Video clip for shot {shot_description.idx} already exists", {"shot_idx": shot_description.idx, "path": video_path})
    else:
        _emit_render_progress(progress, "video_clip_waiting_for_frames", f"Waiting for frames before video clip {shot_description.idx}", {"shot_idx": shot_description.idx})
        await self.frame_events[shot_description.idx]["first_frame"].wait()
        if shot_description.variation_type in ["medium", "large"]:
            await self.frame_events[shot_description.idx]["last_frame"].wait()

        frame_paths = []
        frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "first_frame.png"))
        if shot_description.variation_type in ["medium", "large"]:
            frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png"))

        print(f"🎬 Starting video generation for shot {shot_description.idx}...")
        _emit_render_progress(progress, "video_clip_start", f"Generating video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "frame_count": len(frame_paths)})
        video_output = await self.video_generator.generate_single_video(
            prompt=shot_description.motion_desc + "\n" + shot_description.audio_desc,
            reference_image_paths=frame_paths,
            task_receipt_path=os.path.join(
                self.working_dir, "shots", f"{shot_description.idx}", "video_task.json"
            ),
            progress=_scoped_progress(progress, shot_idx=shot_description.idx, artifact="video_clip"),
        )
        video_output.save(video_path)
        print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")
        _emit_render_progress(progress, "video_clip_done", f"Generated video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "path": video_path})
    return video_path
