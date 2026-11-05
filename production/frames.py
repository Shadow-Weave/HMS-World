"""Single-shot frame production, using the caller's existing state and hooks."""

import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Literal, Tuple

from interfaces import CharacterInScene, ImageOutput
from production.progress import _emit_render_progress


def prepare_last_frame_event(self, shot_idx: int) -> None:
    """Initialize readiness when a planned shot is promoted to use a last frame."""
    self.frame_events[shot_idx]["last_frame"] = asyncio.Event()


async def generate_frame_for_single_shot(
    self,
    shot_idx: int,
    frame_type: Literal["first_frame", "last_frame"],
    first_shot_ff_path_and_text_pair: Tuple[str, str] | None,
    frame_desc: str,
    visible_characters: List[CharacterInScene],
    character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
    available_image_path_and_text_pairs: List[Tuple[str, str]] | None = None,
    continuity_selector: Callable[[List[Tuple[str, str]], str], Any] | None = None,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
) -> ImageOutput:

    if continuity_selector is None:
        raise ValueError("Frame production requires the serial continuity selector")

    frame_image_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}.png")

    if os.path.exists(frame_image_path):
        print(f"🚀 Skipped generating {frame_type} for shot {shot_idx}, already exists.")
        _emit_render_progress(progress, "frame_exists", f"{frame_type} for shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})

    else:
        print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")
        _emit_render_progress(progress, "frame_start", f"Generating {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
        if available_image_path_and_text_pairs is None:
            available_image_path_and_text_pairs = []
            for visible_character in visible_characters:
                identifier_in_scene = visible_character.identifier_in_scene
                registry_item = character_portraits_registry[identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))
            if first_shot_ff_path_and_text_pair is not None:
                available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)

        selector_output_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_selector_output.json")
        reference_image_path_and_text_pairs, prompt = await continuity_selector(
            available_image_path_and_text_pairs, frame_desc,
        )
        selector_output = {
            "reference_image_path_and_text_pairs": reference_image_path_and_text_pairs,
            "text_prompt": prompt,
        }
        with open(selector_output_path, "w", encoding="utf-8") as handle:
            json.dump(selector_output, handle, ensure_ascii=False, indent=4)
        reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

        frame_image: ImageOutput = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            size="1600x900",
        )
        frame_image.save(frame_image_path)
        print(f"☑️ Generated {frame_type} frame for shot {shot_idx}, saved to {frame_image_path}.")
        _emit_render_progress(progress, "frame_done", f"Generated {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})


    self.frame_events[shot_idx][frame_type].set()
    return frame_image_path
