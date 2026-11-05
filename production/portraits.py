"""Portrait production and cache reuse on the caller's working directory."""

import os
from typing import Any, Callable, Dict

from interfaces import CharacterInScene
from production.progress import _emit_render_progress


async def generate_portraits_for_single_character(
    self,
    character: CharacterInScene,
    style: str,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
):
    character_dir = os.path.join(self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
    os.makedirs(character_dir, exist_ok=True)
    _emit_render_progress(progress, "character_portrait_start", f"Generating portraits for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

    front_portrait_path = os.path.join(character_dir, "front.png")
    if os.path.exists(front_portrait_path):
        pass
    else:
        _emit_render_progress(progress, "character_portrait_front_start", f"Generating front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
        front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
        front_portrait_output.save(front_portrait_path)
        _emit_render_progress(progress, "character_portrait_front_done", f"Generated front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": front_portrait_path})


    side_portrait_path = os.path.join(character_dir, "side.png")
    if os.path.exists(side_portrait_path):
        pass
    else:
        _emit_render_progress(progress, "character_portrait_side_start", f"Generating side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
        side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
        side_portrait_output.save(side_portrait_path)
        _emit_render_progress(progress, "character_portrait_side_done", f"Generated side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": side_portrait_path})

    back_portrait_path = os.path.join(character_dir, "back.png")
    if os.path.exists(back_portrait_path):
        pass
    else:
        _emit_render_progress(progress, "character_portrait_back_start", f"Generating back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
        back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
        back_portrait_output.save(back_portrait_path)
        _emit_render_progress(progress, "character_portrait_back_done", f"Generated back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": back_portrait_path})

    print(f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")
    _emit_render_progress(progress, "character_portrait_done", f"Portraits for {character.identifier_in_scene} ready", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

    return {
        character.identifier_in_scene: {
            "front": {
                "path": front_portrait_path,
                "description": f"A front view portrait of {character.identifier_in_scene}.",
            },
            "side": {
                "path": side_portrait_path,
                "description": f"A side view portrait of {character.identifier_in_scene}.",
            },
            "back": {
                "path": back_portrait_path,
                "description": f"A back view portrait of {character.identifier_in_scene}.",
            },
        }
    }
