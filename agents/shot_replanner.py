"""Decide whether one completed shot requires replacing only the next shot."""

import json
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from continuity.shot_sampling import SampledFrame
from continuity.models import EntityRegistry, WorldState
from interfaces import Camera, ShotDescription
from utils.image import image_path_to_b64


class NoChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["no-change"]
    reason: str = Field(min_length=1)


class Replacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["replacement"]
    replacement_shot: ShotDescription
    reason: str = Field(min_length=1)


ReplanDecision = Annotated[NoChange | Replacement, Field(discriminator="outcome")]
_DECISION_ADAPTER = TypeAdapter(ReplanDecision)


def replacement_matches_next_shot(replacement: ShotDescription, next_shot: ShotDescription) -> bool:
    """Compare executable plans; small variations never consume last-frame fields."""
    excluded = {"lf_desc", "lf_vis_char_idxs"} if replacement.variation_type == next_shot.variation_type == "small" else set()
    return replacement.model_dump(exclude=excluded) == next_shot.model_dump(exclude=excluded)


_SYSTEM_PROMPT = """
Decide whether the previous take and the planned next shot form a usable cut.
PREVIOUS_SHOT_VISUAL_DESC is a plan; PREVIOUS_FINAL_STATE and the actual terminal
image describe what happened. Distinguish world location from screen framing.

Judge the cut, not whether the take reproduced its plan exactly. Eyeline match and
match on action apply only when the planned cut claims those relationships.
Off-screen action, ellipsis, reframing, and montage may bridge unshown change;
planned scene changes do not require travel shots. Treat visual-integrity defects
as generation defects rather than story events to canonize.

visual_desc covers the whole shot, ff_desc its opening, and motion_desc its action
and ending. Later actions need not already be present at the opening.
For small variations, lf_desc and lf_vis_char_idxs are unused; the ending belongs
in motion_desc. For medium/large variations, the last frame must match that ending.

Return no-change if the next shot can still fulfill its narrative task.
Return replacement only when concrete evidence shows a conflict that prevents it
or makes the cut misleading. Then make the smallest coherent adjustment within the
supplied Camera. Return the complete revised ShotDescription, with consistent
opening, action, ending, and audio. Preserve idx, cam_idx, is_last, narrative task,
visual style, dialogue, and intentional silence. Keep spoken words in audio_desc.
Do not modify other shots, add cuts, or invent a duration.

Return one direct JSON decision matching the schema, with a brief reason.

{format_instructions}
""".strip()


class ShotReplanner:
    """One multimodal decision call with no position-mutation authority."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def replan(
        self,
        scene_name: str,
        previous_shot_visual_desc: str,
        previous_final_state: WorldState,
        previous_last_sample: SampledFrame,
        next_shot: ShotDescription,
        next_camera: Camera,
        registry: EntityRegistry,
    ) -> NoChange | Replacement:
        if not scene_name.strip() or not previous_shot_visual_desc.strip():
            raise ValueError("ShotReplanner requires non-empty scene_name and previous_shot_visual_desc")
        if next_camera.idx != next_shot.cam_idx:
            raise ValueError("Next Camera idx must match next_shot.cam_idx")
        if next_shot.idx not in next_camera.active_shot_idxs:
            raise ValueError("Next Camera must list next_shot.idx as active")
        expected_previous_shot_idx = next_shot.idx - 1
        if previous_last_sample.shot_idx != expected_previous_shot_idx:
            raise ValueError(
                f"Last sample shot {previous_last_sample.shot_idx} does not precede next shot {next_shot.idx}"
            )
        if not Path(previous_last_sample.asset_path).is_file():
            raise ValueError("ShotReplanner cannot attach the previous last sample image")
        original_next = next_shot.model_copy(deep=True)
        parser = JsonOutputParser()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(format_instructions=_decision_format_instructions())),
            HumanMessage(content=_replan_content(
                scene_name,
                previous_shot_visual_desc,
                previous_final_state,
                previous_last_sample,
                next_shot,
                next_camera,
                registry,
            )),
        ]
        raw_decision = await (self.chat_model | parser).ainvoke(messages)
        decision = _DECISION_ADAPTER.validate_python(raw_decision)
        _validate_decision(decision, original_next)
        if next_shot.model_dump() != original_next.model_dump():
            raise ValueError("ShotReplanner must not mutate the caller's next_shot")
        if isinstance(decision, Replacement) and replacement_matches_next_shot(decision.replacement_shot, original_next):
            return NoChange(outcome="no-change", reason="The proposed replacement is identical to the supplied next shot; there is no change to apply.")
        return decision


def _decision_format_instructions() -> str:
    return "Return direct JSON conforming to this schema:\n" + json.dumps(
        _DECISION_ADAPTER.json_schema(), ensure_ascii=False,
    )


def _replan_content(
    scene_name: str,
    previous_shot_visual_desc: str,
    previous_final_state: WorldState,
    previous_last_sample: SampledFrame,
    next_shot: ShotDescription,
    next_camera: Camera,
    registry: EntityRegistry,
) -> list[dict]:
    entity_names = {
        "characters": [{"entity_id": entity.entity_id, "name": entity.name} for entity in registry.characters],
        "props": [{"entity_id": entity.entity_id, "name": entity.name} for entity in registry.props],
    }
    return [
        {"type": "text", "text": "<SCENE_NAME>\n" + scene_name + "\n</SCENE_NAME>"},
        {"type": "text", "text": "<PREVIOUS_SHOT_VISUAL_DESC>\n" + previous_shot_visual_desc + "\n</PREVIOUS_SHOT_VISUAL_DESC>"},
        {"type": "text", "text": "<NEXT_SHOT>\n" + next_shot.model_dump_json() + "\n</NEXT_SHOT>"},
        {"type": "text", "text": "<NEXT_CAMERA>\n" + next_camera.model_dump_json() + "\n</NEXT_CAMERA>"},
        {"type": "text", "text": "<REGISTRY_ENTITY_NAMES>\n" + json.dumps(entity_names, ensure_ascii=False) + "\n</REGISTRY_ENTITY_NAMES>"},
        {"type": "text", "text": "<PREVIOUS_FINAL_STATE>\n" + previous_final_state.model_dump_json() + "\n</PREVIOUS_FINAL_STATE>"},
        {"type": "text", "text": "Previous shot's actual terminal sampled frame:"},
        {"type": "image_url", "image_url": {"url": image_path_to_b64(previous_last_sample.asset_path, mime=True)}},
    ]


def _validate_decision(decision: NoChange | Replacement, next_shot: ShotDescription) -> None:
    if not decision.reason.strip():
        raise ValueError("ShotReplanner returned an empty reason")
    if isinstance(decision, Replacement):
        replacement = decision.replacement_shot
        if replacement.model_fields_set != set(ShotDescription.model_fields):
            raise ValueError("Replacement must provide a complete ShotDescription")
        if replacement.idx != next_shot.idx:
            raise ValueError("Replacement shot idx must match the next shot")
        if replacement.cam_idx != next_shot.cam_idx:
            raise ValueError("Replacement shot cam_idx must match the next shot")
        if replacement.is_last != next_shot.is_last:
            raise ValueError("Replacement shot is_last must match the next shot")
