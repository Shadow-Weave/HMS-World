"""Write one clean provider prompt from an arranged frame-material context."""

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.continuity_image_retriever import ImageRetrievalResult
from continuity.frame_material_context import FrameMaterialContext, fixed_material_purpose, history_sample_purpose
from continuity.models import InitialDesignSource, ObservedShotSource
from utils.image import image_path_to_b64
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser


class FinalFramePrompt(BaseModel):
    prompt: str = Field(min_length=1)


_SYSTEM_PROMPT = """
You write the final image-generation prompt for exactly one static target frame.
Use the target description, known field state, and attached materials. Return only
one non-empty clean drawing prompt. You may refer to Image N slots, but never
include attachment labels, internal IDs, entity IDs, source labels, shot/sample
references, or XML-like wrappers.

A current first-frame starting state is the baseline for its last frame. Native
ViMax frame candidates are ordinary optional visual references, not Timeline facts.
History samples are actual past images, not proof of every field in their source
shot. Use only visible people, scene, or props relevant to the supplied lineage;
do not import their other people, pose, staging, composition, or performance.
Identity portraits lock only stable identity appearance. The target description
and known field state jointly define the frame; explicit target instructions take
priority, and known state supplies established continuity they do not replace.

Bind each used Image N reference to the identity, setting, or object it supplies.
Several reference views of one character describe the same individual; the
number of reference images does not determine the number of people in the frame.
State the visible people and their placement from the target and known state.
People appearing only in reference backgrounds are not added to the target.
Describe one uninterrupted frame of the scene at a single instant.

Reference images are generation conditions, not pixel locks. Do not enlarge, move,
or attach a small prop, label, photograph, or other detail to a person merely to
make it prominent. If exact text was not supplied, do not invent readable words;
use an illegible mark or shape when needed. Keep visible body parts anatomically
normal and physically connected. Preserve the target's layout and scale.

Write the frame as a coherent composition: visual medium and treatment, camera
and crop, subject placement and facing, visible pose and contact with props, then
relevant setting and light. Keep the requested line/shadow, palette, or material
properties concrete and compatible; do not default to photorealism when another
medium was specified. Express the current visible emotional state through gaze and pose;
do not put a sequence of actions, dialogue, or future events into a static image.

{format_instructions}
""".strip()


class FramePromptWriter:
    """One multimodal call after programmatic context/result validation."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def write(
        self,
        target_description: str,
        context: FrameMaterialContext,
        material_result: ImageRetrievalResult,
    ) -> str:
        if not target_description or not target_description.strip():
            raise ValueError("Frame prompt writer requires a non-empty target description")
        if not isinstance(context, FrameMaterialContext):
            raise TypeError("Frame prompt writer requires a FrameMaterialContext")
        if not isinstance(material_result, ImageRetrievalResult):
            raise TypeError("Frame prompt writer requires an ImageRetrievalResult")
        attachments = _validate_material_result(context, material_result)
        parser = PydanticOutputParser(pydantic_object=FinalFramePrompt)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=_multimodal_content(target_description, context, attachments)),
        ]
        response: FinalFramePrompt = await (self.chat_model | parser).ainvoke(messages)
        prompt = response.prompt.strip()
        if not prompt:
            raise ValueError("Frame prompt writer returned an empty prompt")
        _reject_internal_leaks(prompt, context)
        return prompt


def _validate_material_result(
    context: FrameMaterialContext,
    material_result: ImageRetrievalResult,
) -> list[tuple[str, str, str, str | None]]:
    expected: list[tuple[str, str, str, str | None]] = []
    for item in context.fixed_materials:
        purpose = fixed_material_purpose(item.role)
        expected.append((item.image_id, item.asset_path, purpose, None))
    selected_by_character: dict[str, tuple[str, str]] = {}
    for group in context.portrait_candidates:
        matching = [item for item in material_result.images if item.candidate_id in {candidate.image_id for candidate in group.candidates}]
        if len(matching) != 1:
            raise ValueError(f"Material result must choose exactly one portrait for {group.character_id}")
        candidate = next(candidate for candidate in group.candidates if candidate.image_id == matching[0].candidate_id)
        selected_by_character[group.character_id] = (candidate.image_id, candidate.asset_path)
    for group in context.portrait_candidates:
        image_id, asset_path = selected_by_character[group.character_id]
        expected.append((image_id, asset_path, "character identity portrait", None))
    history_by_id = {
        sample.image_id: (group, sample)
        for group in context.history_sample_groups
        for sample in group.samples
    }
    for item in material_result.images:
        history = history_by_id.get(item.candidate_id)
        if history is None:
            continue
        group, sample = history
        purpose = history_sample_purpose(group, sample)
        if item.purpose != purpose:
            raise ValueError(f"Material result selected history sample {item.candidate_id} with an invalid purpose")
        expected.append((sample.image_id, sample.asset_path, purpose, None))
    native_by_id = {candidate.candidate_id: candidate for candidate in context.native_candidates}
    for item in material_result.images:
        if item.purpose != "native ViMax frame candidate":
            continue
        try:
            candidate = native_by_id[item.candidate_id]
        except KeyError as exc:
            raise ValueError(f"Material result selected an unknown native ViMax candidate {item.candidate_id}") from exc
        expected.append((candidate.candidate_id, candidate.asset_path, item.purpose, candidate.description))
    actual = [(item.candidate_id, item.purpose) for item in material_result.images]
    required = [(image_id, purpose) for image_id, _, purpose, _ in expected]
    if actual != required:
        raise ValueError("Material result must exactly preserve context fixed/portrait/history/native order and purposes")
    if len(actual) < context.required_slots or len(actual) > context.top_k:
        raise ValueError("Material result exceeds the frame material context capacity")
    if len({image_id for image_id, _, _, _ in expected}) != len(expected):
        raise ValueError("Frame material context contains duplicate attachment IDs")
    for image_id, asset_path, _, _ in expected:
        if not Path(asset_path).is_file():
            raise ValueError(f"Frame prompt writer cannot attach missing material {image_id}")
    return expected


def _multimodal_content(
    target_description: str,
    context: FrameMaterialContext,
    attachments: list[tuple[str, str, str, str | None]],
) -> list[dict]:
    content: list[dict] = [{
        "type": "text",
        "text": _state_lineage_text(context),
    }]
    for index, (_, asset_path, purpose, native_description) in enumerate(attachments):
        label = f"Image {index} [role={_role_for_purpose(purpose)}] {purpose}"
        if native_description is not None:
            label += f"\nNative candidate description: {native_description}"
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_path_to_b64(asset_path, mime=True)}})
    content.append({"type": "text", "text": "<TARGET_DESCRIPTION>\n" + target_description + "\n</TARGET_DESCRIPTION>"})
    return content


def _state_lineage_text(context: FrameMaterialContext) -> str:
    lines = ["<KNOWN_FIELD_STATE>"]
    for lineage in context.field_lineage:
        if isinstance(lineage.source, InitialDesignSource):
            source = "initial_design"
        elif isinstance(lineage.source, ObservedShotSource):
            source = f"observed shot {lineage.source.shot_idx}"
        else:
            raise ValueError("Unsupported state-field source")
        lines.append(f"{lineage.entity_id}.{lineage.field}={json.dumps(lineage.value, ensure_ascii=False)} [{source}]")
    lines.append("</KNOWN_FIELD_STATE>")
    return "\n".join(lines)


def _role_for_purpose(purpose: str) -> str:
    if purpose == "current shot first-frame starting state":
        return "current shot first-frame starting state"
    if purpose == "character identity portrait":
        return "character identity portrait"
    if purpose == "native ViMax frame candidate":
        return "native ViMax frame candidate"
    return "observed history frame"


def _reject_internal_leaks(prompt: str, context: FrameMaterialContext) -> None:
    forbidden = {"candidate_id", "source_kind", "initial_design"}
    forbidden.update(item.image_id for item in context.fixed_materials)
    forbidden.update(sample.image_id for group in context.history_sample_groups for sample in group.samples)
    forbidden.update(candidate.image_id for group in context.portrait_candidates for candidate in group.candidates)
    forbidden.update(candidate.candidate_id for candidate in context.native_candidates)
    forbidden.update(lineage.entity_id for lineage in context.field_lineage)
    for token in forbidden:
        if token and token.casefold() in prompt.casefold():
            raise ValueError(f"Final provider prompt leaked internal token {token!r}")
    if re.search(r"\bshot\s+\d+\s+sample\s+\d+\b", prompt, flags=re.IGNORECASE):
        raise ValueError("Final provider prompt leaked an internal shot/sample source")
    if re.search(r"</?[A-Z_]+>", prompt):
        raise ValueError("Final provider prompt leaked an internal wrapper")
