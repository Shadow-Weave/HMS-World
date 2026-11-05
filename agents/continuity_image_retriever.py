"""Arrange already located frame materials and choose portrait views once."""

import random
import hashlib

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from continuity.frame_material_context import FrameMaterialContext, fixed_material_purpose, history_sample_purpose
from utils.image import image_path_to_b64
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser


class RetrievedImage(BaseModel):
    candidate_id: str
    purpose: str = Field(min_length=1)


class ImageRetrievalResult(BaseModel):
    images: list[RetrievedImage]


class PortraitSelection(BaseModel):
    character_id: str
    image_id: str


class FrameSelections(BaseModel):
    selections: list[PortraitSelection]
    history_sample_ids: list[str] = Field(default_factory=list)
    native_candidate_ids: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = """
Choose exactly one identity-portrait image for each listed character. Select only
the supplied image IDs. You may select history sample IDs and native ViMax frame
candidate IDs within the shared stated capacity, or select none of either kind.
Select at most one history sample from each source shot. Return portrait
selections, history sample IDs, and native candidate IDs only.

The target description and state lineage jointly define the current target.
Explicit target instructions take priority; lineage supplies established
continuity facts that the target does not replace. Portraits are identity
references only. Choose their view for useful identity evidence, not as a pose,
expression, gaze, action, scale, or composition instruction.

History samples are actual past images. An observed lineage source locates a
source shot, not a particular sample, and does not prove that every sample shows
the lineage value. Select a history sample only when relevant continuity is
visibly present in that image; never use it as current framing or performance
instruction. Fixed materials are already reserved and must not be selected
again. Native candidates are optional ViMax visual inputs, not historical facts.
Preserve input order.

{format_instructions}
""".strip()


class ContinuityImageRetriever:
    """Material arranger with one optional portrait-view choice call."""

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.same_source_history_approvals = []
        self.correct_invalid_selection_once = False

    async def retrieve(self, context: FrameMaterialContext, target_description: str) -> ImageRetrievalResult:
        if not target_description or not target_description.strip():
            raise ValueError("Frame material arranger requires a non-empty target_description")
        context = _sample_history_context(context)
        fixed = [
            RetrievedImage(candidate_id=item.image_id, purpose=fixed_material_purpose(item.role))
            for item in context.fixed_materials
        ]
        selection_limit = context.top_k - context.required_slots
        if selection_limit < 0:
            raise ValueError("Frame material context required slots exceed top_k")
        if not context.portrait_candidates and (
            selection_limit == 0 or (not _history_candidates(context) and not context.native_candidates)
        ):
            return ImageRetrievalResult(images=fixed)
        correction = None
        for attempt in range(2 if self.correct_invalid_selection_once else 1):
            selections = await self._choose_materials(target_description, context, correction=correction)
            approved = any(
                item.get("shot_idx") == context.shot_idx
                and item.get("frame_type") == context.frame_type
                and item.get("target_sha256") == hashlib.sha256(target_description.encode()).hexdigest()
                and item.get("selections") == selections.model_dump()
                and item.get("reason")
                for item in self.same_source_history_approvals
            )
            try:
                portraits, history, native = _validate_frame_selections(
                    context, selections, allow_same_source_history=approved,
                )
                break
            except ValueError as exc:
                if not self.correct_invalid_selection_once or attempt:
                    raise
                correction = (selections, str(exc))
                print(f"SELECTION_CORRECTION shot={context.shot_idx} frame={context.frame_type} error={exc}", flush=True)
        if approved:
            print(f"USER_APPROVED_SAME_SOURCE_HISTORY shot={context.shot_idx} frame={context.frame_type}", flush=True)
        result = ImageRetrievalResult(images=[*fixed, *portraits, *history, *native])
        if len(result.images) < context.required_slots or len(result.images) > context.top_k:
            raise ValueError("Frame material context slot count changed during portrait arrangement")
        return result

    async def _choose_materials(
        self,
        target_description: str,
        context: FrameMaterialContext,
        *, correction: tuple[FrameSelections, str] | None = None,
    ) -> FrameSelections:
        parser = PydanticOutputParser(pydantic_object=FrameSelections)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=_selection_message(target_description, context)),
        ]
        if correction is not None:
            previous, error = correction
            messages.extend([
                AIMessage(content=previous.model_dump_json()),
                HumanMessage(content=f"Your selection failed validation: {error}. Return a corrected complete selection using the same supplied candidates and target. All original constraints still apply."),
            ])
        return await (self.chat_model | parser).ainvoke(messages)


def _sample_history_context(context: FrameMaterialContext) -> FrameMaterialContext:
    """Offer the last actual sample and one reproducible random sample per source shot."""
    groups = []
    for group in context.history_sample_groups:
        samples = group.samples
        if len(samples) > 1:
            rng = random.Random(f"{context.shot_idx}:{context.frame_type}:{group.source.shot_idx}")
            samples = [samples[rng.randrange(len(samples) - 1)], samples[-1]]
        groups.append(group.model_copy(update={"samples": samples}))
    return context.model_copy(update={"history_sample_groups": groups})


def _selection_message(target_description: str, context: FrameMaterialContext) -> list[dict]:
    import json

    groups = [
        {
            "character_id": group.character_id,
            "candidates": [
                {"image_id": candidate.image_id, "view": candidate.view}
                for candidate in group.candidates
            ],
        }
        for group in context.portrait_candidates
    ]
    text = "\n".join([
        "<CURRENT_ENDPOINT>", f"shot {context.shot_idx} {context.frame_type} frame", "</CURRENT_ENDPOINT>",
        "<TARGET_DESCRIPTION>", target_description, "</TARGET_DESCRIPTION>",
        "<STATE_LINEAGE>", json.dumps([item.model_dump() for item in context.field_lineage], ensure_ascii=False, separators=(",", ":")), "</STATE_LINEAGE>",
        "<RESERVED_FIXED_MATERIAL_ROLES>",
        json.dumps([fixed_material_purpose(item.role) for item in context.fixed_materials], ensure_ascii=False, separators=(",", ":")),
        "</RESERVED_FIXED_MATERIAL_ROLES>",
        "<IDENTITY_PORTRAIT_CANDIDATES>",
        json.dumps(groups, ensure_ascii=False, separators=(",", ":")),
        "</IDENTITY_PORTRAIT_CANDIDATES>",
        "<OPTIONAL_SELECTION_LIMIT>", str(context.top_k - context.required_slots), "</OPTIONAL_SELECTION_LIMIT>",
    ])
    content: list[dict] = [{"type": "text", "text": text}]
    for group, sample in _history_candidates(context):
        content.append({
            "type": "text",
            "text": (
                f"History sample id={sample.image_id}; source shot "
                f"{group.source.shot_idx}; time={sample.time_seconds:g}s"
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_path_to_b64(sample.asset_path, mime=True)},
        })
    for candidate in context.native_candidates:
        content.append({
            "type": "text",
            "text": (
                f"Native candidate id={candidate.candidate_id}; "
                f"description={candidate.description}"
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_path_to_b64(candidate.asset_path, mime=True)},
        })
    return content


def _validate_frame_selections(
    context: FrameMaterialContext,
    selections: FrameSelections,
    *,
    allow_same_source_history: bool = False,
) -> tuple[list[RetrievedImage], list[RetrievedImage], list[RetrievedImage]]:
    expected_groups = context.portrait_candidates
    selected_images: list[RetrievedImage] = []
    for group, selection in zip(expected_groups, selections.selections):
        if selection.character_id != group.character_id:
            raise ValueError("Portrait selections must preserve visible-character order without extras")
        candidates = {candidate.image_id for candidate in group.candidates}
        if selection.image_id not in candidates:
            raise ValueError(f"Portrait selection {selection.image_id} is not a candidate for {group.character_id}")
        selected_images.append(RetrievedImage(
            candidate_id=selection.image_id,
            purpose="character identity portrait",
        ))
    if len({item.candidate_id for item in selected_images}) != len(selected_images):
        raise ValueError("Portrait selections must not reuse one image for multiple characters")
    history_candidates = _history_candidates(context)
    history_by_id = {sample.image_id: (group, sample) for group, sample in history_candidates}
    selected_history_ids = selections.history_sample_ids
    if len(selected_history_ids) != len(set(selected_history_ids)):
        raise ValueError("History samples must not be selected more than once")
    unknown_history = [image_id for image_id in selected_history_ids if image_id not in history_by_id]
    if unknown_history:
        raise ValueError(f"Unknown history sample IDs: {unknown_history}")
    selected_history_shots = [history_by_id[image_id][0].source.shot_idx for image_id in selected_history_ids]
    if not allow_same_source_history and len(selected_history_shots) != len(set(selected_history_shots)):
        raise ValueError("Select at most one history sample from each source shot")
    history_selected = set(selected_history_ids)
    history_images = [
        RetrievedImage(candidate_id=sample.image_id, purpose=history_sample_purpose(group, sample))
        for group, sample in history_candidates
        if sample.image_id in history_selected
    ]
    candidate_ids = [candidate.candidate_id for candidate in context.native_candidates]
    selected_native_ids = selections.native_candidate_ids
    if len(selected_native_ids) != len(set(selected_native_ids)):
        raise ValueError("Native ViMax candidates must not be selected more than once")
    unknown = [candidate_id for candidate_id in selected_native_ids if candidate_id not in candidate_ids]
    if unknown:
        raise ValueError(f"Unknown native ViMax candidates: {unknown}")
    selection_limit = context.top_k - context.required_slots
    if len(selected_history_ids) + len(selected_native_ids) > selection_limit:
        raise ValueError(
            f"Optional history/native selection exceeds remaining capacity {selection_limit}"
        )
    selected = set(selected_native_ids)
    native_images = [
        RetrievedImage(candidate_id=candidate.candidate_id, purpose="native ViMax frame candidate")
        for candidate in context.native_candidates
        if candidate.candidate_id in selected
    ]
    return selected_images, history_images, native_images


def _history_candidates(context: FrameMaterialContext):
    return [
        (group, sample)
        for group in context.history_sample_groups
        for sample in group.samples
    ]
