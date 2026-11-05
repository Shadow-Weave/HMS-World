"""Observe one completed take into a sparse, shot-scoped WorldState."""

import json
import hashlib
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.exceptions import OutputParserException
from continuity.shot_sampling import SampledFrame
from continuity.models import EntityRegistry, ShotPlan, WorldState
from utils.image import image_path_to_b64
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser


_SYSTEM_PROMPT = """
ROLE
You are hms-world's Continuity Observer. Read one completed take as it unfolds and report sparse factual updates describing its endpoint for this shot's participants.

INPUTS
- PARTICIPANT_BASELINE: Previously recorded content for this shot's participants, for continuity context.
- TEMPORAL_BOARD: Observed sequence of actions and state transitions, labelled T0, T1, ... with source times. CHAR and SCENE labels mark identity and setting references outside the take's timeline.
- PARTICIPANT_SCOPE: The only entity IDs you may output.

READING THE TAKE
Follow the temporal frames from the opening to the last sampled moment. Read each participant's appearance, position, interactions, and changes through that sequence. Let the visible progression establish how an action unfolds. Read the currently visible state from these frames; correct outdated baseline values when a change is visible. The baseline supplies continuity for lasting facts outside the current view.

An initial reading can become more specific through a closer view. Record facts already established clearly by the board directly. Use inspect_samples only when a specific visual question remains uncertain or compared readings conflict, and a full-size frame is likely to clarify it. Select up to three T-labelled frames that address that question, passing their numbers as integers (T0 is 0). Read surrounding frames when the relationship unfolds over time, and use the last T frame to locate the endpoint.

Each sample establishes what is visible at its own time; the intervals between samples are not additional observations. Reference images anchor identity and setting, while temporal frames establish the take's behavior. Distinguish visible facts from interpretations of their meaning; retain only interpretations supported by the observed sequence.

Describe the state at the last sampled moment. Earlier frames explain changes and lasting facts, rather than turning the output into a process transcript. Describe holding, support, and quantity distribution as one coherent endpoint across character and prop fields.

STATE TRANSITION LOGIC
The Timeline, not you, owns accumulated state:

1. Output only fields actually established or changed by the TEMPORAL_BOARD.
2. Omit a field when evidence is insufficient: omission means unknown/no-op.
3. Use null only when the board clearly establishes that the field is absent. An empty entity object has no updates.
4. Do not output IDs outside PARTICIPANT_SCOPE. Do not fill in off-screen entities.

OBSERVATION DOMAINS
- Character: Performance, body pose, facing vector, and gaze direction.
- Props & Objects: Contact, holding state, physical support, displacement, and quantity.
- Appearance: Changes in costume, hair, or physical condition.
- Environment: Set dressing modifications, scene state, and visible marks.
- Camera: Framing, angle, and screen-space relations for this shot only.

OUTPUT
Output one strict sparse WorldState JSON matching the provided output contract.

{format_instructions}
""".strip()


_OBSERVER_OUTPUT_CONTRACT = """
SPARSE WORLDSTATE OUTPUT CONTRACT
Only output IDs listed in PARTICIPANT_SCOPE. Omitted IDs, entity null, and empty entity
objects are no-op. Omit unknown fields. Use an explicit null field value only when that
field is confirmed absent. Do not output every Registry entity.

WorldState
- characters: object keyed by in-scope character ID; each value is sparse CharacterContent or null.
- props: object keyed by in-scope prop ID; each value is sparse PropContent or null.
- scenes: object keyed by the one in-scope scene ID; each value is a sparse scene object or null.
- camera: CameraState for this shot only, or null when this shot's camera facts are unknown.

CharacterContent fields: outfit, expression, pose, position, orientation, holding, emotion.
holding: a list of in-scope registered prop IDs, not object names; include established grasp, not mere contact. Holding a member of a registered group uses that group's ID; describe its quantity and distribution in the prop's position. [] confirms no registered prop is held; omit it when unknown.
PropContent fields: position, condition.
position: current holder, support, spatial relation, and established quantity distribution.
condition: physical state such as open, closed, intact, damaged, or illuminated.

CameraState
- shot_scale: string or null.
- angle: string or null.
- composition: string or null.
""".strip()


_INSPECT_SAMPLES_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_samples",
        "description": "View up to three original-size frames by the T numbers printed on the board, when a reading is uncertain or compared readings conflict. T0 is frame 0.",
        "parameters": {
            "type": "object",
            "properties": {"frames": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "maxItems": 3}},
            "required": ["frames"],
            "additionalProperties": False,
        },
    },
}


class ShotObserver:
    """Read a temporal board, optionally inspect samples, and return one update."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def observe(
        self,
        plan: ShotPlan,
        board_path: str | Path,
        samples: list[SampledFrame],
        registry: EntityRegistry,
        reference_slot_map: list[dict[str, str | int]],
    ) -> WorldState:
        shot = plan.shot
        if not samples:
            raise ValueError("ShotObserver requires at least one sampled frame")
        if any(sample.shot_idx != shot.idx for sample in samples):
            raise ValueError(f"ShotObserver samples must all belong to shot {shot.idx}")
        sample_ids = [sample.image_id for sample in samples]
        duplicates = sorted({image_id for image_id in sample_ids if sample_ids.count(image_id) > 1})
        if duplicates:
            raise ValueError(f"ShotObserver samples must have unique stable image IDs: {duplicates}")
        board = Path(board_path)
        if not board.is_file():
            raise ValueError(f"ShotObserver cannot attach missing observation board {board}")
        _validate_slot_maps(samples, reference_slot_map, _participant_scope(plan))
        state = _participant_baseline(plan)
        messages = _observer_messages(
            board,
            samples,
            state,
            _participant_registry(plan, registry),
            _participant_scope(plan),
        )
        parser = PydanticOutputParser(pydantic_object=WorldState)
        messages[0] = SystemMessage(content=_SYSTEM_PROMPT.format(format_instructions=_OBSERVER_OUTPUT_CONTRACT))
        inspecting_model = self.chat_model.bind_tools([_INSPECT_SAMPLES_TOOL], strict=True)
        for round_index in range(3):
            # Two optional inspection rounds, then one terminal-state response.
            model = inspecting_model if round_index < 2 else self.chat_model
            if round_index == 2:
                messages.append(HumanMessage(content="Return the sparse terminal WorldState from the views available now. Omit fields that these views do not establish."))
            response = await model.ainvoke(messages)
            if not response.tool_calls:
                try:
                    parsed: WorldState = parser.invoke(response)
                except OutputParserException:
                    if round_index != 0:
                        raise
                    messages.extend([response, HumanMessage(content=(
                        'The previous response was not a valid WorldState JSON or a function call. '
                        'If you need inspect_samples, use the actual function-call interface with integer '
                        'frame numbers, for example {"frames":[10]}; T10 is a picture label, not a JSON value. '
                        'Otherwise return only the sparse terminal WorldState JSON. '
                        'Do not print XML tool-call tags or an unfinished draft.'
                    ))])
                    continue
                try:
                    return _validated_world_state(parsed, plan)
                except ValueError as exc:
                    approvals = getattr(self, "validation_feedback_approvals", [])
                    receipt = getattr(self, "validation_feedback_receipt", None)
                    matching = [item for item in approvals
                                if item.get("shot_idx") == shot.idx and item.get("reason")
                                and str(exc).startswith(item.get("error_prefix", "Observer output has IDs outside participant scope for characters:"))]
                    if not matching or receipt is None:
                        raise
                    video_digest = hashlib.sha256((board.parent / "video.mp4").read_bytes()).hexdigest()
                    if not any(item.get("video_sha256") == video_digest for item in matching):
                        raise
                    feedback = (
                        f"The previous candidate failed the existing strict validator: {exc}. "
                        "Return a corrected sparse WorldState satisfying the original PARTICIPANT_SCOPE, "
                        "output contract, and all original validators. Preserve the supplied plan, "
                        "evidence, baseline, and registry."
                    )
                    # Consume the explicit approval before submitting the one correction call.
                    receipt({"shot_idx": shot.idx, "video_sha256": video_digest,
                             "error": str(exc), "feedback": feedback})
                    self.validation_feedback_approvals = []
                    corrected = await self.chat_model.ainvoke([
                        *messages, response, HumanMessage(content=feedback),
                    ])
                    return _validated_world_state(parser.invoke(corrected), plan)
            if round_index == 2:
                raise ValueError("Observer requested inspection beyond its two available rounds")
            messages.append(response)
            tool_messages, detail_view = _inspect_samples(response.tool_calls, samples)
            messages.extend(tool_messages)
            messages.append(detail_view)
        raise AssertionError("Observer must return within three model calls")


def _participant_baseline(plan: ShotPlan) -> WorldState:
    """Expose only this execution plan's current participant contents."""
    return WorldState(
        characters={item.entity_id: item.current_content.model_copy(deep=True) for item in plan.characters},
        props={
            item.entity_id: item.current_content.model_copy(deep=True) if item.current_content is not None else None
            for item in plan.props
        },
        scenes={plan.scene.entity_id: dict(plan.scene.current_content or {}) or None},
        camera=None,
    )


def _participant_registry(plan: ShotPlan, registry: EntityRegistry) -> EntityRegistry:
    character_ids = {item.entity_id for item in plan.characters}
    prop_ids = {item.entity_id for item in plan.props}
    return EntityRegistry(
        characters=[item.model_copy(deep=True) for item in registry.characters if item.entity_id in character_ids],
        props=[item.model_copy(deep=True) for item in registry.props if item.entity_id in prop_ids],
        scenes=[item.model_copy(deep=True) for item in registry.scenes if item.entity_id == plan.scene.entity_id],
    )


def _participant_scope(plan: ShotPlan) -> dict[str, list[str]]:
    return {
        "characters": [item.entity_id for item in plan.characters],
        "props": [item.entity_id for item in plan.props],
        "scenes": [plan.scene.entity_id],
    }


def _observer_messages(
    board_path: Path,
    samples: list[SampledFrame],
    previous_state: WorldState,
    registry: EntityRegistry,
    participant_scope: dict[str, list[str]],
) -> list:
    registry_names = {
        "characters": [{"entity_id": item.entity_id, "name": item.name} for item in registry.characters],
        "props": [{"entity_id": item.entity_id, "name": item.name} for item in registry.props],
        "scenes": [{"entity_id": item.entity_id, "name": item.name} for item in registry.scenes],
    }
    content: list[dict] = [
        {"type": "text", "text": "<PARTICIPANT_SCOPE>\n" + json.dumps(participant_scope, ensure_ascii=False) + "\n</PARTICIPANT_SCOPE>"},
        {"type": "text", "text": "<REGISTRY_ENTITY_NAMES>\n" + json.dumps(registry_names, ensure_ascii=False) + "\n</REGISTRY_ENTITY_NAMES>"},
        {"type": "text", "text": "<PARTICIPANT_BASELINE>\n" + previous_state.model_dump_json() + "\n</PARTICIPANT_BASELINE>"},
        {"type": "text", "text": f"TEMPORAL_BOARD: actual frames T0 through T{len(samples) - 1} in chronological order. CHAR and SCENE pictures are references, outside the take's timeline."},
        {"type": "image_url", "image_url": {"url": image_path_to_b64(str(board_path), mime=True)}},
    ]
    return [SystemMessage(content=""), HumanMessage(content=content)]


def _inspect_samples(tool_calls: list[dict], samples: list[SampledFrame]) -> tuple[list[ToolMessage], HumanMessage]:
    selected_frames: set[int] = set()
    tool_messages = []
    for call in tool_calls:
        if call["name"] != "inspect_samples":
            raise ValueError(f"Observer requested unsupported tool: {call['name']}")
        args = call["args"]
        frames = args.get("frames")
        if (
            set(args) != {"frames"}
            or not isinstance(frames, list)
            or not 1 <= len(frames) <= 3
            or any(type(frame) is not int or not 0 <= frame < len(samples) for frame in frames)
        ):
            raise ValueError("Observer inspection requires one to three T-frame numbers from this take")
        selected_frames.update(frames)
        tool_messages.append(ToolMessage(
            content="Requested original-size frames follow below.",
            tool_call_id=call["id"],
            name="inspect_samples",
        ))
    content = []
    for frame, sample in enumerate(samples):
        if frame in selected_frames:
            content.extend([
                {"type": "text", "text": f"T{frame} at {sample.time_seconds:g}s, original image size, on the same take's timeline."},
                {"type": "image_url", "image_url": {"url": image_path_to_b64(sample.asset_path, mime=True), "detail": "high"}},
            ])
    return tool_messages, HumanMessage(content=content)


def _validate_slot_maps(
    samples: list[SampledFrame],
    reference_slot_map: list[dict[str, str | int]],
    participant_scope: dict[str, list[str]],
) -> None:
    if len(reference_slot_map) > 10:
        raise ValueError("Observation board reference slots must leave at least two temporal slots")
    if len(reference_slot_map) + len(samples) != 12:
        raise ValueError("Observation board must contain exactly twelve row-major slots")
    expected_slots = list(range(len(reference_slot_map)))
    actual_slots = [item.get("board_slot") for item in reference_slot_map]
    if actual_slots != expected_slots:
        raise ValueError("Observation board reference slots must be ordered contiguous leading slots")
    for item in reference_slot_map:
        if set(item) != {"board_slot", "kind", "entity_id", "image_id"}:
            raise ValueError("Observation board reference slot map has unsupported fields")
        if item["kind"] not in {"CHARACTER_IDENTITY", "SCENE_HISTORY"}:
            raise ValueError("Observation board reference slot kind is unsupported")
        if not isinstance(item["entity_id"], str) or not isinstance(item["image_id"], str):
            raise ValueError("Observation board reference slot requires stable entity and image IDs")
        if (
            item["kind"] == "CHARACTER_IDENTITY"
            and item["entity_id"] not in participant_scope["characters"]
        ):
            raise ValueError(
                "Observation CHARACTER_IDENTITY slot references an entity outside "
                "this shot's participant scope"
            )
        if (
            item["kind"] == "SCENE_HISTORY"
            and item["entity_id"] not in participant_scope["scenes"]
        ):
            raise ValueError(
                "Observation SCENE_HISTORY slot references an entity outside "
                "this shot's participant scope"
            )


def _validated_world_state(
    parsed: WorldState,
    plan: ShotPlan,
) -> WorldState:
    _require_scope_subset(parsed.characters, {item.entity_id for item in plan.characters}, "characters")
    allowed_props = {item.entity_id for item in plan.props}
    _require_scope_subset(parsed.props, allowed_props, "props")
    _require_scope_subset(parsed.scenes, {plan.scene.entity_id}, "scenes")
    _require_scoped_holding_ids(parsed, allowed_props)
    return parsed.model_copy(deep=True)


def _require_scope_subset(actual: dict, allowed: set[str], kind: str) -> None:
    unexpected = sorted(set(actual) - allowed)
    if unexpected:
        raise ValueError(f"Observer output has IDs outside participant scope for {kind}: {unexpected}")


def _require_scoped_holding_ids(parsed: WorldState, allowed_prop_ids: set[str]) -> None:
    for character_id, content in parsed.characters.items():
        if content is None:
            continue
        if "holding" not in content.model_fields_set or content.holding is None:
            continue
        if len(content.holding) != len(set(content.holding)):
            raise ValueError(f"Observer output has duplicate held prop IDs for {character_id}")
        unknown = set(content.holding) - allowed_prop_ids
        if unknown:
            raise ValueError(
                f"Observer output has held prop IDs outside this shot's participant scope for {character_id}: {sorted(unknown)}"
            )
