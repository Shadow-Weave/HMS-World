"""One-time semantic registration for serial continuity entities."""

import json
import hashlib
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from continuity.models import (
    CharacterContent,
    CharacterEntity,
    EntityRegistry,
    EntityRelation,
    PropContent,
    PropEntity,
    SceneContent,
    SceneEntity,
)
from interfaces import CharacterInScene, ShotDescription
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser


class RelationDraft(BaseModel):
    target_kind: Literal["character", "scene", "prop"]
    target_index: int
    relation: str


class CharacterContentDraft(BaseModel):
    outfit: str | None = None
    expression: str | None = None
    pose: str | None = None
    position: str | None = None
    orientation: str | None = None
    holding_prop_indices: list[int] | None = None
    emotion: str | None = None


class CharacterEntityDraft(BaseModel):
    character_idx: int
    identity: str
    persona: str
    session: str
    stable_appearance: str
    initial_content: CharacterContentDraft = Field(default_factory=CharacterContentDraft)
    relations: list[RelationDraft] = Field(default_factory=list)


class SceneEntityDraft(BaseModel):
    name: str
    initial_content: SceneContent | None = None
    relations: list[RelationDraft] = Field(default_factory=list)


class PropEntityDraft(BaseModel):
    name: str
    initial_content: PropContent | None = None
    relations: list[RelationDraft] = Field(default_factory=list)


class EntityRegistryDraft(BaseModel):
    characters: list[CharacterEntityDraft] = Field(default_factory=list)
    scenes: list[SceneEntityDraft] = Field(default_factory=list)
    props: list[PropEntityDraft] = Field(default_factory=list)


class ReplacementCharacterDraft(BaseModel):
    """The complete minimum registration for one character newly named by a replacement."""

    model_config = ConfigDict(extra="forbid")

    character_idx: int
    name: str = Field(min_length=1)
    static_features: str = Field(min_length=1)
    dynamic_features: str | None = None
    identity: str = Field(min_length=1)
    persona: str = Field(min_length=1)
    session: str = Field(min_length=1)
    stable_appearance: str = Field(min_length=1)
    initial_content: "ReplacementCharacterContentDraft" = Field(default_factory=lambda: ReplacementCharacterContentDraft())


class ReplacementCharacterContentDraft(BaseModel):
    """Replacement fields that are actually materialized; holding stays unknown."""

    model_config = ConfigDict(extra="forbid")

    outfit: str | None = None
    expression: str | None = None
    pose: str | None = None
    position: str | None = None
    orientation: str | None = None
    emotion: str | None = None


class ReplacementPropDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    initial_content: PropContent | None = None


class ReplacementRegistryExtension(BaseModel):
    """Only entities absent from the registry before a replacement is applied."""

    model_config = ConfigDict(extra="forbid")

    characters: list[ReplacementCharacterDraft] = Field(default_factory=list)
    props: list[ReplacementPropDraft] = Field(default_factory=list)


class ShotPropBinding(BaseModel):
    """Stable registry prop identities used by one executable shot."""

    model_config = ConfigDict(extra="forbid")

    shot_idx: int = Field(ge=0)
    prop_entity_ids: list[str] = Field(default_factory=list)


class ShotSceneBinding(BaseModel):
    """Stable registry scene identity used by one executable shot."""

    model_config = ConfigDict(extra="forbid")

    shot_idx: int = Field(ge=0)
    scene_entity_id: str = Field(min_length=1)


_SYSTEM_PROMPT = """
You are the continuity registrar for a short drama. Register only durable semantic facts available before production: each supplied character's identity, persona, current session context, stable appearance, initially known visible content and relationships; plus named scenes and props present in the screenplay. Unknown visual fields must be null. Do not invent future actions, transitions, images, image IDs, or production instructions.

The supplied character index is authoritative and every supplied character must appear exactly once. For a relationship, use target_kind and target_index. Character target_index is the supplied character index. Scene and prop target_index are zero-based positions in their output lists. For holding_prop_indices, use zero-based positions in the props output list.

{format_instructions}
""".strip()

_REPLACEMENT_SYSTEM_PROMPT = """
You are the continuity registrar after a replacement of the next unfilmed shot.
Register only genuinely new characters identified by the supplied unknown indices,
and genuinely new named props needed by that replacement. Return no existing
entity, no image IDs, no production instruction, and no speculative history. Do
not rewrite any supplied existing registry entity. Unknown visual facts are null.

{format_instructions}
""".strip()

_SHOT_PROP_BINDING_SYSTEM_PROMPT = """
You are the entity registrar binding one executable shot plan to an already fixed
prop registry. Return the stable entity IDs of every
registered prop materially present, handled, changed, revealed, or visually
established anywhere in that shot's executable visual plan.

Resolve ordinary descriptions and paraphrases semantically against the supplied
registry. Do not require the shot wording to repeat a registry name. Use only
supplied prop entity IDs, never create an entity, and never bind a prop merely
because it exists in the registry. Ignore audio. Return the supplied shot index
unchanged. An unrelated shot has an empty ID list.

{format_instructions}
""".strip()


_SHOT_SCENE_BINDING_SYSTEM_PROMPT = """
You are the entity registrar binding one executable shot plan to an already fixed
scene registry. Select exactly one registered scene entity ID for the setting
actually shown by this shot's executable visual plan. Resolve setting details
semantically against the supplied registered scenes. Use only a supplied scene
entity ID, never create an entity, and return the supplied shot index unchanged.

{format_instructions}
""".strip()


def _human_prompt(screenplay: str, characters: list[CharacterInScene], user_requirements: str) -> str:
    character_text = "\n".join(
        "\n".join([
            f"Character index: {character.idx}",
            f"Name: {character.identifier_in_scene}",
            f"Visible: {character.is_visible}",
            f"Static appearance: {character.static_features}",
            f"Current known features: {character.dynamic_features}",
        ])
        for character in characters
    )
    return "\n".join([
        "<SCREENPLAY>", screenplay, "</SCREENPLAY>",
        "<HMS_WORLD_CHARACTERS>", character_text, "</HMS_WORLD_CHARACTERS>",
        "<USER_REQUIREMENTS>", user_requirements, "</USER_REQUIREMENTS>",
    ])


def _replacement_extension_prompt(
    replacement_shot: ShotDescription,
    unknown_character_idxs: set[int],
    registry: EntityRegistry,
) -> str:
    """Keep the replacement call bounded to entity-relevant replacement text."""
    replacement_text = [
        "<VISUAL_DESCRIPTION>", replacement_shot.visual_desc, "</VISUAL_DESCRIPTION>",
        "<FIRST_FRAME_DESCRIPTION>", replacement_shot.ff_desc, "</FIRST_FRAME_DESCRIPTION>",
        "<MOTION_DESCRIPTION>", replacement_shot.motion_desc, "</MOTION_DESCRIPTION>",
    ]
    if replacement_shot.variation_type != "small":
        replacement_text.extend([
            "<LAST_FRAME_DESCRIPTION>", replacement_shot.lf_desc, "</LAST_FRAME_DESCRIPTION>",
        ])
    return "\n".join([
        "<REPLACEMENT_ENTITY_TEXT>", *replacement_text, "</REPLACEMENT_ENTITY_TEXT>",
        "<UNKNOWN_CHARACTER_IDXS>", ",".join(str(index) for index in sorted(unknown_character_idxs)), "</UNKNOWN_CHARACTER_IDXS>",
        "<EXISTING_REGISTRY_ENTITIES>",
        json.dumps(
            [{"entity_id": entity.entity_id, "name": entity.name} for entity in [*registry.characters, *registry.scenes, *registry.props]],
            ensure_ascii=False,
        ),
        "</EXISTING_REGISTRY_ENTITIES>",
    ])


def _shot_prop_binding_prompt(shot: ShotDescription, registry: EntityRegistry) -> str:
    executable = {
        "shot_idx": shot.idx,
        "visual_desc": shot.visual_desc,
        "first_frame_desc": shot.ff_desc,
        "motion_desc": shot.motion_desc,
    }
    if shot.variation_type != "small":
        executable["last_frame_desc"] = shot.lf_desc
    prop_payload = [
        {
            "entity_id": prop.entity_id,
            "name": prop.name,
            "initial_content": prop.initial_content.model_dump() if prop.initial_content is not None else None,
        }
        for prop in registry.props
    ]
    return "\n".join([
        "<REGISTERED_PROPS>",
        json.dumps(prop_payload, ensure_ascii=False),
        "</REGISTERED_PROPS>",
        "<EXECUTABLE_SHOT>",
        json.dumps(executable, ensure_ascii=False),
        "</EXECUTABLE_SHOT>",
    ])


def _shot_scene_binding_prompt(shot: ShotDescription, registry: EntityRegistry) -> str:
    executable = {
        "shot_idx": shot.idx,
        "visual_desc": shot.visual_desc,
        "first_frame_desc": shot.ff_desc,
        "motion_desc": shot.motion_desc,
    }
    if shot.variation_type != "small":
        executable["last_frame_desc"] = shot.lf_desc
    scene_payload = [
        {
            "entity_id": scene.entity_id,
            "name": scene.name,
            "initial_content": scene.initial_content,
        }
        for scene in registry.scenes
    ]
    return "\n".join([
        "<REGISTERED_SCENES>",
        json.dumps(scene_payload, ensure_ascii=False),
        "</REGISTERED_SCENES>",
        "<EXECUTABLE_SHOT>",
        json.dumps(executable, ensure_ascii=False),
        "</EXECUTABLE_SHOT>",
    ])


class EntityExtractor:
    """Own semantic entity registration and one-time shot-to-ID binding."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def extract_entities(self, screenplay: str, characters: list[CharacterInScene], user_requirements: str) -> EntityRegistry:
        _validate_character_inputs(characters)
        parser = PydanticOutputParser(pydantic_object=EntityRegistryDraft)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=_human_prompt(screenplay, characters, user_requirements)),
        ]
        chain = self.chat_model.bind(response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "EntityRegistryDraft",
                "schema": EntityRegistryDraft.model_json_schema(),
                "strict": True,
            },
        }) | parser
        draft: EntityRegistryDraft = await chain.ainvoke(messages)
        return materialize_entity_registry(draft, characters)

    async def bind_props_for_shot(
        self,
        shot: ShotDescription,
        registry: EntityRegistry,
    ) -> list[str]:
        """Resolve shot prose to stable prop IDs once, before Timeline projection."""
        if not registry.props:
            return []
        parser = PydanticOutputParser(pydantic_object=ShotPropBinding)
        known_prop_ids = {prop.entity_id for prop in registry.props}
        canonical_prop_order = [prop.entity_id for prop in registry.props]
        messages = [
            SystemMessage(content=_SHOT_PROP_BINDING_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=_shot_prop_binding_prompt(shot, registry)),
        ]
        binding: ShotPropBinding = await (self.chat_model | parser).ainvoke(messages)
        if binding.shot_idx != shot.idx:
            raise ValueError(f"Prop binder must return supplied shot {shot.idx}, got {binding.shot_idx}")
        if len(binding.prop_entity_ids) != len(set(binding.prop_entity_ids)):
            raise ValueError(f"Prop binder returned duplicate IDs for shot {binding.shot_idx}")
        unknown = set(binding.prop_entity_ids) - known_prop_ids
        if unknown:
            raise ValueError(f"Prop binder returned unregistered IDs for shot {binding.shot_idx}: {sorted(unknown)}")
        selected = set(binding.prop_entity_ids)
        return [prop_id for prop_id in canonical_prop_order if prop_id in selected]

    async def bind_scene_for_shot(
        self,
        shot: ShotDescription,
        registry: EntityRegistry,
    ) -> str:
        """Resolve an executable shot to its one stable scene ID before projection."""
        if not registry.scenes:
            raise ValueError("Scene binder requires at least one registered scene")
        if len(registry.scenes) == 1:
            return registry.scenes[0].entity_id
        parser = PydanticOutputParser(pydantic_object=ShotSceneBinding)
        messages = [
            SystemMessage(content=_SHOT_SCENE_BINDING_SYSTEM_PROMPT.format(
                format_instructions=parser.get_format_instructions(),
            )),
            HumanMessage(content=_shot_scene_binding_prompt(shot, registry)),
        ]
        binding: ShotSceneBinding = await (self.chat_model | parser).ainvoke(messages)
        if binding.shot_idx != shot.idx:
            raise ValueError(f"Scene binder must return supplied shot {shot.idx}, got {binding.shot_idx}")
        known_scene_ids = {scene.entity_id for scene in registry.scenes}
        if binding.scene_entity_id not in known_scene_ids:
            raise ValueError(
                f"Scene binder returned unregistered ID for shot {binding.shot_idx}: "
                f"{binding.scene_entity_id!r}"
            )
        return binding.scene_entity_id

    async def extend_for_replacement(
        self,
        replacement_shot: ShotDescription,
        unknown_character_idxs: set[int],
        registry: EntityRegistry,
        characters: list[CharacterInScene],
    ) -> tuple[EntityRegistry, list[CharacterInScene]]:
        """Register only replacement-introduced entities, with stable append-only IDs."""
        _validate_character_inputs(characters)
        unknown = set(unknown_character_idxs)
        existing_indices = {character.idx for character in characters}
        if unknown & existing_indices:
            raise ValueError("Replacement unknown character indices already exist in CharacterInScene")
        replacement_indices = set(replacement_shot.ff_vis_char_idxs)
        if replacement_shot.variation_type != "small":
            replacement_indices.update(replacement_shot.lf_vis_char_idxs)
        if not unknown <= replacement_indices:
            raise ValueError("Replacement extension may only register indices visible in its replacement shot")
        parser = PydanticOutputParser(pydantic_object=ReplacementRegistryExtension)
        messages = [
            SystemMessage(content=_REPLACEMENT_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=_replacement_extension_prompt(replacement_shot, unknown, registry)),
        ]
        extension: ReplacementRegistryExtension = await (self.chat_model | parser).ainvoke(messages)
        try:
            return _apply_replacement_extension(extension, unknown, registry, characters)
        except ValueError as exc:
            digest = hashlib.sha256(replacement_shot.model_dump_json().encode()).hexdigest()
            feedback_approved = any(
                item.get("replacement_sha256") == digest and item.get("reason")
                for item in getattr(self, "replacement_validation_feedback_approvals", [])
            )
            if feedback_approved:
                existing_names = {
                    _normalized_entity_name(entity.name, "existing entity")
                    for entity in [*registry.characters, *registry.scenes, *registry.props]
                }
                duplicate_names = [item.name for item in [*extension.characters, *extension.props]
                                   if _normalized_entity_name(item.name, "candidate entity") in existing_names]
                feedback = (
                    f"The previous candidate failed the existing validator: {exc}. "
                    f"Supplied unknown_character_idxs is {sorted(unknown)}; returned character_idx values "
                    f"were {[item.character_idx for item in extension.characters]}. "
                    f"Returned names already in the supplied registry: {duplicate_names}. "
                    "The existing validator rejects duplicate or already registered entity names. "
                    "Return corrected JSON satisfying all original registration instructions and validators; "
                    "preserve the supplied replacement and existing registry. Register only genuinely new entities."
                )
                receipt = getattr(self, "replacement_validation_feedback_receipt", None)
                if receipt is not None:
                    receipt({"replacement_sha256": digest, "shot_idx": replacement_shot.idx,
                             "error": str(exc), "feedback": feedback})
                corrected: ReplacementRegistryExtension = await (self.chat_model | parser).ainvoke([
                    *messages, AIMessage(content=extension.model_dump_json()), HumanMessage(content=feedback),
                ])
                return _apply_replacement_extension(corrected, unknown, registry, characters)
            approved = any(
                item.get("replacement_sha256") == digest and item.get("reason")
                for item in getattr(self, "keep_registry_on_extension_failure", [])
            )
            if not approved or unknown:
                raise
            receipt = {"replacement_sha256": digest, "error": str(exc),
                       "rejected_extension": extension.model_dump(),
                       "action": "keep_existing_registry", "shot_idx": replacement_shot.idx}
            self.extension_failure_receipt(receipt)
            print(f"USER_APPROVED_KEEP_REGISTRY shot={replacement_shot.idx} error={exc}", flush=True)
            return registry.model_copy(deep=True), []


def materialize_entity_registry(draft: EntityRegistryDraft, characters: list[CharacterInScene]) -> EntityRegistry:
    """Bind model drafts to deterministic IDs without name-based matching."""
    _validate_character_inputs(characters)
    character_drafts = {item.character_idx: item for item in draft.characters}
    expected_indices = {character.idx for character in characters}
    if set(character_drafts) != expected_indices or len(character_drafts) != len(draft.characters):
        raise ValueError("Entity extractor must return each supplied character index exactly once")
    if not draft.scenes:
        raise ValueError("Entity extractor must register at least one scene")
    scene_ids = [f"scene-{idx}" for idx in range(len(draft.scenes))]
    prop_ids = [f"prop-{idx}" for idx in range(len(draft.props))]

    def relations(items: list[RelationDraft]) -> list[EntityRelation]:
        return [_materialize_relation(item, expected_indices, scene_ids, prop_ids) for item in items]

    registry_characters = []
    for character in characters:
        item = character_drafts[character.idx]
        holding = _materialize_holding(item.initial_content.holding_prop_indices, prop_ids)
        registry_characters.append(CharacterEntity(
            entity_id=f"character-{character.idx}", name=character.identifier_in_scene,
            identity=item.identity, persona=item.persona, session=item.session,
            stable_appearance=item.stable_appearance,
            initial_content=CharacterContent(
                outfit=item.initial_content.outfit, expression=item.initial_content.expression,
                pose=item.initial_content.pose, position=item.initial_content.position,
                orientation=item.initial_content.orientation, holding=holding,
                emotion=item.initial_content.emotion,
            ),
            identity_image_ids=[], relations=relations(item.relations),
        ))
    return EntityRegistry(
        characters=registry_characters,
        scenes=[SceneEntity(entity_id=scene_ids[idx], name=item.name, initial_content=item.initial_content, relations=relations(item.relations)) for idx, item in enumerate(draft.scenes)],
        props=[PropEntity(entity_id=prop_ids[idx], name=item.name, initial_content=item.initial_content, relations=relations(item.relations)) for idx, item in enumerate(draft.props)],
    )


def _apply_replacement_extension(
    extension: ReplacementRegistryExtension,
    unknown_character_idxs: set[int],
    registry: EntityRegistry,
    characters: list[CharacterInScene],
) -> tuple[EntityRegistry, list[CharacterInScene]]:
    """Append validated replacement entities without mutating the caller's registry."""
    character_indices = [item.character_idx for item in extension.characters]
    if set(character_indices) != unknown_character_idxs or len(character_indices) != len(extension.characters):
        raise ValueError("Replacement extension must register every unknown character index exactly once")
    existing_character_ids = {entity.entity_id for entity in registry.characters}
    existing_names = {
        _normalized_entity_name(entity.name, "existing entity")
        for entity in [*registry.characters, *registry.scenes, *registry.props]
    }
    new_character_names = [_normalized_entity_name(item.name, "replacement character") for item in extension.characters]
    new_prop_names = [_normalized_entity_name(item.name, "replacement prop") for item in extension.props]
    if (
        len(set(new_character_names)) != len(new_character_names)
        or len(set(new_prop_names)) != len(new_prop_names)
        or set(new_character_names) & existing_names
        or set(new_prop_names) & existing_names
        or set(new_character_names) & set(new_prop_names)
    ):
        raise ValueError("Replacement extension contains duplicate or already registered entity names")
    prop_numbers = []
    for entity in registry.props:
        prefix, separator, suffix = entity.entity_id.partition("prop-")
        if prefix or not separator or not suffix.isdigit():
            raise ValueError(f"Existing prop has non-stable entity_id {entity.entity_id}")
        prop_numbers.append(int(suffix))
    next_prop_number = max(prop_numbers, default=-1) + 1
    added_characters = []
    added_character_entities = []
    for item in extension.characters:
        entity_id = f"character-{item.character_idx}"
        if entity_id in existing_character_ids:
            raise ValueError(f"Replacement extension would overwrite {entity_id}")
        added_characters.append(CharacterInScene(
            idx=item.character_idx,
            identifier_in_scene=item.name,
            is_visible=True,
            static_features=item.static_features,
            dynamic_features=item.dynamic_features,
        ))
        added_character_entities.append(CharacterEntity(
            entity_id=entity_id,
            name=item.name,
            identity=item.identity,
            persona=item.persona,
            session=item.session,
            stable_appearance=item.stable_appearance,
            initial_content=CharacterContent(
                outfit=item.initial_content.outfit,
                expression=item.initial_content.expression,
                pose=item.initial_content.pose,
                position=item.initial_content.position,
                orientation=item.initial_content.orientation,
                holding=None,
                emotion=item.initial_content.emotion,
            ),
        ))
    added_props = [
        PropEntity(
            entity_id=f"prop-{next_prop_number + index}",
            name=item.name,
            initial_content=item.initial_content,
        )
        for index, item in enumerate(extension.props)
    ]
    return EntityRegistry(
        characters=[*registry.characters, *added_character_entities],
        scenes=[scene.model_copy(deep=True) for scene in registry.scenes],
        props=[*registry.props, *added_props],
    ), added_characters


def _normalized_entity_name(name: str, kind: str) -> str:
    if not name or name != name.strip():
        raise ValueError(f"{kind} name must be non-empty and have no surrounding whitespace")
    normalized = name.casefold()
    if not normalized:
        raise ValueError(f"{kind} name normalizes to empty")
    return normalized


def _validate_character_inputs(characters: list[CharacterInScene]) -> None:
    indices = [character.idx for character in characters]
    if len(indices) != len(set(indices)):
        raise ValueError("Production characters must have unique idx values")


def _materialize_relation(draft: RelationDraft, character_indices: set[int], scene_ids: list[str], prop_ids: list[str]) -> EntityRelation:
    if draft.target_kind == "character":
        if draft.target_index not in character_indices:
            raise ValueError(f"Relation references unknown character index {draft.target_index}")
        target_entity_id = f"character-{draft.target_index}"
    elif draft.target_kind == "scene":
        target_entity_id = _indexed_id(scene_ids, draft.target_index, "scene")
    else:
        target_entity_id = _indexed_id(prop_ids, draft.target_index, "prop")
    return EntityRelation(target_entity_id=target_entity_id, relation=draft.relation)


def _materialize_holding(indices: list[int] | None, prop_ids: list[str]) -> list[str] | None:
    if indices is None:
        return None
    return [_indexed_id(prop_ids, index, "prop") for index in indices]


def _indexed_id(ids: list[str], index: int, kind: str) -> str:
    if index < 0 or index >= len(ids):
        raise ValueError(f"Reference to unknown {kind} index {index}")
    return ids[index]
