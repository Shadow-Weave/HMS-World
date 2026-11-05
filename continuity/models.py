"""Entity, shot-plan and state schemas shared by continuity consumers.

State merging and story-order transitions remain in continuity.timeline.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interfaces import ShotDescription


SceneContent: TypeAlias = dict[str, str | None]


class CharacterContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outfit: str | None = Field(default=None, description="Current mutable visible appearance: clothing and wearing state plus conspicuous hair, makeup, dirt, or damage; exclude stable identity.")
    expression: str | None = Field(default=None, description="Visible facial display supported by the current take, without inferred inner motive.")
    pose: str | None = Field(default=None, description="Concrete body and behavior state, including relevant hand configuration and any action still in progress.")
    position: str | None = Field(default=None, description="Physical or narrative location in the world or set, stated relative to registered entities when useful; exclude screen placement.")
    orientation: str | None = Field(default=None, description="Physical facing, gaze, or attention target in the world; exclude screen direction.")
    holding: list[str] | None = Field(default=None, description="Registered prop IDs established as grasped, not merely touched; holding one member of a registered group uses that group's ID. [] confirms no registered prop is held, while null means unknown.")
    emotion: str | None = Field(default=None, description="Performed emotional state only when supported by visible behavior; null when the performance does not establish one.")


class PropContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: str | None = Field(default=None, description="Current holder, support, world location, spatial relation, and established quantity distribution for this registered prop or group.")
    condition: str | None = Field(default=None, description="Current physical state such as open, closed, lit, damaged, wet, consumed, or intact.")


class EntityRelation(BaseModel):
    target_entity_id: str
    relation: str


class CharacterEntity(BaseModel):
    entity_id: str
    name: str
    identity: str
    persona: str
    session: str
    stable_appearance: str
    initial_content: CharacterContent = Field(default_factory=CharacterContent)
    identity_image_ids: list[str] = Field(default_factory=list)
    relations: list[EntityRelation] = Field(default_factory=list)


class SceneEntity(BaseModel):
    entity_id: str
    name: str
    initial_content: SceneContent | None = None
    relations: list[EntityRelation] = Field(default_factory=list)


class PropEntity(BaseModel):
    entity_id: str
    name: str
    initial_content: PropContent | None = None
    relations: list[EntityRelation] = Field(default_factory=list)


class EntityRegistry(BaseModel):
    characters: list[CharacterEntity] = Field(default_factory=list)
    scenes: list[SceneEntity] = Field(default_factory=list)
    props: list[PropEntity] = Field(default_factory=list)


class InitialDesignSource(BaseModel):
    """A field inherited from the entity registry, not an observed image."""

    source_kind: Literal["initial_design"] = "initial_design"


class ObservedShotSource(BaseModel):
    """The observed shot where a field was first added or last changed."""

    source_kind: Literal["observed"] = "observed"
    shot_idx: int = Field(ge=0)


StateFieldSource = Annotated[
    InitialDesignSource | ObservedShotSource,
    Field(discriminator="source_kind"),
]


class PlannedCharacterAtTime(BaseModel):
    entity_id: str
    current_content: CharacterContent = Field(default_factory=CharacterContent)
    field_sources: dict[str, StateFieldSource] = Field(default_factory=dict)
    relations: list[EntityRelation] = Field(default_factory=list)


class PlannedSceneAtTime(BaseModel):
    entity_id: str
    current_content: SceneContent | None = None
    field_sources: dict[str, StateFieldSource] = Field(default_factory=dict)
    relations: list[EntityRelation] = Field(default_factory=list)


class PlannedPropAtTime(BaseModel):
    entity_id: str
    current_content: PropContent | None = None
    field_sources: dict[str, StateFieldSource] = Field(default_factory=dict)
    relations: list[EntityRelation] = Field(default_factory=list)


class ShotPlan(BaseModel):
    shot: ShotDescription
    characters: list[PlannedCharacterAtTime] = Field(default_factory=list)
    scene: PlannedSceneAtTime
    props: list[PlannedPropAtTime] = Field(default_factory=list)
    bound_prop_entity_ids: list[str] | None = None


class CameraState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_scale: str | None = Field(default=None, description="Shot scale shown by the camera.")
    angle: str | None = Field(default=None, description="Camera angle and viewpoint shown.")
    composition: str | None = Field(default=None, description="2D screen placement, framing, screen direction, layering, evidenced eyelines, axis relation, and relevant offscreen state. Anatomical left/right belongs to the character, not this field.")


class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    characters: dict[str, CharacterContent | None] = Field(default_factory=dict)
    props: dict[str, PropContent | None] = Field(default_factory=dict)
    scenes: dict[str, SceneContent | None] = Field(default_factory=dict)
    camera: CameraState | None = None


class PlannedPosition(BaseModel):
    phase: Literal["planned"] = "planned"
    shot_idx: int
    plan: ShotPlan


class ObservedStateSources(BaseModel):
    """Value-free origins for non-null fields in one observed WorldState."""

    model_config = ConfigDict(extra="forbid")

    characters: dict[str, dict[str, StateFieldSource]] = Field(default_factory=dict)
    props: dict[str, dict[str, StateFieldSource]] = Field(default_factory=dict)
    scenes: dict[str, dict[str, StateFieldSource]] = Field(default_factory=dict)


class ObservedPosition(BaseModel):
    phase: Literal["observed"] = "observed"
    shot_idx: int
    state: WorldState
    sources: ObservedStateSources

    @property
    def final_state(self) -> WorldState:
        return self.state

    @model_validator(mode="after")
    def _validate_sources(self):
        _validate_observed_sources(self.state, self.sources)
        return self


TimelinePosition = Annotated[
    PlannedPosition | ObservedPosition,
    Field(discriminator="phase"),
]


def _non_null_values(content) -> dict[str, object]:
    if content is None:
        return {}
    values = content.model_dump() if isinstance(content, BaseModel) else dict(content)
    return {field: value for field, value in values.items() if value is not None}


def _validate_observed_sources(state: WorldState, sources: ObservedStateSources) -> None:
    _validate_entity_source_index("characters", state.characters, sources.characters)
    _validate_entity_source_index("props", state.props, sources.props)
    _validate_entity_source_index("scenes", state.scenes, sources.scenes)


def _validate_entity_source_index(
    kind: str,
    contents: dict,
    source_index: dict[str, dict[str, StateFieldSource]],
) -> None:
    if set(source_index) - set(contents):
        raise ValueError(f"Observed {kind} source index has unknown entities")
    for entity_id, content in contents.items():
        _validate_content_source_index(f"{kind}.{entity_id}", content, source_index.get(entity_id, {}))


def _validate_content_source_index(
    label: str,
    content,
    field_sources: dict[str, StateFieldSource],
) -> None:
    values = _non_null_values(content)
    if set(field_sources) != set(values):
        raise ValueError(f"Observed {label} sources must exactly cover non-null state fields")
