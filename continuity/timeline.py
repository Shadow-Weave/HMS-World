"""Entity-at-time contracts and single-track planned-to-observed transitions."""

from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interfaces import ShotDescription
# Re-export legacy type paths while keeping schemas independent of the state engine.
from continuity.models import (
    SceneContent,
    CharacterContent,
    PropContent,
    EntityRelation,
    CharacterEntity,
    SceneEntity,
    PropEntity,
    EntityRegistry,
    InitialDesignSource,
    ObservedShotSource,
    StateFieldSource,
    PlannedCharacterAtTime,
    PlannedSceneAtTime,
    PlannedPropAtTime,
    ShotPlan,
    CameraState,
    WorldState,
    PlannedPosition,
    ObservedStateSources,
    ObservedPosition,
    TimelinePosition,
    _non_null_values,
    _validate_content_source_index,
    _validate_entity_source_index,
    _validate_observed_sources,
)


class Timeline(BaseModel):
    """A single story-order sequence of planned-or-observed positions only."""

    positions: list[TimelinePosition]

    @model_validator(mode="after")
    def _validate_story_order(self):
        actual = [position.shot_idx for position in self.positions]
        expected = list(range(len(self.positions)))
        if actual != expected:
            raise ValueError(f"Timeline shot_idx values must be contiguous story order {expected}, got {actual}")
        for position in self.positions:
            if isinstance(position, PlannedPosition) and position.plan.shot.idx != position.shot_idx:
                raise ValueError(f"Planned position {position.shot_idx} contains shot {position.plan.shot.idx}")
        return self

    @classmethod
    def initialize(
        cls,
        registry: EntityRegistry,
        shots: list[ShotDescription],
        prop_bindings: Mapping[int, list[str]] | None = None,
        scene_bindings: Mapping[int, str] | None = None,
    ) -> "Timeline":
        if [shot.idx for shot in shots] != list(range(len(shots))):
            raise ValueError("Timeline initialization requires shots in contiguous story order")
        if prop_bindings is None:
            if registry.props:
                raise ValueError("Timeline initialization requires explicit prop bindings when props are registered")
            prop_bindings = {shot.idx: [] for shot in shots}
        if set(prop_bindings) != {shot.idx for shot in shots}:
            raise ValueError("Timeline initialization requires one explicit prop binding for every shot")
        if scene_bindings is None:
            if len(registry.scenes) != 1:
                raise ValueError(
                    "Timeline initialization requires explicit scene bindings when the registry does not contain exactly one scene"
                )
            scene_bindings = {shot.idx: registry.scenes[0].entity_id for shot in shots}
        if set(scene_bindings) != {shot.idx for shot in shots}:
            raise ValueError("Timeline initialization requires one explicit scene binding for every shot")
        known_scene_ids = {scene.entity_id for scene in registry.scenes}
        unknown_scene_ids = set(scene_bindings.values()) - known_scene_ids
        if unknown_scene_ids:
            raise ValueError(f"Timeline scene bindings reference unregistered IDs: {sorted(unknown_scene_ids)}")
        return cls(positions=[
            PlannedPosition(
                shot_idx=shot.idx,
                plan=project_shot_plan(
                    registry,
                    None,
                    shot,
                    prop_entity_ids=prop_bindings[shot.idx],
                    scene_entity_id=scene_bindings[shot.idx],
                ),
            )
            for shot in shots
        ])

    def position(self, shot_idx: int) -> TimelinePosition:
        """Read one explicit history position without exposing storage aliases."""
        return self._stored_position(shot_idx).model_copy(deep=True)

    def refresh_for_execution(self, shot_idx: int, registry: EntityRegistry) -> PlannedPosition:
        current = self._require_planned(shot_idx)
        previous_state = None
        previous_source = None
        if shot_idx > 0:
            previous = self._stored_position(shot_idx - 1)
            if not isinstance(previous, ObservedPosition):
                raise ValueError(f"Shot {shot_idx} cannot refresh until shot {shot_idx - 1} is observed")
            previous_state = previous.final_state
            previous_source = previous.sources
        if current.plan.bound_prop_entity_ids is not None:
            bound_prop_ids = current.plan.bound_prop_entity_ids
        else:
            # A legacy initial plan contains only the shot's bindings. A legacy
            # refreshed plan may also contain props carried from the prior shot.
            was_refreshed = any(
                isinstance(source, ObservedShotSource)
                for projected in [*current.plan.characters, *current.plan.props, current.plan.scene]
                for source in projected.field_sources.values()
            )
            previously_held = (
                {
                    prop_id
                    for content in previous_state.characters.values()
                    if content is not None
                    for prop_id in (content.holding or [])
                }
                if was_refreshed and previous_state is not None else set()
            )
            bound_prop_ids = [
                prop.entity_id for prop in current.plan.props
                if prop.entity_id not in previously_held
            ]
        refreshed = PlannedPosition(
            shot_idx=shot_idx,
            plan=project_shot_plan(
                registry,
                previous_state,
                current.plan.shot,
                previous_source,
                prop_entity_ids=bound_prop_ids,
                scene_entity_id=current.plan.scene.entity_id,
            ),
        )
        self.positions[shot_idx] = refreshed
        return refreshed.model_copy(deep=True)

    def mark_observed(self, shot_idx: int, observation: WorldState) -> ObservedPosition:
        planned = self._require_planned(shot_idx)
        if shot_idx > 0 and not isinstance(self._stored_position(shot_idx - 1), ObservedPosition):
            raise ValueError(f"Shot {shot_idx} cannot be observed before shot {shot_idx - 1}")
        previous = self._stored_position(shot_idx - 1) if shot_idx else None
        state, sources = _materialize_observation(
            observation,
            planned.plan,
            previous if isinstance(previous, ObservedPosition) else None,
            shot_idx,
        )
        observed = ObservedPosition(
            shot_idx=shot_idx,
            state=state,
            sources=sources,
        )
        self.positions[shot_idx] = observed
        return observed.model_copy(deep=True)

    def apply_no_change(self, observed_shot_idx: int) -> None:
        self._require_observed_with_next_planned(observed_shot_idx)

    def apply_replacement(
        self,
        observed_shot_idx: int,
        replacement_shot: ShotDescription,
        registry: EntityRegistry,
        prop_entity_ids: list[str] | None = None,
    ) -> PlannedPosition:
        observed, next_position = self._require_observed_with_next_planned(observed_shot_idx)
        next_idx = observed_shot_idx + 1
        if replacement_shot.idx != next_idx:
            raise ValueError(
                f"Replacement after shot {observed_shot_idx} must target shot {next_idx}, got {replacement_shot.idx}"
            )
        if prop_entity_ids is None:
            if registry.props:
                raise ValueError("Replacement requires explicit prop bindings when props are registered")
            prop_entity_ids = []
        replacement = PlannedPosition(
            shot_idx=next_idx,
            plan=project_shot_plan(
                registry,
                observed.final_state,
                replacement_shot,
                observed.sources,
                prop_entity_ids=prop_entity_ids,
                scene_entity_id=next_position.plan.scene.entity_id,
            ),
        )
        self.positions[next_idx] = replacement
        return replacement.model_copy(deep=True)

    def _stored_position(self, shot_idx: int) -> TimelinePosition:
        if shot_idx < 0 or shot_idx >= len(self.positions):
            raise IndexError(f"Timeline has no shot {shot_idx}")
        return self.positions[shot_idx]

    def _require_planned(self, shot_idx: int) -> PlannedPosition:
        position = self._stored_position(shot_idx)
        if not isinstance(position, PlannedPosition):
            raise ValueError(f"Shot {shot_idx} is already observed and cannot be changed")
        return position

    def _require_observed_with_next_planned(self, observed_shot_idx: int) -> tuple[ObservedPosition, PlannedPosition]:
        position = self._stored_position(observed_shot_idx)
        if not isinstance(position, ObservedPosition):
            raise ValueError(f"Shot {observed_shot_idx} must be observed before updating its next shot")
        next_idx = observed_shot_idx + 1
        if next_idx >= len(self.positions):
            raise ValueError(f"Shot {observed_shot_idx} has no next shot")
        next_position = self._stored_position(next_idx)
        if not isinstance(next_position, PlannedPosition):
            raise ValueError(f"Shot {next_idx} is not planned and cannot be updated")
        return position, next_position


def project_shot_plan(
    registry: EntityRegistry,
    previous_state: WorldState | None,
    shot: ShotDescription,
    previous_source: ObservedStateSources | None = None,
    *,
    prop_entity_ids: list[str],
    scene_entity_id: str | None = None,
) -> ShotPlan:
    """Build known Entity@time content without interpreting planned actions.

    ShotDescription does not carry a scene entity ID.  Single-scene callers
    may omit it; multi-scene callers must bind the shot explicitly.
    """
    if scene_entity_id is None:
        if len(registry.scenes) != 1:
            raise ValueError(
                "Entity@time projection requires exactly one registered scene unless an explicit scene entity ID is supplied"
            )
        scene_entity_id = registry.scenes[0].entity_id
    if previous_state is None and previous_source is not None:
        raise ValueError("Observed field source requires an observed WorldState")
    if previous_state is not None and previous_source is None:
        raise ValueError("Observed WorldState requires its field-source index")
    characters_by_id = {entity.entity_id: entity for entity in registry.characters}
    character_indices = list(shot.ff_vis_char_idxs)
    if shot.variation_type != "small":
        character_indices.extend(shot.lf_vis_char_idxs)
    character_ids = _ordered_unique(f"character-{idx}" for idx in character_indices)
    planned_characters: list[PlannedCharacterAtTime] = []
    for entity_id in character_ids:
        entity = characters_by_id.get(entity_id)
        if entity is None:
            raise ValueError(f"Shot {shot.idx} references unregistered character entity {entity_id}")
        observed_present = previous_state is not None and entity_id in previous_state.characters
        observed = previous_state.characters.get(entity_id) if observed_present else None
        content, field_sources = _merge_character_content(
            entity.initial_content,
            observed,
            previous_source.characters.get(entity_id) if previous_source else None,
            observed_present=observed_present,
        )
        planned_characters.append(PlannedCharacterAtTime(entity_id=entity_id, current_content=content, field_sources=field_sources, relations=[relation.model_copy(deep=True) for relation in entity.relations]))
    scenes_by_id = {entity.entity_id: entity for entity in registry.scenes}
    scene = scenes_by_id.get(scene_entity_id)
    if scene is None:
        raise ValueError(f"Shot {shot.idx} references unregistered scene entity {scene_entity_id}")
    observed_scene_present = previous_state is not None and scene.entity_id in previous_state.scenes
    observed_scene = previous_state.scenes.get(scene.entity_id) if observed_scene_present else None
    scene_content, scene_sources = _merge_scene_content(
        scene.initial_content,
        observed_scene,
        previous_source.scenes.get(scene.entity_id) if previous_source else None,
        observed_present=observed_scene_present,
    )
    planned_scene = PlannedSceneAtTime(entity_id=scene.entity_id, current_content=scene_content, field_sources=scene_sources, relations=[relation.model_copy(deep=True) for relation in scene.relations])
    props_by_id = {entity.entity_id: entity for entity in registry.props}
    planned_props: list[PlannedPropAtTime] = []
    for entity_id in _bound_or_held_prop_ids(registry, previous_state, prop_entity_ids):
        entity = props_by_id[entity_id]
        observed_present = previous_state is not None and entity_id in previous_state.props
        observed = previous_state.props.get(entity_id) if observed_present else None
        content, field_sources = _merge_prop_content(
            entity.initial_content,
            observed,
            previous_source.props.get(entity_id) if previous_source else None,
            observed_present=observed_present,
        )
        planned_props.append(PlannedPropAtTime(entity_id=entity_id, current_content=content, field_sources=field_sources, relations=[relation.model_copy(deep=True) for relation in entity.relations]))
    return ShotPlan(
        shot=shot.model_copy(deep=True),
        characters=planned_characters,
        scene=planned_scene,
        props=planned_props,
        bound_prop_entity_ids=list(prop_entity_ids),
    )


def _ordered_unique(values):
    return list(dict.fromkeys(values))


def _merge_character_content(
    initial: CharacterContent,
    observed: CharacterContent | None,
    observed_source: dict[str, StateFieldSource] | None,
    *,
    observed_present: bool,
) -> tuple[CharacterContent, dict[str, StateFieldSource]]:
    values, field_sources = _merge_content_fields(
        initial.model_dump(),
        observed.model_dump() if observed is not None else None,
        observed_source,
        observed_present=observed_present,
    )
    return CharacterContent.model_validate(values), field_sources


def _merge_prop_content(
    initial: PropContent | None,
    observed: PropContent | None,
    observed_source: dict[str, StateFieldSource] | None,
    *,
    observed_present: bool,
) -> tuple[PropContent | None, dict[str, StateFieldSource]]:
    values, field_sources = _merge_content_fields(
        initial.model_dump() if initial is not None else {},
        observed.model_dump() if observed is not None else None,
        observed_source,
        observed_present=observed_present,
    )
    return (PropContent.model_validate(values) if values else None), field_sources


def _merge_scene_content(
    initial: SceneContent | None,
    observed: SceneContent | None,
    observed_source: dict[str, StateFieldSource] | None,
    *,
    observed_present: bool,
) -> tuple[SceneContent | None, dict[str, StateFieldSource]]:
    values, field_sources = _merge_content_fields(
        dict(initial or {}), observed, observed_source, observed_present=observed_present,
    )
    return values or None, field_sources


def _merge_content_fields(
    initial: dict[str, object],
    observed: dict[str, object] | None,
    observed_source: dict[str, StateFieldSource] | None,
    *,
    observed_present: bool,
) -> tuple[dict[str, object], dict[str, StateFieldSource]]:
    """Use the latest snapshot when present; initial design is only the pre-observation fallback."""
    if not observed_present:
        values = dict(initial)
        field_sources: dict[str, StateFieldSource] = {
            field: InitialDesignSource()
            for field, value in values.items()
            if value is not None
        }
        return values, field_sources

    values = dict(observed or {})
    field_sources = {}
    for field, value in values.items():
        if value is None:
            continue
        if observed_source is None:
            raise ValueError("Observed content requires field sources for non-null fields")
        try:
            field_sources[field] = observed_source[field].model_copy(deep=True)
        except KeyError as exc:
            raise ValueError(f"Observed content field {field!r} has no source") from exc
    return values, field_sources


def _materialize_observation(
    observation: WorldState,
    plan: ShotPlan,
    previous: ObservedPosition | None,
    shot_idx: int,
) -> tuple[WorldState, ObservedStateSources]:
    """Apply this take's sparse facts to one fully materialized story snapshot."""
    _validate_observation_scope(observation, plan)
    state = previous.state.model_copy(deep=True) if previous is not None else WorldState()
    sources = previous.sources.model_copy(deep=True) if previous is not None else ObservedStateSources()

    for item in plan.characters:
        if item.entity_id not in state.characters:
            state.characters[item.entity_id] = item.current_content.model_copy(deep=True)
            _copy_entity_sources(sources.characters, item.entity_id, item.field_sources)
    for item in plan.props:
        if item.entity_id not in state.props:
            state.props[item.entity_id] = item.current_content.model_copy(deep=True) if item.current_content is not None else None
            _copy_entity_sources(sources.props, item.entity_id, item.field_sources)
    if plan.scene.entity_id not in state.scenes:
        state.scenes[plan.scene.entity_id] = dict(plan.scene.current_content or {}) or None
        _copy_entity_sources(sources.scenes, plan.scene.entity_id, plan.scene.field_sources)

    for entity_id, content in observation.characters.items():
        if content is None:
            continue
        current = state.characters[entity_id]
        state.characters[entity_id] = _apply_model_observation(
            current,
            content,
            sources.characters,
            entity_id,
            shot_idx,
            CharacterContent,
        )
    for entity_id, content in observation.props.items():
        if content is None:
            continue
        current = state.props[entity_id]
        state.props[entity_id] = _apply_model_observation(
            current,
            content,
            sources.props,
            entity_id,
            shot_idx,
            PropContent,
        )
    for entity_id, content in observation.scenes.items():
        if content is None:
            continue
        current = state.scenes[entity_id]
        state.scenes[entity_id] = _apply_scene_observation(
            current,
            content,
            sources.scenes,
            entity_id,
            shot_idx,
        )

    # Camera facts are shot-local: null/omission means this shot did not establish one.
    state.camera = observation.camera.model_copy(deep=True) if observation.camera is not None else None
    return state, sources


def _validate_observation_scope(observation: WorldState, plan: ShotPlan) -> None:
    _require_scope_subset("characters", observation.characters, {item.entity_id for item in plan.characters})
    _require_scope_subset("props", observation.props, {item.entity_id for item in plan.props})
    _require_scope_subset("scenes", observation.scenes, {plan.scene.entity_id})


def _require_scope_subset(kind: str, actual: dict, allowed: set[str]) -> None:
    unexpected = sorted(set(actual) - allowed)
    if unexpected:
        raise ValueError(f"Observation {kind} contains IDs outside this shot's participant scope: {unexpected}")


def _copy_entity_sources(
    destination: dict[str, dict[str, StateFieldSource]],
    entity_id: str,
    field_sources: dict[str, StateFieldSource],
) -> None:
    if field_sources:
        destination[entity_id] = {
            field: source.model_copy(deep=True)
            for field, source in field_sources.items()
        }


def _apply_model_observation(
    current: BaseModel | None,
    values: BaseModel | None,
    source_index: dict[str, dict[str, StateFieldSource]],
    entity_id: str,
    shot_idx: int,
    content_type,
) -> BaseModel:
    result = current.model_copy(deep=True) if current is not None else content_type()
    entity_sources = source_index.setdefault(entity_id, {})
    if values is not None:
        for field in values.model_fields_set:
            value = getattr(values, field)
            if getattr(result, field) == value:
                continue
            setattr(result, field, value)
            if value is None:
                entity_sources.pop(field, None)
            else:
                entity_sources[field] = ObservedShotSource(shot_idx=shot_idx)
    if not entity_sources:
        source_index.pop(entity_id, None)
    return result


def _apply_scene_observation(
    current: SceneContent | None,
    values: SceneContent | None,
    source_index: dict[str, dict[str, StateFieldSource]],
    entity_id: str,
    shot_idx: int,
) -> SceneContent:
    result = dict(current or {})
    entity_sources = source_index.setdefault(entity_id, {})
    if values is not None:
        for field, value in values.items():
            if result.get(field) == value:
                continue
            result[field] = value
            if value is None:
                entity_sources.pop(field, None)
            else:
                entity_sources[field] = ObservedShotSource(shot_idx=shot_idx)
    if not entity_sources:
        source_index.pop(entity_id, None)
    return result


def _bound_or_held_prop_ids(
    registry: EntityRegistry,
    previous_state: WorldState | None,
    prop_entity_ids: list[str],
) -> list[str]:
    if len(prop_entity_ids) != len(set(prop_entity_ids)):
        raise ValueError("Shot prop binding contains duplicate entity IDs")
    known_prop_ids = {entity.entity_id for entity in registry.props}
    unknown = set(prop_entity_ids) - known_prop_ids
    if unknown:
        raise ValueError(f"Shot prop binding references unregistered entities: {sorted(unknown)}")
    held: list[str] = []
    if previous_state is not None:
        for content in previous_state.characters.values():
            if content is not None and content.holding:
                held.extend(content.holding)
    unknown_held = set(held) - known_prop_ids
    if unknown_held:
        raise ValueError(f"Previous WorldState holds unregistered prop entities: {sorted(unknown_held)}")
    return _ordered_unique([*prop_entity_ids, *held])
