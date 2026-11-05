"""Build one read-only frame-material context from the single-track Timeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from continuity.history_materials import (
    HistoricalSampleCandidate,
    HistoricalSampleGroup,
    StateFieldLineage,
    historical_sample_groups,
    state_field_lineage,
)
from continuity.image_records import ImageRecord, ImageRecordStore
from continuity.models import EntityRegistry, PlannedCharacterAtTime, PlannedPosition, PlannedPropAtTime, PlannedSceneAtTime
from continuity.timeline import Timeline


FrameType = Literal["first", "last"]


class FixedFrameMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    asset_path: str
    role: Literal["current_first_frame_starting_state"]


class NativeFrameCandidate(BaseModel):
    """An ordinary candidate prepared by ViMax's existing frame production path.

    This is deliberately not Timeline evidence or a fixed Camera material.  The
    native frame selector may choose or omit it just as it did before continuity.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    asset_path: str
    description: str


class IdentityPortraitCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    asset_path: str
    view: Literal["front", "side", "back"]
    role: Literal["character_identity_portrait"] = "character_identity_portrait"


class VisibleCharacterPortraitCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    candidates: list[IdentityPortraitCandidate] = Field(min_length=3, max_length=3)


class FrameMaterialContext(BaseModel):
    """Transient materials, known lineage, and optional history for one endpoint."""

    model_config = ConfigDict(extra="forbid")

    shot_idx: int
    frame_type: FrameType
    fixed_materials: list[FixedFrameMaterial]
    native_candidates: list[NativeFrameCandidate] = Field(default_factory=list)
    field_lineage: list[StateFieldLineage]
    history_sample_groups: list[HistoricalSampleGroup]
    portrait_candidates: list[VisibleCharacterPortraitCandidates]
    required_slots: int
    top_k: int


class FrameMaterialCapacityError(ValueError):
    pass


def fixed_material_purpose(role: str) -> str:
    if role == "current_first_frame_starting_state":
        return "current shot first-frame starting state"
    raise ValueError(f"Unsupported fixed frame material role {role!r}")


def history_sample_purpose(
    group: HistoricalSampleGroup,
    sample: HistoricalSampleCandidate,
) -> str:
    return (
        f"source shot {group.source.shot_idx} at {sample.time_seconds:g}s actual sampled image; "
        "use only continuity visibly present in this image"
    )


def build_frame_material_context(
    position: PlannedPosition,
    frame_type: FrameType,
    timeline: Timeline,
    registry: EntityRegistry,
    observation_sample_paths: Mapping[int, str | Path],
    image_store: ImageRecordStore,
    top_k: int,
    *,
    current_first_frame_id: str | None = None,
) -> FrameMaterialContext:
    """Read known endpoint state lineage and visual candidates without selecting.

    This routine has no prompt, search, model, or persistence behavior.  It reads
    only the current planned projection and formal sample manifests.
    """
    current_position = timeline.position(position.shot_idx)
    if not isinstance(current_position, PlannedPosition):
        raise ValueError(f"Frame material context requires planned Timeline position {position.shot_idx}")
    if current_position.model_dump() != position.model_dump():
        raise ValueError(f"Frame material context position {position.shot_idx} does not match current Timeline plan")
    if frame_type not in {"first", "last"}:
        raise ValueError(f"Unsupported frame_type {frame_type!r}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if frame_type == "first":
        if current_first_frame_id is not None:
            raise ValueError("current_first_frame_id belongs only to a last-frame context")
    else:
        if not current_first_frame_id:
            raise ValueError("Last-frame context requires current_first_frame_id")

    records = {record.image_id: record for record in image_store.all_records()}
    fixed_materials = _fixed_materials(
        position,
        frame_type,
        current_first_frame_id,
        records,
    )
    characters = _visible_planned_characters(position, frame_type)
    field_lineage = _field_lineage(
        characters,
        position.plan.props,
        position.plan.scene,
    )
    relevant_prop_ids = set(
        position.plan.bound_prop_entity_ids
        if position.plan.bound_prop_entity_ids is not None
        else (prop.entity_id for prop in position.plan.props)
    )
    history_groups = historical_sample_groups(
        [item for item in field_lineage if item.entity_id in relevant_prop_ids],
        timeline,
        observation_sample_paths,
        image_store,
    )
    portraits = [_portrait_candidates(character.entity_id, registry, records) for character in characters]
    required_slots = len(fixed_materials) + len(portraits)
    if required_slots > top_k:
        raise FrameMaterialCapacityError(
            "Frame material capacity insufficient: "
            f"fixed={len(fixed_materials)}, portraits={len(portraits)}, "
            f"top_k={top_k}, shortfall={required_slots - top_k}"
        )
    return FrameMaterialContext(
        shot_idx=position.shot_idx,
        frame_type=frame_type,
        fixed_materials=fixed_materials,
        field_lineage=field_lineage,
        history_sample_groups=history_groups,
        portrait_candidates=portraits,
        required_slots=required_slots,
        top_k=top_k,
    )


def _visible_planned_characters(position: PlannedPosition, frame_type: FrameType) -> list[PlannedCharacterAtTime]:
    visible_indices = (
        position.plan.shot.ff_vis_char_idxs
        if frame_type == "first"
        else position.plan.shot.lf_vis_char_idxs
    )
    visible_ids = list(dict.fromkeys(f"character-{index}" for index in visible_indices))
    planned_by_id = {character.entity_id: character for character in position.plan.characters}
    missing = [entity_id for entity_id in visible_ids if entity_id not in planned_by_id]
    if missing:
        raise ValueError(f"Planned position lacks visible character projections: {missing}")
    return [planned_by_id[entity_id] for entity_id in visible_ids]


def _field_lineage(
    characters: list[PlannedCharacterAtTime],
    props: list[PlannedPropAtTime],
    scene: PlannedSceneAtTime,
) -> list[StateFieldLineage]:
    result: list[StateFieldLineage] = []
    for character in characters:
        result.extend(state_field_lineage(character.entity_id, character.current_content, character.field_sources))
    for prop in props:
        result.extend(state_field_lineage(prop.entity_id, prop.current_content, prop.field_sources))
    result.extend(state_field_lineage(scene.entity_id, scene.current_content, scene.field_sources))
    return result


def _fixed_materials(
    position: PlannedPosition,
    frame_type: FrameType,
    current_first_frame_id: str | None,
    records: Mapping[str, ImageRecord],
) -> list[FixedFrameMaterial]:
    if frame_type == "last":
        record = _record(records, current_first_frame_id, "current first frame")
        if record.kind != "generated_first_frame" or record.shot_idx != position.shot_idx:
            raise ValueError("current_first_frame_id is not this shot's generated first frame")
        _require_local_asset(record)
        return [FixedFrameMaterial(
            image_id=record.image_id,
            asset_path=record.asset_path,
            role="current_first_frame_starting_state",
        )]
    return []


def _portrait_candidates(
    character_id: str,
    registry: EntityRegistry,
    records: Mapping[str, ImageRecord],
) -> VisibleCharacterPortraitCandidates:
    entities = [entity for entity in registry.characters if entity.entity_id == character_id]
    if len(entities) != 1:
        raise ValueError(f"Visible character {character_id} has no unique registry entity")
    identity_ids = entities[0].identity_image_ids
    if len(identity_ids) != 3 or len(set(identity_ids)) != 3:
        raise ValueError(f"Visible character {character_id} requires exactly three unique identity image IDs")
    by_view: dict[str, ImageRecord] = {}
    for image_id in identity_ids:
        try:
            record = records[image_id]
        except KeyError as exc:
            raise ValueError(f"Visible character {character_id} references unknown identity image {image_id}") from exc
        if (
            record.kind != "portrait"
            or record.character_id != character_id
            or record.shot_idx is not None
            or record.sample_time_seconds is not None
            or record.view not in {"front", "side", "back"}
        ):
            raise ValueError(f"Invalid identity portrait record {image_id} for {character_id}")
        if record.view in by_view:
            raise ValueError(f"Visible character {character_id} has duplicate {record.view} identity portraits")
        by_view[record.view] = record
    if set(by_view) != {"front", "side", "back"}:
        raise ValueError(f"Visible character {character_id} requires front, side, and back portrait records")
    candidates: list[IdentityPortraitCandidate] = []
    for view in ("front", "side", "back"):
        record = by_view[view]
        _require_local_asset(record)
        candidates.append(IdentityPortraitCandidate(
            image_id=record.image_id,
            asset_path=record.asset_path,
            view=view,
        ))
    return VisibleCharacterPortraitCandidates(character_id=character_id, candidates=candidates)


def _record(records: Mapping[str, ImageRecord], image_id: str | None, role: str) -> ImageRecord:
    if not image_id:
        raise ValueError(f"Missing required {role} image_id")
    try:
        return records[image_id]
    except KeyError as exc:
        raise ValueError(f"Unknown required {role} image_id {image_id}") from exc


def _require_local_asset(record: ImageRecord) -> None:
    if not Path(record.asset_path).is_file():
        raise ValueError(f"Image record {record.image_id} has no local asset")
