"""Read state lineage and real historical sample candidates without conflating them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from continuity.image_records import ImageRecordStore
from continuity.shot_sampling import read_observation_samples
from continuity.models import (
    CharacterContent,
    ObservedPosition,
    ObservedShotSource,
    PropContent,
    SceneContent,
    StateFieldSource,
)
from continuity.timeline import Timeline


StateFieldValue: TypeAlias = str | list[str]
ProjectedContent: TypeAlias = CharacterContent | PropContent | SceneContent | None


class StateFieldLineage(BaseModel):
    """One projected non-null field and its state origin, without image evidence."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    field: str
    value: StateFieldValue
    source: StateFieldSource


class HistoricalSampleCandidate(BaseModel):
    """One actual sampled image available from an observed source shot."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    asset_path: str
    time_seconds: float


class HistoricalSampleGroup(BaseModel):
    """All actual samples from one source shot and the state lineage they relate to."""

    model_config = ConfigDict(extra="forbid")

    source: ObservedShotSource
    lineage: list[StateFieldLineage] = Field(min_length=1)
    samples: list[HistoricalSampleCandidate] = Field(min_length=1)


def state_field_lineage(
    entity_id: str,
    current_content: ProjectedContent,
    field_sources: Mapping[str, StateFieldSource],
) -> list[StateFieldLineage]:
    """Return field-level state origins without resolving any image."""
    values = _validate_current_content_sources(current_content, field_sources)
    return [
        StateFieldLineage(entity_id=entity_id, field=field, value=values[field], source=source)
        for field, source in field_sources.items()
    ]


def historical_sample_groups(
    lineage: list[StateFieldLineage],
    timeline: Timeline,
    observation_sample_paths: Mapping[int, str | Path],
    image_store: ImageRecordStore,
) -> list[HistoricalSampleGroup]:
    """Read all formal sample records for each observed source shot once."""
    grouped: dict[int, list[StateFieldLineage]] = {}
    for item in lineage:
        if isinstance(item.source, ObservedShotSource):
            grouped.setdefault(item.source.shot_idx, []).append(item)
    return [
        HistoricalSampleGroup(
            source=ObservedShotSource(shot_idx=shot_idx),
            lineage=items,
            samples=_samples_for_source_shot(shot_idx, timeline, observation_sample_paths, image_store),
        )
        for shot_idx, items in grouped.items()
    ]


def _validate_current_content_sources(
    current_content: ProjectedContent,
    field_sources: Mapping[str, StateFieldSource],
) -> dict[str, StateFieldValue]:
    if current_content is None:
        values: dict[str, StateFieldValue] = {}
    elif isinstance(current_content, (CharacterContent, PropContent)):
        values = {
            field: value
            for field, value in current_content.model_dump().items()
            if value is not None
        }
    else:
        values = {field: value for field, value in current_content.items() if value is not None}
    missing = sorted(set(values) - set(field_sources))
    extra = sorted(set(field_sources) - set(values))
    if missing or extra:
        raise ValueError(
            "field_sources must exactly cover non-null current fields"
            f" (missing={missing}, extra={extra})"
        )
    return values


def _samples_for_source_shot(
    shot_idx: int,
    timeline: Timeline,
    observation_sample_paths: Mapping[int, str | Path],
    image_store: ImageRecordStore,
) -> list[HistoricalSampleCandidate]:
    position = timeline.position(shot_idx)
    if not isinstance(position, ObservedPosition):
        raise ValueError(f"Observed source shot {shot_idx} is not observed on Timeline")
    try:
        path = Path(observation_sample_paths[shot_idx])
    except KeyError as exc:
        raise ValueError(f"Observed source shot {shot_idx} has no observation sample manifest") from exc
    metadata = read_observation_samples(path)
    samples: list[HistoricalSampleCandidate] = []
    for item in metadata:
        try:
            record = image_store.get(item.image_id)
        except KeyError as exc:
            raise ValueError(f"Observed source image_id {item.image_id} is not a formal image record") from exc
        if (
            record.kind != "shot_sample"
            or record.shot_idx != shot_idx
            or record.sample_time_seconds != item.time_seconds
            or not Path(record.asset_path).is_file()
        ):
            raise ValueError("Observed source sample record does not match shot, time, or local image path")
        samples.append(HistoricalSampleCandidate(
            image_id=record.image_id,
            asset_path=record.asset_path,
            time_seconds=record.sample_time_seconds,
        ))
    return samples
