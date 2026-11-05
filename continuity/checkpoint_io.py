"""Formal serial checkpoint reads, writes, validation and readiness restoration.

Functions operate on the supplied pipeline's working directory and image store.
Timeline remains authoritative; no second checkpoint or state format is introduced.
"""

import asyncio
import json
from pathlib import Path

from continuity.models import EntityRegistry, ObservedPosition, PlannedPosition
from continuity.timeline import Timeline
from continuity.image_records import ImageRecordStore
from continuity.shot_sampling import read_observation_samples
from interfaces import Camera, CharacterInScene, ShotDescription


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_shot_description(path: Path) -> ShotDescription:
    return ShotDescription.model_validate_json(path.read_text(encoding="utf-8"))


def validate_portrait_entry(character: CharacterInScene, entry: dict) -> None:
    for view in ("front", "side", "back"):
        try:
            path = entry[view]["path"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Portrait checkpoint for {character.identifier_in_scene} is missing {view}"
            ) from exc
        if not Path(path).is_file():
            raise ValueError(
                f"Portrait checkpoint for {character.identifier_in_scene} references missing file: {path}"
            )


def save_json(self, relative_path: str, value) -> None:
    path = self.working_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_registry(self, registry: EntityRegistry) -> None:
    self._save("continuity/entity_registry.json", registry.model_dump())


def save_timeline(self, timeline: Timeline) -> None:
    self._save("continuity/timeline.json", timeline.model_dump())


def load_resume_artifacts(self) -> tuple[list[CharacterInScene], EntityRegistry, dict, list[Camera], Timeline]:
    """Load only formal persisted artifacts; Timeline remains authoritative."""
    def read_model(relative_path: str, model):
        path = self.working_dir / relative_path
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Resume requires {path}") from exc

    registry = read_model("continuity/entity_registry.json", EntityRegistry)
    timeline = read_model("continuity/timeline.json", Timeline)
    if not timeline.positions:
        raise ValueError("Resume Timeline must contain at least one position")
    if not registry.scenes:
        raise ValueError("Resume entity registry must contain at least one scene")
    try:
        characters = [CharacterInScene.model_validate(item) for item in json.loads(
            (self.working_dir / "characters.json").read_text(encoding="utf-8")
        )]
        portrait_registry = json.loads((self.working_dir / "character_portraits_registry.json").read_text(encoding="utf-8"))
        cameras = [Camera.model_validate(item) for item in json.loads(
            (self.working_dir / "camera_tree.json").read_text(encoding="utf-8")
        )]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Resume artifacts are missing or malformed") from exc
    self.image_store = ImageRecordStore(self.working_dir / "continuity" / "image_records.jsonl")
    # Parse every formal shot artifact for validation only.  They never
    # replace Timeline's current planned shot.
    for shot_idx in range(len(timeline.positions)):
        shot = read_model(f"shots/{shot_idx}/shot_description.json", ShotDescription)
        if shot.idx != shot_idx:
            raise ValueError(f"Resume shot description index mismatch for shot {shot_idx}")
        position = timeline.position(shot_idx)
        if isinstance(position, PlannedPosition) and shot.model_dump() != position.plan.shot.model_dump():
            raise ValueError(f"Resume planned shot description conflicts with Timeline for shot {shot_idx}")
    self._validate_resume_identity_artifacts(characters, registry, cameras, timeline)
    return characters, registry, portrait_registry, cameras, timeline


def validate_resume_identity_artifacts(
    characters: list[CharacterInScene], registry: EntityRegistry, cameras: list[Camera], timeline: Timeline,
) -> None:
    character_indices = [character.idx for character in characters]
    if character_indices != list(range(len(characters))):
        raise ValueError("Resume characters must be ordered with contiguous stable indices")
    entities = {entity.entity_id: entity for entity in registry.characters}
    if set(entities) != {f"character-{idx}" for idx in character_indices}:
        raise ValueError("Resume characters and Registry character IDs must match")
    for character in characters:
        if entities[f"character-{character.idx}"].name != character.identifier_in_scene:
            raise ValueError(f"Resume character name mismatch for index {character.idx}")
    camera_indices = [camera.idx for camera in cameras]
    if len(camera_indices) != len(set(camera_indices)):
        raise ValueError("Resume camera indices must be unique")
    cameras_by_idx = {camera.idx: camera for camera in cameras}
    membership_counts = {
        shot_idx: sum(shot_idx in camera.active_shot_idxs for camera in cameras)
        for shot_idx in range(len(timeline.positions))
    }
    if any(count != 1 for count in membership_counts.values()):
        raise ValueError("Resume camera tree must assign every shot to exactly one camera")
    unexpected_shots = sorted({
        shot_idx
        for camera in cameras
        for shot_idx in camera.active_shot_idxs
        if shot_idx not in membership_counts
    })
    if unexpected_shots:
        raise ValueError(f"Resume camera tree references unknown shots: {unexpected_shots}")
    for camera in cameras:
        if not camera.active_shot_idxs:
            raise ValueError(f"Resume camera {camera.idx} has no active shots")
        if camera.active_shot_idxs != sorted(camera.active_shot_idxs):
            raise ValueError(f"Resume camera {camera.idx} shots are not in story order")
        if (camera.parent_cam_idx is None) != (camera.parent_shot_idx is None):
            raise ValueError(f"Resume camera {camera.idx} has an incomplete parent binding")
        if camera.parent_cam_idx is not None:
            parent = cameras_by_idx.get(camera.parent_cam_idx)
            if parent is None or camera.parent_shot_idx not in parent.active_shot_idxs:
                raise ValueError(f"Resume camera {camera.idx} parent shot is not on its parent camera")
            if camera.parent_shot_idx >= camera.active_shot_idxs[0]:
                raise ValueError(
                    f"Resume camera {camera.idx} depends on a non-earlier shot and would deadlock serial execution"
                )
    for position in timeline.positions:
        shot = position.plan.shot if isinstance(position, PlannedPosition) else None
        if shot is None:
            continue
        memberships = [camera for camera in cameras if position.shot_idx in camera.active_shot_idxs]
        if len(memberships) != 1 or memberships[0].idx != shot.cam_idx:
            raise ValueError(f"Resume shot {position.shot_idx} camera membership is inconsistent")


def validate_resume_timeline(self, timeline: Timeline, cameras: list[Camera]) -> int:
    """Fail closed unless positions are an observed prefix then planned suffix."""
    first_planned = len(timeline.positions)
    for expected_idx, position in enumerate(timeline.positions):
        if position.shot_idx != expected_idx:
            raise ValueError("Resume Timeline shot indices must be contiguous")
        if isinstance(position, PlannedPosition):
            first_planned = min(first_planned, expected_idx)
            continue
        if not isinstance(position, ObservedPosition) or expected_idx >= first_planned:
            raise ValueError("Resume Timeline must be an observed prefix followed by planned positions")
        self._validate_observed_checkpoint(position, cameras)
    if any(not isinstance(position, PlannedPosition) for position in timeline.positions[first_planned:]):
        raise ValueError("Resume Timeline must not contain observed positions after its planned suffix begins")
    return first_planned


def validate_observed_checkpoint(self, position: ObservedPosition, cameras: list[Camera]) -> None:
    shot_path = self.working_dir / "shots" / str(position.shot_idx) / "shot_description.json"
    video_path = self.working_dir / "shots" / str(position.shot_idx) / "video.mp4"
    observation_samples_path = (
        self.working_dir / "shots" / str(position.shot_idx) / "observation_samples.json"
    )
    try:
        shot = ShotDescription.model_validate_json(shot_path.read_text(encoding="utf-8"))
        metadata = read_observation_samples(observation_samples_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Resume observed shot {position.shot_idx} artifacts are missing or malformed") from exc
    camera = self._camera_for(shot, cameras)
    if position.shot_idx not in camera.active_shot_idxs:
        raise ValueError(f"Resume observed shot {position.shot_idx} is not active on camera {camera.idx}")
    if shot.idx != position.shot_idx or not video_path.is_file():
        raise ValueError(f"Resume observed shot {position.shot_idx} video or description mismatch")
    first_frame = self.working_dir / "shots" / str(position.shot_idx) / "first_frame.png"
    last_frame = self.working_dir / "shots" / str(position.shot_idx) / "last_frame.png"
    if not first_frame.is_file() or (shot.variation_type in {"medium", "large"} and not last_frame.is_file()):
        raise ValueError(f"Resume observed shot {position.shot_idx} frame artifacts are missing")
    for item in metadata:
        try:
            record = self.image_store.get(item.image_id)
        except KeyError as exc:
            raise ValueError(f"Resume observed shot {position.shot_idx} is missing image record {item.image_id}") from exc
        if (
            record.kind != "shot_sample" or record.shot_idx != position.shot_idx
            or record.sample_time_seconds != item.time_seconds or not Path(record.asset_path).is_file()
        ):
            raise ValueError(f"Resume observed shot {position.shot_idx} sample record mismatch")


def restore_native_frame_events(self, timeline: Timeline) -> None:
    """Recreate only ViMax's per-shot readiness events from formal frame files."""
    restored = {}
    for position in timeline.positions:
        shot_idx = position.shot_idx
        shot = position.plan.shot if isinstance(position, PlannedPosition) else ShotDescription.model_validate_json(
            (self.working_dir / "shots" / str(shot_idx) / "shot_description.json").read_text(encoding="utf-8")
        )
        events = {"first_frame": asyncio.Event()}
        if (self.working_dir / "shots" / str(shot_idx) / "first_frame.png").is_file():
            events["first_frame"].set()
        if shot.variation_type in {"medium", "large"}:
            events["last_frame"] = asyncio.Event()
            if (self.working_dir / "shots" / str(shot_idx) / "last_frame.png").is_file():
                events["last_frame"].set()
        restored[shot_idx] = events
    self.production.frame_events = restored


def observation_sample_paths(self, timeline: Timeline) -> dict[int, Path]:
    """Map observed shots to their sole ordered observation-sample manifests."""
    return {
        position.shot_idx: (
            self.working_dir / "shots" / str(position.shot_idx) / "observation_samples.json"
        )
        for position in timeline.positions
        if isinstance(position, ObservedPosition)
    }


def native_shot_descriptions(
    self,
    timeline: Timeline,
    current_position: PlannedPosition,
) -> list[ShotDescription]:
    """Pass the single-track plan to ViMax, loading only already-observed history."""
    result: list[ShotDescription] = []
    for stored_position in timeline.positions:
        shot_idx = stored_position.shot_idx
        if isinstance(stored_position, PlannedPosition):
            result.append(stored_position.plan.shot)
            continue
        path = self.working_dir / "shots" / str(shot_idx) / "shot_description.json"
        try:
            result.append(ShotDescription.model_validate_json(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise ValueError(f"Missing production shot description {path}") from exc
    # Refresh returns the authoritative execution position; explicitly put
    # that shot into the native list so no persisted artifact can replace a
    # replan within this iteration.
    result[current_position.shot_idx] = current_position.plan.shot
    if result[current_position.shot_idx] is not current_position.plan.shot:
        raise AssertionError("Native production must receive the current Timeline plan shot")
    return result
