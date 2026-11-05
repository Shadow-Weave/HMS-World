"""ViMax-Bench's strict dataset contract and serial-pipeline adapter.

The benchmark already supplies shot-level first-frame and motion prompts.  This
module never sends those prompts through ViMax storyboard generation or shot
decomposition.  It adds only the identity bindings required by the local serial
continuity protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, stop_after_attempt

from agents.entity_extractor import EntityExtractor
from continuity.models import EntityRegistry
from interfaces import CharacterInScene, ShotDescription
from utils.retry import after_func
from utils.robust_json_parser import (
    TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DATA_ROOT = Path(__file__).resolve().parent / "data"


ConsistencyType = Literal["Type A", "Type B", "Type C"]
RUN_FORMAT = "vimax-benchmark-serial-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkIndexModelConfig(_StrictModel):
    model: str
    api: str


class BenchmarkStoryRef(_StrictModel):
    id: int = Field(ge=1)
    consistency_type: ConsistencyType = Field(alias="type")
    theme: str = Field(min_length=1)
    file: str = Field(min_length=1)


class BenchmarkIndex(_StrictModel):
    total_stories: int = Field(ge=1)
    generation_config: BenchmarkIndexModelConfig = Field(alias="model_config")
    stories: list[BenchmarkStoryRef]


class BenchmarkShot(_StrictModel):
    shot_id: int = Field(ge=1)
    first_frame: str = Field(min_length=1)
    video_prompt: str = Field(min_length=1)


class BenchmarkScene(_StrictModel):
    scene_num: int = Field(ge=1)
    shots: list[BenchmarkShot] = Field(min_length=1)


class BenchmarkMetadata(_StrictModel):
    theme_key: str = Field(min_length=1)
    theme_description: str = Field(min_length=1)
    consistency_type: ConsistencyType
    requested_scenes: int = Field(ge=1)
    requested_shots: int = Field(ge=1)


class BenchmarkStory(_StrictModel):
    story_overview: str = Field(min_length=1)
    consistency_type: ConsistencyType
    scenes: list[BenchmarkScene] = Field(min_length=1)
    metadata: BenchmarkMetadata

    @model_validator(mode="after")
    def _validate_shape(self):
        expected_scenes = list(range(1, len(self.scenes) + 1))
        actual_scenes = [scene.scene_num for scene in self.scenes]
        if actual_scenes != expected_scenes:
            raise ValueError(
                f"scene_num values must be contiguous from 1: expected {expected_scenes}, got {actual_scenes}"
            )
        shots = [shot for scene in self.scenes for shot in scene.shots]
        expected_shots = list(range(1, len(shots) + 1))
        actual_shots = [shot.shot_id for shot in shots]
        if actual_shots != expected_shots:
            raise ValueError(
                f"shot_id values must be globally contiguous from 1: expected {expected_shots}, got {actual_shots}"
            )
        if self.metadata.consistency_type != self.consistency_type:
            raise ValueError("metadata consistency_type does not match story consistency_type")
        if self.metadata.requested_scenes != len(self.scenes):
            raise ValueError("metadata requested_scenes does not match the scene array")
        if self.metadata.requested_shots != len(shots):
            raise ValueError("metadata requested_shots does not match the shot arrays")
        return self


@dataclass(frozen=True)
class LoadedBenchmarkStory:
    ref: BenchmarkStoryRef
    story: BenchmarkStory
    path: Path
    sha256: str


@dataclass(frozen=True)
class BenchmarkCatalog:
    root: Path
    index_path: Path
    index: BenchmarkIndex
    stories: tuple[LoadedBenchmarkStory, ...]


def _read_model(path: Path, model_type):
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read benchmark JSON: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid benchmark JSON contract: {path}: {exc}") from exc


def load_catalog(root: str | Path) -> BenchmarkCatalog:
    """Load all indexed stories and reject a partial or altered dataset."""
    requested = Path(root)
    benchmark_root = requested.resolve()
    index_path = benchmark_root / "benchmark_index.json"
    index = _read_model(index_path, BenchmarkIndex)
    if index.total_stories != len(index.stories):
        raise ValueError("benchmark_index total_stories does not match stories")
    expected_ids = list(range(1, len(index.stories) + 1))
    if [item.id for item in index.stories] != expected_ids:
        raise ValueError("benchmark_index story IDs must be unique and contiguous from 1")
    themes = [item.theme for item in index.stories]
    files = [item.file for item in index.stories]
    if len(themes) != len(set(themes)) or len(files) != len(set(files)):
        raise ValueError("benchmark_index themes and files must be unique")
    for file_name in files:
        if Path(file_name).name != file_name or not file_name.endswith(".json"):
            raise ValueError(f"benchmark_index contains an unsafe story file: {file_name}")

    actual_files = {
        path.name
        for path in benchmark_root.glob("*.json")
        if path.name != "benchmark_index.json"
    }
    declared_files = set(files)
    if actual_files != declared_files:
        raise ValueError(
            "benchmark story-file set differs from benchmark_index; "
            f"missing={sorted(declared_files - actual_files)}, extra={sorted(actual_files - declared_files)}"
        )

    loaded = []
    for ref in index.stories:
        path = benchmark_root / ref.file
        raw = path.read_bytes()
        story = _read_model(path, BenchmarkStory)
        if story.consistency_type != ref.consistency_type:
            raise ValueError(f"Story {ref.id} consistency type differs from benchmark_index")
        if story.metadata.theme_key != ref.theme:
            raise ValueError(f"Story {ref.id} theme_key differs from benchmark_index")
        loaded.append(LoadedBenchmarkStory(
            ref=ref,
            story=story,
            path=path,
            sha256=hashlib.sha256(raw).hexdigest(),
        ))
    return BenchmarkCatalog(
        root=benchmark_root,
        index_path=index_path,
        index=index,
        stories=tuple(loaded),
    )


def select_stories(
    catalog: BenchmarkCatalog,
    *,
    ids: list[int] | None = None,
    themes: list[str] | None = None,
    run_all: bool = False,
) -> list[LoadedBenchmarkStory]:
    ids = ids or []
    themes = themes or []
    if run_all and (ids or themes):
        raise ValueError("--all cannot be combined with story IDs or themes")
    if not run_all and not ids and not themes:
        raise ValueError("Select at least one story ID or theme, or use --all")
    if len(ids) != len(set(ids)) or len(themes) != len(set(themes)):
        raise ValueError("Benchmark selectors must be unique")
    if run_all:
        return list(catalog.stories)
    by_id = {item.ref.id: item for item in catalog.stories}
    by_theme = {item.ref.theme: item for item in catalog.stories}
    unknown_ids = [story_id for story_id in ids if story_id not in by_id]
    unknown_themes = [theme for theme in themes if theme not in by_theme]
    if unknown_ids or unknown_themes:
        raise ValueError(
            f"Unknown benchmark selectors: ids={unknown_ids}, themes={unknown_themes}"
        )
    selected = [by_id[story_id] for story_id in ids]
    selected.extend(by_theme[theme] for theme in themes)
    selected_by_id = {item.ref.id: item for item in selected}
    if len(selected_by_id) != len(selected):
        raise ValueError("The same benchmark story was selected more than once")
    return [selected_by_id[story_id] for story_id in sorted(selected_by_id)]


class BenchmarkShotBindingDraft(_StrictModel):
    shot_id: int = Field(ge=1)
    visible_character_idxs: list[int] = Field(default_factory=list)
    camera_group: int = Field(ge=0)


class BenchmarkSceneRegistrationDraft(_StrictModel):
    scene_num: int = Field(ge=1)
    shots: list[BenchmarkShotBindingDraft]


_SCENE_ANALYZER_SYSTEM_PROMPT = """
You are analyzing one already-authored ViMax-Bench narrative group. The
benchmark's first_frame and video_prompt strings are immutable: do not rewrite,
expand, summarize, or replace them.

Bind visible named characters to the supplied stable character indices.

camera_group is a zero-based scene-local camera-position group. Reuse a group
only when viewpoint, angle, scale, and focus can plausibly reuse the same camera;
groups must be contiguous from zero in first-appearance order.

{format_instructions}
""".strip()


def benchmark_character_source(story: BenchmarkStory) -> str:
    """Expose official narrative text with truthful benchmark structure labels."""
    sections = [
        f"STORY OVERVIEW\n{story.story_overview}",
        (
            "STRUCTURE NOTE\n"
            "BENCHMARK GROUP is an organizational narrative group, not a physical scene. "
            "Each distinct physical setting described by a shot is a separate scene."
        ),
    ]
    for scene in story.scenes:
        sections.append(f"BENCHMARK GROUP {scene.scene_num}")
        for shot in scene.shots:
            sections.append(
                f"SHOT {shot.shot_id} FIRST FRAME\n{shot.first_frame}\n"
                f"SHOT {shot.shot_id} MOTION\n{shot.video_prompt}"
            )
    return "\n\n".join(sections)


class BenchmarkSceneAnalyzer:
    """Analyze benchmark-only character visibility and camera grouping."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def analyze(
        self,
        scene: BenchmarkScene,
        characters: list[CharacterInScene],
    ) -> BenchmarkSceneRegistrationDraft:
        parser = PydanticOutputParser(pydantic_object=BenchmarkSceneRegistrationDraft)
        payload = {
            "scene": scene.model_dump(),
            "characters": [character.model_dump() for character in characters],
        }
        messages = [
            SystemMessage(content=_SCENE_ANALYZER_SYSTEM_PROMPT.format(
                format_instructions=parser.get_format_instructions(),
            )),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        draft = await (self.chat_model | parser).ainvoke(messages)
        _validate_scene_registration(draft, scene, characters)
        return draft


def _normalized_name(name: str, label: str) -> str:
    if name != name.strip() or not name:
        raise ValueError(f"{label} must be non-empty without surrounding whitespace")
    return name.casefold()


def _validate_scene_registration(
    draft: BenchmarkSceneRegistrationDraft,
    scene: BenchmarkScene,
    characters: list[CharacterInScene],
) -> None:
    if draft.scene_num != scene.scene_num:
        raise ValueError(
            f"Scene analyzer returned scene {draft.scene_num}, expected {scene.scene_num}"
        )
    expected_shots = [shot.shot_id for shot in scene.shots]
    if [item.shot_id for item in draft.shots] != expected_shots:
        raise ValueError("Scene analyzer must bind every supplied shot exactly once and in order")
    valid_character_idxs = {character.idx for character in characters}
    camera_groups = []
    for item in draft.shots:
        if len(item.visible_character_idxs) != len(set(item.visible_character_idxs)):
            raise ValueError(f"Shot {item.shot_id} has duplicate visible character indices")
        unknown_characters = set(item.visible_character_idxs) - valid_character_idxs
        if unknown_characters:
            raise ValueError(
                f"Shot {item.shot_id} references unknown character indices: {sorted(unknown_characters)}"
            )
        if item.camera_group not in camera_groups:
            camera_groups.append(item.camera_group)
    if camera_groups != list(range(len(camera_groups))):
        raise ValueError("Scene camera groups must be contiguous from zero in first-appearance order")


class BenchmarkPlan(_StrictModel):
    registry: EntityRegistry
    shots: list[ShotDescription]
    prop_bindings: dict[int, list[str]]
    scene_bindings: dict[int, str]
    scene_registrations: list[BenchmarkSceneRegistrationDraft]


def _validate_characters(characters: list[CharacterInScene]) -> None:
    if [character.idx for character in characters] != list(range(len(characters))):
        raise ValueError("Benchmark characters must have contiguous zero-based indices")
    names = [_normalized_name(character.identifier_in_scene, "character name") for character in characters]
    if len(names) != len(set(names)):
        raise ValueError("Benchmark character names must be unique")


async def prepare_story(
    story: BenchmarkStory,
    characters: list[CharacterInScene],
    analyzer: BenchmarkSceneAnalyzer,
    entity_extractor: EntityExtractor,
) -> BenchmarkPlan:
    """Bind official shots to serial entities without changing benchmark prose."""
    _validate_characters(characters)
    registry = await entity_extractor.extract_entities(
        benchmark_character_source(story),
        characters,
        "",
    )
    shots: list[ShotDescription] = []
    registrations = []
    next_camera_idx = 0

    for scene in story.scenes:
        draft = await analyzer.analyze(scene, characters)
        registrations.append(draft)

        camera_map: dict[int, int] = {}
        for official_shot, binding in zip(scene.shots, draft.shots, strict=True):
            camera_idx = camera_map.get(binding.camera_group)
            if camera_idx is None:
                camera_idx = next_camera_idx
                next_camera_idx += 1
                camera_map[binding.camera_group] = camera_idx
            shot_idx = official_shot.shot_id - 1
            shots.append(ShotDescription(
                idx=shot_idx,
                is_last=official_shot.shot_id == story.metadata.requested_shots,
                cam_idx=camera_idx,
                visual_desc=(
                    f"Opening frame:\n{official_shot.first_frame}\n\n"
                    f"Action over the shot:\n{official_shot.video_prompt}"
                ),
                variation_type="small",
                variation_reason=(
                    "ViMax-Bench supplies one first-frame keyframe and one motion prompt; "
                    "no separate last-frame target may be invented."
                ),
                ff_desc=official_shot.first_frame,
                ff_vis_char_idxs=binding.visible_character_idxs,
                lf_desc="",
                lf_vis_char_idxs=[],
                motion_desc=official_shot.video_prompt,
                audio_desc="",
            ))

    expected = list(range(story.metadata.requested_shots))
    if [shot.idx for shot in shots] != expected:
        raise ValueError("Prepared benchmark shots are not in contiguous story order")
    prop_bindings = {
        shot.idx: await entity_extractor.bind_props_for_shot(shot, registry)
        for shot in shots
    }
    scene_bindings = {
        shot.idx: await entity_extractor.bind_scene_for_shot(shot, registry)
        for shot in shots
    }
    return BenchmarkPlan(
        registry=registry,
        shots=shots,
        prop_bindings=prop_bindings,
        scene_bindings=scene_bindings,
        scene_registrations=registrations,
    )
