import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Tuple, Literal, Type, TypeVar
from agents import (
    CameraImageGenerator,
    CharacterExtractor,
    CharacterPortraitsGenerator,
    StoryboardArtist,
)
import yaml
from interfaces.camera import validate_serial_camera_parents
from interfaces import Camera, CharacterInScene, ImageOutput, ShotBriefDescription, ShotDescription
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend
from utils.local_inference import validate_local_providers
from production import cameras, frames, portraits, videos


TModel = TypeVar("TModel")


def _normalize_model_list(items: Any, model_cls: Type[TModel], field_name: str) -> List[TModel]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise TypeError(f"{field_name} must be a list, got {type(items).__name__}")
    normalized: List[TModel] = []
    for idx, item in enumerate(items):
        if isinstance(item, model_cls):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(model_cls.model_validate(item))
        else:
            raise TypeError(f"{field_name}[{idx}] must be {model_cls.__name__} or dict, got {type(item).__name__}")
    return normalized


def _group_shots_into_cameras(shot_descriptions: List[ShotDescription]) -> List[Camera]:
    cameras_by_idx: Dict[int, Camera] = {}
    for shot_description in shot_descriptions:
        camera = cameras_by_idx.get(shot_description.cam_idx)
        if camera is None:
            camera = Camera(idx=shot_description.cam_idx, active_shot_idxs=[])
            cameras_by_idx[shot_description.cam_idx] = camera
        camera.active_shot_idxs.append(shot_description.idx)
    return list(cameras_by_idx.values())


def _pipeline_print(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)


class Script2VideoPipeline:
    """Single-shot production services; SerialDemoPipeline owns execution order."""

    def __init__(
        self,
        chat_model: str,
        image_generator,
        video_generator,
        working_dir: str,
    ):

        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator

        self.character_extractor = CharacterExtractor(chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(image_generator=self.image_generator)
        self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model)
        self.camera_image_generator = CameraImageGenerator(chat_model=self.chat_model, image_generator=self.image_generator, video_generator=self.video_generator)

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.frame_events = {}


    @classmethod
    def init_from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls.init_from_config_data(config)

    @classmethod
    def init_from_config_data(cls, config: dict):
        """Construct a pipeline from already-parsed configuration data."""
        validate_local_providers(config)
        chat_model_args = config["chat_model"]["init_args"]
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(config)

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config["working_dir"],
        )


    def concatenate_rendered_shots(self, shot_idxs: List[int], progress=None) -> str:
        """Concatenate already rendered clips in explicit story order."""
        return videos.concatenate_rendered_shots(self, shot_idxs=shot_idxs, progress=progress)


    async def generate_frames_for_single_camera(
        self,
        camera: Camera,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        serial_shot_idx: int | None = None,
        serial_frame_type: Literal["first_frame", "last_frame"] | None = None,
        continuity_selector: Callable[[List[Tuple[str, str]], str], Any] | None = None,
    ):
        return await cameras.generate_frames_for_single_camera(
            self,
            camera=camera,
            shot_descriptions=shot_descriptions,
            characters=characters,
            character_portraits_registry=character_portraits_registry,
            progress=progress,
            serial_shot_idx=serial_shot_idx,
            serial_frame_type=serial_frame_type,
            continuity_selector=continuity_selector,
        )


    async def _generate_child_camera_material(
        self,
        camera: Camera,
        shot_description: ShotDescription,
        parent_shot_visual_desc: str,
        parent_first_frame_path: str,
        *,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> str:
        """Create the transition-derived first-frame reference for a child camera."""
        return await cameras._generate_child_camera_material(
            self,
            camera=camera,
            shot_description=shot_description,
            parent_shot_visual_desc=parent_shot_visual_desc,
            parent_first_frame_path=parent_first_frame_path,
            progress=progress,
        )

    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> str:
        return await videos.generate_video_for_single_shot(
            self,
            shot_description=shot_description,
            progress=progress,
        )

    async def generate_frame_for_single_shot(
        self,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        first_shot_ff_path_and_text_pair: Tuple[str, str] | None,
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        available_image_path_and_text_pairs: List[Tuple[str, str]] | None = None,
        continuity_selector: Callable[[List[Tuple[str, str]], str], Any] | None = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> ImageOutput:
        return await frames.generate_frame_for_single_shot(
            self,
            shot_idx=shot_idx,
            frame_type=frame_type,
            first_shot_ff_path_and_text_pair=first_shot_ff_path_and_text_pair,
            frame_desc=frame_desc,
            visible_characters=visible_characters,
            character_portraits_registry=character_portraits_registry,
            available_image_path_and_text_pairs=available_image_path_and_text_pairs,
            continuity_selector=continuity_selector,
            progress=progress,
        )


    def prepare_last_frame_event(self, shot_idx: int) -> None:
        """Prepare readiness for a shot newly promoted to first-and-last-frame production."""
        frames.prepare_last_frame_event(self, shot_idx)


    async def construct_camera_tree(
        self,
        shot_descriptions: List[ShotDescription],
        quiet: bool = False,
    ):
        camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")

        if os.path.exists(camera_tree_path):
            with open(camera_tree_path, "r", encoding="utf-8") as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            validate_serial_camera_parents(camera_tree)
            _pipeline_print(quiet, f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
            return camera_tree

        shot_descriptions = _normalize_model_list(shot_descriptions, ShotDescription, "shot_descriptions")
        cameras = _group_shots_into_cameras(shot_descriptions)

        camera_tree = await self.camera_image_generator.construct_camera_tree(cameras=cameras, shot_descs=shot_descriptions)
        camera_tree = _normalize_model_list(camera_tree, Camera, "camera_tree")
        validate_serial_camera_parents(camera_tree)
        with open(camera_tree_path, "w", encoding="utf-8") as f:
            json.dump([camera.model_dump() for camera in camera_tree], f, ensure_ascii=False, indent=4)
        _pipeline_print(quiet, f"✅ Constructed camera tree and saved to {camera_tree_path}.")
        return camera_tree


    async def extract_characters(
        self,
        script: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(character) for character in characters]
            _pipeline_print(quiet, f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Extracted {len(characters)} characters from script and saved to {save_path}.")

        return characters


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        return await portraits.generate_portraits_for_single_character(
            self,
            character=character,
            style=style,
            progress=progress,
        )


    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        quiet: bool = False,
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            _pipeline_print(quiet, f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            _pipeline_print(quiet, f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.design_storyboard(
                script=script,
                characters=characters,
                user_requirement=user_requirement,
                retry_timeout=150,
            )
            storyboard = _normalize_model_list(storyboard, ShotBriefDescription, "storyboard")
            with open(storyboard_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Designed storyboard and saved to {storyboard_path}.")

        return storyboard


    async def decompose_visual_description_for_single_shot_brief_description(
        self,
        shot_brief_description: ShotBriefDescription,
        characters: List[CharacterInScene],
        user_requirement: str = "",
        quiet: bool = False,
    ):
        shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_brief_description.idx}", "shot_description.json")
        os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)

        if os.path.exists(shot_description_path):
            with open(shot_description_path, 'r', encoding='utf-8') as f:
                shot_description = ShotDescription.model_validate(json.load(f))
            _pipeline_print(quiet, f"🚀 Loaded shot {shot_brief_description.idx} description from existing file.")
        else:
            shot_description = await self.storyboard_artist.decompose_visual_description(
                shot_brief_desc=shot_brief_description,
                characters=characters,
                user_requirement=user_requirement,
                retry_timeout=120,
            )
            shot_description = _normalize_model_list([shot_description], ShotDescription, "shot_description")[0]
            with open(shot_description_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Decomposed visual description for shot {shot_brief_description.idx} and saved to {shot_description_path}.")

        if shot_description.variation_type in ["medium", "large"]:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
                "last_frame": asyncio.Event(),
            }
        else:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
            }

        return shot_description
