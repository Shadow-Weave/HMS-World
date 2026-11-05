"""Thin, strictly serial continuity demo orchestration."""

import json
import asyncio
from pathlib import Path

from pydantic import TypeAdapter

from agents.continuity_image_retriever import ContinuityImageRetriever, ImageRetrievalResult
from agents.entity_extractor import EntityExtractor
from agents.frame_prompt_writer import FramePromptWriter
from agents.shot_observer import ShotObserver
from agents.shot_replanner import NoChange, Replacement, ShotReplanner, replacement_matches_next_shot
from continuity.image_records import ImageRecord, ImageRecordStore
from continuity import checkpoint_io
from continuity.frame_material_context import (
    NativeFrameCandidate,
    build_frame_material_context,
    fixed_material_purpose,
    history_sample_purpose,
)
from continuity.shot_sampling import (
    build_observation_board,
    image_records_for_observed_samples,
    last_observation_sample_from_manifest,
    sample_video,
    write_observation_samples,
)
from continuity.models import EntityRegistry, ObservedPosition, PlannedPosition
from continuity.timeline import Timeline
from interfaces import Camera, CharacterInScene, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline


class SerialDemoPipeline:
    """Connect accepted roles in story order; it owns no new execution abstraction."""

    def __init__(self, production: Script2VideoPipeline, entity_extractor: EntityExtractor, retriever: ContinuityImageRetriever,
                 prompt_writer: FramePromptWriter, observer: ShotObserver, replanner: ShotReplanner,
                 *, max_reference_images: int, progress=None):
        if max_reference_images < 1:
            raise ValueError("max_reference_images must be positive")
        self.production = production
        self.entity_extractor = entity_extractor
        self.retriever = retriever
        self.prompt_writer = prompt_writer
        self.observer = observer
        self.replanner = replanner
        self.max_reference_images = max_reference_images
        self.progress = progress
        self.working_dir = Path(production.working_dir)
        self.image_store = ImageRecordStore(self.working_dir / "continuity" / "image_records.jsonl")

    async def run(self, screenplay: str, user_requirements: str, style: str, *, resume: bool = False) -> str:
        if resume:
            if screenplay or user_requirements:
                raise ValueError("resume does not accept screenplay or user_requirements input")
            characters, registry, portrait_registry, cameras, timeline = self._load_resume_artifacts()
            start_shot_idx = self._validate_resume_timeline(timeline, cameras)
            self._restore_native_frame_events(timeline)
        else:
            characters, registry, portrait_registry, cameras, timeline = await self._start_fresh(
                screenplay, user_requirements, style,
            )
            start_shot_idx = 0

        return await self._run_timeline(
            characters, registry, portrait_registry, cameras, timeline, style,
            start_shot_idx=start_shot_idx, resume=resume,
        )

    async def run_preplanned(
        self,
        characters: list[CharacterInScene],
        registry: EntityRegistry,
        shots: list[ShotDescription],
        cameras: list[Camera],
        prop_bindings: dict[int, list[str]],
        scene_bindings: dict[int, str],
        style: str,
    ) -> str:
        """Run an already-authored shot plan without storyboard regeneration."""
        timeline = self.prepare_preplanned(
            characters, registry, shots, cameras, prop_bindings, scene_bindings,
        )
        portrait_registry = await self._ensure_character_portraits(
            characters, registry, {}, style,
        )
        return await self._run_timeline(
            characters, registry, portrait_registry, cameras, timeline, style,
            start_shot_idx=0, resume=False,
        )

    async def resume_preplanned(self, style: str) -> str:
        """Resume a persisted preplanned run, including interrupted portraits."""
        characters, registry, portrait_registry, cameras, timeline = self._load_resume_artifacts()
        start_shot_idx = self._validate_resume_timeline(timeline, cameras)
        self._restore_native_frame_events(timeline)
        if start_shot_idx < len(timeline.positions):
            portrait_registry = await self._ensure_character_portraits(
                characters, registry, portrait_registry, style,
            )
        return await self._run_timeline(
            characters, registry, portrait_registry, cameras, timeline, style,
            start_shot_idx=start_shot_idx, resume=True,
        )

    def prepare_preplanned(
        self,
        characters: list[CharacterInScene],
        registry: EntityRegistry,
        shots: list[ShotDescription],
        cameras: list[Camera],
        prop_bindings: dict[int, list[str]],
        scene_bindings: dict[int, str],
    ) -> Timeline:
        """Validate and checkpoint a fixed shot plan before any media call."""
        timeline = Timeline.initialize(
            registry,
            shots,
            prop_bindings,
            scene_bindings,
        )
        self._validate_resume_identity_artifacts(characters, registry, cameras, timeline)
        self._save("characters.json", [character.model_dump() for character in characters])
        self._save("character_portraits_registry.json", {})
        self._save_registry(registry)
        self._save("camera_tree.json", [camera.model_dump() for camera in cameras])
        for shot in shots:
            self._save(f"shots/{shot.idx}/shot_description.json", shot.model_dump())
        self._save_timeline(timeline)
        self.production.frame_events = {
            shot.idx: {
                "first_frame": asyncio.Event(),
                **(
                    {"last_frame": asyncio.Event()}
                    if shot.variation_type in {"medium", "large"}
                    else {}
                ),
            }
            for shot in shots
        }
        return timeline


    async def _run_timeline(
        self,
        characters: list[CharacterInScene],
        registry: EntityRegistry,
        portrait_registry: dict,
        cameras: list[Camera],
        timeline: Timeline,
        style: str,
        *,
        start_shot_idx: int,
        resume: bool,
    ) -> str:
        shot_count = len(timeline.positions)
        if start_shot_idx == shot_count:
            return self.production.concatenate_rendered_shots(list(range(shot_count)))
        if start_shot_idx:
            registry, characters, portrait_registry = await self._ensure_replan_after_observed(
                start_shot_idx - 1, timeline, registry, characters, portrait_registry, cameras, style,
            )
        for shot_idx in range(start_shot_idx, shot_count):
            self._emit_progress("serial_shot_start", f"Producing shot {shot_idx + 1}/{shot_count}", shot_idx=shot_idx, shot_count=shot_count)
            position = timeline.refresh_for_execution(shot_idx, registry)
            self._save_timeline(timeline)
            await self._produce_and_observe_position(
                position, timeline, registry, characters, portrait_registry, cameras,
            )
            self._emit_progress("timeline_updated", f"Observed shot {shot_idx + 1}; Timeline updated", shot_idx=shot_idx)
            if shot_idx + 1 < shot_count:
                self._emit_progress("replanning", f"Reviewing the plan for shot {shot_idx + 2}", shot_idx=shot_idx + 1)
                registry, characters, portrait_registry = await self._ensure_replan_after_observed(
                    shot_idx, timeline, registry, characters, portrait_registry, cameras, style,
                    allow_existing=resume,
                )
        self._emit_progress("concat_start", "Assembling the observed shots")
        result = self.production.concatenate_rendered_shots(list(range(shot_count)))
        self._emit_progress("render_done", "Serial video completed", path=result)
        return result

    def _emit_progress(self, stage: str, message: str, **metadata) -> None:
        callback = getattr(self, "progress", None)
        if callback is not None:
            callback(stage, message, metadata)

    async def plan(self, screenplay: str, user_requirements: str) -> Timeline:
        """Prepare the serial text plan for review without generating any media."""
        if (self.working_dir / "continuity/timeline.json").exists():
            raise ValueError("A serial plan already exists; resume it or create a new working directory")
        characters, registry = await self._extract_entities(screenplay, user_requirements)
        self._save("character_portraits_registry.json", {})
        cameras, timeline = await self._plan_shots(screenplay, user_requirements, characters, registry)
        self._emit_progress("narrative_planned", "Serial text plan ready for review", shot_count=len(timeline.positions))
        return timeline

    async def _extract_entities(self, screenplay: str, user_requirements: str):
        self._emit_progress("extract_characters", "Extracting characters and registering visual entities")
        characters = await self.production.extract_characters(screenplay)
        registry = await self.entity_extractor.extract_entities(screenplay, characters, user_requirements)
        self._save("characters.json", [character.model_dump() for character in characters])
        self._save_registry(registry)
        return characters, registry

    async def _start_fresh(self, screenplay: str, user_requirements: str, style: str):
        characters, registry = await self._extract_entities(screenplay, user_requirements)
        portrait_registry = await self._generate_and_register_portraits(characters, registry, style)
        self._save("character_portraits_registry.json", portrait_registry)
        self._save_registry(registry)
        cameras, timeline = await self._plan_shots(screenplay, user_requirements, characters, registry)
        return characters, registry, portrait_registry, cameras, timeline

    async def _plan_shots(self, screenplay, user_requirements, characters, registry):
        self._emit_progress("design_storyboard", "Planning serial shots and camera dependencies")
        briefs = await self.production.design_storyboard(screenplay, characters, user_requirements)
        self._save("storyboard.json", [brief.model_dump() for brief in briefs])
        shots = []
        for brief in briefs:
            shot = await self.production.decompose_visual_description_for_single_shot_brief_description(
                brief,
                characters,
                user_requirement=user_requirements,
            )
            shots.append(shot)
            self._save(f"shots/{shot.idx}/shot_description.json", shot.model_dump())
        cameras = await self.production.construct_camera_tree(shots)
        self._save("camera_tree.json", [camera.model_dump() for camera in cameras])
        prop_bindings = {
            shot.idx: await self.entity_extractor.bind_props_for_shot(shot, registry)
            for shot in shots
        }
        scene_bindings = {
            shot.idx: await self.entity_extractor.bind_scene_for_shot(shot, registry)
            for shot in shots
        }
        timeline = Timeline.initialize(registry, shots, prop_bindings, scene_bindings)
        self._validate_resume_identity_artifacts(characters, registry, cameras, timeline)
        self._save_timeline(timeline)
        return cameras, timeline

    async def _ensure_character_portraits(
        self,
        characters: list[CharacterInScene],
        registry: EntityRegistry,
        portrait_registry: dict,
        style: str,
    ) -> dict:
        """Generate missing portraits one character at a time with checkpoints."""
        result = dict(portrait_registry)
        for character in characters:
            existing = result.get(character.identifier_in_scene)
            if existing is not None:
                self._validate_portrait_entry(character, existing)
                continue
            generated = await self._generate_and_register_portraits([character], registry, style)
            result.update(generated)
            self._save("character_portraits_registry.json", result)
            self._save_registry(registry)
        return result

    @staticmethod
    def _validate_portrait_entry(character: CharacterInScene, entry: dict) -> None:
        return checkpoint_io.validate_portrait_entry(character=character, entry=entry)

    async def produce_position(
        self,
        position: PlannedPosition,
        timeline: Timeline,
        registry: EntityRegistry,
        characters: list[CharacterInScene],
        portrait_registry: dict,
        cameras: list[Camera],
    ) -> str:
        """Render, observe and commit one planned position."""
        shot = position.plan.shot
        self._emit_progress("frames_start", f"Producing frames for shot {shot.idx + 1}", shot_idx=shot.idx)
        self._validate_registered_visible_characters(shot, registry)
        camera = self._camera_for(shot, cameras)
        first_context = build_frame_material_context(
            position, "first", timeline, registry, self._observation_sample_paths(timeline), self.image_store,
            self.max_reference_images,
        )
        await self.production.generate_frames_for_single_camera(
            camera=camera,
            shot_descriptions=self._native_shot_descriptions(timeline, position),
            characters=characters,
            character_portraits_registry=portrait_registry,
            serial_shot_idx=shot.idx,
            serial_frame_type="first_frame",
            continuity_selector=self._continuity_selector(
                shot, "first", first_context, f"shots/{shot.idx}/selected_first_frame_images.json",
                f"shots/{shot.idx}/first_frame_prompt.json",
            ),
        )
        first_path = str(self.working_dir / "shots" / str(shot.idx) / "first_frame.png")
        if not Path(first_path).is_file():
            raise FileNotFoundError(f"Production did not produce first frame for shot {shot.idx}")
        first_id = f"shot-{shot.idx}-first-frame"
        self.image_store.append(ImageRecord(image_id=first_id, kind="generated_first_frame", asset_path=first_path, character_id=None, shot_idx=shot.idx, sample_time_seconds=None, view=None, summary="actual generated first frame"))
        last_context = None
        if shot.variation_type in {"medium", "large"}:
            self.production.prepare_last_frame_event(shot.idx)
            last_context = build_frame_material_context(
                position, "last", timeline, registry, self._observation_sample_paths(timeline), self.image_store,
                self.max_reference_images, current_first_frame_id=first_id,
            )
            await self.production.generate_frames_for_single_camera(
                camera=camera,
                shot_descriptions=self._native_shot_descriptions(timeline, position),
                characters=characters,
                character_portraits_registry=portrait_registry,
                serial_shot_idx=shot.idx,
                serial_frame_type="last_frame",
                continuity_selector=self._continuity_selector(
                    shot, "last", last_context, f"shots/{shot.idx}/selected_last_frame_images.json",
                    f"shots/{shot.idx}/last_frame_prompt.json",
                ),
            )
        reference_slots = self._observation_reference_slots(
            shot,
            position.plan.scene.entity_id,
            first_context,
            last_context,
        )
        if len(reference_slots) > 10:
            raise ValueError("Observation board references must leave at least two temporal slots")
        self._emit_progress("video_clip_start", f"Generating video for shot {shot.idx + 1}", shot_idx=shot.idx)
        video_path = await self.production.generate_video_for_single_shot(shot)
        await self.observe_position(position, timeline, registry, video_path, reference_slots)
        return video_path

    async def observe_position(
        self, position: PlannedPosition, timeline: Timeline, registry: EntityRegistry,
        video_path: str | Path, reference_slots: list[tuple[str, str, str, str]],
    ) -> ObservedPosition:
        """Sample, observe and commit one completed take through the shared production contract."""
        shot = position.plan.shot
        self._emit_progress("observing", f"Observing shot {shot.idx + 1}", shot_idx=shot.idx)
        samples = sample_video(video_path, shot.idx, self.working_dir / "shots", len(reference_slots))
        board_path = build_observation_board(
            [
                (f"CHAR {entity_id}" if kind == "CHARACTER_IDENTITY" else f"SCENE {entity_id}", asset_path)
                for kind, entity_id, _image_id, asset_path in reference_slots
            ],
            samples,
            self.working_dir / "shots" / str(shot.idx) / "observation_board.png",
        )
        reference_slot_map = [
            {"board_slot": index, "kind": kind, "entity_id": entity_id, "image_id": image_id}
            for index, (kind, entity_id, image_id, _asset_path) in enumerate(reference_slots)
        ]
        observed_state = await self.observer.observe(position.plan, board_path, samples, registry, reference_slot_map)
        write_observation_samples(
            self.working_dir / "shots" / str(shot.idx) / "observation_samples.json",
            samples,
        )
        for record in image_records_for_observed_samples(samples, shot.idx):
            self.image_store.append(record)
        observed = timeline.mark_observed(shot.idx, observed_state)
        self._save_timeline(timeline)
        return observed

    async def _produce_and_observe_position(
        self,
        position: PlannedPosition,
        timeline: Timeline,
        registry: EntityRegistry,
        characters: list[CharacterInScene],
        portrait_registry: dict,
        cameras: list[Camera],
    ) -> None:
        await self.produce_position(
            position,
            timeline,
            registry,
            characters,
            portrait_registry,
            cameras,
        )

    def _save(self, relative_path: str, value) -> None:
        return checkpoint_io.save_json(self, relative_path=relative_path, value=value)

    def _save_registry(self, registry: EntityRegistry) -> None:
        return checkpoint_io.save_registry(self, registry=registry)

    def _save_timeline(self, timeline: Timeline) -> None:
        return checkpoint_io.save_timeline(self, timeline=timeline)

    def _load_resume_artifacts(self) -> tuple[list[CharacterInScene], EntityRegistry, dict, list[Camera], Timeline]:
        """Load only formal persisted artifacts; Timeline remains authoritative."""
        return checkpoint_io.load_resume_artifacts(self)

    @staticmethod
    def _validate_resume_identity_artifacts(
        characters: list[CharacterInScene], registry: EntityRegistry, cameras: list[Camera], timeline: Timeline,
    ) -> None:
        return checkpoint_io.validate_resume_identity_artifacts(
            characters=characters,
            registry=registry,
            cameras=cameras,
            timeline=timeline,
        )

    def _validate_resume_timeline(self, timeline: Timeline, cameras: list[Camera]) -> int:
        """Fail closed unless positions are an observed prefix then planned suffix."""
        return checkpoint_io.validate_resume_timeline(self, timeline=timeline, cameras=cameras)

    def _validate_observed_checkpoint(self, position: ObservedPosition, cameras: list[Camera]) -> None:
        return checkpoint_io.validate_observed_checkpoint(self, position=position, cameras=cameras)

    def _restore_native_frame_events(self, timeline: Timeline) -> None:
        """Recreate only ViMax's per-shot readiness events from formal frame files."""
        return checkpoint_io.restore_native_frame_events(self, timeline=timeline)

    async def _ensure_replan_after_observed(
        self, shot_idx: int, timeline: Timeline, registry: EntityRegistry,
        characters: list[CharacterInScene], portrait_registry: dict, cameras: list[Camera], style: str,
        *, allow_existing: bool = True,
    ) -> tuple[EntityRegistry, list[CharacterInScene], dict]:
        """Persist or verify the one allowed replan after an observed shot."""
        if shot_idx + 1 >= len(timeline.positions):
            return registry, characters, portrait_registry
        observed = timeline.position(shot_idx)
        next_position = timeline.position(shot_idx + 1)
        if not isinstance(observed, ObservedPosition) or not isinstance(next_position, PlannedPosition):
            raise ValueError("Replan checkpoint requires observed shot followed by planned shot")
        replan_path = self.working_dir / "shots" / str(shot_idx) / "replan.json"
        if allow_existing and replan_path.exists():
            try:
                payload = checkpoint_io.read_json(replan_path)
                decision = TypeAdapter(NoChange | Replacement).validate_python(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Resume replan checkpoint for shot {shot_idx} is malformed") from exc
            formal_next = checkpoint_io.read_shot_description(
                self.working_dir / "shots" / str(shot_idx + 1) / "shot_description.json"
            )
            if isinstance(decision, NoChange):
                if formal_next.model_dump() != next_position.plan.shot.model_dump():
                    raise ValueError(f"Resume no-change replan after {shot_idx} conflicts with Timeline")
                return registry, characters, portrait_registry
            replacement = decision.replacement_shot
            if replacement_matches_next_shot(replacement, next_position.plan.shot):
                if formal_next.model_dump() != next_position.plan.shot.model_dump():
                    raise ValueError(f"Resume applied replacement after {shot_idx} conflicts with formal shot artifact")
                return registry, characters, portrait_registry
            if formal_next.model_dump() != next_position.plan.shot.model_dump():
                raise ValueError(f"Resume saved replacement after {shot_idx} has no coherent pre-apply checkpoint")
            return await self._apply_replacement_decision(
                shot_idx, decision, timeline, registry, characters, portrait_registry, style,
            )

        previous_shot = checkpoint_io.read_shot_description(
            self.working_dir / "shots" / str(shot_idx) / "shot_description.json"
        )
        next_shot = next_position.plan.shot
        scenes_by_id = {scene.entity_id: scene for scene in registry.scenes}
        try:
            next_scene_name = scenes_by_id[next_position.plan.scene.entity_id].name
        except KeyError as exc:
            raise ValueError(
                f"Next shot {next_shot.idx} references an unregistered scene"
            ) from exc
        decision = await self.replanner.replan(
            next_scene_name, previous_shot.visual_desc, observed.state,
            last_observation_sample_from_manifest(
                self.working_dir / "shots" / str(shot_idx) / "observation_samples.json",
                self.image_store,
                shot_idx,
            ),
            next_shot, self._camera_for(next_shot, cameras), registry,
        )
        self._save(f"shots/{shot_idx}/replan.json", decision.model_dump())
        if isinstance(decision, NoChange):
            timeline.apply_no_change(shot_idx)
            return registry, characters, portrait_registry
        return await self._apply_replacement_decision(
            shot_idx, decision, timeline, registry, characters, portrait_registry, style,
        )

    async def _apply_replacement_decision(
        self, shot_idx: int, decision: Replacement, timeline: Timeline, registry: EntityRegistry,
        characters: list[CharacterInScene], portrait_registry: dict, style: str,
    ) -> tuple[EntityRegistry, list[CharacterInScene], dict]:
        """Apply one persisted replacement to the sole next Timeline position."""
        unknown = self._unknown_character_idxs(decision.replacement_shot, registry)
        registry, added = await self.entity_extractor.extend_for_replacement(
            decision.replacement_shot, unknown, registry, characters,
        )
        characters.extend(added)
        portrait_registry.update(await self._generate_and_register_portraits(added, registry, style))
        self._save("characters.json", [character.model_dump() for character in characters])
        self._save("character_portraits_registry.json", portrait_registry)
        self._save_registry(registry)
        prop_entity_ids = await self.entity_extractor.bind_props_for_shot(decision.replacement_shot, registry)
        timeline.apply_replacement(
            shot_idx,
            decision.replacement_shot,
            registry,
            prop_entity_ids,
        )
        self._save(f"shots/{decision.replacement_shot.idx}/shot_description.json", decision.replacement_shot.model_dump())
        self._save_timeline(timeline)
        return registry, characters, portrait_registry

    def _observation_sample_paths(self, timeline: Timeline) -> dict[int, Path]:
        """Map observed shots to their sole ordered observation-sample manifests."""
        return checkpoint_io.observation_sample_paths(self, timeline=timeline)

    def _native_shot_descriptions(
        self,
        timeline: Timeline,
        current_position: PlannedPosition,
    ) -> list[ShotDescription]:
        """Pass the single-track plan to ViMax, loading only already-observed history."""
        return checkpoint_io.native_shot_descriptions(
            self,
            timeline=timeline,
            current_position=current_position,
        )

    def _observation_reference_slots(self, shot: ShotDescription, scene_entity_id: str, first_context, last_context):
        """Read only Observer-relevant materials from formal endpoint receipts."""
        selected_characters: dict[str, tuple[str, str]] = {}
        selected_scene: tuple[str, str] | None = None
        endpoint_contexts = [("first", first_context)]
        if last_context is not None:
            endpoint_contexts.append(("last", last_context))
        for frame_type, context in endpoint_contexts:
            path = self.working_dir / "shots" / str(shot.idx) / f"selected_{frame_type}_frame_images.json"
            if not path.exists():
                continue
            try:
                result = ImageRetrievalResult.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Missing or invalid selected {frame_type} frame receipt {path}") from exc
            fixed_by_id = {item.image_id: item for item in context.fixed_materials}
            portraits_by_id = {
                candidate.image_id: (group.character_id, candidate.asset_path)
                for group in context.portrait_candidates
                for candidate in group.candidates
            }
            history_by_id = {
                sample.image_id: (group, sample)
                for group in context.history_sample_groups
                for sample in group.samples
            }
            for item in result.images:
                portrait = portraits_by_id.get(item.candidate_id)
                if portrait is not None:
                    if item.purpose != "character identity portrait":
                        raise ValueError(f"Selected portrait {item.candidate_id} has an invalid purpose")
                    selected_characters.setdefault(item.candidate_id, portrait)
                    continue
                history = history_by_id.get(item.candidate_id)
                if history is not None:
                    group, sample = history
                    if item.purpose != history_sample_purpose(group, sample):
                        raise ValueError(f"Selected history sample {item.candidate_id} has an invalid purpose")
                    if selected_scene is None and any(
                        lineage.entity_id == scene_entity_id for lineage in group.lineage
                    ):
                        selected_scene = (sample.image_id, sample.asset_path)
                    continue
                fixed = fixed_by_id.get(item.candidate_id)
                if fixed is not None:
                    if item.purpose != fixed_material_purpose(fixed.role):
                        raise ValueError(f"Selected fixed material {item.candidate_id} has an invalid purpose")
                    continue
                if item.purpose == "native ViMax frame candidate":
                    continue
                if item.purpose == "character identity portrait":
                    raise ValueError(f"Selected portrait {item.candidate_id} is not in this endpoint context")
                raise ValueError(f"Selected material {item.candidate_id} is not in this endpoint context")
        slots = [
            ("CHARACTER_IDENTITY", entity_id, image_id, asset_path)
            for image_id, (entity_id, asset_path) in selected_characters.items()
        ]
        if selected_scene is not None:
            slots.append(("SCENE_HISTORY", scene_entity_id, *selected_scene))
        return slots

    def _continuity_selector(
        self,
        shot: ShotDescription,
        frame_type: str,
        context,
        selected_path: str,
        prompt_path: str,
    ):
        """Adapt ViMax-prepared candidates to the continuity material contract.

        It deliberately receives only candidate pairs already assembled by the
        native frame coroutine.  It does not read or interpret Camera topology.
        """
        async def select(available_image_path_and_text_pairs, frame_description):
            if context.shot_idx != shot.idx or context.frame_type != frame_type:
                raise ValueError(
                    f"Continuity context does not match Timeline shot {shot.idx} {frame_type} endpoint"
                )
            expected_description = shot.ff_desc if frame_type == "first" else shot.lf_desc
            if frame_description != expected_description:
                raise ValueError(
                    f"Native {frame_type} frame description does not match Timeline shot {shot.idx}"
                )
            deterministic_paths = {
                str(Path(item.asset_path).resolve())
                for item in [
                    *context.fixed_materials,
                    *(
                        sample
                        for group in context.history_sample_groups
                        for sample in group.samples
                    ),
                    *(candidate for group in context.portrait_candidates for candidate in group.candidates),
                ]
            }
            native_candidates = []
            native_paths = set()
            for index, (asset_path, description) in enumerate(available_image_path_and_text_pairs):
                resolved_path = str(Path(asset_path).resolve())
                if resolved_path in deterministic_paths or resolved_path in native_paths:
                    continue
                native_paths.add(resolved_path)
                native_candidates.append(NativeFrameCandidate(
                    candidate_id=f"native-{shot.idx}-{frame_type}-{index}",
                    asset_path=asset_path,
                    description=description,
                ))
            prepared_context = context.model_copy(update={"native_candidates": native_candidates})
            result = await self.retriever.retrieve(prepared_context, frame_description)
            self._save(selected_path, result.model_dump())
            prompt = await self.prompt_writer.write(frame_description, prepared_context, result)
            native_by_id = {candidate.candidate_id: candidate for candidate in native_candidates}
            references = []
            for item in result.images:
                candidate = native_by_id.get(item.candidate_id)
                if candidate is None:
                    asset_path = self.image_store.get(item.candidate_id).asset_path
                else:
                    asset_path = candidate.asset_path
                references.append((asset_path, item.purpose))
            self._save(prompt_path, {"prompt": prompt})
            return references, prompt

        return select

    async def _generate_and_register_portraits(
        self,
        characters,
        registry: EntityRegistry,
        style: str,
    ) -> dict:
        """Produce one new character at a time, then register real portrait paths."""
        portrait_registry = {}
        entities = {entity.entity_id: entity for entity in registry.characters}
        for character in characters:
            entity_id = f"character-{character.idx}"
            try:
                entity = entities[entity_id]
            except KeyError as exc:
                raise ValueError(f"Portrait character {entity_id} is not registered") from exc
            portraits = await self.production.generate_portraits_for_single_character(character, style)
            try:
                portrait_registry.update(portraits)
                by_view = portraits[character.identifier_in_scene]
            except KeyError as exc:
                raise ValueError(f"Production portraits missing character {character.identifier_in_scene}") from exc
            for view in ("front", "side", "back"):
                try:
                    item = by_view[view]
                    path = item["path"]
                    summary = item["description"]
                except KeyError as exc:
                    raise ValueError(f"Production portraits missing {view} view for {character.identifier_in_scene}") from exc
                image_id = f"portrait-character-{character.idx}-{view}"
                self.image_store.append(ImageRecord(
                    image_id=image_id,
                    kind="portrait",
                    asset_path=path,
                    character_id=entity.entity_id,
                    shot_idx=None,
                    sample_time_seconds=None,
                    view=view,
                    summary=summary,
                ))
                if image_id not in entity.identity_image_ids:
                    entity.identity_image_ids.append(image_id)
        return portrait_registry

    @staticmethod
    def _validate_registered_visible_characters(shot: ShotDescription, registry: EntityRegistry) -> None:
        """Fail before any downstream provider sees an unregistered character.

        Character indices are the authoritative visible-character references in
        the shared ShotDescription contract.
        """
        indices = list(shot.ff_vis_char_idxs)
        if shot.variation_type != "small":
            indices.extend(shot.lf_vis_char_idxs)
        registered = {character.entity_id for character in registry.characters}
        unknown = [f"character-{idx}" for idx in dict.fromkeys(indices) if f"character-{idx}" not in registered]
        if unknown:
            raise ValueError(f"Shot {shot.idx} references unregistered entities: {unknown}")

    @staticmethod
    def _unknown_character_idxs(shot: ShotDescription, registry: EntityRegistry) -> set[int]:
        indices = set(shot.ff_vis_char_idxs)
        if shot.variation_type != "small":
            indices.update(shot.lf_vis_char_idxs)
        registered = {entity.entity_id for entity in registry.characters}
        return {index for index in indices if f"character-{index}" not in registered}

    @staticmethod
    def _camera_for(shot: ShotDescription, cameras: list[Camera]) -> Camera:
        matching = [camera for camera in cameras if camera.idx == shot.cam_idx]
        if len(matching) != 1:
            raise ValueError(f"Shot {shot.idx} requires exactly one camera {shot.cam_idx}")
        return matching[0]
