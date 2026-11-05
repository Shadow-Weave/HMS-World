#!/usr/bin/env python3
"""Run the indexed ViMax-Bench stories through hms-world's serial pipeline."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_serial_demo import load_serial_demo  # noqa: E402
from evals.vimax.catalog import (  # noqa: E402
    RUN_FORMAT,
    BenchmarkSceneAnalyzer,
    LoadedBenchmarkStory,
    benchmark_character_source,
    load_catalog,
    prepare_story,
    select_stories,
)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _case_name(item: LoadedBenchmarkStory) -> str:
    return f"{item.ref.id:02d}-{item.ref.theme}"


def _manifest(item: LoadedBenchmarkStory, status: str) -> dict:
    return {
        "format": RUN_FORMAT,
        "status": status,
        "benchmark_story_id": item.ref.id,
        "theme": item.ref.theme,
        "consistency_type": item.ref.consistency_type,
        "source_file": item.ref.file,
        "source_sha256": item.sha256,
        "scene_count": len(item.story.scenes),
        "shot_count": item.story.metadata.requested_shots,
        "input_protocol": "official first_frame -> image; official video_prompt -> video",
        "storyboard_regenerated": False,
        "shot_prompts_rewritten_during_initial_adaptation": False,
    }


def _read_manifest(case_root: Path, item: LoadedBenchmarkStory) -> dict:
    path = case_root / "benchmark_run.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resume without a valid benchmark manifest: {path}") from exc
    expected = _manifest(item, manifest.get("status", ""))
    for key in (
        "format",
        "benchmark_story_id",
        "theme",
        "consistency_type",
        "source_file",
        "source_sha256",
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Resume benchmark manifest mismatch for {key}")
    return manifest


def _assert_prompt_identity(item: LoadedBenchmarkStory, shots) -> None:
    official = [shot for scene in item.story.scenes for shot in scene.shots]
    if len(official) != len(shots):
        raise ValueError("Prepared shot count differs from the benchmark story")
    for source, prepared in zip(official, shots, strict=True):
        if prepared.idx != source.shot_id - 1:
            raise ValueError(f"Prepared shot index differs at benchmark shot {source.shot_id}")
        if prepared.ff_desc != source.first_frame:
            raise ValueError(f"Prepared first-frame prompt changed at benchmark shot {source.shot_id}")
        if prepared.motion_desc != source.video_prompt:
            raise ValueError(f"Prepared video prompt changed at benchmark shot {source.shot_id}")


async def _prepare_case(
    item: LoadedBenchmarkStory,
    *,
    case_root: Path,
    config_path: Path,
    style: str,
    execute: bool,
) -> dict:
    case_root.mkdir(parents=True)
    manifest = _manifest(item, "preparing")
    _write_json(case_root / "benchmark_run.json", manifest)
    _write_json(case_root / "benchmark_input.json", item.story.model_dump())

    try:
        pipeline = load_serial_demo(
            str(config_path),
            output=str(case_root),
        )
        characters = await pipeline.production.extract_characters(
            benchmark_character_source(item.story),
            quiet=True,
        )
        plan = await prepare_story(
            item.story,
            characters,
            BenchmarkSceneAnalyzer(pipeline.production.chat_model),
            pipeline.entity_extractor,
        )
        _assert_prompt_identity(item, plan.shots)
        _write_json(
            case_root / "benchmark_scene_registrations.json",
            [registration.model_dump() for registration in plan.scene_registrations],
        )
        cameras = await pipeline.production.construct_camera_tree(plan.shots, quiet=True)
        if execute:
            manifest["status"] = "running"
            _write_json(case_root / "benchmark_run.json", manifest)
            final_video = await pipeline.run_preplanned(
                characters,
                plan.registry,
                plan.shots,
                cameras,
                plan.prop_bindings,
                plan.scene_bindings,
                style,
            )
            manifest.update({"status": "complete", "final_video": final_video})
        else:
            pipeline.prepare_preplanned(
                characters,
                plan.registry,
                plan.shots,
                cameras,
                plan.prop_bindings,
                plan.scene_bindings,
            )
            manifest["status"] = "prepared-only"
        manifest.update({
            "character_count": len(characters),
            "registered_scene_count": len(plan.registry.scenes),
            "registered_prop_count": len(plan.registry.props),
            "camera_count": len(cameras),
        })
        _write_json(case_root / "benchmark_run.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(case_root / "benchmark_run.json", manifest)
        raise


async def _resume_case(
    item: LoadedBenchmarkStory,
    *,
    case_root: Path,
    config_path: Path,
    style: str,
) -> dict:
    manifest = _read_manifest(case_root, item)
    pipeline = load_serial_demo(
        str(config_path),
        output=str(case_root),
    )
    manifest["status"] = "running"
    _write_json(case_root / "benchmark_run.json", manifest)
    try:
        final_video = await pipeline.resume_preplanned(style)
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(case_root / "benchmark_run.json", manifest)
        raise
    manifest.update({"status": "complete", "final_video": final_video})
    manifest.pop("error_type", None)
    manifest.pop("error", None)
    _write_json(case_root / "benchmark_run.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official ViMax-Bench shot prompts through the strict serial continuity pipeline. "
            "Exactly one action must be selected."
        ),
    )
    parser.add_argument("story_ids", nargs="*", type=int, help="Indexed story IDs")
    parser.add_argument("--theme", action="append", default=[], help="Exact indexed theme; repeatable")
    parser.add_argument("--all", action="store_true", help="Select all 35 indexed stories")
    parser.add_argument("--benchmark-root", default="evals/vimax/data")
    parser.add_argument("--config", default="configs/serial_demo_local_h3.yaml")
    parser.add_argument("--style", default="realistic live action")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Run root containing one subdirectory per story",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List the validated index; no provider calls")
    action.add_argument("--validate-only", action="store_true", help="Validate selected dataset inputs; no provider calls or writes")
    action.add_argument("--prepare-only", action="store_true", help="Create and checkpoint the serial text plan; no image/video calls")
    action.add_argument("--execute", action="store_true", help="Prepare and render selected stories sequentially")
    action.add_argument("--resume", action="store_true", help="Resume selected cases from an existing output root")
    args = parser.parse_args(argv)
    if args.list and (args.story_ids or args.theme or args.all):
        parser.error("--list does not accept story selectors")
    if args.all and (args.story_ids or args.theme):
        parser.error("--all cannot be combined with story IDs or --theme")
    if not args.list and not args.all and not args.story_ids and not args.theme:
        parser.error("select story IDs/--theme or use --all")
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir")
    return args


async def amain(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    catalog = load_catalog(_project_path(args.benchmark_root))
    if args.list:
        return {
            "total_stories": catalog.index.total_stories,
            "json_files": catalog.index.total_stories + 1,
            "stories": [
                {
                    "id": item.ref.id,
                    "type": item.ref.consistency_type,
                    "theme": item.ref.theme,
                    "scenes": len(item.story.scenes),
                    "shots": item.story.metadata.requested_shots,
                }
                for item in catalog.stories
            ],
        }
    selected = select_stories(
        catalog,
        ids=args.story_ids,
        themes=args.theme,
        run_all=args.all,
    )
    if args.validate_only:
        return {
            "status": "valid",
            "selected": [
                {
                    "id": item.ref.id,
                    "theme": item.ref.theme,
                    "type": item.ref.consistency_type,
                    "scenes": len(item.story.scenes),
                    "shots": item.story.metadata.requested_shots,
                    "sha256": item.sha256,
                }
                for item in selected
            ],
        }

    config_path = _project_path(args.config)
    if not config_path.is_file():
        raise ValueError(f"Missing hms-world config: {config_path}")
    if args.output_dir:
        output_root = _project_path(args.output_dir)
    else:
        output_root = (
            PROJECT_ROOT
            / ".working_dir"
            / "vimax_benchmark"
            / "runs"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    if args.resume:
        if not output_root.is_dir():
            raise ValueError(f"Resume output root does not exist: {output_root}")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError(f"Fresh benchmark output directory must be empty: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for item in selected:
        case_root = output_root / _case_name(item)
        if args.resume:
            result = await _resume_case(
                item,
                case_root=case_root,
                config_path=config_path,
                style=args.style,
            )
        else:
            result = await _prepare_case(
                item,
                case_root=case_root,
                config_path=config_path,
                style=args.style,
                execute=args.execute,
            )
        results.append(result)

    summary = {
        "format": RUN_FORMAT,
        "status": "complete" if args.execute or args.resume else "prepared-only",
        "output_root": str(output_root),
        "cases": results,
    }
    _write_json(output_root / "vimax_benchmark_run.json", summary)
    return summary


def main() -> None:
    try:
        print(json.dumps(asyncio.run(amain()), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
