"""Run the real, strictly serial continuity demo from an existing hms-world config."""

import argparse
import asyncio
from pathlib import Path
import sys

import yaml
from langchain.chat_models import init_chat_model

from agents.continuity_image_retriever import ContinuityImageRetriever
from agents.entity_extractor import EntityExtractor
from agents.frame_prompt_writer import FramePromptWriter
from agents.shot_observer import ShotObserver
from agents.shot_replanner import ShotReplanner
from pipelines.script2video_pipeline import Script2VideoPipeline
from pipelines.serial_demo_pipeline import SerialDemoPipeline
from utils.local_inference import validate_local_providers as _validate_local_providers

_CONTINUITY_KEYS = {
    "enabled",
    "serial",
    "max_reference_images",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hms-world's real serial continuity demo.")
    parser.add_argument("--config", default="configs/serial_demo_local_h3.yaml", help="Local hms-world YAML configuration.")
    script_group = parser.add_mutually_exclusive_group(required=False)
    script_group.add_argument("--script")
    script_group.add_argument("--script-file")
    requirement_group = parser.add_mutually_exclusive_group(required=False)
    requirement_group.add_argument("--user-requirement")
    requirement_group.add_argument("--user-requirement-file")
    parser.add_argument("--style", required=True)
    parser.add_argument("--output", default="", help="Optional gitignored working directory override.")
    parser.add_argument("--resume", action="store_true", help="Resume one existing serial working directory from its Timeline checkpoint.")
    return parser.parse_args(argv)


def load_serial_demo(config_path: str, output: str = "", *, config_overrides: dict | None = None) -> SerialDemoPipeline:
    """Validate local configuration before constructing or calling any provider."""
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config_overrides:
        config.update(config_overrides)
    _validate_local_providers(config)
    continuity = _validate_continuity_config(config)
    _validate_provider_credentials(config)
    production = (
        Script2VideoPipeline.init_from_config_data(config)
        if config_overrides
        else Script2VideoPipeline.init_from_config(config_path)
    )
    if output:
        production.working_dir = output
        Path(output).mkdir(parents=True, exist_ok=True)
    chat_model = production.chat_model
    vision_model = chat_model
    if "vision_chat_model" in config:
        vision_model = init_chat_model(**_init_args(config, "vision_chat_model"))
    return SerialDemoPipeline(
        production,
        EntityExtractor(chat_model),
        ContinuityImageRetriever(vision_model),
        FramePromptWriter(vision_model),
        ShotObserver(vision_model),
        ShotReplanner(vision_model),
        max_reference_images=continuity["max_reference_images"],
    )


def _validate_continuity_config(config: dict) -> dict:
    continuity = config.get("continuity")
    if not isinstance(continuity, dict):
        raise ValueError("Missing continuity configuration")
    unknown = set(continuity) - _CONTINUITY_KEYS
    if unknown:
        raise ValueError(f"Unsupported continuity configuration keys: {sorted(unknown)}")
    if continuity.get("enabled") is not True:
        raise ValueError("continuity.enabled must be true")
    if continuity.get("serial") is not True:
        raise ValueError("continuity.serial must be true")
    top_k = continuity.get("max_reference_images")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("continuity.max_reference_images must be a positive integer")
    return {
        "max_reference_images": top_k,
    }


def _validate_provider_credentials(config: dict) -> None:
    """Report absent configuration keys without reading or printing their values."""
    missing = []
    sections = ["chat_model", "image_generator", "video_generator"]
    if "vision_chat_model" in config:
        sections.append("vision_chat_model")
    for section in sections:
        key = _init_args(config, section).get("api_key")
        if not isinstance(key, str) or not key.strip():
            missing.append(f"{section}.init_args.api_key")
    if missing:
        raise ValueError("Missing provider credentials: " + ", ".join(missing))


def _init_args(config: dict, section: str) -> dict:
    value = config.get(section)
    if not isinstance(value, dict) or not isinstance(value.get("init_args"), dict):
        raise ValueError(f"Missing {section}.init_args configuration")
    return value["init_args"]


def _input_value(literal: str | None, file_path: str | None, label: str) -> str:
    if literal is not None:
        return literal
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {label} file: {file_path}") from exc


async def amain(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    if args.resume and not args.output.strip():
        raise ValueError("--resume requires an explicit non-empty --output existing working directory")
    if args.resume and any((args.script, args.script_file, args.user_requirement, args.user_requirement_file)):
        raise ValueError("--resume cannot be combined with script or user-requirement input")
    if not args.resume and (
        (args.script is None and args.script_file is None)
        or (args.user_requirement is None and args.user_requirement_file is None)
    ):
        raise ValueError("--script/--script-file and --user-requirement/--user-requirement-file are required unless --resume")
    pipeline = load_serial_demo(args.config, args.output)
    script = "" if args.resume else _input_value(args.script, args.script_file, "script")
    user_requirement = "" if args.resume else _input_value(args.user_requirement, args.user_requirement_file, "user requirement")
    return await pipeline.run(script, user_requirement, args.style, resume=args.resume)


def main() -> None:
    try:
        print(asyncio.run(amain()))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
