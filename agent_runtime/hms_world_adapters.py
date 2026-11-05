"""Web/TUI tools for the same serial plan and execution used by the CLI."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import TypeAdapter
from tenacity import RetryError

from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from main_serial_demo import load_serial_demo
from .config import llm_api_key, llm_base_url, llm_model, llm_model_provider, serial_config_path
from .models import ToolResult
from .tools import ToolArgumentSchema, ToolRuntimeContext, ToolSpec


def build_hms_world_adapter_specs(workspace_root: str | Path, session_index: Any) -> list[ToolSpec]:
    adapter = HmsWorldAdapters(Path(workspace_root), session_index)
    return [
        ToolSpec(
            name="hms_world_narrative_planning",
            description=(
                "Prepare a serial video plan from a story or screenplay under serial/. "
                "Writes characters, storyboard, shots, camera tree, entity registry and Timeline without generating media. "
                "Use script for the agreed story text; idea is a compatible alternative for a story premise. "
                "Pass the active session_id. An existing plan is reused without overwriting it. "
                "revision_target and revision_instruction create a revised plan in a NEW session while preserving the original. "
                "Do not render unless the user requests rendering or end-to-end generation."
            ),
            handler=adapter.hms_world_narrative_planning,
            schema={
                "session_id": ToolArgumentSchema(str, default=""),
                "script": ToolArgumentSchema(str, default=""),
                "idea": ToolArgumentSchema(str, default=""),
                "user_requirement": ToolArgumentSchema(str, default=""),
                "style": ToolArgumentSchema(str, default=""),
                "revision_target": ToolArgumentSchema(str, default=""),
                "revision_instruction": ToolArgumentSchema(str, default=""),
            },
        ),
        ToolSpec(
            name="hms_world_render_video",
            description=(
                "Render the selected serial plan, or resume its persisted Timeline after interruption. "
                "Each shot is generated, observed and committed before replanning the next shot. "
                "Requires hms_world_narrative_planning first. Legacy non-serial sessions remain readable but are not auto-converted."
            ),
            handler=adapter.hms_world_render_video,
            schema={"session_id": ToolArgumentSchema(str, default="")},
        ),
    ]


class HmsWorldAdapters:
    def __init__(self, workspace_root: Path, session_index: Any) -> None:
        self.workspace_root = workspace_root.resolve()
        self.session_index = session_index

    def _pipeline(self, session_id: str, runtime=None):
        root = self.session_index.working_dir(session_id) / "serial"
        pipeline = load_serial_demo(
            str(serial_config_path(self.workspace_root)), output=str(root),
            config_overrides={"working_dir": str(root)},
        )
        pipeline.progress = _pipeline_progress(runtime, session_id)
        return pipeline

    def _result(self, name: str, session_id: str, **payload) -> ToolResult:
        value = {"session_id": session_id, **payload}
        return ToolResult(name, True, json.dumps(value, ensure_ascii=False, indent=2), value)

    def _failure(self, name: str, session_id: str, exc: Exception) -> ToolResult:
        cause = _unwrap_retry_error(exc)
        message = _sanitize_error_text(str(cause))
        self.session_index.update_stage(session_id, "error", message)
        return ToolResult(name, False, message, {
            "session_id": session_id, "error_type": "serial_workflow_failed",
            "error": message, "retryable": _is_retryable_render_error(cause),
        })

    async def hms_world_narrative_planning(self, args: dict[str, Any], runtime=None) -> ToolResult:
        name = "hms_world_narrative_planning"
        script, idea = (str(args.get(key) or "").strip() for key in ("script", "idea"))
        if script and idea:
            return ToolResult(name, False, "Pass script or idea, not both.", {"error_type": "invalid_input"})
        requirements = str(args.get("user_requirement") or "").strip()
        style = str(args.get("style") or "").strip()
        session = self._resolve_session(str(args.get("session_id") or ""), idea=idea, script=script,
                                        user_requirement=requirements, style=style)
        session_id = session["session_id"]
        if args.get("revision_target"):
            return await self._revise_plan(session, args, runtime)
        root = self.session_index.working_dir(session_id) / "serial"
        if (root / "continuity/timeline.json").exists():
            return self._result(name, session_id, ready_for_render=True, reused=True)
        source = script or idea or str(session.get("idea") or "").strip()
        if not source:
            return ToolResult(name, False, "Provide a story or screenplay to plan.", {"error_type": "missing_input", "session_id": session_id})
        # Old artifacts stay readable; creating a serial plan must be explicit.
        work = self.session_index.working_dir(session_id)
        if not (script or idea) and any(p.is_file() for folder in ("idea2video", "script2video", "novel2video") for p in (work / folder).rglob("*")):
            return ToolResult(name, False, "This is a legacy session. Start a new serial session with its agreed story text.", {"error_type": "legacy_session", "session_id": session_id})
        style = style or session.get("style") or "Cinematic, coherent, 16:9"
        self._update_session_metadata(session_id, idea=source, user_requirement=requirements, style=style)
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / "script.txt").write_text(source, encoding="utf-8")
            self.session_index.update_stage(session_id, "narrative_planning", "Preparing the serial text plan")
            pipeline = self._pipeline(session_id, runtime)
            await _run_planning_step("Preparing serial plan", "planning", pipeline.plan(
                _planning_context(root, source), requirements or session.get("user_requirement", "")), runtime, {"session_id": session_id})
            self.session_index.update_stage(session_id, "narrative_planned", "Serial plan ready for review")
            return self._result(name, session_id, ready_for_render=True, working_dir=str(root.relative_to(self.workspace_root)))
        except Exception as exc:
            return self._failure(name, session_id, exc)

    async def hms_world_render_video(self, args: dict[str, Any], runtime=None) -> ToolResult:
        name = "hms_world_render_video"
        session = self.session_index.get(str(args["session_id"])) if args.get("session_id") else self.session_index.active()
        if session is None:
            return ToolResult(name, False, "No active session to render.", {"error_type": "missing_session"})
        session_id = session["session_id"]
        work = self.session_index.working_dir(session_id)
        if not (work / "serial/continuity/timeline.json").is_file():
            return ToolResult(name, False, "A serial Timeline is required. Plan a new serial session first; legacy plans are not automatically converted.", {"error_type": "dependency_missing", "session_id": session_id})
        try:
            self.session_index.update_stage(session_id, "rendering", "Rendering the serial Timeline")
            _write_render_status(work, status="rendering", payload={"session_id": session_id, "render_started": True, "render_completed": False})
            pipeline = self._pipeline(session_id, runtime)
            with _suppress_pipeline_output():
                final_video = await pipeline.resume_preplanned(str(session.get("style") or "Cinematic, coherent, 16:9"))
            payload = {"render_mode": "serial", "render_started": True, "render_completed": True,
                       "final_video_path": str(Path(final_video).relative_to(self.workspace_root)), "missing": []}
            self.session_index.update_stage(session_id, "rendered", "Serial video completed")
            _write_render_status(work, status="rendered", payload={"session_id": session_id, **payload})
            return self._result(name, session_id, **payload)
        except Exception as exc:
            result = self._failure(name, session_id, exc)
            _write_render_status(work, status="error", payload=result.metadata)
            return result

    async def _revise_plan(self, session, args, runtime):
        name = "hms_world_narrative_planning"
        session_id = session["session_id"]
        work = self.session_index.working_dir(session_id)
        relative = str(args["revision_target"])
        instruction = str(args.get("revision_instruction") or "").strip()
        if not instruction:
            return ToolResult(name, False, "revision_instruction is required.", {"error_type": "missing_revision_instruction"})
        try:
            target = _resolve_artifact_path(work, relative)
            serial_root = work / "serial"
            subpath = target.relative_to(serial_root)
            allowed = {"script.txt", "characters.json", "storyboard.json", "camera_tree.json"}
            is_shot = len(subpath.parts) == 3 and subpath.parts[0] == "shots" and subpath.parts[1].isdigit() and subpath.name == "shot_description.json"
            if subpath.as_posix() not in allowed and not is_shot:
                raise ValueError("Revise only the serial script, characters, storyboard, shot description or camera plan")
            before = target.read_text(encoding="utf-8")
            revised = await _revise_artifact_with_llm(_build_chat_model(self.workspace_root), relative, before, instruction)
            schemas = {"characters.json": list[CharacterInScene], "storyboard.json": list[ShotBriefDescription], "camera_tree.json": list[Camera], "shot_description.json": ShotDescription}
            if target.name in schemas:
                TypeAdapter(schemas[target.name]).validate_json(revised)
            if not revised.strip():
                raise ValueError("Revision must not be empty")
            source = revised if subpath.as_posix() == "script.txt" else (serial_root / "script.txt").read_text(encoding="utf-8")
            revision = self.session_index.create(idea=source, user_requirement=session.get("user_requirement", ""), style=session.get("style", ""), project_name=f"{session.get('project_name') or session_id} revision")
            session_id = revision["session_id"]
            destination = self.session_index.working_dir(session_id) / "serial"
            destination.mkdir(parents=True, exist_ok=True)
            # Copy only approved text; a revision gets its own Timeline and media.
            if subpath.as_posix() not in {"script.txt", "characters.json"}:
                shutil.copy2(serial_root / "characters.json", destination / "characters.json")
                shutil.copy2(serial_root / "storyboard.json", destination / "storyboard.json")
                if subpath.as_posix() != "storyboard.json":
                    for path in serial_root.glob("shots/*/shot_description.json"):
                        out = destination / path.relative_to(serial_root)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, out)
            out = destination / subpath
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(revised, encoding="utf-8")
            (destination / "script.txt").write_text(source, encoding="utf-8")
            context = _planning_context(destination, source)
            pipeline = self._pipeline(session_id, runtime)
            await _run_planning_step("Preparing revised serial plan", "planning", pipeline.plan(context, revision.get("user_requirement", "")), runtime, {"session_id": session_id})
            self.session_index.update_stage(session_id, "narrative_planned", "Revised serial plan ready for review")
            return self._result(name, session_id, ready_for_render=True, revision_of=session["session_id"], revised=[relative])
        except Exception as exc:
            return self._failure(name, session_id, exc)

    def _resolve_session(self, session_id: str, *, idea: str, script: str, user_requirement: str, style: str) -> dict[str, Any]:
        requested_source = idea or script
        if session_id:
            session = self.session_index.get(session_id)
            if session is None:
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style, session_id=session_id)
            elif requested_source and _is_new_source_for_session(session, requested_source):
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                self.session_index.set_active(session_id)
        else:
            if requested_source:
                active = self.session_index.active()
                if active is not None and self._session_is_empty(active):
                    session = self.session_index.set_active(active["session_id"])
                else:
                    session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                session = self.session_index.active() or self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
        work = self.session_index.working_dir(session["session_id"])
        has_artifacts = any(path.is_file() for path in work.rglob("*"))
        changed_settings = any(value and value != str(session.get(key) or "")
                               for key, value in (("user_requirement", user_requirement), ("style", style)))
        legacy_source = requested_source and any((work / folder).is_dir() for folder in ("idea2video", "script2video", "novel2video"))
        if has_artifacts and (changed_settings or legacy_source):
            session = self.session_index.create(
                idea=requested_source or session.get("idea", ""),
                user_requirement=user_requirement or session.get("user_requirement", ""),
                style=style or session.get("style", ""),
            )
        self._update_session_metadata(session["session_id"], idea=requested_source, user_requirement=user_requirement, style=style)
        return self.session_index.get(session["session_id"]) or session


    def _session_is_empty(self, session: dict[str, Any]) -> bool:
        if str(session.get("idea") or "").strip():
            return False
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return False
        return not any(path.is_file() for path in self.session_index.working_dir(session_id).rglob("*"))


    def _update_session_metadata(self, session_id: str, *, idea: str, user_requirement: str, style: str) -> None:
        data = self.session_index.load()
        record = data.get("sessions", {}).get(session_id)
        if not isinstance(record, dict):
            return
        if idea and not record.get("idea"):
            record["idea"] = idea
        if user_requirement:
            record["user_requirement"] = user_requirement
        if style:
            record["style"] = style
        self.session_index.save(data)
def _planning_context(root: Path, source: str) -> str:
    """Include reviewed text when registering entities, including on a retry."""
    artifacts = [root / "storyboard.json", *sorted(root.glob("shots/*/shot_description.json"))]
    reviewed = [f"{path.relative_to(root)}:\n{path.read_text(encoding='utf-8')}"
                for path in artifacts if path.is_file()]
    if reviewed:
        return source + "\nReviewed plan artifacts:\n" + "\n".join(reviewed)
    return source


class _DiscardStream:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


@contextmanager
def _suppress_pipeline_output():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with redirect_stdout(_PIPELINE_OUTPUT_SINK), redirect_stderr(_PIPELINE_OUTPUT_SINK):
            yield
    finally:
        logging.disable(previous_disable_level)


def _narrative_step_timeout_seconds() -> float:
    raw = os.environ.get("HMS_WORLD_NARRATIVE_STEP_TIMEOUT_SECONDS", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


async def _run_planning_step(
    message: str,
    stage: str,
    awaitable: Any,
    runtime: ToolRuntimeContext | None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    timeout_seconds = _narrative_step_timeout_seconds()
    event_metadata = dict(metadata or {})
    event_metadata["timeout_seconds"] = timeout_seconds
    if runtime:
        runtime.emit_progress(message, stage=stage, metadata=event_metadata)
        await asyncio.sleep(0)
    try:
        with _suppress_pipeline_output():
            if timeout_seconds <= 0:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{message} timed out after {timeout_seconds:g}s") from exc
    except Exception as exc:
        raise RuntimeError(f"{message} failed: {exc}") from exc


def _is_new_source_for_session(session: dict[str, Any], requested_source: str) -> bool:
    current = str(session.get("idea") or "").strip()
    requested = requested_source.strip()
    if not current or not requested:
        return False
    return current != requested


def _llm_request_timeout_seconds() -> float:
    raw = os.environ.get("HMS_WORLD_LLM_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


def _narrative_max_tokens() -> int:
    raw = os.environ.get("HMS_WORLD_NARRATIVE_MAX_TOKENS", "4096")
    try:
        return max(256, int(raw))
    except ValueError:
        return 4096


def _pipeline_progress(runtime: ToolRuntimeContext | None, session_id: str):
    if runtime is None:
        return None

    def emit(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        payload = dict(metadata or {})
        payload["session_id"] = session_id
        runtime.raise_if_cancelled()
        runtime.emit_progress(message, stage=stage, metadata=payload)

    return emit


def _build_chat_model(workspace_root: str | Path = ".") -> Any:
    api_key = llm_api_key(workspace_root)
    if not api_key:
        raise RuntimeError("Configure a chat API key in the serial YAML, agent.local.yaml llm.api_key or HMS_WORLD_LLM_API_KEY")
    return init_chat_model(
        model=llm_model(workspace_root),
        model_provider=llm_model_provider(workspace_root),
        api_key=api_key,
        base_url=llm_base_url(workspace_root),
        timeout=_llm_request_timeout_seconds(),
        max_retries=0,
        max_completion_tokens=_narrative_max_tokens(),
    )


def _unwrap_retry_error(exc: Exception) -> Exception:
    if isinstance(exc, RetryError):
        try:
            return exc.last_attempt.exception() or exc
        except Exception:
            return exc
    return exc


def _is_retryable_render_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, AttributeError):
        return False
    if "http 403" in text or "key limit exceeded" in text or "quota" in text:
        return False
    return True


def _sanitize_error_text(text: str) -> str:
    sanitized = text
    for marker in ("workspaces/default/keys/",):
        if marker in sanitized:
            prefix, rest = sanitized.split(marker, 1)
            key_id = []
            for char in rest:
                if char.isalnum() or char in "-_":
                    key_id.append(char)
                    continue
                break
            sanitized = prefix + marker + "<redacted>" + rest[len(key_id):]
    if "sk-" in sanitized:
        prefix, rest = sanitized.split("sk-", 1)
        token = []
        for char in rest:
            if char.isalnum() or char in "-_":
                token.append(char)
                continue
            break
        sanitized = prefix + "sk-<redacted>" + rest[len(token):]
    return sanitized


def _write_render_status(working_dir: Path, *, status: str, payload: dict[str, Any]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        **payload,
    }
    (working_dir / "render_status.json").write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    with (working_dir / "render_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _resolve_artifact_path(working_dir: Path, revision_target: str) -> Path:
    rel = Path(revision_target)
    if rel.is_absolute():
        raise ValueError(f"revision_target must be relative to session working_dir: {revision_target}")
    path = (working_dir / rel).resolve()
    if path != working_dir and working_dir not in path.parents:
        raise ValueError(f"revision_target escapes session working_dir: {revision_target}")
    return path


async def _revise_artifact_with_llm(chat_model: Any, target: str, current_text: str, instruction: str) -> str:
    prompt = (
        "Revise this hms-world structured artifact exactly as requested. "
        "Return only the complete replacement file content, with no Markdown fences or explanation. "
        "If the file is JSON, preserve valid JSON and the existing schema shape.\n\n"
        f"Target: {target}\n"
        f"Revision instruction: {instruction}\n\n"
        "Current file content:\n"
        f"{current_text}"
    )
    if hasattr(chat_model, "ainvoke"):
        response = await chat_model.ainvoke(prompt)
    elif hasattr(chat_model, "invoke"):
        response = chat_model.invoke(prompt)
    else:
        raise RuntimeError("chat_model does not support invoke/ainvoke for revision mode")
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return _strip_markdown_fences(str(content).strip())


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


_PIPELINE_OUTPUT_SINK = _DiscardStream()
