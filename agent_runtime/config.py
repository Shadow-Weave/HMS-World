from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LLM_MODEL = "qwen3-vl-30b-a3b"
DEFAULT_LLM_MODEL_PROVIDER = "openai"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:18936/v1"


@lru_cache(maxsize=4)
def load_agent_config(workspace_root: str | Path = ".") -> dict[str, Any]:
    path = Path(workspace_root).resolve() / "configs" / "agent.local.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid configs/agent.local.yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("configs/agent.local.yaml must be a YAML mapping")
    return payload


def config_value(section: str, key: str, env_names: list[str], default: str = "", workspace_root: str | Path = ".") -> str:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    section_payload = load_agent_config(workspace_root).get(section, {})
    if isinstance(section_payload, dict):
        value = section_payload.get(key)
        if isinstance(value, str) and value:
            return value
    if section == "llm":
        path = serial_config_path(workspace_root)
        if path.is_file():
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            value = payload.get("chat_model", {}).get("init_args", {}).get(key)
            if isinstance(value, str) and value:
                return value
    return default


def llm_model(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model", ["HMS_WORLD_LLM_MODEL"], DEFAULT_LLM_MODEL, workspace_root)


def llm_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model_provider", ["HMS_WORLD_LLM_MODEL_PROVIDER"], DEFAULT_LLM_MODEL_PROVIDER, workspace_root)


def llm_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "base_url", ["HMS_WORLD_LLM_BASE_URL"], DEFAULT_LLM_BASE_URL, workspace_root)


def llm_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "api_key", ["HMS_WORLD_LLM_API_KEY", "HMS_WORLD_API_KEY"], "", workspace_root)


def serial_config_path(workspace_root: str | Path = ".") -> Path:
    root = Path(workspace_root).resolve()
    configured = os.environ.get("HMS_WORLD_SERIAL_CONFIG") or load_agent_config(root).get("serial", {}).get("config") or "configs/serial_demo_local_h3.yaml"
    path = Path(configured)
    return path if path.is_absolute() else root / path
