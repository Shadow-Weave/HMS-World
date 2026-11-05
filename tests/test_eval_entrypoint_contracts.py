"""ViMax evaluation entrypoints and existing checkpoint compatibility."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.vimax import catalog, runner as vimax


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", ["evals/vimax/runner.py"])
@pytest.mark.parametrize("args", [["--list"], ["29", "--validate-only"]])
def test_vimax_commands_find_data_from_another_cwd(tmp_path, entrypoint, args):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / entrypoint), *args],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30,
    )
    payload = json.loads(result.stdout)
    if args == ["--list"]:
        assert payload["total_stories"] == 35 and payload["json_files"] == 36
    else:
        assert payload["status"] == "valid"
        assert payload["selected"][0]["id"] == 29
        item = catalog.load_catalog(catalog.BENCHMARK_DATA_ROOT).stories[28]
        assert payload["selected"][0]["sha256"] == item.sha256


@pytest.mark.parametrize("source_matches", [True, False])
def test_vimax_resumes_existing_manifest_and_rejects_changed_input(tmp_path, monkeypatch, source_matches):
    item = catalog.load_catalog(PROJECT_ROOT / "evals/vimax/data").stories[0]
    manifest = vimax._manifest(item, "prepared-only")
    if not source_matches:
        manifest["source_sha256"] = "changed-input"
    manifest_path = tmp_path / "benchmark_run.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = SimpleNamespace(resume_preplanned=AsyncMock(return_value="existing-final.mp4"))
    loader = MagicMock(return_value=pipeline)
    monkeypatch.setattr(vimax, "load_serial_demo", loader)
    run = vimax._resume_case(
        item, case_root=tmp_path, config_path=tmp_path / "config.yaml", style="style",
    )
    if not source_matches:
        with pytest.raises(ValueError, match="source_sha256"):
            asyncio.run(run)
        loader.assert_not_called()
    else:
        result = asyncio.run(run)
        pipeline.resume_preplanned.assert_awaited_once_with("style")
        saved = json.loads(manifest_path.read_text())
        assert saved["format"] == "vimax-benchmark-serial-v1"
        assert saved["source_sha256"] == item.sha256
        assert saved["status"] == result["status"] == "complete"
