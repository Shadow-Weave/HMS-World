"""Existing production progress callbacks and scope metadata."""

from typing import Any, Dict


def _emit_render_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _scoped_progress(progress, **scope):
    if progress is None:
        return None

    def emit(stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
        payload = dict(scope)
        payload.update(metadata or {})
        _emit_render_progress(progress, stage, message, payload)

    return emit
