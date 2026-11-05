"""Rendering contracts and the local-only network boundary."""
from utils.local_inference import install_loopback_guard

install_loopback_guard()

from .protocols import ImageGenerator, VideoGenerator
from .render_backend import RenderBackend

__all__ = ["ImageGenerator", "VideoGenerator", "RenderBackend"]
