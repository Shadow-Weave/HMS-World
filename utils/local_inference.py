"""This deployment permits inference only through local SSH forwards."""
import ipaddress
import os
import socket
from urllib.parse import urlsplit


LOCAL_GENERATORS = {
    "image_generator": {"tools.image_generator_comfy_qwen.ImageGeneratorComfyQwen",
                        "tools.image_generator_comfy_flux.ImageGeneratorComfyFlux"},
    "video_generator": {"tools.video_generator_comfy_h3.VideoGeneratorComfyH3"},
}


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def require_local_endpoint(url, role="inference"):
    endpoint = urlsplit(str(url or ""))
    if (endpoint.scheme not in {"http", "https"} or not _loopback(endpoint.hostname)
            or endpoint.username or endpoint.password):
        raise ValueError(f"local_only rejects non-loopback endpoint for {role}")


def validate_local_providers(config):
    for role, expected in LOCAL_GENERATORS.items():
        if config.get(role, {}).get("class_path") not in expected:
            raise ValueError(f"local_only rejects non-local {role}")
    for role in ["chat_model", "vision_chat_model", *LOCAL_GENERATORS]:
        if role == "vision_chat_model" and role not in config:
            continue
        args = config.get(role, {}).get("init_args", {})
        require_local_endpoint(args.get("base_url"), role)
        if role in {"chat_model", "vision_chat_model"} and args.get("model_provider", "openai") != "openai":
            raise ValueError(f"local_only rejects non-local provider for {role}")


def install_loopback_guard():
    """Block external TCP connections in this Python process, including redirects."""
    if getattr(socket, "_hms_world_loopback_only", False):
        return
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    # An empty proxy environment makes macOS clients fall back to system proxies.
    # Keep this process direct; socket guards below still reject external hosts.
    os.environ["NO_PROXY"] = "*"
    for method in ("connect", "connect_ex"):
        original = getattr(socket.socket, method)

        def guarded(self, address, _original=original):
            if self.family in (socket.AF_INET, socket.AF_INET6) and not _loopback(address[0]):
                raise PermissionError("hms-world local-only policy blocks external network connections")
            return _original(self, address)

        setattr(socket.socket, method, guarded)
    socket._hms_world_loopback_only = True
