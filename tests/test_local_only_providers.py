from copy import deepcopy
from pathlib import Path
import unittest
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml
import main_serial_demo
from tools.render_backend import RenderBackend
from utils.local_inference import require_local_endpoint


class LocalOnlyProvidersTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((Path(__file__).parents[1]/"configs/serial_demo_local_h3.example.yaml").read_text())

    def test_accepts_only_our_local_pipeline(self):
        main_serial_demo._validate_local_providers(self.config)

    def test_both_local_image_adapters_validate_for_saved_checkpoints(self):
        for backend in ("comfy_qwen.ImageGeneratorComfyQwen", "comfy_flux.ImageGeneratorComfyFlux"):
            self.config["image_generator"]["class_path"] = "tools.image_generator_" + backend
            main_serial_demo._validate_local_providers(self.config)

    def test_rejects_cloud_endpoint_in_every_model_role(self):
        for role in ["chat_model", "image_generator", "video_generator"]:
            with self.subTest(role=role):
                value = deepcopy(self.config)
                value[role]["init_args"]["base_url"] = "https://cloud.example/v1"
                with self.assertRaisesRegex(ValueError, "non-loopback"):
                    main_serial_demo._validate_local_providers(value)

    def test_rejects_cloud_image_backend_even_with_loopback_proxy(self):
        self.config["image_generator"]["class_path"] = "unsupported.CloudImageGenerator"
        with self.assertRaisesRegex(ValueError, "non-local image_generator"):
            main_serial_demo._validate_local_providers(self.config)


    def test_common_render_factory_rejects_cloud_even_without_local_flag(self):
        self.config.pop("local_only")
        self.config["image_generator"]["class_path"] = "unsupported.CloudImageGenerator"
        with self.assertRaisesRegex(ValueError, "non-local image_generator"):
            RenderBackend.from_config(self.config)

    def test_rejects_cloud_provider_preset_with_a_local_url(self):
        self.config["chat_model"]["init_args"]["model_provider"] = "minimax"
        with self.assertRaisesRegex(ValueError, "non-local provider"):
            main_serial_demo._validate_local_providers(self.config)

    def test_external_socket_connections_are_blocked_before_connecting(self):
        for method in ("connect", "connect_ex"):
            with self.subTest(method=method), socket.socket() as connection:
                with self.assertRaisesRegex(PermissionError, "local-only"):
                    getattr(connection, method)(("203.0.113.1", 443))

    def test_loopback_connection_still_works(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client.connect(listener.getsockname())
            peer, _ = listener.accept()
            with peer:
                client.sendall(b"local")
                self.assertEqual(peer.recv(5), b"local")

    def test_rejects_cloud_redirect_endpoint(self):
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            require_local_endpoint("https://cloud.example/generated/image.png")

    def test_local_http_bypasses_a_system_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"local")

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            # Simulate the system HTTP proxy without allowing it to receive data.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": "http://127.0.0.1:1"}))
            with opener.open(f"http://127.0.0.1:{server.server_port}/", timeout=2) as response:
                self.assertEqual(response.read(), b"local")
            thread.join(timeout=2)
