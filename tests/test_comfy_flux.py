import asyncio
from io import BytesIO
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from PIL import Image
import pytest

from tools.image_generator_comfy_flux import ImageGeneratorComfyFlux, build_workflow, MODEL


def test_flux_keeps_every_whole_reference_in_order_on_both_conditionings():
    names = [f"history-or-portrait-{i}.png" for i in range(8)]
    prompt = "Image 0 is a portrait. Preserve the kettle from Image 7. Two chefs plate food."
    graph = build_workflow(prompt, names, 123)
    assert graph["4"]["inputs"]["text"] == prompt
    assert graph["1"]["inputs"]["unet_name"] == MODEL
    assert graph["7"]["inputs"]["steps"] == 50
    positive, negative = ["4", 0], ["5", 0]
    for index, name in enumerate(names):
        load, scale, encode, pos, neg = [str(100 + index * 10 + i) for i in range(5)]
        assert graph[load]["inputs"]["image"] == name
        assert graph[scale]["inputs"]["image"] == [load, 0]
        assert graph[encode]["inputs"]["pixels"] == [scale, 0]
        assert graph[pos]["inputs"] == {"conditioning": positive, "latent": [encode, 0]}
        assert graph[neg]["inputs"] == {"conditioning": negative, "latent": [encode, 0]}
        positive, negative = [pos, 0], [neg, 0]
    assert graph["10"]["inputs"]["positive"] == positive
    assert graph["10"]["inputs"]["negative"] == negative
    assert not any("crop" in node["class_type"].lower() for node in graph.values())


def test_text_only_and_invalid_sampling():
    graph = build_workflow("A chef", [], 7)
    assert graph["10"]["inputs"]["positive"] == ["4", 0]
    assert graph["8"]["inputs"]["noise_seed"] == 7
    for kwargs in [{"steps": 0}, {"cfg": 0}, {"cfg": float("nan")}]:
        with pytest.raises(ValueError):
            build_workflow("", [], 7, **kwargs)
    with pytest.raises(ValueError, match="eight"):
        build_workflow("", ["a"] * 9, 7)
    with pytest.raises(ValueError, match="non-loopback"):
        ImageGeneratorComfyFlux("https://cloud.example")


@pytest.fixture
def comfy():
    state = {"submissions": [], "uploads": 0}
    buf = BytesIO()
    Image.new("RGB", (1280, 736), "red").save(buf, format="PNG")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/upload/image":
                state["uploads"] += 1
                value = {"name": f"ref-{state['uploads']}.png"}
            elif self.path == "/prompt":
                state["submissions"].append(json.loads(body)["prompt"])
                value = {"prompt_id": f"task-{len(state['submissions'])}"}
            else:
                raise AssertionError(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if self.path.startswith("/history/") else "image/png")
            self.end_headers()
            if self.path.startswith("/history/"):
                task = self.path.rsplit("/", 1)[1]
                self.wfile.write(json.dumps({task: {"status": {"completed": True}, "outputs": {
                    "13": {"images": [{"filename": "image.png", "subfolder": "", "type": "output"}]}}}}).encode())
            else:
                assert self.path.startswith("/view?")
                self.wfile.write(buf.getvalue())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join()


def test_receipt_resume_never_resubmits_and_rejects_changed_reference(tmp_path, comfy):
    url, state = comfy
    reference = tmp_path / "prop-history.png"
    Image.new("RGB", (32, 40), "green").save(reference)
    receipt = tmp_path / "task.json"
    generator = ImageGeneratorComfyFlux(url)
    for _ in range(2):
        result = asyncio.run(generator.generate_single_image("Keep the kettle.", [reference], task_receipt_path=receipt))
        assert result.data.size == (1280, 720)
    assert len(state["submissions"]) == state["uploads"] == 1
    saved = json.loads(receipt.read_text())
    assert saved["reference_images"][0]["path"] == str(reference)
    assert saved["original_prompt"] == saved["provider_prompt"] == "Keep the kettle."
    assert saved["model"] == MODEL and saved["completed_at"] >= saved["started_at"]
    Image.new("RGB", (32, 40), "blue").save(reference)
    with pytest.raises(ValueError, match="receipt differs"):
        asyncio.run(generator.generate_single_image("Keep the kettle.", [reference], task_receipt_path=receipt))
    assert len(state["submissions"]) == 1


def test_regeneration_uses_new_seed_and_audits_every_request(tmp_path, comfy):
    url, state = comfy
    generator = ImageGeneratorComfyFlux(url, receipt_dir=tmp_path)
    for _ in range(2):
        asyncio.run(generator.generate_single_image("Unchanged prompt"))
    assert len(state["submissions"]) == 2
    seeds = [graph["8"]["inputs"]["noise_seed"] for graph in state["submissions"]]
    assert seeds[0] != seeds[1]
    assert len(list(tmp_path.glob("*.json"))) == 2
