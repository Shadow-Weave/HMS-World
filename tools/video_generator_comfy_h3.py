"""Run the native MiniMax H3 ComfyUI graph through its local HTTP interface."""

import asyncio
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time

import aiohttp
import imageio_ffmpeg
from PIL import Image, ImageOps

from interfaces.video_output import VideoOutput
from utils.local_inference import require_local_endpoint


def normalize_frame(path: str) -> bytes:
    """Fit the whole source into 720p, then add the H3 alignment border."""
    with Image.open(path) as source:
        image = ImageOps.pad(
            ImageOps.exif_transpose(source).convert("RGB"), (1280, 720),
            method=Image.Resampling.LANCZOS, color="black",
        )
    image = ImageOps.expand(image, border=(0, 8, 0, 8), fill="black")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_workflow(template: dict, prompt: str, frames: list[str], seed: int) -> dict:
    if not 1 <= len(frames) <= 2:
        raise ValueError("H3 requires a first frame and accepts one optional last frame")
    workflow = copy.deepcopy(template)
    workflow["7"]["inputs"]["image"] = frames[0]
    workflow["8"]["inputs"].update(prompt=prompt, width=1280, height=736, length=124)
    workflow["8"]["inputs"].pop("last_frame", None)
    if len(frames) == 2:
        workflow["18"] = {"class_type": "LoadImage", "inputs": {"image": frames[1]}}
        workflow["8"]["inputs"]["last_frame"] = ["18", 0]
    workflow["10"]["inputs"]["noise_seed"] = seed
    workflow["17"]["inputs"]["filename_prefix"] = f"production-bench/{seed}"
    return workflow


class VideoGeneratorComfyH3:
    def __init__(self, base_url, workflow_path, *, api_key="local", seed=20261019,
                 size="1280x720", duration_seconds=5, rate_limiter=None,
                 poll_interval_seconds=2, poll_timeout_seconds=900):
        if size != "1280x720" or int(duration_seconds) != 5:
            raise ValueError("This H3 workflow produces 1280x720, five-second clips")
        require_local_endpoint(base_url, "video generator")
        self.base_url = base_url.rstrip("/")
        self.template = json.loads(Path(workflow_path).read_text())
        self.seed = seed
        self.rate_limiter = rate_limiter
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds

    async def generate_single_video(self, prompt="", reference_image_paths=None, **kwargs):
        seed = kwargs.get("seed", self.seed)
        references = list(reference_image_paths or [])
        if not 1 <= len(references) <= 2:
            raise ValueError("H3 requires one or two reference frames")
        normalized = [normalize_frame(path) for path in references]
        fingerprint = hashlib.sha256(json.dumps({
            "endpoint": self.base_url, "prompt": prompt, "template": self.template,
            "seed": seed, "frames": [hashlib.sha256(b).hexdigest() for b in normalized],
        }, sort_keys=True).encode()).hexdigest()
        receipt_path = Path(kwargs["task_receipt_path"]) if kwargs.get("task_receipt_path") else None
        receipt = json.loads(receipt_path.read_text()) if receipt_path and receipt_path.exists() else None
        if receipt and receipt.get("request_fingerprint") != fingerprint:
            raise ValueError("Existing H3 task receipt does not match this request")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            if not receipt:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                names = []
                for data in normalized:
                    form = aiohttp.FormData()
                    form.add_field("image", data, filename=hashlib.sha256(data).hexdigest()+".png", content_type="image/png")
                    async with session.post(self.base_url+"/upload/image", data=form) as response:
                        response.raise_for_status()
                        uploaded = await response.json()
                    names.append("/".join(filter(None, [uploaded.get("subfolder"), uploaded["name"]])))
                workflow = build_workflow(self.template, prompt, names, seed)
                async with session.post(self.base_url+"/prompt", json={"prompt": workflow}) as response:
                    payload = await response.json()
                    if response.status >= 400 or "prompt_id" not in payload:
                        raise RuntimeError(f"ComfyUI rejected H3 workflow: {payload}")
                receipt = {"prompt_id": payload["prompt_id"], "request_fingerprint": fingerprint,
                           "workflow": workflow, "started_at": time.time()}
                if receipt_path:
                    receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    receipt_path.write_text(json.dumps(receipt, indent=2))
            deadline = time.monotonic()+self.poll_timeout_seconds
            while time.monotonic() < deadline:
                async with session.get(self.base_url+"/history/"+receipt["prompt_id"]) as response:
                    response.raise_for_status()
                    history = (await response.json()).get(receipt["prompt_id"])
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI H3 failed: {status.get('messages')}")
                    if status.get("completed"):
                        break
                await asyncio.sleep(self.poll_interval_seconds)
            else:
                raise TimeoutError("ComfyUI H3 task timed out; saved receipt can be resumed")
            output = history["outputs"]["17"]
            files = output.get("images") or output.get("videos") or output.get("gifs")
            if not files:
                raise RuntimeError("ComfyUI H3 completed without a saved video")
            async with session.get(self.base_url+"/view", params={
                k: files[0][k] for k in ("filename", "subfolder", "type")
            }) as response:
                response.raise_for_status()
                video = await response.read()
        with tempfile.TemporaryDirectory(prefix="production-h3-") as directory:
            source, target = Path(directory)/"native.mp4", Path(directory)/"720p.mp4"
            source.write_bytes(video)
            process = await asyncio.create_subprocess_exec(
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(source),
                "-vf", "crop=1280:720:0:8", "-t", "5", "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-c:a", "aac", "-movflags", "+faststart", str(target),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, error = await process.communicate()
            if process.returncode:
                raise RuntimeError("H3 output conversion failed: "+error.decode())
            result = target.read_bytes()
        if receipt_path:
            receipt.update(completed_at=time.time(), history=history, output_bytes=len(result))
            receipt_path.write_text(json.dumps(receipt, indent=2))
        return VideoOutput(fmt="bytes", ext="mp4", data=result)
