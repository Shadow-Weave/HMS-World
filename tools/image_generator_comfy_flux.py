"""Local FLUX.2 klein 9B Base with ordered, whole-image references."""
import asyncio
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
import uuid

import aiohttp
from PIL import Image, ImageOps

from interfaces.image_output import ImageOutput
from utils.local_inference import require_local_endpoint


MODEL = "flux-2-klein-base-9b.safetensors"
TEXT_ENCODER = "qwen_3_8b_fp8mixed.safetensors"
VAE = "flux2-vae.safetensors"


def build_workflow(prompt, references, seed, width=1280, height=736, *, steps=50, cfg=4.0):
    if len(references) > 8:
        raise ValueError("This deployment accepts at most eight reference images")
    if not isinstance(steps, int) or steps < 1 or not math.isfinite(cfg) or cfg <= 0:
        raise ValueError("FLUX requires positive steps and cfg")
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "cfg": cfg}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["10", 0], "sampler": ["9", 0], "sigmas": ["7", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"production-flux/{seed}"}},
    }
    positive, negative = ["4", 0], ["5", 0]
    for index, name in enumerate(references):
        load, scale, encode, pos, neg = [str(100 + index * 10 + offset) for offset in range(5)]
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        # Resize the entire image using FLUX's native reference preprocessing.
        graph[scale] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": [load, 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}}
        graph[encode] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        graph[pos] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [encode, 0]}}
        graph[neg] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [encode, 0]}}
        positive, negative = [pos, 0], [neg, 0]
    graph["10"]["inputs"].update(positive=positive, negative=negative)
    return graph


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class ImageGeneratorComfyFlux:
    def __init__(self, base_url, *, api_key="local", rate_limiter=None, seed=20261019,
                 steps=50, cfg=4.0, poll_interval_seconds=1, poll_timeout_seconds=2400,
                 receipt_dir=None):
        require_local_endpoint(base_url, "image generator")
        build_workflow("", [], seed, steps=steps, cfg=cfg)
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.seed, self.steps, self.cfg = seed, steps, cfg
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        self.receipt_dir = Path(receipt_dir) if receipt_dir else None

    async def generate_single_image(self, prompt, reference_image_paths=None, aspect_ratio="16:9", **kwargs):
        references = list(reference_image_paths or [])
        if len(references) > 8:
            raise ValueError("This deployment accepts at most eight reference images")
        refs = [{"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                for path in references]
        request = {"endpoint": self.base_url, "provider_prompt": prompt,
                   "refs": [ref["sha256"] for ref in refs], "seed": self.seed,
                   "model": MODEL, "text_encoder": TEXT_ENCODER, "vae": VAE,
                   "steps": self.steps, "cfg": self.cfg, "canvas": [1280, 736]}
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        receipt_path = Path(kwargs["task_receipt_path"]) if kwargs.get("task_receipt_path") else None
        if receipt_path is None and self.receipt_dir:
            receipt_path = self.receipt_dir / (uuid.uuid4().hex + ".json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path and receipt_path.exists() else None
        if receipt and receipt.get("fingerprint") != fingerprint:
            raise ValueError("Image receipt differs from this request")
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), trust_env=False) as session:
            if not receipt:
                uploaded = []
                for path in references:
                    with Image.open(path) as source:
                        im = ImageOps.exif_transpose(source).convert("RGB")
                        im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                        data = BytesIO()
                        im.save(data, format="PNG")
                    form = aiohttp.FormData()
                    form.add_field("image", data.getvalue(), filename=hashlib.sha256(data.getvalue()).hexdigest()+".png", content_type="image/png")
                    async with session.post(self.base_url+"/upload/image", data=form) as response:
                        response.raise_for_status()
                        item = await response.json()
                    uploaded.append("/".join(filter(None, [item.get("subfolder"), item["name"]])))
                seed = self.seed + uuid.uuid4().int % (2**32)
                graph = build_workflow(prompt, uploaded, seed, steps=self.steps, cfg=self.cfg)
                async with session.post(self.base_url+"/prompt", json={"prompt": graph}) as response:
                    payload = await response.json()
                    if response.status >= 400 or "prompt_id" not in payload:
                        raise RuntimeError(f"Local FLUX workflow rejected: {payload}")
                receipt = {**request, "fingerprint": fingerprint, "prompt_id": payload["prompt_id"],
                           "original_prompt": prompt, "reference_images": refs, "sample_seed": seed,
                           "workflow": graph, "started_at": time.time()}
                if receipt_path:
                    _atomic_json(receipt_path, receipt)
            deadline = time.monotonic() + self.poll_timeout_seconds
            while time.monotonic() < deadline:
                async with session.get(self.base_url+"/history/"+receipt["prompt_id"]) as response:
                    response.raise_for_status()
                    history = (await response.json()).get(receipt["prompt_id"])
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"Local FLUX generation failed: {status.get('messages')}")
                    if status.get("completed"):
                        break
                await asyncio.sleep(self.poll_interval_seconds)
            else:
                raise TimeoutError("Local FLUX image task timed out; task receipt retained")
            item = history["outputs"]["13"]["images"][0]
            async with session.get(self.base_url+"/view", params={k: item[k] for k in ("filename", "subfolder", "type")}) as response:
                response.raise_for_status()
                raw = await response.read()
        with Image.open(BytesIO(raw)) as source:
            if source.size != (1280, 736):
                raise ValueError(f"Unexpected FLUX image dimensions: {source.size}")
            im = source.convert("RGB").crop((0, 8, 1280, 728))
        if receipt_path:
            output = BytesIO()
            im.save(output, format="PNG")
            receipt.update(completed_at=time.time(), history=history,
                           output_sha256=hashlib.sha256(output.getvalue()).hexdigest())
            _atomic_json(receipt_path, receipt)
        return ImageOutput(fmt="pil", ext="png", data=im)
