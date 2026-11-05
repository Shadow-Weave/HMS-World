"""Native local Qwen-Image-2.1 generation and reference-image editing."""
import asyncio
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
import uuid

import aiohttp
from PIL import Image, ImageOps

from interfaces.image_output import ImageOutput
from utils.local_inference import require_local_endpoint


def prepare_qwen_prompt(prompt: str, reference_count: int) -> str:
    """Translate ViMax's zero-based Image N labels to Qwen's native slots."""
    def replace(match):
        index = int(match.group(1))
        if index >= reference_count:
            raise ValueError(f"Prompt refers to Image {index}, but only {reference_count} references were supplied")
        return f"<image{index+1}>"

    converted = re.sub(r"(?<![<\w])Image\s+(\d+)\b", replace, prompt, flags=re.IGNORECASE)
    for match in re.finditer(r"<image(\d+)>", converted, flags=re.IGNORECASE):
        if not 1 <= int(match.group(1)) <= reference_count:
            raise ValueError("Prompt refers to a Qwen image slot that was not supplied")
    return converted


def _sampling_config(model_variant, steps):
    if model_variant not in {"turbo", "standard"}:
        raise ValueError("Qwen model_variant must be turbo or standard")
    steps = (8 if model_variant == "turbo" else 25) if steps is None else steps
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Qwen steps must be a positive integer")
    if model_variant == "turbo" and steps != 8:
        raise ValueError("Turbo uses its fixed eight-step schedule; select standard for other step counts")
    checkpoint = ("qwen_image_2.1_turbo_int8_convrot.safetensors" if model_variant == "turbo"
                  else "qwen_image_2.1_bf16.safetensors")
    return checkpoint, steps


def build_workflow(prompt, references, seed, width=1280, height=736, *, model_variant="turbo", steps=None):
    if len(references) > 8:
        raise ValueError("This deployment accepts at most eight reference images")
    checkpoint, steps = _sampling_config(model_variant, steps)
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 960, "vae": ["3", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "7": {"class_type": "CFGGuider", "inputs": {"model": ["6", 0], "positive": ["4", 0], "negative": ["4", 1], "cfg": 1.0}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["7", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["5", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "production-local/"+str(seed)}},
    }
    if model_variant == "standard":
        graph["10"] = {"class_type": "BasicScheduler", "inputs": {
            "model": ["6", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0,
        }}
    for i, name in enumerate(references, 1):
        node = str(20+i)
        graph[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["4"]["inputs"][f"images.image_{i}"] = [node, 0]
    return graph


class ImageGeneratorComfyQwen:
    def __init__(self, base_url, *, api_key="local", rate_limiter=None, seed=20261019,
                 poll_interval_seconds=1, poll_timeout_seconds=1200, model_variant="turbo", steps=None):
        require_local_endpoint(base_url, "image generator")
        _, self.steps = _sampling_config(model_variant, steps)
        self.model_variant = model_variant
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.seed = seed
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds

    async def generate_single_image(self, prompt, reference_image_paths=None, aspect_ratio="16:9", **kwargs):
        references = list(reference_image_paths or [])
        if len(references) > 8:
            raise ValueError("This deployment accepts at most eight reference images")
        provider_prompt = prepare_qwen_prompt(prompt, len(references))
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        receipt_path = Path(kwargs["task_receipt_path"]) if kwargs.get("task_receipt_path") else None
        fingerprint = hashlib.sha256(json.dumps({"endpoint": self.base_url, "provider_prompt": provider_prompt, "refs": [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in references], "seed": self.seed, "model_variant": self.model_variant, "steps": self.steps}, sort_keys=True).encode()).hexdigest()
        receipt = json.loads(receipt_path.read_text()) if receipt_path and receipt_path.exists() else None
        if receipt and receipt.get("fingerprint") != fingerprint:
            raise ValueError("Image receipt differs from this request")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), trust_env=False) as session:
            if not receipt:
                uploaded = []
                for path in references:
                    with Image.open(path) as source:
                        im = ImageOps.exif_transpose(source).convert("RGB")
                        im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                        data = BytesIO(); im.save(data, format="PNG")
                    form = aiohttp.FormData()
                    form.add_field("image", data.getvalue(), filename=hashlib.sha256(data.getvalue()).hexdigest()+".png", content_type="image/png")
                    async with session.post(self.base_url+"/upload/image", data=form) as response:
                        response.raise_for_status(); item = await response.json()
                    uploaded.append("/".join(filter(None, [item.get("subfolder"), item["name"]])))
                seed = self.seed + (int(fingerprint[:8], 16) if receipt_path else uuid.uuid4().int % (2**32))
                graph = build_workflow(provider_prompt, uploaded, seed, model_variant=self.model_variant, steps=self.steps)
                async with session.post(self.base_url+"/prompt", json={"prompt": graph}) as response:
                    payload = await response.json()
                    if response.status >= 400 or "prompt_id" not in payload:
                        raise RuntimeError(f"Local Qwen workflow rejected: {payload}")
                receipt = {"fingerprint": fingerprint, "prompt_id": payload["prompt_id"], "original_prompt": prompt,
                           "provider_prompt": provider_prompt, "model_variant": self.model_variant,
                           "steps": self.steps, "workflow": graph, "started_at": time.time()}
                if receipt_path:
                    receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    receipt_path.write_text(json.dumps(receipt, indent=2))
            deadline = time.monotonic()+self.poll_timeout_seconds
            while time.monotonic() < deadline:
                async with session.get(self.base_url+"/history/"+receipt["prompt_id"]) as response:
                    response.raise_for_status(); history = (await response.json()).get(receipt["prompt_id"])
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"Local Qwen generation failed: {status.get('messages')}")
                    if status.get("completed"):
                        break
                await asyncio.sleep(self.poll_interval_seconds)
            else:
                raise TimeoutError("Local Qwen image task timed out")
            item = history["outputs"]["13"]["images"][0]
            async with session.get(self.base_url+"/view", params={k: item[k] for k in ("filename", "subfolder", "type")}) as response:
                response.raise_for_status(); raw = await response.read()
        with Image.open(BytesIO(raw)) as source:
            im = source.convert("RGB")
            # Match the 720p video canvas; model dimensions are multiples of 32.
            im = im.crop((0, 8, 1280, 728))
        if receipt_path:
            receipt.update(completed_at=time.time(), history=history)
            receipt_path.write_text(json.dumps(receipt, indent=2))
        return ImageOutput(fmt="pil", ext="png", data=im)
