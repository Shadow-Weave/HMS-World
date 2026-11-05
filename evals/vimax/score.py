#!/usr/bin/env python3
"""Score existing ViMax-Bench shots with ViCLIP and measure local inference time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def consistency_metrics(features: np.ndarray, scene_numbers: list[int]) -> tuple[dict, np.ndarray]:
    """Average distinct shot pairs, using the benchmark's authored scene groups."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(scene_numbers) or len(features) < 2:
        raise ValueError("Scoring requires at least two shot features with matching scene labels")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms == 0):
        raise ValueError("ViCLIP features must be finite and nonzero")
    normalized = features / norms
    similarities = normalized @ normalized.T
    rows, cols = np.triu_indices(len(features), k=1)
    pair_values = similarities[rows, cols]
    labels = np.asarray(scene_numbers)
    same_scene = labels[rows] == labels[cols]

    def average(values):
        return float(values.mean()) if len(values) else None

    return {
        "IC": average(pair_values[same_scene]),
        "CC": average(pair_values[~same_scene]),
        "GC": average(pair_values),
        "intra_scene_pairs": int(same_scene.sum()),
        "cross_scene_pairs": int((~same_scene).sum()),
        "all_pairs": int(len(pair_values)),
    }, similarities


def read_video(path: Path) -> tuple[list, dict]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open shot video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 8:
        raise ValueError(f"ViCLIP requires at least eight frames: {path}")
    step = len(frames) // 8
    return frames, {
        "frame_count": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps if fps > 0 else None,
        "sampled_frame_indices": list(range(0, step * 8, step)),
    }


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Existing case directory with benchmark_input.json and shots/")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / ".working_dir/viclip-runtime")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    case_dir = args.case_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    story = json.loads((case_dir / "benchmark_input.json").read_text())
    shots = [(scene["scene_num"], shot) for scene in story["scenes"] for shot in scene["shots"]]
    paths = [case_dir / "shots" / str(shot["shot_id"] - 1) / "video.mp4" for _, shot in shots]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Full-story scoring requires every shot video; missing={missing}")

    import torch

    source = runtime_dir / "source"
    sys.path.insert(0, str(source))
    from viclip import frames2tensor, get_vid_feat, get_viclip

    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("This PyTorch runtime does not have an available MPS device")
    device = torch.device(device_name)

    def synchronize():
        if device_name == "mps":
            torch.mps.synchronize()

    load_started = time.perf_counter()
    weight_path = runtime_dir / "models/ViClip-InternVid-10M-FLT.pth"
    model = get_viclip(size="l", pretrain=str(weight_path))["viclip"].eval().to(device)
    synchronize()
    model_load_seconds = time.perf_counter() - load_started
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(case_dir),
        "theme": story["metadata"]["theme_key"],
        "consistency_type": story["consistency_type"],
        "shot_count": len(shots),
        "model": "ViCLIP-L-14 / InternVid-10M-FLT",
        "weights": str(weight_path),
        "source_manifest": json.loads((runtime_dir / "source_manifest.json").read_text()),
        "torch_version": torch.__version__,
        "device": device_name,
        "dtype": "float32",
        "sampling": "Official frames2tensor: 8 frames, floor-stride from frame zero, resize to 224x224, RGB ImageNet normalization",
        "scene_grouping": "Original benchmark_input.json scene_num; generation registry scenes are not used",
        "model_load_seconds": model_load_seconds,
        "shots": [],
    }
    print(json.dumps({"event": "model_loaded", "device": device_name, "seconds": model_load_seconds, "peak_rss_mb": peak_rss_mb()}), flush=True)
    features = []
    for index, ((scene_num, shot), path) in enumerate(zip(shots, paths, strict=True)):
        shot_started = time.perf_counter()
        frames, video_info = read_video(path)
        decode_seconds = time.perf_counter() - shot_started
        preprocess_started = time.perf_counter()
        # Official normalization uses float64 NumPy intermediates. Cast to the
        # official float32 result on CPU before transfer: MPS rejects float64.
        tensor = frames2tensor(frames, device=torch.device("cpu")).to(device)
        synchronize()
        preprocess_seconds = time.perf_counter() - preprocess_started
        del frames
        inference_started = time.perf_counter()
        with torch.inference_mode():
            feature = get_vid_feat(tensor, model)
        synchronize()
        inference_seconds = time.perf_counter() - inference_started
        vector = feature.detach().cpu().numpy().reshape(-1)
        features.append(vector)
        del tensor, feature
        timing = {
            "shot_id": shot["shot_id"],
            "scene_num": scene_num,
            "video": str(path),
            **video_info,
            "decode_seconds": decode_seconds,
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - shot_started,
            "feature_dimensions": int(len(vector)),
            "feature_norm": float(np.linalg.norm(vector)),
            "peak_rss_mb": peak_rss_mb(),
        }
        if device_name == "mps":
            timing["mps_allocated_mb"] = torch.mps.current_allocated_memory() / 1024**2
            timing["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024**2
        report["shots"].append(timing)
        if index == 0:
            write_json(output_dir / "first_shot_timing.json", timing)
        write_json(output_dir / "progress.json", report)
        print(json.dumps({"event": "shot_encoded", **timing}), flush=True)

    metric_started = time.perf_counter()
    feature_array = np.stack(features)
    metrics, similarities = consistency_metrics(feature_array, [scene_num for scene_num, _ in shots])
    np.save(output_dir / "shot_features.npy", feature_array)
    np.save(output_dir / "shot_similarities.npy", similarities)
    report["metrics"] = metrics
    report["metric_and_save_seconds"] = time.perf_counter() - metric_started
    report["total_seconds_including_import_and_model_load"] = time.perf_counter() - started
    report["total_shot_processing_seconds"] = sum(shot["total_seconds"] for shot in report["shots"])
    report["median_inference_seconds_after_first_shot"] = statistics.median(shot["inference_seconds"] for shot in report["shots"][1:])
    report["peak_rss_mb"] = peak_rss_mb()
    write_json(output_dir / "scores.json", report)
    print(json.dumps({"event": "complete", "metrics": metrics, "total_seconds": report["total_seconds_including_import_and_model_load"], "report": str(output_dir / "scores.json")}), flush=True)


if __name__ == "__main__":
    main()
