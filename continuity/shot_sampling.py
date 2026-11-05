"""One-pass video sampling and board construction for shot observation."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from continuity.image_records import ImageRecord, ImageRecordStore


class SampledFrame(BaseModel):
    shot_idx: int
    image_id: str
    asset_path: str
    time_seconds: float


class ObservationSampleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    image_id: str


_OBSERVATION_BOARD_ROWS = 3
_OBSERVATION_BOARD_COLUMNS = 4
_OBSERVATION_BOARD_SLOT_COUNT = _OBSERVATION_BOARD_ROWS * _OBSERVATION_BOARD_COLUMNS
_OBSERVATION_BOARD_CELL_WIDTH = 512
_OBSERVATION_BOARD_CELL_HEIGHT = 288


def sample_video(
    video_path: str | Path,
    shot_idx: int,
    shots_dir: str | Path,
    reference_count: int,
) -> list[SampledFrame]:
    """Write board temporal slots from one metadata-planned sequential decode.

    The board has twelve row-major slots.  ``reference_count`` reserves its
    leading slots, so the remaining temporal samples include the first and last
    metadata-addressable video frames.  Metadata or decode disagreement fails
    closed; this routine never switches to a second decoding strategy.
    """
    sample_count = _sample_count_for_references(reference_count)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode video for sampling: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has invalid FPS metadata: {video_path}")
        if (
            not math.isfinite(frame_count_value)
            or frame_count_value < 1
            or not frame_count_value.is_integer()
        ):
            raise ValueError(f"Video has invalid frame-count metadata: {video_path}")
        frame_count = int(frame_count_value)
        if frame_count < sample_count:
            raise ValueError(
                f"Video metadata reports only {frame_count} frames but observation board requires {sample_count} unique samples"
            )
        target_indices = _evenly_spaced_frame_indices(frame_count, sample_count)
        samples_dir = Path(shots_dir) / str(shot_idx) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        samples: list[SampledFrame] = []
        target_position = 0
        frame_index = 0
        while target_position < len(target_indices):
            decoded, frame = capture.read()
            if not decoded or frame is None:
                raise ValueError(
                    "Video decode ended before all metadata-planned observation-board samples were written"
                )
            if frame_index == target_indices[target_position]:
                samples.append(_write_sample(shot_idx, frame_index, frame, fps, samples_dir))
                target_position += 1
            frame_index += 1
        return samples
    finally:
        capture.release()


def build_observation_board(
    reference_slots: list[tuple[str, str | Path]],
    samples: list[SampledFrame],
    output_path: str | Path,
) -> Path:
    """Write the fixed 3×4 board in deterministic row-major slot order.

    The first slots are formal entity references; every remaining slot is one
    actual sampled frame.  This is intentionally a dedicated Observer board,
    not a reusable collage abstraction.
    """
    expected_samples = _sample_count_for_references(len(reference_slots))
    if len(samples) != expected_samples:
        raise ValueError(
            f"Observation board requires {expected_samples} samples for {len(reference_slots)} references, got {len(samples)}"
        )
    if len({sample.image_id for sample in samples}) != len(samples):
        raise ValueError("Observation board samples must have unique image IDs")
    labels_and_paths = [
        *((label, str(path)) for label, path in reference_slots),
        *((f"T{index} {_seconds_label(sample.time_seconds)}s", sample.asset_path) for index, sample in enumerate(samples)),
    ]
    cells = [_observation_board_cell(path, label) for label, path in labels_and_paths]
    rows = [np.concatenate(cells[offset:offset + _OBSERVATION_BOARD_COLUMNS], axis=1)
            for offset in range(0, _OBSERVATION_BOARD_SLOT_COUNT, _OBSERVATION_BOARD_COLUMNS)]
    board = np.concatenate(rows, axis=0)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), board):
        raise ValueError(f"Failed to write observation board {path}")
    return path


def _sample_count_for_references(reference_count: int) -> int:
    if not isinstance(reference_count, int) or not 0 <= reference_count <= _OBSERVATION_BOARD_SLOT_COUNT - 2:
        raise ValueError(
            "Observation board reference_count must leave at least two slots for the first and last actual video frames"
        )
    return _OBSERVATION_BOARD_SLOT_COUNT - reference_count


def _evenly_spaced_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < sample_count:
        raise ValueError("Cannot select unique evenly spaced frames from fewer decoded frames")
    return [index * (frame_count - 1) // (sample_count - 1) for index in range(sample_count)]


def _seconds_label(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Observation board sample time must be finite and non-negative")
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def _observation_board_cell(asset_path: str, label: str) -> np.ndarray:
    image = cv2.imread(asset_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Observation board cannot read image {asset_path}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Observation board image has invalid dimensions {asset_path}")
    label_height = 28
    image_height = _OBSERVATION_BOARD_CELL_HEIGHT - label_height
    scale = min(_OBSERVATION_BOARD_CELL_WIDTH / width, image_height / height)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    cell = np.zeros((_OBSERVATION_BOARD_CELL_HEIGHT, _OBSERVATION_BOARD_CELL_WIDTH, 3), dtype=np.uint8)
    y_offset = (image_height - resized.shape[0]) // 2
    x_offset = (_OBSERVATION_BOARD_CELL_WIDTH - resized.shape[1]) // 2
    cell[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    cv2.putText(
        cell,
        label,
        (8, _OBSERVATION_BOARD_CELL_HEIGHT - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cell


def _write_sample(
    shot_idx: int,
    frame_index: int,
    frame,
    fps: float,
    samples_dir: Path,
) -> SampledFrame:
    image_id = f"shot-{shot_idx}-sample-frame-{frame_index}"
    output_path = samples_dir / f"frame-{frame_index}.png"
    if not cv2.imwrite(str(output_path), frame):
        raise ValueError(f"Failed to write sampled PNG {output_path}")
    return SampledFrame(
        shot_idx=shot_idx,
        image_id=image_id,
        asset_path=str(output_path),
        time_seconds=frame_index / fps,
    )


def observation_sample_metadata(samples: list[SampledFrame]) -> list[ObservationSampleMetadata]:
    return [
        ObservationSampleMetadata(index=index, time_seconds=sample.time_seconds, image_id=sample.image_id)
        for index, sample in enumerate(samples)
    ]


def write_observation_samples(path: str | Path, samples: list[SampledFrame]) -> None:
    """Write the ordered observation-input sample manifest without state data."""
    Path(path).write_text(
        json.dumps([item.model_dump() for item in observation_sample_metadata(samples)], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_observation_samples(path: str | Path) -> list[ObservationSampleMetadata]:
    """Read one ordered sample manifest and reject ambiguous sample identities."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = [ObservationSampleMetadata.model_validate(item) for item in payload]
    except (OSError, json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise ValueError(f"Cannot read valid observation samples {path}") from exc
    if not metadata or [item.index for item in metadata] != list(range(len(metadata))):
        raise ValueError("Observation sample indices must be non-empty contiguous order")
    if len({item.image_id for item in metadata}) != len(metadata):
        raise ValueError("Observation samples must have unique image IDs")
    if any(current.time_seconds <= previous.time_seconds for previous, current in zip(metadata, metadata[1:])):
        raise ValueError("Observation sample times must be strictly increasing")
    return metadata


def image_records_for_observed_samples(
    samples: list[SampledFrame],
    shot_idx: int,
) -> list[ImageRecord]:
    """Build one factual ImageRecord per observed shot sample; never append them."""
    if any(sample.shot_idx != shot_idx for sample in samples):
        raise ValueError(f"Sampled frames do not all belong to shot {shot_idx}")
    return [
        ImageRecord(
            image_id=sample.image_id,
            kind="shot_sample",
            asset_path=sample.asset_path,
            character_id=None,
            shot_idx=shot_idx,
            sample_time_seconds=sample.time_seconds,
            view=None,
            summary=f"actual sampled frame from shot {shot_idx} at {sample.time_seconds:g}s",
        )
        for sample in samples
    ]


def last_observation_sample_from_manifest(
    observation_samples_path: str | Path,
    image_store: ImageRecordStore,
    shot_idx: int,
) -> SampledFrame:
    """Recover the last actual sample from one observed shot's sample manifest."""
    metadata_items = read_observation_samples(observation_samples_path)
    last = metadata_items[-1]
    try:
        record = image_store.get(last.image_id)
    except KeyError as exc:
        raise ValueError(f"Observation tail image_id {last.image_id} is not a formal image record") from exc
    if (
        record.kind != "shot_sample"
        or record.shot_idx != shot_idx
        or record.sample_time_seconds != last.time_seconds
        or not Path(record.asset_path).is_file()
    ):
        raise ValueError("Observation tail record does not match its shot, time, or local image path")
    return SampledFrame(
        shot_idx=record.shot_idx,
        image_id=record.image_id,
        asset_path=record.asset_path,
        time_seconds=record.sample_time_seconds,
    )
