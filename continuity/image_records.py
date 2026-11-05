"""Stable JSONL image records for local protocol resolution."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ImageRecord(BaseModel):
    image_id: str
    kind: Literal["portrait", "camera", "shot_sample", "generated_first_frame"]
    asset_path: str
    character_id: str | None
    shot_idx: int | None
    sample_time_seconds: float | None
    view: Literal["front", "side", "back"] | None
    summary: str


class ImageRecordStore:
    """Append and read stable local image records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: ImageRecord) -> None:
        if not record.image_id:
            raise ValueError("ImageRecord.image_id must be non-empty")
        existing = {item.image_id: item for item in self._read_records()}
        previous = existing.get(record.image_id)
        if previous is not None:
            if previous.model_dump() != record.model_dump():
                raise ValueError(f"Conflicting ImageRecord for stable image_id {record.image_id}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")

    def get(self, image_id: str) -> ImageRecord:
        records = {record.image_id: record for record in self._read_records()}
        try:
            return records[image_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"Unknown stable image_id {image_id}") from exc

    def all_records(self) -> list[ImageRecord]:
        """Return full local records for deterministic protocol resolution."""
        return [record.model_copy(deep=True) for record in self._read_records()]

    def _read_records(self) -> list[ImageRecord]:
        if not self.path.exists():
            return []
        records: list[ImageRecord] = []
        seen: dict[str, ImageRecord] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                    record = ImageRecord.model_validate(payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"Malformed image_records JSONL at line {line_number}") from exc
                if not record.image_id:
                    raise ValueError(f"Malformed image_records JSONL at line {line_number}: empty image_id")
                previous = seen.get(record.image_id)
                if previous is not None:
                    if previous.model_dump() != record.model_dump():
                        raise ValueError(f"Conflicting ImageRecord for stable image_id {record.image_id}")
                    raise ValueError(f"Duplicate ImageRecord for stable image_id {record.image_id}")
                seen[record.image_id] = record
                records.append(record)
        return records
