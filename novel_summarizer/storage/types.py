from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class InsertResult:
    id: int
    inserted: bool


@dataclass
class ChapterRow:
    id: int
    idx: int
    title: str


@dataclass
class ChunkRow:
    id: int
    idx: int
    text: str
    chunk_hash: str


@dataclass
class NarrationRow:
    id: int
    book_id: int
    chapter_id: int
    chapter_idx: int
    narration_text: str
    key_events_json: str | None
    prompt_version: str
    model: str
    input_hash: str


@dataclass
class NarrationOutputRow:
    id: int
    narration_id: int
    book_id: int
    chapter_id: int
    chapter_idx: int
    prompt_version: str
    model: str
    input_hash: str
    payload_json: str


@dataclass
class SummaryRow:
    id: int
    scope: str
    ref_id: int
    summary_type: str
    prompt_version: str
    model: str
    input_hash: str
    content: str


@dataclass
class BookRow:
    id: int
    title: str | None
    author: str | None
    book_hash: str
    source_path: str | None


@dataclass
class CharacterRow:
    id: int
    book_id: int
    canonical_name: str
    aliases_json: str
    first_chapter_idx: int | None
    last_chapter_idx: int | None
    status: str
    location: str | None
    abilities_json: str | None
    relationships_json: str | None
    motivation: str | None
    notes: str | None


@dataclass
class ItemRow:
    id: int
    book_id: int
    name: str
    owner_name: str | None
    first_chapter_idx: int | None
    last_chapter_idx: int | None
    description: str | None
    status: str


@dataclass
class PlotEventRow:
    id: int
    book_id: int
    chapter_idx: int
    event_summary: str
    involved_characters_json: str | None
    event_type: str | None
    impact: str | None


@dataclass
class SearchHitRow:
    source_type: str
    source_id: int
    chapter_idx: int
    chapter_title: str
    text: str
    score: float | None = None


@dataclass
class WorldFactRow:
    id: int
    book_id: int
    fact_key: str
    fact_value: str
    confidence: float
    source_chapter_idx: int | None
    source_excerpt: str | None


@dataclass
class WorldStateCheckpointRow:
    id: int
    book_id: int
    chapter_idx: int
    step_size: int
    snapshot_json: str
    snapshot_hash: str
    created_at: datetime
    updated_at: datetime
