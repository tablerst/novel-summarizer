from __future__ import annotations

from dataclasses import dataclass


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
