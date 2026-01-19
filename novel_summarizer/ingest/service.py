from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import book_hash as compute_book_hash
from novel_summarizer.domain.hashing import chapter_hash as compute_chapter_hash
from novel_summarizer.domain.hashing import chunk_hash as compute_chunk_hash
from novel_summarizer.ingest.parser import load_text, normalize_text, parse_chapters
from novel_summarizer.ingest.splitter import split_text
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo


@dataclass
class IngestStats:
    book_id: int
    book_hash: str
    chapters_total: int
    chapters_inserted: int
    chunks_total: int
    chunks_inserted: int


async def ingest_book(
    input_path: Path,
    config: AppConfigRoot,
    title: str | None = None,
    author: str | None = None,
    chapter_regex_override: str | None = None,
) -> IngestStats:
    logger.info("Reading novel text from %s", input_path)
    raw_text = load_text(input_path, config.ingest.encoding)
    normalized = normalize_text(raw_text, config.ingest.cleanup)

    book_hash_value = compute_book_hash(normalized)
    chapter_regex = chapter_regex_override or config.ingest.chapter_regex
    chapters = parse_chapters(normalized, chapter_regex)
    logger.info("Parsed %d chapters", len(chapters))

    split_params = (
        f"size={config.split.chunk_size_tokens};"
        f"overlap={config.split.chunk_overlap_tokens};"
        f"min={config.split.min_chunk_tokens}"
    )

    chapters_inserted = 0
    chunks_inserted = 0
    chunks_total = 0

    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        book_result = await repo.get_or_create_book(title, author, book_hash_value, str(input_path))

        for chapter in chapters:
            chapter_hash_value = compute_chapter_hash(book_hash_value, chapter.title, chapter.text)
            chapter_result = await repo.upsert_chapter(
                book_id=book_result.id,
                idx=chapter.idx,
                title=chapter.title,
                chapter_hash=chapter_hash_value,
                start_pos=chapter.start_pos,
                end_pos=chapter.end_pos,
            )
            if chapter_result.inserted:
                chapters_inserted += 1

            chunks = split_text(
                chapter.text,
                config.split.chunk_size_tokens,
                config.split.chunk_overlap_tokens,
                config.split.min_chunk_tokens,
            )
            chunks_total += len(chunks)

            for chunk in chunks:
                chunk_hash_value = compute_chunk_hash(chapter_hash_value, chunk.text, split_params)
                chunk_result = await repo.upsert_chunk(
                    chapter_id=chapter_result.id,
                    idx=chunk.idx,
                    chunk_hash=chunk_hash_value,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    start_pos=chunk.start_pos,
                    end_pos=chunk.end_pos,
                )
                if chunk_result.inserted:
                    chunks_inserted += 1

    return IngestStats(
        book_id=book_result.id,
        book_hash=book_hash_value,
        chapters_total=len(chapters),
        chapters_inserted=chapters_inserted,
        chunks_total=chunks_total,
        chunks_inserted=chunks_inserted,
    )
