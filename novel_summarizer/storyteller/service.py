from __future__ import annotations

from dataclasses import dataclass

import orjson
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo
from novel_summarizer.storyteller.graph import build_storyteller_graph
from novel_summarizer.storyteller.prompts.narration import NARRATION_PROMPT_VERSION
from novel_summarizer.storyteller.state import StorytellerState

STORYTELLER_MVP_MODEL = "storyteller-mvp"


@dataclass
class StorytellStats:
    book_id: int
    chapters_total: int
    chapters_processed: int
    chapters_skipped: int


def _filter_chapters(chapters, from_chapter: int | None, to_chapter: int | None):
    selected = []
    for chapter in chapters:
        if from_chapter is not None and chapter.idx < from_chapter:
            continue
        if to_chapter is not None and chapter.idx > to_chapter:
            continue
        selected.append(chapter)
    return selected


async def _chapter_text(repo: SQLAlchemyRepo, chapter_id: int) -> str:
    chunks = await repo.list_chunks(chapter_id)
    if not chunks:
        return ""
    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.idx)
    return "\n".join(chunk.text for chunk in ordered_chunks).strip()


async def storytell_book(
    *,
    book_id: int,
    config: AppConfigRoot,
    from_chapter: int | None = None,
    to_chapter: int | None = None,
) -> StorytellStats:
    chapters_processed = 0
    chapters_skipped = 0

    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        chapters = await repo.list_chapters(book_id)
        selected_chapters = _filter_chapters(chapters, from_chapter=from_chapter, to_chapter=to_chapter)
        graph = build_storyteller_graph(repo=repo, config=config, book_id=book_id)

        for chapter in selected_chapters:
            chapter_text = await _chapter_text(repo, chapter.id)
            if not chapter_text:
                chapters_skipped += 1
                continue

            input_hash = sha256_text(f"{chapter.id}:{chapter.idx}:{chapter_text}")
            existing = await repo.get_narration(
                chapter_id=chapter.id,
                prompt_version=NARRATION_PROMPT_VERSION,
                model=STORYTELLER_MVP_MODEL,
                input_hash=input_hash,
            )
            if existing is not None:
                chapters_skipped += 1
                continue

            state: StorytellerState = {
                "book_id": book_id,
                "chapter_id": chapter.id,
                "chapter_idx": chapter.idx,
                "chapter_title": chapter.title,
                "chapter_text": chapter_text,
            }
            final_state = await graph.ainvoke(state)

            narration = str(final_state.get("narration") or "").strip()
            if not narration:
                logger.warning("No narration generated for chapter_id=%s", chapter.id)
                chapters_skipped += 1
                continue

            key_events = final_state.get("key_events") or []
            await repo.upsert_narration(
                book_id=book_id,
                chapter_id=chapter.id,
                chapter_idx=chapter.idx,
                narration_text=narration,
                key_events_json=orjson.dumps(key_events).decode("utf-8"),
                prompt_version=NARRATION_PROMPT_VERSION,
                model=STORYTELLER_MVP_MODEL,
                input_hash=input_hash,
            )
            chapters_processed += 1

    return StorytellStats(
        book_id=book_id,
        chapters_total=len(selected_chapters),
        chapters_processed=chapters_processed,
        chapters_skipped=chapters_skipped,
    )