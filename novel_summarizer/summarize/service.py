from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import orjson
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.embeddings.service import embed_book_chunks, retrieve_evidence
from novel_summarizer.llm.cache import SimpleCache
from novel_summarizer.llm.factory import OpenAIChatClient, make_cache_key
from novel_summarizer.llm.prompts import (
    BOOK_PROMPT_VERSION,
    CHAPTER_PROMPT_VERSION,
    CHUNK_PROMPT_VERSION,
    STORY_PROMPT_VERSION,
    book_summary_prompts,
    chapter_summary_prompts,
    chunk_summary_prompts,
    story_summary_prompts,
)
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo


@dataclass
class SummarizeStats:
    book_id: int
    chunks_total: int
    chunks_new: int
    chapters_total: int
    chapters_new: int
    book_summary_new: int
    characters_new: int
    timeline_new: int
    story_new: int


def _safe_load_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty JSON text")
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        cleaned = _sanitize_json_text(text)
        try:
            return orjson.loads(cleaned)
        except orjson.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return orjson.loads(cleaned[start : end + 1])
            raise


def _sanitize_json_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _estimate_tokens(text: str) -> int:
    return len(text)


def _chunk_items_by_size(items: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for item in items:
        item_size = _estimate_tokens(orjson.dumps(item).decode("utf-8"))
        if current and current_size + item_size > max_chars:
            groups.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += item_size
    if current:
        groups.append(current)
    return groups


def _reduce_chapter_summaries_for_book(
    *,
    client: OpenAIChatClient,
    config: AppConfigRoot,
    chapter_summaries: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    current = chapter_summaries
    while True:
        payload = orjson.dumps(current).decode("utf-8")
        if _estimate_tokens(payload) <= max_chars or len(current) <= 1:
            return current

        groups = _chunk_items_by_size(current, max_chars)
        reduced: list[dict[str, Any]] = []
        for group in groups:
            group_hash = sha256_text(orjson.dumps(group).decode("utf-8"))
            partial = _book_summary(
                client=client,
                config=config,
                chapter_summaries=group,
                input_hash=group_hash,
                evidence=None,
            )
            reduced.append(
                {
                    "summary": partial.get("summary", ""),
                    "events": partial.get("timeline", []),
                    "characters": partial.get("characters", []),
                    "open_questions": [],
                    "chapter_idx": group[0].get("chapter_idx"),
                    "chapter_title": f"章节{group[0].get('chapter_idx')}~{group[-1].get('chapter_idx')}汇总",
                }
            )
        current = reduced


def _dump_json(data: dict[str, Any]) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")


def _format_evidence(evidence_items: list[dict[str, Any]]) -> str:
    if not evidence_items:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(evidence_items, start=1):
        chunk_id = item.get("chunk_id")
        chapter_idx = item.get("chapter_idx")
        chapter_title = item.get("chapter_title")
        text = str(item.get("text", "")).replace("\n", " ").strip()
        if len(text) > 300:
            text = text[:300] + "..."
        lines.append(
            f"[{idx}] chunk_id={chunk_id} 章节{chapter_idx} {chapter_title}: {text}"
        )
    return "\n".join(lines)


def _chunk_summary(
    *,
    client: OpenAIChatClient,
    config: AppConfigRoot,
    chunk_text: str,
    input_hash: str,
) -> dict[str, Any]:
    system, user = chunk_summary_prompts(
        language=config.summarize.language,
        style=config.summarize.style,
        include_quotes=config.summarize.include_quotes,
        with_citations=config.summarize.with_citations.enabled,
    )
    user = user.format(chunk=chunk_text)
    cache_key = make_cache_key(
        "chunk",
        config.llm.chat_model,
        CHUNK_PROMPT_VERSION,
        input_hash,
        str(config.llm.temperature),
    )
    response, summary_obj = client.complete_json(system, user, cache_key, _safe_load_json)
    logger.debug("Chunk summary cached=%s", response.cached)
    return summary_obj


def _chapter_summary(
    *,
    client: OpenAIChatClient,
    config: AppConfigRoot,
    chunk_summaries: list[dict[str, Any]],
    input_hash: str,
    evidence: str | None = None,
) -> dict[str, Any]:
    system, user = chapter_summary_prompts(
        language=config.summarize.language,
        style=config.summarize.style,
        word_range=config.summarize.chapter_summary_words,
        with_citations=config.summarize.with_citations.enabled,
        evidence=evidence,
    )
    chunk_json = orjson.dumps(chunk_summaries).decode("utf-8")
    user = user.format(chunk_summaries=chunk_json)
    cache_key = make_cache_key(
        "chapter",
        config.llm.chat_model,
        CHAPTER_PROMPT_VERSION,
        input_hash,
        str(config.llm.temperature),
    )
    response, summary_obj = client.complete_json(system, user, cache_key, _safe_load_json)
    logger.debug("Chapter summary cached=%s", response.cached)
    return summary_obj


def _book_summary(
    *,
    client: OpenAIChatClient,
    config: AppConfigRoot,
    chapter_summaries: list[dict[str, Any]],
    input_hash: str,
    evidence: str | None = None,
) -> dict[str, Any]:
    system, user = book_summary_prompts(
        language=config.summarize.language,
        style=config.summarize.style,
        word_range=config.summarize.book_summary_words,
        with_citations=config.summarize.with_citations.enabled,
        evidence=evidence,
    )
    chapter_json = orjson.dumps(chapter_summaries).decode("utf-8")
    user = user.format(chapter_summaries=chapter_json)
    cache_key = make_cache_key(
        "book",
        config.llm.chat_model,
        BOOK_PROMPT_VERSION,
        input_hash,
        str(config.llm.temperature),
    )
    response, summary_obj = client.complete_json(system, user, cache_key, _safe_load_json)
    logger.debug("Book summary cached=%s", response.cached)
    return summary_obj


def _story_summary(
    *,
    client: OpenAIChatClient,
    config: AppConfigRoot,
    chapter_summaries: list[dict[str, Any]],
    input_hash: str,
    evidence: str | None = None,
) -> dict[str, Any]:
    if not config.summarize.story_words:
        raise ValueError("story_words is required to generate story summary")
    system, user = story_summary_prompts(
        language=config.summarize.language,
        style=config.summarize.style,
        word_range=config.summarize.story_words,
        with_citations=config.summarize.with_citations.enabled,
        evidence=evidence,
    )
    chapter_json = orjson.dumps(chapter_summaries).decode("utf-8")
    user = user.format(chapter_summaries=chapter_json)
    cache_key = make_cache_key(
        "story",
        config.llm.chat_model,
        STORY_PROMPT_VERSION,
        input_hash,
        str(config.llm.temperature),
    )
    response, summary_obj = client.complete_json(system, user, cache_key, _safe_load_json)
    logger.debug("Story summary cached=%s", response.cached)
    return summary_obj


async def summarize_book(book_id: int, config: AppConfigRoot) -> SummarizeStats:
    cache = SimpleCache(config.cache.enabled, config.cache.backend, config.app.data_dir, config.cache.ttl_seconds)
    client = OpenAIChatClient(config, cache)

    if config.summarize.with_citations.enabled:
        await embed_book_chunks(book_id=book_id, config=config)

    chunks_total = 0
    chunks_new = 0
    chapters_new = 0
    book_summary_new = 0
    characters_new = 0
    timeline_new = 0
    story_new = 0

    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        chapters = await repo.list_chapters(book_id)
        chapter_summary_objects_all: list[dict[str, Any]] = []
        for chapter in chapters:
            chunks = await repo.list_chunks(chapter.id)
            chunks_total += len(chunks)
            chunk_summary_objects: list[dict[str, Any]] = []

            for chunk in chunks:
                existing = await repo.get_summary(
                    scope="chunk",
                    ref_id=chunk.id,
                    summary_type="chunk_summary",
                    prompt_version=CHUNK_PROMPT_VERSION,
                    model=config.llm.chat_model,
                    input_hash=chunk.chunk_hash,
                )
                if existing:
                    chunk_summary_objects.append(_safe_load_json(existing.content))
                    continue

                summary_obj = _chunk_summary(
                    client=client,
                    config=config,
                    chunk_text=chunk.text,
                    input_hash=chunk.chunk_hash,
                )
                await repo.upsert_summary(
                    scope="chunk",
                    ref_id=chunk.id,
                    summary_type="chunk_summary",
                    prompt_version=CHUNK_PROMPT_VERSION,
                    model=config.llm.chat_model,
                    input_hash=chunk.chunk_hash,
                    content=_dump_json(summary_obj),
                )
                chunks_new += 1
                chunk_summary_objects.append(summary_obj)

            chapter_input_hash = sha256_text(orjson.dumps(chunk_summary_objects).decode("utf-8"))
            existing_chapter = await repo.get_summary(
                scope="chapter",
                ref_id=chapter.id,
                summary_type="chapter_summary",
                prompt_version=CHAPTER_PROMPT_VERSION,
                model=config.llm.chat_model,
                input_hash=chapter_input_hash,
            )
            if existing_chapter:
                chapter_summary_obj = _safe_load_json(existing_chapter.content)
            else:
                evidence_text = None
                if config.summarize.with_citations.enabled:
                    query_text = orjson.dumps(chunk_summary_objects).decode("utf-8")
                    evidence_items = retrieve_evidence(
                        book_id=book_id,
                        config=config,
                        query_text=query_text,
                        top_k=config.summarize.with_citations.top_k,
                        chapter_id=chapter.id,
                    )
                    evidence_text = _format_evidence(evidence_items)

                chapter_summary_obj = _chapter_summary(
                    client=client,
                    config=config,
                    chunk_summaries=chunk_summary_objects,
                    input_hash=chapter_input_hash,
                    evidence=evidence_text,
                )
                await repo.upsert_summary(
                    scope="chapter",
                    ref_id=chapter.id,
                    summary_type="chapter_summary",
                    prompt_version=CHAPTER_PROMPT_VERSION,
                    model=config.llm.chat_model,
                    input_hash=chapter_input_hash,
                    content=_dump_json(chapter_summary_obj),
                )
                chapters_new += 1

            chapter_summary_for_book = dict(chapter_summary_obj)
            chapter_summary_for_book["chapter_idx"] = chapter.idx
            chapter_summary_for_book["chapter_title"] = chapter.title
            chapter_summary_objects_all.append(chapter_summary_for_book)

        if chapter_summary_objects_all:
            book_source_objects = chapter_summary_objects_all
            max_book_chars = 60_000
            if _estimate_tokens(orjson.dumps(book_source_objects).decode("utf-8")) > max_book_chars:
                logger.info("Book input too large; reducing chapter summaries before final summary")
                book_source_objects = _reduce_chapter_summaries_for_book(
                    client=client,
                    config=config,
                    chapter_summaries=book_source_objects,
                    max_chars=max_book_chars,
                )

            book_input_hash = sha256_text(orjson.dumps(book_source_objects).decode("utf-8"))
            has_book_summary = await repo.get_summary(
                scope="book",
                ref_id=book_id,
                summary_type="book_summary",
                prompt_version=BOOK_PROMPT_VERSION,
                model=config.llm.chat_model,
                input_hash=book_input_hash,
            )
            has_characters = await repo.get_summary(
                scope="book",
                ref_id=book_id,
                summary_type="characters",
                prompt_version=BOOK_PROMPT_VERSION,
                model=config.llm.chat_model,
                input_hash=book_input_hash,
            )
            has_timeline = await repo.get_summary(
                scope="book",
                ref_id=book_id,
                summary_type="timeline",
                prompt_version=BOOK_PROMPT_VERSION,
                model=config.llm.chat_model,
                input_hash=book_input_hash,
            )
            has_story = None
            should_generate_story = config.summarize.story_words is not None
            if should_generate_story:
                has_story = await repo.get_summary(
                    scope="book",
                    ref_id=book_id,
                    summary_type="story",
                    prompt_version=STORY_PROMPT_VERSION,
                    model=config.llm.chat_model,
                    input_hash=book_input_hash,
                )

            needs_book = not (has_book_summary and has_characters and has_timeline)
            needs_story = should_generate_story and not has_story

            if needs_book or needs_story:
                evidence_text = None
                if config.summarize.with_citations.enabled:
                    query_text = orjson.dumps(book_source_objects).decode("utf-8")
                    evidence_items = retrieve_evidence(
                        book_id=book_id,
                        config=config,
                        query_text=query_text,
                        top_k=config.summarize.with_citations.top_k,
                    )
                    evidence_text = _format_evidence(evidence_items)

                if needs_book:
                    book_obj = _book_summary(
                        client=client,
                        config=config,
                        chapter_summaries=book_source_objects,
                        input_hash=book_input_hash,
                        evidence=evidence_text,
                    )

                    book_payload = dict(book_obj)
                    characters_payload = {"characters": book_obj.get("characters", [])}
                    timeline_payload = {"events": book_obj.get("timeline", [])}

                    book_result = await repo.upsert_summary(
                        scope="book",
                        ref_id=book_id,
                        summary_type="book_summary",
                        prompt_version=BOOK_PROMPT_VERSION,
                        model=config.llm.chat_model,
                        input_hash=book_input_hash,
                        content=_dump_json(book_payload),
                    )
                    characters_result = await repo.upsert_summary(
                        scope="book",
                        ref_id=book_id,
                        summary_type="characters",
                        prompt_version=BOOK_PROMPT_VERSION,
                        model=config.llm.chat_model,
                        input_hash=book_input_hash,
                        content=_dump_json(characters_payload),
                    )
                    timeline_result = await repo.upsert_summary(
                        scope="book",
                        ref_id=book_id,
                        summary_type="timeline",
                        prompt_version=BOOK_PROMPT_VERSION,
                        model=config.llm.chat_model,
                        input_hash=book_input_hash,
                        content=_dump_json(timeline_payload),
                    )
                    book_summary_new += int(book_result.inserted)
                    characters_new += int(characters_result.inserted)
                    timeline_new += int(timeline_result.inserted)

                if needs_story:
                    story_obj = _story_summary(
                        client=client,
                        config=config,
                        chapter_summaries=book_source_objects,
                        input_hash=book_input_hash,
                        evidence=evidence_text,
                    )
                    story_result = await repo.upsert_summary(
                        scope="book",
                        ref_id=book_id,
                        summary_type="story",
                        prompt_version=STORY_PROMPT_VERSION,
                        model=config.llm.chat_model,
                        input_hash=book_input_hash,
                        content=_dump_json(dict(story_obj)),
                    )
                    story_new += int(story_result.inserted)

    cache.close()
    return SummarizeStats(
        book_id=book_id,
        chunks_total=chunks_total,
        chunks_new=chunks_new,
        chapters_total=len(chapters),
        chapters_new=chapters_new,
        book_summary_new=book_summary_new,
        characters_new=characters_new,
        timeline_new=timeline_new,
        story_new=story_new,
    )
