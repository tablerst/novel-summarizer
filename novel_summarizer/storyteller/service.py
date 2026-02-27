from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

import orjson
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.embeddings.service import prepare_retrieval_assets
from novel_summarizer.llm.cache import SimpleCache
from novel_summarizer.llm.factory import OpenAIChatClient
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo
from novel_summarizer.storyteller.graph import build_storyteller_graph
from novel_summarizer.storyteller.nodes import entity_extract, memory_retrieve
from novel_summarizer.storyteller.prompts.narration import NARRATION_PROMPT_VERSION
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storyteller.tiering import build_tier_overrides, decide_tier, has_storyteller_memory_retrieval

STORYTELLER_MVP_MODEL = "storyteller-mvp"


@dataclass
class StorytellStats:
    book_id: int
    chapters_total: int
    chapters_processed: int
    chapters_skipped: int
    llm_calls_estimated: int
    refine_llm_calls_estimated: int
    llm_cache_hits: int
    llm_cache_misses: int
    input_tokens_estimated: int
    output_tokens_estimated: int
    refine_input_tokens_estimated: int
    refine_output_tokens_estimated: int
    consistency_warnings: int
    consistency_actions: int
    evidence_supported_claims: int
    evidence_unsupported_claims: int
    runtime_seconds: float


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
    started_at = time.perf_counter()
    run_log = logger.bind(node="storyteller_service", chapter_id="-", chapter_idx="-")
    chapters_processed = 0
    chapters_skipped = 0
    llm_calls_estimated = 0
    refine_llm_calls_estimated = 0
    llm_cache_hits = 0
    input_tokens_estimated = 0
    output_tokens_estimated = 0
    refine_input_tokens_estimated = 0
    refine_output_tokens_estimated = 0
    consistency_warnings = 0
    consistency_actions = 0
    evidence_supported_claims = 0
    evidence_unsupported_claims = 0
    selected_chapters = []
    cache = SimpleCache(config.cache.enabled, config.cache.backend, config.app.data_dir, config.cache.ttl_seconds)
    entity_llm_client: OpenAIChatClient | None = None
    narration_llm_client: OpenAIChatClient | None = None
    refine_llm_client: OpenAIChatClient | None = None
    model_identifier = STORYTELLER_MVP_MODEL

    try:
        narration_llm_client = OpenAIChatClient(config=config, cache=cache, route="storyteller_narration")
        model_identifier = narration_llm_client.model_identifier
    except Exception as exc:  # noqa: BLE001
        run_log.warning("Storyteller narration LLM disabled; fallback mode enabled: {}", exc)

    try:
        entity_llm_client = OpenAIChatClient(config=config, cache=cache, route="storyteller_entity")
    except Exception as exc:  # noqa: BLE001
        run_log.warning("Storyteller entity LLM disabled; extraction fallback mode enabled: {}", exc)

    try:
        refine_llm_client = OpenAIChatClient(config=config, cache=cache, route="storyteller_refine")
    except Exception as exc:  # noqa: BLE001
        run_log.warning("Storyteller refine LLM disabled; refine fallback mode enabled: {}", exc)

    try:
        async with session_scope() as session:
            repo = SQLAlchemyRepo(session)

            if has_storyteller_memory_retrieval(config):
                try:
                    await prepare_retrieval_assets(book_id=book_id, config=config)
                except Exception as exc:  # noqa: BLE001
                    run_log.warning("Retrieval assets prebuild failed for storyteller retrieval: {}", exc)

            chapters = await repo.list_chapters(book_id)
            selected_chapters = _filter_chapters(chapters, from_chapter=from_chapter, to_chapter=to_chapter)
            graph = build_storyteller_graph(
                repo=repo,
                config=config,
                book_id=book_id,
                entity_llm_client=entity_llm_client,
                narration_llm_client=narration_llm_client,
                refine_llm_client=refine_llm_client,
            )
            prefetch_window = max(0, int(config.storyteller.prefetch_window))
            prefetch_tasks: dict[int, asyncio.Task[dict]] = {}

            async def _prefetch_state(chapter_row) -> dict:
                pre_text = await _chapter_text(repo, chapter_row.id)
                if not pre_text:
                    return {}

                pre_tier = decide_tier(
                    chapter_idx=chapter_row.idx,
                    chapter_title=chapter_row.title,
                    chapter_text=pre_text,
                    config=config,
                )
                pre_overrides = build_tier_overrides(tier=pre_tier, config=config)
                pre_state: StorytellerState = {
                    "book_id": book_id,
                    "chapter_id": chapter_row.id,
                    "chapter_idx": chapter_row.idx,
                    "chapter_title": chapter_row.title,
                    "chapter_text": pre_text,
                    "tier": pre_tier,
                    "storyteller_overrides": pre_overrides,
                }

                pre_state.update(
                    await entity_extract.run(
                        pre_state,
                        config=config,
                        llm_client=entity_llm_client,
                    )
                )
                pre_state.update(
                    await memory_retrieve.run(
                        pre_state,
                        config=config,
                        book_id=book_id,
                    )
                )
                return dict(pre_state)

            def _schedule_prefetch(current_idx: int) -> None:
                if prefetch_window <= 0:
                    return
                for offset in range(1, prefetch_window + 1):
                    target_idx = current_idx + offset
                    if target_idx >= len(selected_chapters):
                        break
                    target = selected_chapters[target_idx]
                    if target.id in prefetch_tasks:
                        continue
                    prefetch_tasks[target.id] = asyncio.create_task(_prefetch_state(target))

            run_log.info("Storyteller chapter loop started book_id={} chapters_selected={}", book_id, len(selected_chapters))

            for chapter_position, chapter in enumerate(selected_chapters):
                chapter_log = run_log.bind(chapter_id=chapter.id, chapter_idx=chapter.idx)

                _schedule_prefetch(chapter_position)

                prefetched_state: dict | None = None
                prefetch_task = prefetch_tasks.pop(chapter.id, None)
                if prefetch_task is not None:
                    try:
                        prefetched_state = await prefetch_task
                    except Exception as exc:  # noqa: BLE001
                        chapter_log.warning("Prefetch task failed, fallback to synchronous path: {}", exc)

                chapter_text = str((prefetched_state or {}).get("chapter_text") or "")
                if not chapter_text:
                    chapter_text = await _chapter_text(repo, chapter.id)
                if not chapter_text:
                    chapter_log.warning("Chapter text empty; skipped")
                    chapters_skipped += 1
                    continue

                tier = str((prefetched_state or {}).get("tier") or "")
                if not tier:
                    tier = decide_tier(
                        chapter_idx=chapter.idx,
                        chapter_title=chapter.title,
                        chapter_text=chapter_text,
                        config=config,
                    )
                tier_overrides = (prefetched_state or {}).get("storyteller_overrides")
                if not isinstance(tier_overrides, dict):
                    tier_overrides = build_tier_overrides(tier=tier, config=config)
                tier_overrides_json = orjson.dumps(tier_overrides, option=orjson.OPT_SORT_KEYS).decode("utf-8")

                input_hash = sha256_text(
                    (
                        f"{chapter.id}:{chapter.idx}:{chapter_text}:{config.storyteller.style}:"
                        f"{tier}:{tier_overrides_json}:"
                        f"{config.llm.routes.storyteller_narration_chat}:{config.llm.routes.storyteller_refine_chat}"
                    )
                )
                existing = await repo.get_narration(
                    chapter_id=chapter.id,
                    prompt_version=NARRATION_PROMPT_VERSION,
                    model=model_identifier,
                    input_hash=input_hash,
                )
                if existing is not None:
                    chapter_log.info("Narration already exists for current input hash; skipped")
                    chapters_skipped += 1
                    continue

                state: StorytellerState = {
                    "book_id": book_id,
                    "chapter_id": chapter.id,
                    "chapter_idx": chapter.idx,
                    "chapter_title": chapter.title,
                    "chapter_text": chapter_text,
                    "tier": tier,
                    "storyteller_overrides": tier_overrides,
                }
                if prefetched_state:
                    for key in (
                        "entities_mentioned",
                        "locations_mentioned",
                        "items_mentioned",
                        "entity_llm_calls",
                        "entity_llm_cache_hit",
                        "awakened_memories",
                    ):
                        if key in prefetched_state:
                            state[key] = prefetched_state[key]
                chapter_log.debug("Invoking storyteller graph tier={}", tier)
                try:
                    final_state = await graph.ainvoke(state)
                except Exception:
                    chapter_log.exception("Storyteller graph invocation failed")
                    raise

                narration = str(final_state.get("narration") or "").strip()
                if not narration:
                    chapter_log.warning("No narration generated")
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
                    model=model_identifier,
                    input_hash=input_hash,
                )

                entity_calls = int(final_state.get("entity_llm_calls") or 0)
                narration_calls = int(final_state.get("narration_llm_calls") or 0)
                refine_calls = int(final_state.get("refine_llm_calls") or 0)
                llm_calls_estimated += entity_calls + narration_calls + refine_calls
                refine_llm_calls_estimated += refine_calls
                if bool(final_state.get("entity_llm_cache_hit")):
                    llm_cache_hits += entity_calls
                if bool(final_state.get("narration_llm_cache_hit")):
                    llm_cache_hits += narration_calls
                if bool(final_state.get("refine_llm_cache_hit")):
                    llm_cache_hits += refine_calls
                input_tokens_estimated += int(final_state.get("input_tokens_estimated") or 0)
                output_tokens_estimated += int(final_state.get("output_tokens_estimated") or 0)
                refine_input_tokens_estimated += int(final_state.get("refine_input_tokens_estimated") or 0)
                refine_output_tokens_estimated += int(final_state.get("refine_output_tokens_estimated") or 0)
                consistency_warnings += len(final_state.get("consistency_warnings") or [])
                consistency_actions += len(final_state.get("consistency_actions") or [])
                evidence_report = final_state.get("evidence_report") or {}
                evidence_supported_claims += int(evidence_report.get("supported_claims") or 0)
                evidence_unsupported_claims += int(evidence_report.get("unsupported_claims") or 0)
                chapters_processed += 1
                chapter_log.info(
                    "Chapter narration persisted key_events={} llm_calls={} cache_hits={} warnings={} actions={}",
                    len(key_events),
                    entity_calls + narration_calls + refine_calls,
                    int(bool(final_state.get("entity_llm_cache_hit")))
                    + int(bool(final_state.get("narration_llm_cache_hit")))
                    + int(bool(final_state.get("refine_llm_cache_hit"))),
                    len(final_state.get("consistency_warnings") or []),
                    len(final_state.get("consistency_actions") or []),
                )

            if prefetch_tasks:
                for task in prefetch_tasks.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*prefetch_tasks.values(), return_exceptions=True)
    finally:
        cache.close()

    runtime_seconds = time.perf_counter() - started_at
    llm_cache_misses = max(0, llm_calls_estimated - llm_cache_hits)

    return StorytellStats(
        book_id=book_id,
        chapters_total=len(selected_chapters),
        chapters_processed=chapters_processed,
        chapters_skipped=chapters_skipped,
        llm_calls_estimated=llm_calls_estimated,
        refine_llm_calls_estimated=refine_llm_calls_estimated,
        llm_cache_hits=llm_cache_hits,
        llm_cache_misses=llm_cache_misses,
        input_tokens_estimated=input_tokens_estimated,
        output_tokens_estimated=output_tokens_estimated,
        refine_input_tokens_estimated=refine_input_tokens_estimated,
        refine_output_tokens_estimated=refine_output_tokens_estimated,
        consistency_warnings=consistency_warnings,
        consistency_actions=consistency_actions,
        evidence_supported_claims=evidence_supported_claims,
        evidence_unsupported_claims=evidence_unsupported_claims,
        runtime_seconds=runtime_seconds,
    )