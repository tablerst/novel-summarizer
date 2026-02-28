from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import time

import orjson
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.embeddings.service import prepare_retrieval_assets, retrieve_hybrid_memories_batched
from novel_summarizer.llm.cache import SimpleCache
from novel_summarizer.llm.factory import OpenAIChatClient
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo
from novel_summarizer.storyteller.graph import build_storyteller_graph
from novel_summarizer.storyteller.nodes import consistency_check, entity_extract, evidence_verify, memory_retrieve, state_lookup, state_update
from novel_summarizer.storyteller.nodes.storyteller_generate_step import run_batch as run_step_batch
from novel_summarizer.storyteller.prompts.narration import NARRATION_PROMPT_VERSION
from novel_summarizer.storyteller.prompts.step_narration import STEP_NARRATION_PROMPT_VERSION
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storyteller.step_utils import align_from_chapter, align_to_chapter, iter_step_ranges
from novel_summarizer.storyteller.tiering import (
    build_tier_overrides,
    decide_tier,
    effective_storyteller_value,
    has_storyteller_memory_retrieval,
)

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


def _chapter_by_idx(chapters: list) -> dict[int, object]:
    mapped: dict[int, object] = {}
    for chapter in chapters:
        mapped[int(chapter.idx)] = chapter
    return mapped


def _world_state_hash(base_lookup: dict) -> str:
    """Computes a stable hash for world_state fields used as step baseline."""

    payload = {
        "character_states": base_lookup.get("character_states") or [],
        "item_states": base_lookup.get("item_states") or [],
        "recent_events": base_lookup.get("recent_events") or [],
        "world_facts": base_lookup.get("world_facts") or [],
    }
    serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return sha256_text(serialized)


async def _snapshot_world_state(repo: SQLAlchemyRepo, *, book_id: int) -> tuple[str, str]:
    characters = [asdict(row) for row in await repo.list_character_states(book_id=book_id)]
    items = [asdict(row) for row in await repo.list_item_states(book_id=book_id)]
    plot_events = [asdict(row) for row in await repo.list_plot_events_by_book(book_id=book_id)]
    world_facts = [asdict(row) for row in await repo.list_world_facts(book_id=book_id, limit=10_000)]

    # Sort for stable hashes.
    characters.sort(key=lambda item: str(item.get("canonical_name") or ""))
    items.sort(key=lambda item: str(item.get("name") or ""))
    plot_events.sort(key=lambda item: (int(item.get("chapter_idx") or 0), int(item.get("id") or 0)))
    world_facts.sort(key=lambda item: str(item.get("fact_key") or ""))

    snapshot_obj = {
        "characters": characters,
        "items": items,
        "plot_events": plot_events,
        "world_facts": world_facts,
    }
    snapshot_json = orjson.dumps(snapshot_obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return snapshot_json, sha256_text(snapshot_json)


async def _ensure_world_state_at(
    *,
    repo: SQLAlchemyRepo,
    config: AppConfigRoot,
    book_id: int,
    target_chapter_idx: int,
    step_size: int,
    chapters_by_idx: dict[int, object],
    graph,
    entity_llm_client: OpenAIChatClient | None,
    narration_llm_client: OpenAIChatClient | None,
    refine_llm_client: OpenAIChatClient | None,
    model_identifier: str,
) -> None:
    """Ensures DB world_state represents the boundary state at target_chapter_idx.

    This is used by step execution to restore/rebuild the baseline state at S-1.
    """

    if target_chapter_idx <= 0:
        # Chapter 0 baseline: clear to empty state.
        await repo.clear_world_state_for_book(book_id=book_id)
        return

    resume_mode = str(config.storyteller.step_resume_mode)
    if resume_mode != "restore":
        return

    checkpoint = await repo.get_latest_world_state_checkpoint_at_or_before(
        book_id=book_id,
        chapter_idx=target_chapter_idx,
    )

    start_idx = 1
    if checkpoint is not None:
        await repo.restore_world_state_checkpoint(checkpoint=checkpoint)
        start_idx = int(checkpoint.chapter_idx) + 1
        if int(checkpoint.chapter_idx) >= target_chapter_idx:
            return
    else:
        await repo.clear_world_state_for_book(book_id=book_id)

    # Fallback rebuild: replay chapters to reconstruct world_state to the desired boundary.
    for idx in range(start_idx, target_chapter_idx + 1):
        chapter = chapters_by_idx.get(idx)
        if chapter is None:
            raise ValueError(f"Missing chapter idx={idx} while rebuilding world_state")
        chapter_text = await _chapter_text(repo, chapter.id)
        if not chapter_text:
            continue

        tier = decide_tier(
            chapter_idx=chapter.idx,
            chapter_title=chapter.title,
            chapter_text=chapter_text,
            config=config,
        )
        tier_overrides = build_tier_overrides(tier=tier, config=config)

        state: StorytellerState = {
            "book_id": book_id,
            "chapter_id": chapter.id,
            "chapter_idx": chapter.idx,
            "chapter_title": chapter.title,
            "chapter_text": chapter_text,
            "tier": tier,
            "storyteller_overrides": tier_overrides,
        }

        # Prefer replay from persisted structured outputs to avoid expensive LLM calls.
        output_row = await repo.get_latest_narration_output_for_chapter(chapter_id=chapter.id)
        if output_row is not None:
            try:
                payload = orjson.loads(output_row.payload_json)
            except Exception:  # noqa: BLE001
                payload = None
            if isinstance(payload, dict):
                state.update(
                    {
                        "entities_mentioned": payload.get("entities_mentioned") or [],
                        "key_events": payload.get("key_events") or [],
                        "character_updates": payload.get("character_updates") or [],
                        "new_items": payload.get("new_items") or [],
                    }
                )
                state.update(await state_lookup.run(state, repo=repo, config=config, book_id=book_id))
                await state_update.run(state, repo=repo, config=config, book_id=book_id)
                continue

        # Fallback: invoke full graph (includes state_update) when outputs are missing.
        await graph.ainvoke(state)

    if bool(config.storyteller.step_checkpoint_enabled):
        snapshot_json, snapshot_hash = await _snapshot_world_state(repo, book_id=book_id)
        await repo.upsert_world_state_checkpoint(
            book_id=book_id,
            chapter_idx=target_chapter_idx,
            step_size=step_size,
            snapshot_json=snapshot_json,
            snapshot_hash=snapshot_hash,
        )


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
            chapters_by_idx = _chapter_by_idx(chapters)

            step_size = max(1, int(config.storyteller.step_size))
            effective_from = from_chapter
            effective_to = to_chapter
            if step_size > 1 and str(config.storyteller.step_align) == "auto":
                if effective_from is not None:
                    effective_from = align_from_chapter(from_chapter=int(effective_from), step_size=step_size)
                if effective_to is not None:
                    max_idx = max((int(ch.idx) for ch in chapters), default=0)
                    if max_idx > 0:
                        effective_to = align_to_chapter(
                            to_chapter=int(effective_to),
                            step_size=step_size,
                            max_chapter_idx=max_idx,
                        )

            selected_chapters = _filter_chapters(chapters, from_chapter=effective_from, to_chapter=effective_to)

            full_graph = build_storyteller_graph(
                repo=repo,
                config=config,
                book_id=book_id,
                entity_llm_client=entity_llm_client,
                narration_llm_client=narration_llm_client,
                refine_llm_client=refine_llm_client,
            )

            if step_size > 1:
                if not selected_chapters:
                    run_log.warning("No chapters selected for step execution")
                else:
                    run_log.info(
                        "Storyteller step mode enabled step_size={} chapters_selected={} (from={}, to={})",
                        step_size,
                        len(selected_chapters),
                        effective_from,
                        effective_to,
                    )

                # Ensure baseline state for the first step.
                first_idx = int(selected_chapters[0].idx) if selected_chapters else 0
                if first_idx > 0:
                    first_step_start = align_from_chapter(from_chapter=first_idx, step_size=step_size)
                    await _ensure_world_state_at(
                        repo=repo,
                        config=config,
                        book_id=book_id,
                        target_chapter_idx=first_step_start - 1,
                        step_size=step_size,
                        chapters_by_idx=chapters_by_idx,
                        graph=full_graph,
                        entity_llm_client=entity_llm_client,
                        narration_llm_client=narration_llm_client,
                        refine_llm_client=refine_llm_client,
                        model_identifier=model_identifier,
                    )

                # Step loop (no prefetch in v1 implementation).
                max_selected_idx = max((int(ch.idx) for ch in selected_chapters), default=0)
                step_ranges = iter_step_ranges(
                    start_chapter=int(selected_chapters[0].idx) if selected_chapters else 1,
                    end_chapter=max_selected_idx,
                    step_size=step_size,
                )

                for step_start, step_end in step_ranges:
                    step_log = run_log.bind(chapter_idx=f"{step_start}-{step_end}")
                    # Baseline world_state at S-1 (DB should currently be at S-1).
                    base_lookup = await state_lookup.run(
                        {
                            "book_id": book_id,
                            "chapter_idx": step_start,
                            "entities_mentioned": [],
                            "items_mentioned": [],
                        },
                        repo=repo,
                        config=config,
                        book_id=book_id,
                    )
                    base_hash = _world_state_hash(base_lookup)

                    # Build per-chapter states for this step.
                    step_chapter_states: list[StorytellerState] = []
                    for chapter in selected_chapters:
                        if int(chapter.idx) < step_start or int(chapter.idx) > step_end:
                            continue

                        chapter_log = step_log.bind(chapter_id=chapter.id, chapter_idx=chapter.idx)
                        chapter_text = await _chapter_text(repo, chapter.id)
                        if not chapter_text:
                            chapter_log.warning("Chapter text empty; skipped")
                            chapters_skipped += 1
                            continue

                        tier = decide_tier(
                            chapter_idx=chapter.idx,
                            chapter_title=chapter.title,
                            chapter_text=chapter_text,
                            config=config,
                        )
                        tier_overrides = build_tier_overrides(tier=tier, config=config)

                        # In step mode we want narration to be final; disable refine node side-effects.
                        storyteller_overrides = dict(tier_overrides)
                        storyteller_overrides["refine_enabled"] = False

                        state: StorytellerState = {
                            "book_id": book_id,
                            "chapter_id": chapter.id,
                            "chapter_idx": chapter.idx,
                            "chapter_title": chapter.title,
                            "chapter_text": chapter_text,
                            "tier": tier,
                            "storyteller_overrides": storyteller_overrides,
                            # Prefill baseline world_state for the whole step.
                            **base_lookup,
                        }

                        # Per-chapter entity extraction happens first; memory retrieval can be batched per step.
                        state.update(await entity_extract.run(state, config=config, llm_client=entity_llm_client))
                        step_chapter_states.append(state)

                    if not step_chapter_states:
                        step_log.warning("No chapters prepared for step; skipped")
                        continue

                    step_chapter_states.sort(key=lambda item: int(item.get("chapter_idx") or 0))
                    anchor_state = step_chapter_states[-1]
                    anchor_chapter_id = int(anchor_state.get("chapter_id") or 0)
                    anchor_chapter_idx = int(anchor_state.get("chapter_idx") or 0)

                    step_input_obj = {
                        "step_size": step_size,
                        "step_start": step_start,
                        "step_end": step_end,
                        "base_hash": base_hash,
                        "style": config.storyteller.style,
                        "narration_route": config.llm.routes.storyteller_narration_chat,
                        "refine_route": config.llm.routes.storyteller_refine_chat,
                        "chapters": [
                            {
                                "chapter_id": int(st.get("chapter_id") or 0),
                                "chapter_idx": int(st.get("chapter_idx") or 0),
                                "chapter_title": str(st.get("chapter_title") or ""),
                                "chapter_text": str(st.get("chapter_text") or ""),
                                "tier": str(st.get("tier") or ""),
                                "storyteller_overrides": st.get("storyteller_overrides") or {},
                                "entities_mentioned": st.get("entities_mentioned") or [],
                            }
                            for st in step_chapter_states
                        ],
                    }
                    step_input_hash = sha256_text(orjson.dumps(step_input_obj, option=orjson.OPT_SORT_KEYS).decode("utf-8"))

                    existing_step_payload: dict[str, object] | None = None
                    existing_step = await repo.get_narration(
                        chapter_id=anchor_chapter_id,
                        prompt_version=STEP_NARRATION_PROMPT_VERSION,
                        model=model_identifier,
                        input_hash=step_input_hash,
                    )
                    if existing_step is not None:
                        out_row = await repo.get_narration_output(narration_id=existing_step.id)
                        if out_row is not None:
                            try:
                                parsed_payload = orjson.loads(out_row.payload_json)
                            except Exception:  # noqa: BLE001
                                parsed_payload = None
                            if isinstance(parsed_payload, dict):
                                existing_step_payload = parsed_payload

                    if existing_step_payload is not None:
                        step_state_from_cache: StorytellerState = {
                            "book_id": book_id,
                            "chapter_id": anchor_chapter_id,
                            "chapter_idx": anchor_chapter_idx,
                            "entities_mentioned": existing_step_payload.get("entities_mentioned") or [],
                            "key_events": existing_step_payload.get("key_events") or [],
                            "character_updates": existing_step_payload.get("character_updates") or [],
                            "new_items": existing_step_payload.get("new_items") or [],
                        }
                        step_state_from_cache.update(
                            await state_lookup.run(step_state_from_cache, repo=repo, config=config, book_id=book_id)
                        )
                        await state_update.run(step_state_from_cache, repo=repo, config=config, book_id=book_id)
                        chapters_skipped += len(step_chapter_states)

                        if bool(config.storyteller.step_checkpoint_enabled) and step_end > 0:
                            snapshot_json, snapshot_hash = await _snapshot_world_state(repo, book_id=book_id)
                            await repo.upsert_world_state_checkpoint(
                                book_id=book_id,
                                chapter_idx=step_end,
                                step_size=step_size,
                                snapshot_json=snapshot_json,
                                snapshot_hash=snapshot_hash,
                            )
                        continue

                    # Batched memory retrieval (Phase 4).
                    step_memory_mode = str(config.storyteller.step_memory_mode)
                    if step_memory_mode == "off":
                        for st in step_chapter_states:
                            st["awakened_memories"] = []
                    else:
                        query_texts: list[str] = []
                        chapter_idxs: list[int] = []
                        keyword_terms_list: list[list[str]] = []
                        top_ks: list[int] = []

                        for st in step_chapter_states:
                            chapter_idx = int(st.get("chapter_idx") or 0)
                            top_k = int(effective_storyteller_value(st, config, "memory_top_k", config.storyteller.memory_top_k))
                            if top_k <= 0:
                                st["awakened_memories"] = []
                                continue

                            chapter_text = str(st.get("chapter_text") or "")
                            entities = [str(item) for item in (st.get("entities_mentioned") or [])]
                            locations = [str(item) for item in (st.get("locations_mentioned") or [])]
                            items = [str(item) for item in (st.get("items_mentioned") or [])]

                            query_text = "\n".join(
                                [
                                    f"chapter_idx={chapter_idx}",
                                    f"entities={', '.join(entities)}",
                                    f"locations={', '.join(locations)}",
                                    f"items={', '.join(items)}",
                                    chapter_text[:2000],
                                ]
                            )

                            query_texts.append(query_text)
                            chapter_idxs.append(chapter_idx)
                            keyword_terms_list.append(entities + locations + items)
                            top_ks.append(top_k)

                        if query_texts:
                            max_top_k = max(top_ks) if top_ks else 0
                            # per_step_shared: reuse a single shared query text, but keep per-chapter filtering by current_chapter_idx.
                            if step_memory_mode == "per_step_shared":
                                shared = query_texts[0]
                                query_texts = [shared for _ in query_texts]

                            retrieved = await retrieve_hybrid_memories_batched(
                                book_id=book_id,
                                config=config,
                                query_texts=query_texts,
                                top_k=max(max_top_k, 1),
                                current_chapter_idxs=chapter_idxs,
                                keyword_terms_list=keyword_terms_list,
                            )

                            idx_to_memories: dict[int, list[dict]] = {}
                            for idx, candidates in zip(chapter_idxs, retrieved, strict=True):
                                idx_to_memories[idx] = candidates

                            for st in step_chapter_states:
                                chapter_idx = int(st.get("chapter_idx") or 0)
                                top_k = int(
                                    effective_storyteller_value(st, config, "memory_top_k", config.storyteller.memory_top_k)
                                )
                                if top_k <= 0:
                                    st["awakened_memories"] = []
                                    continue

                                candidates = idx_to_memories.get(chapter_idx, [])
                                memories: list[dict] = []
                                for item in candidates:
                                    source_chapter_idx = int(item.get("chapter_idx") or 0)
                                    source_type = str(item.get("source_type") or "chunk")
                                    source_id = int(item.get("source_id") or 0)

                                    memories.append(
                                        {
                                            "source_id": source_id,
                                            "chapter_idx": source_chapter_idx,
                                            "chapter_title": item.get("chapter_title"),
                                            "source_type": source_type,
                                            "score": float(item.get("score", 0.0)),
                                            "text": str(item.get("text", ""))[:600],
                                        }
                                    )
                                    if len(memories) >= top_k:
                                        break

                                st["awakened_memories"] = memories

                    # Batch narration generation (Phase 3).
                    base_world_state_obj = {
                        "character_states": base_lookup.get("character_states") or [],
                        "item_states": base_lookup.get("item_states") or [],
                        "recent_events": base_lookup.get("recent_events") or [],
                        "world_facts": base_lookup.get("world_facts") or [],
                    }
                    step_output = await run_step_batch(
                        step_chapter_states,
                        config=config,
                        llm_client=narration_llm_client,
                        base_world_state=base_world_state_obj,
                    )

                    merged_entities: list[str] = []
                    seen_entities: set[str] = set()
                    merged_memories: list[dict] = []
                    seen_memories: set[tuple[int, int, str]] = set()
                    for st in step_chapter_states:
                        for raw in st.get("entities_mentioned") or []:
                            entity = str(raw).strip()
                            if not entity or entity in seen_entities:
                                continue
                            seen_entities.add(entity)
                            merged_entities.append(entity)

                        for memory in st.get("awakened_memories") or []:
                            source_id = int(memory.get("source_id") or 0)
                            chapter_idx = int(memory.get("chapter_idx") or 0)
                            source_type = str(memory.get("source_type") or "")
                            key = (source_id, chapter_idx, source_type)
                            if key in seen_memories:
                                continue
                            seen_memories.add(key)
                            merged_memories.append(memory)

                    step_state: StorytellerState = {
                        "book_id": book_id,
                        "chapter_id": anchor_chapter_id,
                        "chapter_idx": anchor_chapter_idx,
                        "chapter_title": str(anchor_state.get("chapter_title") or ""),
                        "chapter_text": "\n\n".join(str(st.get("chapter_text") or "") for st in step_chapter_states),
                        "entities_mentioned": merged_entities,
                        "character_states": base_lookup.get("character_states") or [],
                        "item_states": base_lookup.get("item_states") or [],
                        "recent_events": base_lookup.get("recent_events") or [],
                        "world_facts": base_lookup.get("world_facts") or [],
                        "awakened_memories": merged_memories,
                    }
                    step_state.update(step_output)

                    # Post-process once per step aggregate.
                    step_state.update(await consistency_check.run(step_state, config=config))
                    step_state.update(await evidence_verify.run(step_state, config=config))

                    narration = str(step_state.get("narration") or "").strip()
                    if not narration:
                        step_log.warning("No narration generated for step")
                        chapters_skipped += len(step_chapter_states)
                        continue

                    key_events = step_state.get("key_events") or []
                    await repo.upsert_narration(
                        book_id=book_id,
                        chapter_id=anchor_chapter_id,
                        chapter_idx=anchor_chapter_idx,
                        narration_text=narration,
                        key_events_json=orjson.dumps(key_events).decode("utf-8"),
                        prompt_version=STEP_NARRATION_PROMPT_VERSION,
                        model=model_identifier,
                        input_hash=step_input_hash,
                    )

                    narration_row = await repo.get_narration(
                        chapter_id=anchor_chapter_id,
                        prompt_version=STEP_NARRATION_PROMPT_VERSION,
                        model=model_identifier,
                        input_hash=step_input_hash,
                    )
                    if narration_row is not None:
                        payload_json = orjson.dumps(
                            {
                                "step_start_chapter_idx": step_state.get("step_start_chapter_idx"),
                                "step_end_chapter_idx": step_state.get("step_end_chapter_idx"),
                                "entities_mentioned": step_state.get("entities_mentioned") or [],
                                "key_events": step_state.get("key_events") or [],
                                "character_updates": step_state.get("character_updates") or [],
                                "new_items": step_state.get("new_items") or [],
                            },
                            option=orjson.OPT_SORT_KEYS,
                        ).decode("utf-8")
                        await repo.upsert_narration_output(
                            narration_id=narration_row.id,
                            book_id=book_id,
                            chapter_id=anchor_chapter_id,
                            chapter_idx=anchor_chapter_idx,
                            prompt_version=STEP_NARRATION_PROMPT_VERSION,
                            model=model_identifier,
                            input_hash=step_input_hash,
                            payload_json=payload_json,
                        )

                    # Stats aggregation.
                    entity_calls_total = sum(int(st.get("entity_llm_calls") or 0) for st in step_chapter_states)
                    narration_calls = int(step_state.get("narration_llm_calls") or 0)
                    refine_calls = int(step_state.get("refine_llm_calls") or 0)
                    llm_calls_estimated += entity_calls_total + narration_calls + refine_calls
                    refine_llm_calls_estimated += refine_calls
                    if entity_calls_total > 0:
                        llm_cache_hits += sum(
                            int(st.get("entity_llm_calls") or 0)
                            for st in step_chapter_states
                            if bool(st.get("entity_llm_cache_hit"))
                        )
                    if bool(step_state.get("narration_llm_cache_hit")):
                        llm_cache_hits += narration_calls
                    if bool(step_state.get("refine_llm_cache_hit")):
                        llm_cache_hits += refine_calls
                    input_tokens_estimated += int(step_state.get("input_tokens_estimated") or 0)
                    output_tokens_estimated += int(step_state.get("output_tokens_estimated") or 0)
                    refine_input_tokens_estimated += int(step_state.get("refine_input_tokens_estimated") or 0)
                    refine_output_tokens_estimated += int(step_state.get("refine_output_tokens_estimated") or 0)
                    consistency_warnings += len(step_state.get("consistency_warnings") or [])
                    consistency_actions += len(step_state.get("consistency_actions") or [])
                    evidence_report = step_state.get("evidence_report") or {}
                    evidence_supported_claims += int(evidence_report.get("supported_claims") or 0)
                    evidence_unsupported_claims += int(evidence_report.get("unsupported_claims") or 0)
                    chapters_processed += len(step_chapter_states)

                    # Apply world_state mutation once for this step aggregate.
                    step_state.update(await state_lookup.run(step_state, repo=repo, config=config, book_id=book_id))
                    await state_update.run(step_state, repo=repo, config=config, book_id=book_id)

                    if bool(config.storyteller.step_checkpoint_enabled) and step_end > 0:
                        snapshot_json, snapshot_hash = await _snapshot_world_state(repo, book_id=book_id)
                        await repo.upsert_world_state_checkpoint(
                            book_id=book_id,
                            chapter_idx=step_end,
                            step_size=step_size,
                            snapshot_json=snapshot_json,
                            snapshot_hash=snapshot_hash,
                        )
                        step_log.info("Step checkpoint persisted chapter_idx={} step_size={}", step_end, step_size)

                # Step mode done.
                prefetch_tasks = {}
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

            graph = full_graph
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