from __future__ import annotations

from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.embeddings.service import retrieve_hybrid_memories
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storyteller.tiering import effective_storyteller_value


async def run(state: StorytellerState, *, config: AppConfigRoot, book_id: int) -> dict:
    if state.get("awakened_memories") is not None:
        return {}

    chapter_idx = int(state.get("chapter_idx") or 0)
    top_k = int(effective_storyteller_value(state, config, "memory_top_k", config.storyteller.memory_top_k))
    if top_k <= 0:
        return {"awakened_memories": []}

    chapter_text = str(state.get("chapter_text") or "")
    entities = [str(item) for item in (state.get("entities_mentioned") or [])]
    locations = [str(item) for item in (state.get("locations_mentioned") or [])]
    items = [str(item) for item in (state.get("items_mentioned") or [])]
    query_text = "\n".join(
        [
            f"chapter_idx={chapter_idx}",
            f"entities={', '.join(entities)}",
            f"locations={', '.join(locations)}",
            f"items={', '.join(items)}",
            chapter_text[:2000],
        ]
    )

    try:
        candidates = await retrieve_hybrid_memories(
            book_id=book_id,
            config=config,
            query_text=query_text,
            top_k=max(top_k, 1),
            current_chapter_idx=chapter_idx,
            keyword_terms=entities + locations + items,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory retrieve failed, fallback to empty memories: {}", exc)
        return {"awakened_memories": []}

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

    return {"awakened_memories": memories}