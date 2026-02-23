from __future__ import annotations

from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.embeddings.service import retrieve_evidence
from novel_summarizer.storyteller.state import StorytellerState


async def run(state: StorytellerState, *, config: AppConfigRoot, book_id: int) -> dict:
    chapter_idx = int(state.get("chapter_idx") or 0)
    top_k = int(config.storyteller.memory_top_k)
    if top_k <= 0:
        return {"awakened_memories": []}

    chapter_text = str(state.get("chapter_text") or "")
    entities = state.get("entities_mentioned") or []
    query_text = "\n".join(
        [
            f"chapter_idx={chapter_idx}",
            f"entities={', '.join(entities)}",
            chapter_text[:2000],
        ]
    )

    try:
        candidates = retrieve_evidence(
            book_id=book_id,
            config=config,
            query_text=query_text,
            top_k=max(top_k * 3, top_k),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory retrieve failed, fallback to empty memories: {}", exc)
        return {"awakened_memories": []}

    memories: list[dict] = []
    for item in candidates:
        source_chapter_idx = item.get("chapter_idx")
        if source_chapter_idx is None:
            continue
        try:
            source_chapter_idx = int(source_chapter_idx)
        except (TypeError, ValueError):
            continue
        if source_chapter_idx >= chapter_idx:
            continue

        memories.append(
            {
                "chunk_id": item.get("chunk_id"),
                "chapter_idx": source_chapter_idx,
                "chapter_title": item.get("chapter_title"),
                "source_type": "chunk",
                "text": str(item.get("text", ""))[:600],
            }
        )
        if len(memories) >= top_k:
            break

    return {"awakened_memories": memories}