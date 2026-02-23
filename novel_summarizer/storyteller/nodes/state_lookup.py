from __future__ import annotations

from dataclasses import asdict

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storage.repo import SQLAlchemyRepo


async def run(state: StorytellerState, *, repo: SQLAlchemyRepo, config: AppConfigRoot, book_id: int) -> dict:
    entity_names = state.get("entities_mentioned") or None
    item_names = state.get("items_mentioned") or None

    characters = await repo.list_character_states(book_id=book_id, canonical_names=entity_names)
    items = await repo.list_item_states(book_id=book_id, names=item_names)
    recent_events = await repo.list_recent_plot_events(
        book_id=book_id,
        chapter_idx=state.get("chapter_idx"),
        window=config.storyteller.recent_events_window,
    )

    return {
        "character_states": [asdict(row) for row in characters],
        "item_states": [asdict(row) for row in items],
        "recent_events": [asdict(row) for row in recent_events],
    }