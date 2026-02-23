from __future__ import annotations

import orjson

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storage.repo import SQLAlchemyRepo


async def run(state: StorytellerState, *, repo: SQLAlchemyRepo, config: AppConfigRoot, book_id: int) -> dict:
    _ = config
    chapter_idx = int(state.get("chapter_idx", 0))

    event_writes = 0
    for key_event in state.get("key_events", []):
        event_summary = str(key_event.get("what") or "")
        if not event_summary:
            continue
        involved = key_event.get("who")
        involved_json = None
        if involved:
            involved_json = orjson.dumps([str(involved)]).decode("utf-8")
        await repo.insert_plot_event(
            book_id=book_id,
            chapter_idx=chapter_idx,
            event_summary=event_summary,
            involved_characters_json=involved_json,
            event_type="narration_draft",
            impact=str(key_event.get("impact") or ""),
        )
        event_writes += 1

    return {"mutations_applied": {"plot_events_inserted": event_writes}}