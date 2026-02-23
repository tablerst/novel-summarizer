from __future__ import annotations

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState


async def run(state: StorytellerState, *, config: AppConfigRoot, book_id: int) -> dict:
    _ = (state, config, book_id)
    # MVP placeholder: vector persistence for narrations is planned for M2.
    return {"memory_committed": True}