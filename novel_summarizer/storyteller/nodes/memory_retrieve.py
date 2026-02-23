from __future__ import annotations

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState


async def run(state: StorytellerState, *, config: AppConfigRoot, book_id: int) -> dict:
    _ = (state, config, book_id)
    # MVP placeholder: retrieval integration with LanceDB will be implemented in M1/M2.
    return {"awakened_memories": []}