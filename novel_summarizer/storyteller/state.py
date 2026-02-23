from __future__ import annotations

from typing import NotRequired, TypedDict


class StorytellerState(TypedDict):
    # Inputs
    book_id: int
    chapter_id: int
    chapter_idx: int
    chapter_title: str
    chapter_text: str

    # Node outputs
    entities_mentioned: NotRequired[list[str]]
    locations_mentioned: NotRequired[list[str]]
    items_mentioned: NotRequired[list[str]]

    character_states: NotRequired[list[dict]]
    item_states: NotRequired[list[dict]]
    recent_events: NotRequired[list[dict]]

    awakened_memories: NotRequired[list[dict]]

    narration: NotRequired[str]
    key_events: NotRequired[list[dict]]
    character_updates: NotRequired[list[dict]]
    new_items: NotRequired[list[dict]]

    mutations_applied: NotRequired[dict]
    memory_committed: NotRequired[bool]