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
    world_facts: NotRequired[list[dict]]

    awakened_memories: NotRequired[list[dict]]

    narration: NotRequired[str]
    key_events: NotRequired[list[dict]]
    character_updates: NotRequired[list[dict]]
    new_items: NotRequired[list[dict]]
    consistency_warnings: NotRequired[list[str]]
    consistency_actions: NotRequired[list[str]]
    evidence_report: NotRequired[dict]

    # Runtime report metrics
    entity_llm_calls: NotRequired[int]
    entity_llm_cache_hit: NotRequired[bool]
    narration_llm_calls: NotRequired[int]
    narration_llm_cache_hit: NotRequired[bool]
    refine_llm_calls: NotRequired[int]
    refine_llm_cache_hit: NotRequired[bool]
    input_tokens_estimated: NotRequired[int]
    output_tokens_estimated: NotRequired[int]
    refine_input_tokens_estimated: NotRequired[int]
    refine_output_tokens_estimated: NotRequired[int]

    mutations_applied: NotRequired[dict]
    memory_committed: NotRequired[bool]