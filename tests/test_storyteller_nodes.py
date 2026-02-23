from __future__ import annotations

import asyncio
from typing import cast

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.nodes import entity_extract, memory_retrieve, storyteller_generate
from novel_summarizer.storyteller.state import StorytellerState


class _FakeLLMClient:
    model_identifier = "fake/provider/model"

    def complete_json(self, system_prompt, user_prompt, cache_key, parser):
        _ = (system_prompt, user_prompt, cache_key)
        text = '{"characters":["韩立","韩立"],"locations":["天南"],"items":["掌天瓶"],"key_phrases":["筑基"]}'
        return None, parser(text)


def test_safe_load_json_dict_parses_code_fence() -> None:
    payload = """```json
    {"narration":"ok"}
    ```"""
    parsed = safe_load_json_dict(payload)
    assert parsed == {"narration": "ok"}


def test_entity_extract_uses_llm_and_normalizes() -> None:
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
        "chapter_id": 1,
        "chapter_idx": 1,
        "chapter_title": "第1章",
        "chapter_text": "韩立在天南得到掌天瓶。",
        },
    )
    result = asyncio.run(entity_extract.run(state, config=config, llm_client=_FakeLLMClient()))

    assert result["entities_mentioned"] == ["韩立"]
    assert result["locations_mentioned"] == ["天南"]
    assert result["items_mentioned"] == ["掌天瓶"]


def test_memory_retrieve_filters_future_and_current(monkeypatch) -> None:
    config = AppConfigRoot()

    def _fake_retrieve_evidence(*, book_id, config, query_text, top_k):
        _ = (book_id, config, query_text, top_k)
        return [
            {"chapter_idx": 1, "chapter_title": "第1章", "chunk_id": 101, "text": "前情A"},
            {"chapter_idx": 3, "chapter_title": "第3章", "chunk_id": 301, "text": "当前章内容"},
            {"chapter_idx": 8, "chapter_title": "第8章", "chunk_id": 801, "text": "未来剧情"},
            {"chapter_idx": 2, "chapter_title": "第2章", "chunk_id": 201, "text": "前情B"},
        ]

    monkeypatch.setattr(memory_retrieve, "retrieve_evidence", _fake_retrieve_evidence)

    state = cast(
        StorytellerState,
        {
        "chapter_idx": 3,
        "chapter_text": "本章文本",
        "entities_mentioned": ["韩立"],
        },
    )
    result = asyncio.run(memory_retrieve.run(state, config=config, book_id=1))
    memories = result["awakened_memories"]

    assert len(memories) == 2
    assert all(int(item["chapter_idx"]) < 3 for item in memories)
    assert [item["chunk_id"] for item in memories] == [101, 201]


def test_storyteller_generate_fallback_produces_narration() -> None:
    config = AppConfigRoot()
    chapter_text = "甲" * 100
    state = cast(
        StorytellerState,
        {
        "chapter_idx": 1,
        "chapter_title": "第1章",
        "chapter_text": chapter_text,
        "character_states": [],
        "item_states": [],
        "recent_events": [],
        "awakened_memories": [],
        },
    )

    result = asyncio.run(storyteller_generate.run(state, config=config, llm_client=None))

    assert result["narration"]
    assert len(result["narration"]) == 50
    assert isinstance(result["key_events"], list)
