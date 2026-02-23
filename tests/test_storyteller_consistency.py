from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.nodes import consistency_check, state_update
from novel_summarizer.storyteller.state import StorytellerState


def test_consistency_check_dedup_and_alias_normalization() -> None:
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
            "chapter_idx": 10,
            "recent_events": [{"event_summary": "宗门大比开启"}],
            "key_events": [
                {"who": "韩立", "what": "宗门大比开启", "where": "山门", "outcome": "开始", "impact": "晋升"},
                {"who": "韩立", "what": "拿到掌天瓶", "where": "秘境", "outcome": "获宝", "impact": "战力提升"},
                {"who": "韩立", "what": "拿到掌天瓶", "where": "秘境", "outcome": "获宝", "impact": "战力提升"},
            ],
            "character_states": [
                {
                    "canonical_name": "韩立",
                    "aliases_json": '["韩跑跑"]',
                }
            ],
            "character_updates": [
                {"name": "韩跑跑", "change_type": "status", "before": "炼气", "after": "筑基", "evidence": "破境成功"},
                {"name": "韩立", "change_type": "location", "before": "天南", "after": "天南", "evidence": "无变化"},
            ],
        },
    )

    result = asyncio.run(consistency_check.run(state, config=config))

    assert len(result["key_events"]) == 1
    assert result["key_events"][0]["what"] == "拿到掌天瓶"
    assert len(result["character_updates"]) == 1
    assert result["character_updates"][0]["name"] == "韩立"
    assert any("Normalized character alias" in item for item in result["consistency_actions"])
    assert any("Dropped duplicated key_event" in item for item in result["consistency_warnings"])


@dataclass
class _FakeInsertResult:
    id: int
    inserted: bool


class _FakeRepo:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.world_facts: list[dict[str, Any]] = []
        self.characters: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []

    async def insert_plot_event(self, **kwargs):
        self.events.append(kwargs)
        return _FakeInsertResult(id=len(self.events), inserted=True)

    async def upsert_world_fact(self, **kwargs):
        self.world_facts.append(kwargs)
        return _FakeInsertResult(id=len(self.world_facts), inserted=True)

    async def upsert_character_state(self, **kwargs):
        self.characters.append(kwargs)
        return _FakeInsertResult(id=len(self.characters), inserted=True)

    async def upsert_item_state(self, **kwargs):
        self.items.append(kwargs)
        return _FakeInsertResult(id=len(self.items), inserted=True)


def test_state_update_writes_world_facts_and_normalized_character_updates() -> None:
    repo = _FakeRepo()
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
            "chapter_idx": 12,
            "key_events": [
                {"who": "韩立", "what": "斩杀妖兽", "impact": "获得材料"},
            ],
            "character_states": [
                {
                    "canonical_name": "韩立",
                    "aliases_json": '["韩跑跑"]',
                    "first_chapter_idx": 1,
                    "status": "active",
                    "location": "天南",
                    "abilities_json": None,
                    "relationships_json": None,
                    "motivation": None,
                    "notes": None,
                }
            ],
            "entities_mentioned": ["韩跑跑"],
            "character_updates": [
                {
                    "name": "韩立",
                    "name_raw": "韩跑跑",
                    "change_type": "status",
                    "before": "active",
                    "after": "injured",
                    "evidence": "大战受伤",
                }
            ],
            "new_items": [
                {"name": "掌天瓶", "owner": "韩立", "description": "神秘小瓶"},
            ],
        },
    )

    result = asyncio.run(state_update.run(state, repo=cast(Any, repo), config=config, book_id=1))

    assert result["mutations_applied"]["plot_events_inserted"] == 1
    assert result["mutations_applied"]["characters_upserted"] >= 2
    assert result["mutations_applied"]["items_upserted"] == 1
    assert result["mutations_applied"]["world_facts_upserted"] >= 3
    assert any(item["fact_key"].startswith("event:12:") for item in repo.world_facts)
    assert any(item["fact_key"] == "character:韩立:status" for item in repo.world_facts)
