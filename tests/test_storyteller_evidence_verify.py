from __future__ import annotations

import asyncio
from typing import cast

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.nodes import evidence_verify
from novel_summarizer.storyteller.state import StorytellerState


def test_evidence_verify_filters_unsupported_claims() -> None:
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
            "chapter_text": "韩立在秘境中斩杀妖兽，获得掌天瓶。",
            "awakened_memories": [
                {"source_type": "chunk", "text": "前文提到韩立需要突破瓶颈。"},
            ],
            "key_events": [
                {"who": "韩立", "what": "斩杀妖兽", "where": "秘境", "outcome": "获胜", "impact": "获得材料"},
                {"who": "韩立", "what": "飞升灵界", "where": "未知", "outcome": "成功", "impact": "剧情终章"},
            ],
            "character_updates": [
                {"name": "韩立", "change_type": "status", "before": "炼气", "after": "筑基", "evidence": "突破瓶颈"},
            ],
            "new_items": [
                {"name": "掌天瓶", "owner": "韩立", "description": "神秘小瓶"},
            ],
            "consistency_warnings": [],
            "consistency_actions": [],
        },
    )

    result = asyncio.run(evidence_verify.run(state, config=config))

    assert len(result["key_events"]) == 1
    assert result["key_events"][0]["what"] == "斩杀妖兽"
    assert len(result["character_updates"]) == 1
    assert len(result["new_items"]) == 1
    assert result["evidence_report"]["total_claims"] == 4
    assert result["evidence_report"]["supported_claims"] == 3
    assert result["evidence_report"]["unsupported_claims"] == 1
    assert any("Evidence rejected key_event" in msg for msg in result["consistency_warnings"])


def test_evidence_verify_uses_memory_when_chapter_lacks_support() -> None:
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
            "chapter_text": "本章以氛围描写为主。",
            "awakened_memories": [
                {"source_type": "narration", "text": "韩立在前章已与南宫婉结盟。"},
            ],
            "key_events": [
                {"who": "韩立", "what": "与南宫婉结盟", "where": "天南", "outcome": "联盟成立", "impact": "势力增强"},
            ],
            "character_updates": [],
            "new_items": [],
        },
    )

    result = asyncio.run(evidence_verify.run(state, config=config))

    assert len(result["key_events"]) == 1
    assert result["key_events"][0]["evidence_source_type"] == "narration"
    assert result["evidence_report"]["supported_claims"] == 1
