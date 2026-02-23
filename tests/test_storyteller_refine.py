from __future__ import annotations

import asyncio
from typing import cast

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.nodes import refine_narration
from novel_summarizer.storyteller.state import StorytellerState


class _FakeLLMClient:
    model_identifier = "fake/provider/refine-model"

    def complete_json(self, system_prompt, user_prompt, cache_key, parser):
        _ = (system_prompt, user_prompt, cache_key)
        payload = parser('{"narration":"润色后的说书稿。"}')
        return object(), payload


def test_refine_narration_disabled_keeps_text() -> None:
    config = AppConfigRoot.model_validate({"storyteller": {"refine_enabled": False}})
    state = cast(StorytellerState, {"narration": "初稿说书稿。"})

    result = asyncio.run(refine_narration.run(state, config=config, llm_client=_FakeLLMClient()))

    assert result["refine_llm_calls"] == 0
    assert result["refine_output_tokens_estimated"] > 0
    assert "narration" not in result


def test_refine_narration_enabled_uses_llm_output() -> None:
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
            "chapter_id": 1,
            "chapter_idx": 1,
            "narration": "初稿说书稿。",
            "key_events": [{"what": "韩立获宝"}],
            "character_updates": [{"name": "韩立", "after": "筑基"}],
        },
    )

    result = asyncio.run(refine_narration.run(state, config=config, llm_client=_FakeLLMClient()))

    assert result["narration"] == "润色后的说书稿。"
    assert result["refine_llm_calls"] == 1
    assert result["refine_input_tokens_estimated"] > 0
    assert result["refine_output_tokens_estimated"] > 0


def test_refine_narration_no_client_fallback() -> None:
    config = AppConfigRoot()
    state = cast(StorytellerState, {"narration": "初稿说书稿。"})

    result = asyncio.run(refine_narration.run(state, config=config, llm_client=None))

    assert result["refine_llm_calls"] == 0
    assert result["refine_llm_cache_hit"] is False
