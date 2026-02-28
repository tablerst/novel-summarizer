from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.nodes.storyteller_generate_step import run_batch
from novel_summarizer.storyteller.state import StorytellerState


class _FakeStepLLM:
    model_identifier = "fake/provider/model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, system_prompt: str, user_prompt: str, cache_key: str, parser, *, context=None):
        _ = (system_prompt, cache_key, context)
        self.calls.append(user_prompt)

        obj = (
            "{"
            '"step_start_chapter_idx": 1,'
            '"step_end_chapter_idx": 2,'
            '"narration": "step-narration",'
            '"key_events": [],'
            '"character_updates": [],'
            '"new_items": []'
            "}"
        )
        return None, parser(obj)


def test_safe_load_json_dict_parses_code_fence() -> None:
    payload = """```json
    {"a": 1}
    ```"""
    assert safe_load_json_dict(payload) == {"a": 1}


def test_step_generate_returns_aggregate_output() -> None:
    async def _run() -> None:
        config = AppConfigRoot()
        llm = _FakeStepLLM()

        states = cast(
            list[StorytellerState],
            [
                {
                    "chapter_idx": 1,
                    "chapter_title": "第1章",
                    "chapter_text": "甲" * 10,
                    "awakened_memories": [],
                },
                {
                    "chapter_idx": 2,
                    "chapter_title": "第2章",
                    "chapter_text": "乙" * 10,
                    "awakened_memories": [],
                },
            ],
        )

        outputs = await run_batch(
            states,
            config=config,
            llm_client=cast(Any, llm),
            base_world_state={"character_states": [], "item_states": [], "recent_events": [], "world_facts": []},
        )

        assert outputs["step_start_chapter_idx"] == 1
        assert outputs["step_end_chapter_idx"] == 2
        assert outputs["narration"] == "step-narration"
        assert len(llm.calls) == 1

    asyncio.run(_run())


def test_step_generate_llm_disabled_falls_back_to_draft() -> None:
    async def _run() -> None:
        config = AppConfigRoot.model_validate({"storyteller": {"narration_ratio": (0.1, 0.2)}})
        state = cast(
            StorytellerState,
            {
                "chapter_idx": 1,
                "chapter_title": "第1章",
                "chapter_text": "甲" * 100,
                "awakened_memories": [],
            },
        )
        outputs = await run_batch(
            [state],
            config=config,
            llm_client=None,
            base_world_state={"character_states": [], "item_states": [], "recent_events": [], "world_facts": []},
        )
        assert outputs["narration"]
        assert outputs["narration_llm_calls"] == 0

    asyncio.run(_run())


def test_step_generate_llm_failure_falls_back_to_draft() -> None:
    class _BrokenLLM:
        model_identifier = "fake/provider/model"

        def complete_json(self, system_prompt: str, user_prompt: str, cache_key: str, parser, *, context=None):
            _ = (system_prompt, user_prompt, cache_key, parser, context)
            raise RuntimeError("boom")

    config = AppConfigRoot()

    states = cast(
        list[StorytellerState],
        [
            {"chapter_idx": 1, "chapter_title": "第1章", "chapter_text": "甲" * 10, "awakened_memories": []},
            {"chapter_idx": 2, "chapter_title": "第2章", "chapter_text": "乙" * 10, "awakened_memories": []},
        ],
    )

    outputs = asyncio.run(
        run_batch(
            states,
            config=config,
            llm_client=cast(Any, _BrokenLLM()),
            base_world_state={"character_states": [], "item_states": [], "recent_events": [], "world_facts": []},
        )
    )
    assert outputs["narration"]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "[]",
        "{]",
    ],
)
def test_safe_load_json_dict_rejects_bad_payloads(bad: str) -> None:
    with pytest.raises(Exception):
        safe_load_json_dict(bad)
