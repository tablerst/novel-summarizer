from __future__ import annotations

from typing import cast

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storyteller.tiering import (
    build_tier_overrides,
    decide_tier,
    effective_storyteller_value,
    has_storyteller_memory_retrieval,
)


def test_decide_tier_disabled_follows_global_preset() -> None:
    config = AppConfigRoot.model_validate(
        {
            "storyteller": {
                "narration_preset": "short",
                "tiering": {"enabled": False},
            }
        }
    )

    tier = decide_tier(
        chapter_idx=10,
        chapter_title="第十章",
        chapter_text="普通章节文本",
        config=config,
    )

    assert tier == "short"


def test_decide_tier_promotes_to_long_on_keyword() -> None:
    config = AppConfigRoot.model_validate(
        {
            "storyteller": {
                "tiering": {
                    "enabled": True,
                    "default_tier": "short",
                    "long_keyword_triggers": ["大战", "突破"],
                }
            }
        }
    )

    tier = decide_tier(
        chapter_idx=3,
        chapter_title="第3章 决战前夕",
        chapter_text="宗门大战一触即发。",
        config=config,
    )

    assert tier == "long"


def test_build_tier_overrides_uses_global_values_when_disabled() -> None:
    config = AppConfigRoot.model_validate(
        {
            "storyteller": {
                "narration_ratio": (0.33, 0.44),
                "memory_top_k": 6,
                "include_key_dialogue": False,
                "include_inner_thoughts": True,
                "refine_enabled": False,
                "entity_extract_mode": "fallback",
                "tiering": {"enabled": False},
            }
        }
    )

    overrides = build_tier_overrides(tier="medium", config=config)

    assert overrides["narration_ratio"] == (0.33, 0.44)
    assert overrides["memory_top_k"] == 6
    assert overrides["include_key_dialogue"] is False
    assert overrides["include_inner_thoughts"] is True
    assert overrides["refine_enabled"] is False
    assert overrides["entity_extract_mode"] == "fallback"


def test_effective_storyteller_value_prefers_state_override() -> None:
    config = AppConfigRoot()
    state = cast(
        StorytellerState,
        {
            "storyteller_overrides": {
                "memory_top_k": 0,
            }
        },
    )

    value = effective_storyteller_value(state, config, "memory_top_k")

    assert value == 0


def test_has_storyteller_memory_retrieval_checks_tier_profiles() -> None:
    config = AppConfigRoot.model_validate(
        {
            "storyteller": {
                "tiering": {
                    "enabled": True,
                    "short": {
                        "narration_ratio": (0.1, 0.2),
                        "memory_top_k": 0,
                        "include_key_dialogue": False,
                        "include_inner_thoughts": False,
                        "refine_enabled": False,
                        "entity_extract_mode": "fallback",
                    },
                    "medium": {
                        "narration_ratio": (0.2, 0.3),
                        "memory_top_k": 3,
                        "include_key_dialogue": True,
                        "include_inner_thoughts": False,
                        "refine_enabled": False,
                        "entity_extract_mode": "llm",
                    },
                    "long": {
                        "narration_ratio": (0.4, 0.5),
                        "memory_top_k": 8,
                        "include_key_dialogue": True,
                        "include_inner_thoughts": True,
                        "refine_enabled": True,
                        "entity_extract_mode": "llm",
                    },
                },
            }
        }
    )

    assert has_storyteller_memory_retrieval(config) is True

    zero_memory_config = AppConfigRoot.model_validate(
        {
            "storyteller": {
                "tiering": {
                    "enabled": True,
                    "short": {
                        "narration_ratio": (0.1, 0.2),
                        "memory_top_k": 0,
                        "include_key_dialogue": False,
                        "include_inner_thoughts": False,
                        "refine_enabled": False,
                        "entity_extract_mode": "fallback",
                    },
                    "medium": {
                        "narration_ratio": (0.2, 0.3),
                        "memory_top_k": 0,
                        "include_key_dialogue": True,
                        "include_inner_thoughts": False,
                        "refine_enabled": False,
                        "entity_extract_mode": "llm",
                    },
                    "long": {
                        "narration_ratio": (0.4, 0.5),
                        "memory_top_k": 0,
                        "include_key_dialogue": True,
                        "include_inner_thoughts": True,
                        "refine_enabled": True,
                        "entity_extract_mode": "llm",
                    },
                },
            }
        }
    )

    assert has_storyteller_memory_retrieval(zero_memory_config) is False
