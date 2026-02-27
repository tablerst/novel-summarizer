from __future__ import annotations

from typing import Any, Literal

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState

TierName = Literal["short", "medium", "long"]


def decide_tier(
    *,
    chapter_idx: int,
    chapter_title: str,
    chapter_text: str,
    config: AppConfigRoot,
) -> TierName:
    tiering = config.storyteller.tiering

    if not tiering.enabled:
        return config.storyteller.narration_preset

    if tiering.long_every_n > 0 and chapter_idx % tiering.long_every_n == 0:
        return "long"

    if tiering.long_min_chars > 0 and len(chapter_text) >= tiering.long_min_chars:
        return "long"

    if tiering.long_keyword_triggers:
        haystack = f"{chapter_title}\n{chapter_text[:4000]}".lower()
        for keyword in tiering.long_keyword_triggers:
            key = keyword.strip().lower()
            if key and key in haystack:
                return "long"

    return tiering.default_tier


def build_tier_overrides(*, tier: TierName, config: AppConfigRoot) -> dict[str, Any]:
    if not config.storyteller.tiering.enabled:
        return {
            "narration_ratio": config.storyteller.narration_ratio,
            "memory_top_k": config.storyteller.memory_top_k,
            "include_key_dialogue": config.storyteller.include_key_dialogue,
            "include_inner_thoughts": config.storyteller.include_inner_thoughts,
            "refine_enabled": config.storyteller.refine_enabled,
            "entity_extract_mode": config.storyteller.entity_extract_mode,
        }

    profile = getattr(config.storyteller.tiering, tier)
    return {
        "narration_ratio": profile.narration_ratio,
        "memory_top_k": profile.memory_top_k,
        "include_key_dialogue": profile.include_key_dialogue,
        "include_inner_thoughts": profile.include_inner_thoughts,
        "refine_enabled": profile.refine_enabled,
        "entity_extract_mode": profile.entity_extract_mode,
    }


def effective_storyteller_value(state: StorytellerState, config: AppConfigRoot, key: str, default: Any = None) -> Any:
    overrides = state.get("storyteller_overrides") or {}
    if isinstance(overrides, dict) and key in overrides:
        return overrides[key]
    if hasattr(config.storyteller, key):
        return getattr(config.storyteller, key)
    return default


def has_storyteller_memory_retrieval(config: AppConfigRoot) -> bool:
    tiering = config.storyteller.tiering
    if tiering.enabled:
        return any(getattr(tiering, tier).memory_top_k > 0 for tier in ("short", "medium", "long"))

    return config.storyteller.memory_top_k > 0
