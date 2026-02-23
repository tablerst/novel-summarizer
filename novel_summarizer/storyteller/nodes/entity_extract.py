from __future__ import annotations

import re
from typing import Any

from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.llm.factory import make_cache_key
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.prompts.entity import ENTITY_PROMPT_VERSION, entity_prompt
from novel_summarizer.storyteller.state import StorytellerState


_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,8}")


def _unique(values: list[str], max_items: int = 20) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= max_items:
            break
    return output


def _fallback_entities(text: str) -> dict[str, list[str]]:
    tokens = _CJK_TOKEN_PATTERN.findall(text)
    return {
        "characters": _unique(tokens, max_items=16),
        "locations": [],
        "items": [],
        "key_phrases": _unique(tokens, max_items=20),
    }


def _normalize_list_field(payload: dict[str, Any], key: str, max_items: int = 32) -> list[str]:
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        return []
    values = [str(item).strip() for item in raw if str(item).strip()]
    return _unique(values, max_items=max_items)


async def run(state: StorytellerState, *, config: AppConfigRoot, llm_client: Any | None = None) -> dict:
    text = state.get("chapter_text", "")
    fallback = _fallback_entities(text)

    if llm_client is None:
        return {
            "entities_mentioned": fallback["characters"],
            "locations_mentioned": fallback["locations"],
            "items_mentioned": fallback["items"],
        }

    try:
        system, user_template = entity_prompt(config.storyteller.language)
        user = user_template.format(chapter_text=text)
        input_hash = sha256_text(
            f"{state.get('chapter_id')}::{state.get('chapter_idx')}::{state.get('chapter_title')}::{text}"
        )
        cache_key = make_cache_key(
            "storyteller_entity",
            llm_client.model_identifier,
            ENTITY_PROMPT_VERSION,
            input_hash,
            str(config.storyteller.entity_temperature),
        )
        _, payload = llm_client.complete_json(system, user, cache_key, safe_load_json_dict)
        return {
            "entities_mentioned": _normalize_list_field(payload, "characters", max_items=16),
            "locations_mentioned": _normalize_list_field(payload, "locations", max_items=16),
            "items_mentioned": _normalize_list_field(payload, "items", max_items=16),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Entity extraction fallback due to LLM error: {}", exc)
        return {
            "entities_mentioned": fallback["characters"],
            "locations_mentioned": fallback["locations"],
            "items_mentioned": fallback["items"],
        }