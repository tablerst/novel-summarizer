from __future__ import annotations

from typing import Any

import orjson
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.llm.factory import make_cache_key
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.prompts.refine import REFINE_PROMPT_VERSION, refine_prompt
from novel_summarizer.storyteller.state import StorytellerState


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


def _normalize_text(text: Any) -> str:
    return str(text or "").strip()


async def run(state: StorytellerState, *, config: AppConfigRoot, llm_client: Any | None = None) -> dict:
    narration = _normalize_text(state.get("narration"))
    if not narration:
        return {
            "refine_llm_calls": 0,
            "refine_llm_cache_hit": False,
            "refine_input_tokens_estimated": 0,
            "refine_output_tokens_estimated": 0,
        }

    input_tokens = _estimate_tokens(narration)

    if not config.storyteller.refine_enabled:
        return {
            "refine_llm_calls": 0,
            "refine_llm_cache_hit": False,
            "refine_input_tokens_estimated": input_tokens,
            "refine_output_tokens_estimated": _estimate_tokens(narration),
        }

    if llm_client is None:
        return {
            "refine_llm_calls": 0,
            "refine_llm_cache_hit": False,
            "refine_input_tokens_estimated": input_tokens,
            "refine_output_tokens_estimated": _estimate_tokens(narration),
        }

    try:
        system, user_template = refine_prompt(language=config.storyteller.language, style=config.storyteller.style)
        user = user_template.format(
            key_events=orjson.dumps(state.get("key_events", [])).decode("utf-8"),
            character_updates=orjson.dumps(state.get("character_updates", [])).decode("utf-8"),
            draft_narration=narration,
        )

        input_hash = sha256_text(
            orjson.dumps(
                {
                    "chapter_id": state.get("chapter_id"),
                    "chapter_idx": state.get("chapter_idx"),
                    "narration": narration,
                    "key_events": state.get("key_events", []),
                    "character_updates": state.get("character_updates", []),
                    "style": config.storyteller.style,
                }
            ).decode("utf-8")
        )
        cache_key = make_cache_key(
            "storyteller_refine",
            llm_client.model_identifier,
            REFINE_PROMPT_VERSION,
            input_hash,
            str(config.storyteller.refine_temperature),
        )
        llm_response, payload = llm_client.complete_json(system, user, cache_key, safe_load_json_dict)
        cache_hit = bool(getattr(llm_response, "cached", False))
        refined = _normalize_text(payload.get("narration")) or narration

        return {
            "narration": refined,
            "refine_llm_calls": 1,
            "refine_llm_cache_hit": cache_hit,
            "refine_input_tokens_estimated": input_tokens,
            "refine_output_tokens_estimated": _estimate_tokens(refined),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Narration refine fallback due to LLM error: {}", exc)
        return {
            "refine_llm_calls": 1,
            "refine_llm_cache_hit": False,
            "refine_input_tokens_estimated": input_tokens,
            "refine_output_tokens_estimated": _estimate_tokens(narration),
        }