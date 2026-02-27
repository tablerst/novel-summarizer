from __future__ import annotations

from typing import Any

import orjson
from loguru import logger
from pydantic import BaseModel, ConfigDict

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.llm.factory import make_cache_key
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.prompts.refine import REFINE_PROMPT_VERSION, refine_prompt
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storyteller.tiering import effective_storyteller_value


class RefineOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    narration: str = ""


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


def _normalize_text(text: Any) -> str:
    return str(text or "").strip()


async def run(state: StorytellerState, *, config: AppConfigRoot, llm_client: Any | None = None) -> dict:
    narration = _normalize_text(state.get("narration"))
    refine_enabled = bool(effective_storyteller_value(state, config, "refine_enabled", config.storyteller.refine_enabled))
    refine_temperature = float(
        effective_storyteller_value(state, config, "refine_temperature", config.storyteller.refine_temperature)
    )
    language = str(effective_storyteller_value(state, config, "language", config.storyteller.language))
    style = str(effective_storyteller_value(state, config, "style", config.storyteller.style))
    if not narration:
        return {
            "refine_llm_calls": 0,
            "refine_llm_cache_hit": False,
            "refine_input_tokens_estimated": 0,
            "refine_output_tokens_estimated": 0,
        }

    input_tokens = _estimate_tokens(narration)

    if not refine_enabled:
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
        system, user_template = refine_prompt(language=language, style=style)
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
                    "style": style,
                }
            ).decode("utf-8")
        )
        cache_key = make_cache_key(
            "storyteller_refine",
            llm_client.model_identifier,
            REFINE_PROMPT_VERSION,
            input_hash,
            str(refine_temperature),
        )
        if hasattr(llm_client, "complete_structured_async"):
            llm_response, payload_obj = await llm_client.complete_structured_async(
                system,
                user,
                cache_key,
                RefineOutput,
                method="function_calling",
            )
            payload = payload_obj.model_dump(mode="python")
        elif hasattr(llm_client, "complete_structured"):
            llm_response, payload_obj = llm_client.complete_structured(
                system,
                user,
                cache_key,
                RefineOutput,
                method="function_calling",
            )
            payload = payload_obj.model_dump(mode="python")
        else:
            if hasattr(llm_client, "complete_json_async"):
                llm_response, payload = await llm_client.complete_json_async(
                    system,
                    user,
                    cache_key,
                    safe_load_json_dict,
                )
            else:
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