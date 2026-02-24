from __future__ import annotations

from typing import Any

import orjson
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.llm.factory import make_cache_key
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.prompts.narration import NARRATION_PROMPT_VERSION, narration_prompt
from novel_summarizer.storyteller.state import StorytellerState


class KeyEventOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    who: str = ""
    what: str = ""
    where: str = ""
    outcome: str = ""
    impact: str = ""


class CharacterUpdateOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    change_type: str = ""
    before: str = ""
    after: str = ""
    evidence: str = ""


class NewItemOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    owner: str = ""
    description: str = ""


class NarrationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    narration: str = ""
    key_events: list[KeyEventOutput] = Field(default_factory=list)
    character_updates: list[CharacterUpdateOutput] = Field(default_factory=list)
    new_items: list[NewItemOutput] = Field(default_factory=list)


def _draft_narration(text: str, ratio: tuple[float, float]) -> str:
    if not text:
        return ""
    target_len = max(1, int(len(text) * ratio[1]))
    return text[:target_len].strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Coarse estimate for Chinese-heavy text: ~2 chars/token
    return max(1, len(text) // 2)


def _normalize_dict_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append({str(k): v for k, v in item.items()})
    return normalized


async def run(state: StorytellerState, *, config: AppConfigRoot, llm_client: Any | None = None) -> dict:
    chapter_text = state.get("chapter_text", "")
    fallback_narration = _draft_narration(chapter_text, config.storyteller.narration_ratio)
    chapter_id = state.get("chapter_id")
    chapter_idx = state.get("chapter_idx")
    node_log = logger.bind(node="storyteller_generate", chapter_id=chapter_id, chapter_idx=chapter_idx)

    if llm_client is None:
        narration = fallback_narration
        key_events = []
        if narration:
            key_events.append(
                {
                    "who": "unknown",
                    "what": f"Chapter {state.get('chapter_idx')} draft narration generated",
                    "where": "unknown",
                    "outcome": "draft_generated",
                    "impact": "world_state_pending",
                }
            )

        return {
            "narration": narration,
            "key_events": key_events,
            "character_updates": [],
            "new_items": [],
            "narration_llm_calls": 0,
            "narration_llm_cache_hit": False,
            "input_tokens_estimated": _estimate_tokens(chapter_text),
            "output_tokens_estimated": _estimate_tokens(narration),
        }

    try:
        system, user_template = narration_prompt(
            language=config.storyteller.language,
            style=config.storyteller.style,
            narration_ratio=config.storyteller.narration_ratio,
            include_key_dialogue=config.storyteller.include_key_dialogue,
            include_inner_thoughts=config.storyteller.include_inner_thoughts,
        )
        user = user_template.format(
            chapter_title=state.get("chapter_title", ""),
            chapter_text=chapter_text,
            character_states=orjson.dumps(state.get("character_states", [])).decode("utf-8"),
            item_states=orjson.dumps(state.get("item_states", [])).decode("utf-8"),
            recent_events=orjson.dumps(state.get("recent_events", [])).decode("utf-8"),
            awakened_memories=orjson.dumps(state.get("awakened_memories", [])).decode("utf-8"),
        )

        input_hash = sha256_text(
            orjson.dumps(
                {
                    "chapter_id": state.get("chapter_id"),
                    "chapter_idx": state.get("chapter_idx"),
                    "chapter_title": state.get("chapter_title"),
                    "chapter_text": chapter_text,
                    "character_states": state.get("character_states", []),
                    "item_states": state.get("item_states", []),
                    "recent_events": state.get("recent_events", []),
                    "awakened_memories": state.get("awakened_memories", []),
                    "style": config.storyteller.style,
                    "ratio": config.storyteller.narration_ratio,
                }
            ).decode("utf-8")
        )
        cache_key = make_cache_key(
            "storyteller_generate",
            llm_client.model_identifier,
            NARRATION_PROMPT_VERSION,
            input_hash,
            str(config.storyteller.narration_temperature),
        )
        llm_context = {
            "node": "storyteller_generate",
            "chapter_id": chapter_id,
            "chapter_idx": chapter_idx,
            "input_hash": input_hash,
        }
        run_log = node_log.bind(cache_key=cache_key[:12], input_hash=input_hash[:12])
        run_log.debug("Invoking storyteller narration generation")
        if hasattr(llm_client, "complete_structured"):
            llm_response, payload_obj = llm_client.complete_structured(
                system,
                user,
                cache_key,
                NarrationOutput,
                method="function_calling",
                context=llm_context,
            )
            payload = payload_obj.model_dump(mode="python")
        else:
            llm_response, payload = llm_client.complete_json(
                system,
                user,
                cache_key,
                safe_load_json_dict,
                context=llm_context,
            )
        cache_hit = bool(getattr(llm_response, "cached", False))

        narration = str(payload.get("narration", "")).strip() or fallback_narration
        key_events = _normalize_dict_list(payload.get("key_events", []))
        character_updates = _normalize_dict_list(payload.get("character_updates", []))
        new_items = _normalize_dict_list(payload.get("new_items", []))

        if not key_events and narration:
            key_events = [
                {
                    "who": "unknown",
                    "what": f"Chapter {state.get('chapter_idx')} narration generated",
                    "where": "unknown",
                    "outcome": "generated",
                    "impact": "state_update_pending",
                }
            ]

        return {
            "narration": narration,
            "key_events": key_events,
            "character_updates": character_updates,
            "new_items": new_items,
            "narration_llm_calls": 1,
            "narration_llm_cache_hit": cache_hit,
            "input_tokens_estimated": _estimate_tokens(chapter_text),
            "output_tokens_estimated": _estimate_tokens(narration),
        }
    except Exception as exc:  # noqa: BLE001
        node_log.exception("Storyteller generation fallback due to LLM error: {}", exc)
        narration = fallback_narration
        key_events = []
        if narration:
            key_events.append(
                {
                    "who": "unknown",
                    "what": f"Chapter {state.get('chapter_idx')} draft narration generated",
                    "where": "unknown",
                    "outcome": "draft_generated",
                    "impact": "world_state_pending",
                }
            )

        return {
            "narration": narration,
            "key_events": key_events,
            "character_updates": [],
            "new_items": [],
            "narration_llm_calls": 1,
            "narration_llm_cache_hit": False,
            "input_tokens_estimated": _estimate_tokens(chapter_text),
            "output_tokens_estimated": _estimate_tokens(narration),
        }