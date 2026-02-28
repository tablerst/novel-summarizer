from __future__ import annotations

from typing import Any

import orjson
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.llm.factory import make_cache_key
from novel_summarizer.storyteller.json_utils import safe_load_json_dict
from novel_summarizer.storyteller.prompts.step_narration import STEP_NARRATION_PROMPT_VERSION, step_narration_prompt
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storyteller.tiering import effective_storyteller_value


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


def _draft_narration(text: str, ratio: tuple[float, float]) -> str:
    if not text:
        return ""
    target_len = max(1, int(len(text) * ratio[1]))
    return text[:target_len].strip()


def _normalize_dict_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append({str(k): v for k, v in item.items()})
    return out


def _merge_entities(states: list[StorytellerState]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for st in states:
        for raw in st.get("entities_mentioned") or []:
            entity = str(raw).strip()
            if not entity or entity in seen:
                continue
            seen.add(entity)
            merged.append(entity)
    return merged


async def _invoke_llm(
    *,
    states: list[StorytellerState],
    config: AppConfigRoot,
    llm_client: Any,
    base_world_state: dict[str, Any],
    narration_temperature: float,
) -> dict[str, Any]:
    language = str(effective_storyteller_value(states[0], config, "language", config.storyteller.language))
    style = str(effective_storyteller_value(states[0], config, "style", config.storyteller.style))
    system, user_template = step_narration_prompt(language=language, style=style)

    ordered_states = sorted(states, key=lambda s: int(s.get("chapter_idx") or 0))
    step_start = int(ordered_states[0].get("chapter_idx") or 0)
    step_end = int(ordered_states[-1].get("chapter_idx") or 0)

    chapters_payload: list[dict[str, Any]] = []
    for st in ordered_states:
        ratio = tuple(effective_storyteller_value(st, config, "narration_ratio", config.storyteller.narration_ratio))
        include_key_dialogue = bool(
            effective_storyteller_value(st, config, "include_key_dialogue", config.storyteller.include_key_dialogue)
        )
        include_inner_thoughts = bool(
            effective_storyteller_value(st, config, "include_inner_thoughts", config.storyteller.include_inner_thoughts)
        )
        chapters_payload.append(
            {
                "chapter_idx": int(st.get("chapter_idx") or 0),
                "chapter_title": str(st.get("chapter_title") or ""),
                "chapter_text": str(st.get("chapter_text") or ""),
                "awakened_memories": st.get("awakened_memories") or [],
                "constraints": {
                    "narration_ratio": ratio,
                    "include_key_dialogue": include_key_dialogue,
                    "include_inner_thoughts": include_inner_thoughts,
                },
            }
        )

    input_hash = sha256_text(
        orjson.dumps(
            {
                "base_world_state": base_world_state,
                "chapters": chapters_payload,
                "style": style,
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8")
    )
    cache_key = make_cache_key(
        "storyteller_generate_step",
        llm_client.model_identifier,
        STEP_NARRATION_PROMPT_VERSION,
        input_hash,
        str(narration_temperature),
    )

    user = user_template.format(
        step_start=step_start,
        step_end=step_end,
        base_world_state=orjson.dumps(base_world_state, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
        chapters=orjson.dumps(chapters_payload, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
    )

    node_log = logger.bind(node="storyteller_generate_step", chapter_idx=f"{step_start}-{step_end}")
    node_log.debug("Invoking step narration generation batch_size={} cache_key={} input_hash={}", len(states), cache_key[:12], input_hash[:12])

    if hasattr(llm_client, "complete_json_async"):
        llm_response, payload = await llm_client.complete_json_async(system, user, cache_key, safe_load_json_dict)
    else:
        llm_response, payload = llm_client.complete_json(system, user, cache_key, safe_load_json_dict)

    cache_hit = bool(getattr(llm_response, "cached", False))

    combined_text = "\n\n".join(str(st.get("chapter_text") or "") for st in ordered_states).strip()
    ratio = tuple(
        effective_storyteller_value(ordered_states[0], config, "narration_ratio", config.storyteller.narration_ratio)
    )
    narration = str(payload.get("narration") or "").strip() or _draft_narration(combined_text, ratio)

    return {
        "step_start_chapter_idx": int(payload.get("step_start_chapter_idx") or step_start),
        "step_end_chapter_idx": int(payload.get("step_end_chapter_idx") or step_end),
        "narration": narration,
        "key_events": _normalize_dict_list(payload.get("key_events", [])),
        "character_updates": _normalize_dict_list(payload.get("character_updates", [])),
        "new_items": _normalize_dict_list(payload.get("new_items", [])),
        "entities_mentioned": _merge_entities(ordered_states),
        "narration_llm_calls": 1,
        "narration_llm_cache_hit": cache_hit,
        "input_tokens_estimated": _estimate_tokens(combined_text),
        "output_tokens_estimated": _estimate_tokens(narration),
    }


async def run_batch(
    states: list[StorytellerState],
    *,
    config: AppConfigRoot,
    llm_client: Any | None,
    base_world_state: dict[str, Any],
) -> dict[str, Any]:
    """Generate one aggregated narration for a step worth of chapters in one LLM call."""

    if not states:
        return {
            "step_start_chapter_idx": 0,
            "step_end_chapter_idx": 0,
            "narration": "",
            "key_events": [],
            "character_updates": [],
            "new_items": [],
            "entities_mentioned": [],
            "narration_llm_calls": 0,
            "narration_llm_cache_hit": False,
            "input_tokens_estimated": 0,
            "output_tokens_estimated": 0,
        }

    ordered = sorted(states, key=lambda s: int(s.get("chapter_idx") or 0))
    step_start = int(ordered[0].get("chapter_idx") or 0)
    step_end = int(ordered[-1].get("chapter_idx") or 0)
    combined_text = "\n\n".join(str(st.get("chapter_text") or "") for st in ordered).strip()
    ratio = tuple(effective_storyteller_value(ordered[0], config, "narration_ratio", config.storyteller.narration_ratio))
    fallback_narration = _draft_narration(combined_text, ratio)

    narration_temperature = float(
        effective_storyteller_value(ordered[0], config, "narration_temperature", config.storyteller.narration_temperature)
    )

    if llm_client is None:
        return {
            "step_start_chapter_idx": step_start,
            "step_end_chapter_idx": step_end,
            "narration": fallback_narration,
            "key_events": [],
            "character_updates": [],
            "new_items": [],
            "entities_mentioned": _merge_entities(ordered),
            "narration_llm_calls": 0,
            "narration_llm_cache_hit": False,
            "input_tokens_estimated": _estimate_tokens(combined_text),
            "output_tokens_estimated": _estimate_tokens(fallback_narration),
        }

    try:
        return await _invoke_llm(
            states=ordered,
            config=config,
            llm_client=llm_client,
            base_world_state=base_world_state,
            narration_temperature=narration_temperature,
        )
    except Exception as exc:  # noqa: BLE001
        logger.bind(node="storyteller_generate_step", chapter_idx=f"{step_start}-{step_end}").warning(
            "Step aggregate generation failed; fallback to draft narration: {}",
            exc,
        )
        return {
            "step_start_chapter_idx": step_start,
            "step_end_chapter_idx": step_end,
            "narration": fallback_narration,
            "key_events": [],
            "character_updates": [],
            "new_items": [],
            "entities_mentioned": _merge_entities(ordered),
            "narration_llm_calls": 1,
            "narration_llm_cache_hit": False,
            "input_tokens_estimated": _estimate_tokens(combined_text),
            "output_tokens_estimated": _estimate_tokens(fallback_narration),
        }
