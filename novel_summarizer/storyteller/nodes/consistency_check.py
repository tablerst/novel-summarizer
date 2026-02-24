from __future__ import annotations

from typing import Any

from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name_key(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def _build_character_alias_index(character_states: list[dict[str, Any]]) -> dict[str, str]:
    alias_index: dict[str, str] = {}
    for character in character_states:
        canonical = _normalize_text(character.get("canonical_name"))
        if not canonical:
            continue
        alias_index[_normalize_name_key(canonical)] = canonical

        aliases_raw = character.get("aliases_json")
        if isinstance(aliases_raw, str):
            aliases = [alias.strip() for alias in aliases_raw.strip("[]").replace('"', "").split(",") if alias.strip()]
        elif isinstance(aliases_raw, list):
            aliases = [str(alias).strip() for alias in aliases_raw if str(alias).strip()]
        else:
            aliases = []

        for alias in aliases:
            alias_index[_normalize_name_key(alias)] = canonical

    return alias_index


async def run(state: StorytellerState, *, config: AppConfigRoot) -> dict:
    _ = config
    node_log = logger.bind(
        node="consistency_check",
        chapter_id=state.get("chapter_id"),
        chapter_idx=state.get("chapter_idx"),
    )
    try:
        warnings: list[str] = []
        actions: list[str] = []

        recent_events = state.get("recent_events", [])
        recent_event_set = {
            _normalize_text(item.get("event_summary"))
            for item in recent_events
            if _normalize_text(item.get("event_summary"))
        }

        sanitized_events: list[dict[str, Any]] = []
        seen_event_texts: set[str] = set()
        for event in state.get("key_events", []):
            what = _normalize_text(event.get("what"))
            if not what:
                warnings.append("Dropped empty key_event")
                continue
            if what in seen_event_texts or what in recent_event_set:
                warnings.append(f"Dropped duplicated key_event: {what}")
                continue
            seen_event_texts.add(what)
            sanitized_events.append(
                {
                    "who": _normalize_text(event.get("who")),
                    "what": what,
                    "where": _normalize_text(event.get("where")),
                    "outcome": _normalize_text(event.get("outcome")),
                    "impact": _normalize_text(event.get("impact")),
                }
            )

        if len(sanitized_events) > 20:
            warnings.append("Too many key_events; truncated to 20")
            sanitized_events = sanitized_events[:20]

        alias_index = _build_character_alias_index(state.get("character_states", []))
        sanitized_updates: list[dict[str, Any]] = []
        for update in state.get("character_updates", []):
            name_raw = _normalize_text(update.get("name"))
            if not name_raw:
                warnings.append("Dropped character_update without name")
                continue
            canonical = alias_index.get(_normalize_name_key(name_raw), name_raw)
            if canonical != name_raw:
                actions.append(f"Normalized character alias '{name_raw}' -> '{canonical}'")

            change_type = _normalize_text(update.get("change_type")) or "status"
            before = _normalize_text(update.get("before"))
            after = _normalize_text(update.get("after"))
            if before and after and before == after:
                warnings.append(f"Dropped no-op character_update for '{canonical}' ({change_type})")
                continue

            sanitized_updates.append(
                {
                    "name": canonical,
                    "name_raw": name_raw,
                    "change_type": change_type,
                    "before": before,
                    "after": after,
                    "evidence": _normalize_text(update.get("evidence")),
                }
            )

        node_log.info(
            "Consistency check completed key_events_in={} key_events_out={} updates_in={} updates_out={} warnings={} actions={}",
            len(state.get("key_events", [])),
            len(sanitized_events),
            len(state.get("character_updates", [])),
            len(sanitized_updates),
            len(warnings),
            len(actions),
        )

        return {
            "key_events": sanitized_events,
            "character_updates": sanitized_updates,
            "consistency_warnings": warnings,
            "consistency_actions": actions,
        }
    except Exception:
        node_log.exception("Consistency check node failed")
        raise