from __future__ import annotations

from typing import Any

import orjson

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storage.repo import SQLAlchemyRepo


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def _parse_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            loaded = orjson.loads(text)
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
        except orjson.JSONDecodeError:
            pass
        return [part.strip() for part in text.replace("[", "").replace("]", "").replace('"', "").split(",") if part.strip()]
    return [str(raw).strip()]


def _build_character_lookup(character_states: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_canonical: dict[str, dict[str, Any]] = {}
    alias_to_canonical: dict[str, str] = {}
    for item in character_states:
        canonical = str(item.get("canonical_name") or "").strip()
        if not canonical:
            continue
        by_canonical[canonical] = item
        alias_to_canonical[_normalize_name(canonical)] = canonical
        aliases = _parse_json_list(item.get("aliases_json"))
        for alias in aliases:
            alias_to_canonical[_normalize_name(alias)] = canonical
    return by_canonical, alias_to_canonical


def _resolve_canonical(name: str, alias_to_canonical: dict[str, str]) -> str:
    key = _normalize_name(name)
    return alias_to_canonical.get(key, name)


async def run(state: StorytellerState, *, repo: SQLAlchemyRepo, config: AppConfigRoot, book_id: int) -> dict:
    _ = config
    chapter_idx = int(state.get("chapter_idx", 0))

    character_states = state.get("character_states", [])
    by_canonical, alias_to_canonical = _build_character_lookup(character_states)

    character_writes = 0
    item_writes = 0
    world_fact_writes = 0

    event_writes = 0
    for key_event in state.get("key_events", []):
        event_summary = str(key_event.get("what") or "")
        if not event_summary:
            continue
        involved = key_event.get("who")
        involved_json = None
        if involved:
            involved_json = orjson.dumps([str(involved)]).decode("utf-8")
        await repo.insert_plot_event(
            book_id=book_id,
            chapter_idx=chapter_idx,
            event_summary=event_summary,
            involved_characters_json=involved_json,
            event_type="narration_draft",
            impact=str(key_event.get("impact") or ""),
        )
        event_writes += 1

        event_fact_key = f"event:{chapter_idx}:{sha256_text(event_summary)[:12]}"
        await repo.upsert_world_fact(
            book_id=book_id,
            fact_key=event_fact_key,
            fact_value=event_summary,
            confidence=0.7,
            source_chapter_idx=chapter_idx,
            source_excerpt=event_summary[:300],
        )
        world_fact_writes += 1

    for raw_name in state.get("entities_mentioned", []) or []:
        name = str(raw_name).strip()
        if not name:
            continue
        canonical = _resolve_canonical(name, alias_to_canonical)
        existing = by_canonical.get(canonical)
        aliases = _parse_json_list(existing.get("aliases_json") if existing else None)
        if canonical != name and name not in aliases:
            aliases.append(name)
        if canonical == name:
            aliases = [alias for alias in aliases if alias != canonical]
        aliases_json = orjson.dumps(sorted(set(aliases))).decode("utf-8")

        await repo.upsert_character_state(
            book_id=book_id,
            canonical_name=canonical,
            aliases_json=aliases_json,
            first_chapter_idx=existing.get("first_chapter_idx") if existing else chapter_idx,
            last_chapter_idx=chapter_idx,
            status=str((existing or {}).get("status") or "active"),
            location=(existing or {}).get("location"),
            abilities_json=(existing or {}).get("abilities_json"),
            relationships_json=(existing or {}).get("relationships_json"),
            motivation=(existing or {}).get("motivation"),
            notes=(existing or {}).get("notes"),
        )
        by_canonical[canonical] = {
            **(existing or {}),
            "canonical_name": canonical,
            "aliases_json": aliases_json,
            "last_chapter_idx": chapter_idx,
        }
        alias_to_canonical[_normalize_name(canonical)] = canonical
        alias_to_canonical[_normalize_name(name)] = canonical
        character_writes += 1

    for update in state.get("character_updates", []) or []:
        raw_name = str(update.get("name_raw") or update.get("name") or "").strip()
        name = str(update.get("name") or raw_name).strip()
        if not name:
            continue
        canonical = _resolve_canonical(name, alias_to_canonical)
        existing = by_canonical.get(canonical)

        aliases = _parse_json_list(existing.get("aliases_json") if existing else None)
        if raw_name and raw_name != canonical and raw_name not in aliases:
            aliases.append(raw_name)
        aliases_json = orjson.dumps(sorted(set(aliases))).decode("utf-8")

        change_type = str(update.get("change_type") or "status").strip().lower()
        after = str(update.get("after") or "").strip()
        status = str((existing or {}).get("status") or "active")
        location = (existing or {}).get("location")
        if change_type == "status" and after:
            status = after
        if change_type == "location" and after:
            location = after

        await repo.upsert_character_state(
            book_id=book_id,
            canonical_name=canonical,
            aliases_json=aliases_json,
            first_chapter_idx=existing.get("first_chapter_idx") if existing else chapter_idx,
            last_chapter_idx=chapter_idx,
            status=status,
            location=location,
            abilities_json=(existing or {}).get("abilities_json"),
            relationships_json=(existing or {}).get("relationships_json"),
            motivation=(existing or {}).get("motivation"),
            notes=(existing or {}).get("notes"),
        )
        character_writes += 1

        await repo.upsert_world_fact(
            book_id=book_id,
            fact_key=f"character:{canonical}:status",
            fact_value=status,
            confidence=0.85,
            source_chapter_idx=chapter_idx,
            source_excerpt=str(update.get("evidence") or "")[:300],
        )
        world_fact_writes += 1

        if location:
            await repo.upsert_world_fact(
                book_id=book_id,
                fact_key=f"character:{canonical}:location",
                fact_value=str(location),
                confidence=0.8,
                source_chapter_idx=chapter_idx,
                source_excerpt=str(update.get("evidence") or "")[:300],
            )
            world_fact_writes += 1

    for new_item in state.get("new_items", []) or []:
        name = str(new_item.get("name") or "").strip()
        if not name:
            continue
        owner = str(new_item.get("owner") or "").strip() or None
        description = str(new_item.get("description") or "").strip() or None
        await repo.upsert_item_state(
            book_id=book_id,
            name=name,
            owner_name=owner,
            first_chapter_idx=chapter_idx,
            last_chapter_idx=chapter_idx,
            description=description,
            status="active",
        )
        item_writes += 1

        if owner:
            await repo.upsert_world_fact(
                book_id=book_id,
                fact_key=f"item:{name}:owner",
                fact_value=owner,
                confidence=0.75,
                source_chapter_idx=chapter_idx,
                source_excerpt=(description or owner)[:300],
            )
            world_fact_writes += 1

    return {
        "mutations_applied": {
            "plot_events_inserted": event_writes,
            "characters_upserted": character_writes,
            "items_upserted": item_writes,
            "world_facts_upserted": world_fact_writes,
        }
    }