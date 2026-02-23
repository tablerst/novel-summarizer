from __future__ import annotations

import re
from typing import Any

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{2,20}")


def _normalize(text: Any) -> str:
    return str(text or "").strip()


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text))


def _claim_text_from_event(event: dict[str, Any]) -> str:
    parts = [
        _normalize(event.get("who")),
        _normalize(event.get("what")),
        _normalize(event.get("where")),
        _normalize(event.get("outcome")),
        _normalize(event.get("impact")),
    ]
    return " ".join(part for part in parts if part)


def _claim_text_from_update(update: dict[str, Any]) -> str:
    parts = [
        _normalize(update.get("name")),
        _normalize(update.get("change_type")),
        _normalize(update.get("after")),
        _normalize(update.get("evidence")),
    ]
    return " ".join(part for part in parts if part)


def _claim_text_from_item(item: dict[str, Any]) -> str:
    parts = [
        _normalize(item.get("name")),
        _normalize(item.get("owner")),
        _normalize(item.get("description")),
    ]
    return " ".join(part for part in parts if part)


def _source_snippet(text: str, length: int = 120) -> str:
    return text[:length].strip()


def _best_support_score(
    claim_text: str,
    sources: list[dict[str, str]],
    key_phrases: list[str] | None = None,
) -> tuple[float, str, str]:
    if not claim_text:
        return 0.0, "", ""

    claim_tokens = _tokens(claim_text)
    best_score = 0.0
    best_source_type = ""
    best_snippet = ""

    for source in sources:
        source_text = source["text"]
        if not source_text:
            continue

        for phrase in key_phrases or []:
            phrase = phrase.strip()
            if phrase and phrase in source_text:
                return 1.0, source["source_type"], _source_snippet(source_text)

        if claim_text and claim_text in source_text:
            return 1.0, source["source_type"], _source_snippet(source_text)

        source_tokens = _tokens(source_text)
        if not claim_tokens:
            score = 0.0
        else:
            overlap = len(claim_tokens.intersection(source_tokens))
            score = overlap / max(len(claim_tokens), 1)

        if score > best_score:
            best_score = score
            best_source_type = source["source_type"]
            best_snippet = _source_snippet(source_text)

    return best_score, best_source_type, best_snippet


def _build_sources(state: StorytellerState, max_snippets: int) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    chapter_text = _normalize(state.get("chapter_text"))
    if chapter_text:
        sources.append({"source_type": "chapter", "text": chapter_text})

    for item in (state.get("awakened_memories") or [])[:max_snippets]:
        memory_text = _normalize(item.get("text"))
        if not memory_text:
            continue
        source_type = _normalize(item.get("source_type")) or "memory"
        sources.append({"source_type": source_type, "text": memory_text})
    return sources


async def run(state: StorytellerState, *, config: AppConfigRoot) -> dict:
    min_score = float(config.storyteller.evidence_min_support_score)
    sources = _build_sources(state, max_snippets=int(config.storyteller.evidence_max_snippets))

    warnings = list(state.get("consistency_warnings") or [])
    actions = list(state.get("consistency_actions") or [])

    supported_events: list[dict[str, Any]] = []
    supported_updates: list[dict[str, Any]] = []
    supported_items: list[dict[str, Any]] = []

    total_claims = 0
    supported_claims = 0

    for event in state.get("key_events", []):
        total_claims += 1
        claim_text = _claim_text_from_event(event)
        key_phrases = [
            _normalize(event.get("what")),
        ]
        score, source_type, snippet = _best_support_score(claim_text, sources, key_phrases=key_phrases)
        if score < min_score:
            warnings.append(f"Evidence rejected key_event: {event.get('what', '')}")
            continue
        supported_claims += 1
        enriched = dict(event)
        enriched["evidence_source_type"] = source_type
        enriched["evidence_quote"] = snippet
        enriched["evidence_score"] = round(score, 4)
        supported_events.append(enriched)

    for update in state.get("character_updates", []):
        total_claims += 1
        claim_text = _claim_text_from_update(update)
        key_phrases = [
            _normalize(update.get("evidence")),
            _normalize(update.get("after")),
        ]
        score, source_type, snippet = _best_support_score(claim_text, sources, key_phrases=key_phrases)
        if score < min_score:
            warnings.append(f"Evidence rejected character_update: {update.get('name', '')}")
            continue
        supported_claims += 1
        enriched = dict(update)
        enriched["evidence_source_type"] = source_type
        enriched["evidence_quote"] = snippet
        enriched["evidence_score"] = round(score, 4)
        supported_updates.append(enriched)

    for item in state.get("new_items", []):
        total_claims += 1
        claim_text = _claim_text_from_item(item)
        key_phrases = [
            _normalize(item.get("name")),
            _normalize(item.get("description")),
            _normalize(item.get("owner")),
        ]
        score, source_type, snippet = _best_support_score(claim_text, sources, key_phrases=key_phrases)
        if score < min_score:
            warnings.append(f"Evidence rejected new_item: {item.get('name', '')}")
            continue
        supported_claims += 1
        enriched = dict(item)
        enriched["evidence_source_type"] = source_type
        enriched["evidence_quote"] = snippet
        enriched["evidence_score"] = round(score, 4)
        supported_items.append(enriched)

    unsupported_claims = max(0, total_claims - supported_claims)
    if unsupported_claims > 0:
        actions.append(f"Evidence filtered unsupported claims: {unsupported_claims}")

    return {
        "key_events": supported_events,
        "character_updates": supported_updates,
        "new_items": supported_items,
        "consistency_warnings": warnings,
        "consistency_actions": actions,
        "evidence_report": {
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
        },
    }