from __future__ import annotations

ENTITY_PROMPT_VERSION = "v0-mvp"


def entity_prompt(language: str) -> tuple[str, str]:
    system = "You are a strict NER extractor for novel chapters. Always return JSON only."
    user = (
        f"Language: {language}\n"
        "Extract characters, locations, items, and key phrases from this chapter text.\n"
        'Return JSON: {"characters": [], "locations": [], "items": [], "key_phrases": []}'
    )
    return system, user