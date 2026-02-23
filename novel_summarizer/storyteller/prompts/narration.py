from __future__ import annotations

NARRATION_PROMPT_VERSION = "v0-mvp"


def narration_prompt(language: str, style: str) -> tuple[str, str]:
    system = (
        "You are an expert storyteller. Rewrite each chapter with immersive narration while preserving "
        "core events and character dynamics. Return JSON only."
    )
    user = (
        f"Language: {language}\n"
        f"Style: {style}\n"
        'Return JSON: {"narration": "...", "key_events": [], "character_updates": [], "new_items": []}'
    )
    return system, user