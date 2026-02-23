from __future__ import annotations

STATE_MUTATION_PROMPT_VERSION = "v0-mvp"


def state_mutation_prompt(language: str) -> tuple[str, str]:
    system = "You are a strict state mutation extractor. Return JSON only."
    user = (
        f"Language: {language}\n"
        'Return JSON: {"character_mutations": [], "item_mutations": [], "new_facts": []}'
    )
    return system, user