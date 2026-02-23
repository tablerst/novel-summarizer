from __future__ import annotations

import re

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState


_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,8}")


def _unique(values: list[str], max_items: int = 20) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= max_items:
            break
    return output


async def run(state: StorytellerState, *, config: AppConfigRoot) -> dict:
    _ = config
    text = state.get("chapter_text", "")
    tokens = _CJK_TOKEN_PATTERN.findall(text)

    return {
        "entities_mentioned": _unique(tokens, max_items=16),
        "locations_mentioned": [],
        "items_mentioned": [],
    }