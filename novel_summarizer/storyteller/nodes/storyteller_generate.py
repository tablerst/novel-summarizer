from __future__ import annotations

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storyteller.state import StorytellerState


def _draft_narration(text: str, ratio: tuple[float, float]) -> str:
    if not text:
        return ""
    target_len = max(1, int(len(text) * ratio[1]))
    return text[:target_len].strip()


async def run(state: StorytellerState, *, config: AppConfigRoot) -> dict:
    chapter_text = state.get("chapter_text", "")
    narration = _draft_narration(chapter_text, config.storyteller.narration_ratio)

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
    }