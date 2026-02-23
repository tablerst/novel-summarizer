from __future__ import annotations

from novel_summarizer.storyteller.prompts.narration import narration_prompt


def test_narration_prompt_template_format_safe() -> None:
    _, template = narration_prompt(
        language="zh",
        style="说书人",
        narration_ratio=(0.4, 0.5),
        include_key_dialogue=True,
        include_inner_thoughts=True,
    )

    rendered = template.format(
        character_states="[]",
        item_states="[]",
        recent_events="[]",
        awakened_memories="[]",
        chapter_title="第一章",
        chapter_text="韩立登场。",
    )

    assert '"key_events": [{"who":"string"' in rendered
    assert "{chapter_text}" not in rendered
