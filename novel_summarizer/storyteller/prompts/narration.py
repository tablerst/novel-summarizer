from __future__ import annotations

NARRATION_PROMPT_VERSION = "v0-mvp"


def narration_prompt(
    *,
    language: str,
    style: str,
    narration_ratio: tuple[float, float],
    include_key_dialogue: bool,
    include_inner_thoughts: bool,
) -> tuple[str, str]:
    system = (
        "你是一位资深评书艺人/剧情解说作者。"
        "你的目标不是压缩，而是重写：在不偏离事实的前提下，用强叙事表达重写本章。"
        "只输出严格有效 JSON，不要输出 markdown，不要输出解释。"
    )

    dialogue_rule = "保留关键对白。" if include_key_dialogue else "尽量少引用对白。"
    thought_rule = "保留人物关键心理活动。" if include_inner_thoughts else "心理活动仅保留必要信息。"
    ratio_text = f"{narration_ratio[0]:.2f}~{narration_ratio[1]:.2f}"

    user = (
        f"语言：{language}\n"
        f"风格：{style}\n"
        f"篇幅比例目标（相对原文）：{ratio_text}\n"
        f"额外要求：{dialogue_rule}{thought_rule}\n\n"
        "请综合以下上下文生成本章说书稿：\n"
        "1) 世界观状态（硬约束，优先级最高）\n"
        "{character_states}\n\n"
        "2) 道具状态\n"
        "{item_states}\n\n"
        "3) 最近关键事件\n"
        "{recent_events}\n\n"
        "4) 被唤醒的前情记忆\n"
        "{awakened_memories}\n\n"
        "5) 本章原文\n"
        "标题：{chapter_title}\n"
        "{chapter_text}\n\n"
        "输出 JSON schema：\n"
        "{{\n"
        '  "narration": "string",\n'
        '  "key_events": [{{"who":"string","what":"string","where":"string","outcome":"string","impact":"string"}}],\n'
        '  "character_updates": [{{"name":"string","change_type":"status|location|ability|relationship","before":"string","after":"string","evidence":"string"}}],\n'
        '  "new_items": [{{"name":"string","owner":"string","description":"string"}}]\n'
        "}}\n"
    )
    return system, user