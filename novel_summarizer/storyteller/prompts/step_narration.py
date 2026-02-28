from __future__ import annotations

STEP_NARRATION_PROMPT_VERSION = "v1-step-aggregate"


def step_narration_prompt(*, language: str, style: str) -> tuple[str, str]:
    system = (
        "你是一位资深评书艺人/剧情解说作者。"
        "你的目标不是压缩，而是重写：在不偏离事实的前提下，对一个 step 范围进行整体重写。"
        "你将一次处理多个章节，但只能输出一个 step 级聚合结果，且遵守同一份世界观硬约束。"
        "只输出严格有效 JSON 对象，不要输出 markdown，不要输出解释。"
    )

    user = (
        f"语言：{language}\n"
        f"风格：{style}\n\n"
        "你会收到：\n"
        "- step 基准世界观状态（硬约束，所有章节共享）\n"
        "- 多个章节的原文与该章的唤醒前情（软约束）\n\n"
        "step 范围：第 {step_start} 章 到 第 {step_end} 章。\n"
        "请输出一个 step 级说书稿（不要逐章拆分输出）。\n\n"
        "step 基准世界观状态（硬约束，所有章节共享）：\n"
        "{base_world_state}\n\n"
        "chapters（用于汇总，不要引用 step 范围外未来信息）：\n"
        "{chapters}\n\n"
        "输出 JSON schema（单个对象）：\n"
        "  {{\n"
        '    "step_start_chapter_idx": 1,\n'
        '    "step_end_chapter_idx": 8,\n'
        '    "narration": "string",\n'
        '    "key_events": [{{"who":"string","what":"string","where":"string","outcome":"string","impact":"string"}}],\n'
        '    "character_updates": [{{"name":"string","change_type":"status|location|ability|relationship","before":"string","after":"string","evidence":"string"}}],\n'
        '    "new_items": [{{"name":"string","owner":"string","description":"string"}}]\n'
        "  }}\n"
    )
    return system, user
