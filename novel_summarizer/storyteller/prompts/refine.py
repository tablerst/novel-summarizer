from __future__ import annotations

REFINE_PROMPT_VERSION = "v0-refine"


def refine_prompt(*, language: str, style: str) -> tuple[str, str]:
    system = (
        "你是一位小说叙事润色编辑。"
        "请在不改变事实的前提下，优化叙事连贯性、节奏和文风统一性。"
        "只输出严格 JSON，不要输出 markdown。"
    )
    user = (
        f"语言：{language}\n"
        f"目标风格：{style}\n\n"
        "你会收到初稿和结构化约束，请仅做润色，不新增虚构事实。\n"
        "关键事件（不可丢失）：\n"
        "{key_events}\n\n"
        "人物更新（不可丢失）：\n"
        "{character_updates}\n\n"
        "初稿：\n"
        "{draft_narration}\n\n"
        "输出 JSON schema：\n"
        "{{\n"
        '  "narration": "string"\n'
        "}}\n"
    )
    return system, user