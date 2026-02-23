from __future__ import annotations

ENTITY_PROMPT_VERSION = "v0-mvp"


def entity_prompt(language: str) -> tuple[str, str]:
    system = (
        "你是一个严谨的命名实体抽取器。"
        "只输出严格有效 JSON，不要输出 markdown，不要输出解释。"
    )
    user = (
        f"语言：{language}\n"
        "请从以下章节文本中提取：人物、地点、道具/法宝、关键术语。\n"
        "同义词或别名请保留原文写法，不要臆造。\n"
        "输出字段要求：\n"
        '- characters: string[]\n'
        '- locations: string[]\n'
        '- items: string[]\n'
        '- key_phrases: string[]\n'
        '仅输出 JSON：{{"characters": [], "locations": [], "items": [], "key_phrases": []}}\n\n'
        "<chapter_text>\n"
        "{chapter_text}\n"
        "</chapter_text>\n"
    )
    return system, user