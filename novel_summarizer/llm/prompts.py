from __future__ import annotations

CHUNK_PROMPT_VERSION = "v1"
CHAPTER_PROMPT_VERSION = "v2"
BOOK_PROMPT_VERSION = "v2"
STORY_PROMPT_VERSION = "v1"


def chunk_summary_prompts(
    *,
    language: str,
    style: str,
    include_quotes: bool,
    with_citations: bool,
) -> tuple[str, str]:
    system = (
        "你是一个严谨的长文本总结器。只输出严格有效的 JSON，不要输出 markdown。"
        "不要添加任何额外说明。"
    )

    quotes_instruction = "如果文本里有关键对话可简要引用。" if include_quotes else "不要引用原文句子。"
    citations_instruction = (
        "输出字段中包含 citations，但此阶段没有证据检索，请输出空数组。"
        if with_citations
        else "不要输出 citations 字段。"
    )

    user = (
        "请基于以下小说片段输出 JSON：\n"
        "字段要求：\n"
        "- summary_points: 字符串数组，提炼关键要点\n"
        "- events: 事件数组，每项包含 who/what/where/outcome（字符串即可）\n"
        "- characters_mentions: 主要人物姓名数组\n"
        "- open_questions: 不确定或需要回溯的问题数组\n"
        f"{citations_instruction}\n"
        f"语言：{language}；风格：{style}。{quotes_instruction}\n"
        "请严格输出 JSON。\n\n"
        "<chunk>\n"
        "{chunk}\n"
        "</chunk>\n"
    )
    return system, user


def chapter_summary_prompts(
    *,
    language: str,
    style: str,
    word_range: tuple[int, int],
    with_citations: bool,
    evidence: str | None = None,
) -> tuple[str, str]:
    system = (
        "你是一个严谨的章节总结器。只输出严格有效的 JSON，不要输出 markdown。"
        "不要添加任何额外说明。"
    )

    citations_instruction = (
        "输出字段中包含 citations，数组元素包含 chunk_id 与 quote（可选）。"
        if with_citations
        else "不要输出 citations 字段。"
    )

    evidence_block = ""
    if with_citations and evidence:
        evidence_block = (
            "证据片段如下，请只基于证据输出 citations：\n"
            f"{evidence}\n"
        )
    elif with_citations:
        evidence_block = "当前未检索到证据片段，citations 置空。\n"

    user = (
        "请基于以下 chunk 级摘要 JSON 列表，生成章节级 JSON：\n"
        "字段要求：\n"
        "- summary: 章节总结（自然语言，控制在指定字数范围）\n"
        "- events: 事件数组（延续并合并要点）\n"
        "- characters: 主要人物数组\n"
        "- open_questions: 未解决的问题数组\n"
        f"{citations_instruction}\n"
        f"语言：{language}；风格：{style}；字数范围：{word_range[0]}~{word_range[1]}。\n"
        f"{evidence_block}"
        "请严格输出 JSON。\n\n"
        "<chunk_summaries_json>\n"
        "{chunk_summaries}\n"
        "</chunk_summaries_json>\n"
    )
    return system, user


def book_summary_prompts(
    *,
    language: str,
    style: str,
    word_range: tuple[int, int],
    with_citations: bool,
    evidence: str | None = None,
) -> tuple[str, str]:
    system = (
        "你是一个严谨的全书总结器。只输出严格有效的 JSON，不要输出 markdown。"
        "不要添加任何额外说明。"
    )

    citations_instruction = (
        "输出字段中包含 citations，数组元素包含 chunk_id 与 quote（可选）。"
        if with_citations
        else "不要输出 citations 字段。"
    )

    evidence_block = ""
    if with_citations and evidence:
        evidence_block = (
            "证据片段如下，请只基于证据输出 citations：\n"
            f"{evidence}\n"
        )
    elif with_citations:
        evidence_block = "当前未检索到证据片段，citations 置空。\n"

    user = (
        "请基于以下章节摘要 JSON 列表，生成全书级 JSON：\n"
        "字段要求：\n"
        "- summary: 全书总结（自然语言）\n"
        "- characters: 人物数组（每项包含 name, aliases[], relationships, motivation, changes）\n"
        "- timeline: 事件数组（每项包含 chapter_idx, event, impact）\n"
        "- themes: 主题数组（字符串）\n"
        f"{citations_instruction}\n"
        f"语言：{language}；风格：{style}；字数范围：{word_range[0]}~{word_range[1]}。\n"
        f"{evidence_block}"
        "请严格输出 JSON。\n\n"
        "<chapter_summaries_json>\n"
        "{chapter_summaries}\n"
        "</chapter_summaries_json>\n"
    )
    return system, user


def story_summary_prompts(
    *,
    language: str,
    style: str,
    word_range: tuple[int, int],
    with_citations: bool,
    evidence: str | None = None,
) -> tuple[str, str]:
    system = (
        "你是一个说书人口吻的长篇叙述生成器。只输出严格有效的 JSON，不要输出 markdown。"
        "不要添加任何额外说明。"
    )

    citations_instruction = (
        "输出字段中包含 citations，数组元素包含 chunk_id 与 quote（可选）。"
        if with_citations
        else "不要输出 citations 字段。"
    )

    evidence_block = ""
    if with_citations and evidence:
        evidence_block = (
            "证据片段如下，请只基于证据输出 citations：\n"
            f"{evidence}\n"
        )
    elif with_citations:
        evidence_block = "当前未检索到证据片段，citations 置空。\n"

    user = (
        "请基于以下章节摘要 JSON 列表，生成说书人风格的连贯叙事稿：\n"
        "字段要求：\n"
        "- story: 连贯叙述，按剧情推进，分段输出，不要写列表或小标题\n"
        f"{citations_instruction}\n"
        f"语言：{language}；风格：{style}；字数范围：{word_range[0]}~{word_range[1]}。\n"
        f"{evidence_block}"
        "请严格输出 JSON。\n\n"
        "<chapter_summaries_json>\n"
        "{chapter_summaries}\n"
        "</chapter_summaries_json>\n"
    )
    return system, user
