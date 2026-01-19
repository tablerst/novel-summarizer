from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo


@dataclass
class ExportResult:
    output_dir: Path
    book_summary_path: Path
    characters_path: Path
    timeline_path: Path
    story_path: Path


def _safe_load_json(text: str) -> dict[str, Any] | list[Any]:
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return orjson.loads(text[start : end + 1])
        raise


def _render_book_summary(book_title: str | None, summary: str) -> str:
    title = book_title or "(未命名)"
    return f"# {title}\n\n{summary}\n"


def _render_story(story: str | None) -> str:
    if not story:
        return "# 说书稿\n\n暂无说书稿数据。\n"
    return f"# 说书稿\n\n{story}\n"


def _render_characters(characters: list[dict[str, Any]]) -> str:
    if not characters:
        return "# 人物表\n\n暂无人物数据。\n"

    lines = [
        "# 人物表",
        "",
        "| 姓名 | 别名 | 关系 | 动机/目标 | 变化 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in characters:
        name = str(item.get("name", ""))
        aliases = ", ".join(item.get("aliases", []) or [])
        relations = str(item.get("relationships", ""))
        motivation = str(item.get("motivation", ""))
        changes = str(item.get("changes", ""))
        lines.append(f"| {name} | {aliases} | {relations} | {motivation} | {changes} |")
    lines.append("")
    return "\n".join(lines)


def _render_timeline(events: list[dict[str, Any]]) -> str:
    if not events:
        return "# 时间线\n\n暂无事件数据。\n"

    lines = ["# 时间线", ""]
    for idx, event in enumerate(events, start=1):
        chapter_idx = event.get("chapter_idx")
        event_text = str(event.get("event", ""))
        impact = str(event.get("impact", ""))
        prefix = f"{idx}. "
        if chapter_idx is not None:
            prefix += f"[第{chapter_idx}章] "
        line = f"{prefix}{event_text}"
        if impact:
            line += f"（影响：{impact}）"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


async def export_book_markdown(book_id: int, config: AppConfigRoot) -> ExportResult:
    output_root = config.app.output_dir
    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        book = await repo.get_book(book_id)
        output_dir = (output_root / book.book_hash).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        book_summary_row = await repo.get_latest_summary("book", book_id, "book_summary")
        characters_row = await repo.get_latest_summary("book", book_id, "characters")
        timeline_row = await repo.get_latest_summary("book", book_id, "timeline")
        story_row = await repo.get_latest_summary("book", book_id, "story")

        if not book_summary_row:
            raise ValueError("Book summary not found. Run summarize first.")

        book_summary_obj = _safe_load_json(book_summary_row.content)
        if isinstance(book_summary_obj, dict):
            summary_text = str(book_summary_obj.get("summary", book_summary_row.content))
        else:
            summary_text = book_summary_row.content

        characters: list[dict[str, Any]] = []
        if characters_row:
            characters_obj = _safe_load_json(characters_row.content)
            if isinstance(characters_obj, dict):
                characters = characters_obj.get("characters", []) or []

        events: list[dict[str, Any]] = []
        if timeline_row:
            timeline_obj = _safe_load_json(timeline_row.content)
            if isinstance(timeline_obj, dict):
                events = timeline_obj.get("events", []) or []

        story_text = ""
        if story_row:
            story_obj = _safe_load_json(story_row.content)
            if isinstance(story_obj, dict):
                story_text = str(story_obj.get("story", story_row.content))
            else:
                story_text = str(story_obj)

    book_summary_path = output_dir / "book_summary.md"
    characters_path = output_dir / "characters.md"
    timeline_path = output_dir / "timeline.md"
    story_path = output_dir / "story.md"

    book_summary_path.write_text(_render_book_summary(book.title, summary_text), encoding="utf-8")
    characters_path.write_text(_render_characters(characters), encoding="utf-8")
    timeline_path.write_text(_render_timeline(events), encoding="utf-8")
    story_path.write_text(_render_story(story_text), encoding="utf-8")

    return ExportResult(
        output_dir=output_dir,
        book_summary_path=book_summary_path,
        characters_path=characters_path,
        timeline_path=timeline_path,
        story_path=story_path,
    )
