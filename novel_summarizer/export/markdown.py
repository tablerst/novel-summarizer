from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

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
    mode: str = "legacy"
    chapters_dir: Path | None = None
    full_story_path: Path | None = None
    world_state_path: Path | None = None


def _safe_load_json(text: str) -> dict[str, Any] | list[Any]:
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return orjson.loads(text[start : end + 1])
        raise


def _safe_filename(text: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", text).strip()
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized or "untitled"


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            loaded = orjson.loads(stripped)
            if isinstance(loaded, list):
                return [str(item) for item in loaded if str(item).strip()]
        except orjson.JSONDecodeError:
            pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [str(value)]


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
        name = str(item.get("name") or item.get("canonical_name") or "")
        aliases = ", ".join(_coerce_list(item.get("aliases") or item.get("aliases_json")))
        relations = str(item.get("relationships") or item.get("relationships_json") or "")
        motivation = str(item.get("motivation") or "")
        changes = str(item.get("changes") or item.get("status") or "")
        lines.append(f"| {name} | {aliases} | {relations} | {motivation} | {changes} |")
    lines.append("")
    return "\n".join(lines)


def _render_timeline(events: list[dict[str, Any]]) -> str:
    if not events:
        return "# 时间线\n\n暂无事件数据。\n"

    lines = ["# 时间线", ""]
    for idx, event in enumerate(events, start=1):
        chapter_idx = event.get("chapter_idx")
        event_text = str(event.get("event") or event.get("event_summary") or "")
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


def _render_chapter_narration(chapter_idx: int, chapter_title: str, narration: str) -> str:
    return f"# 第{chapter_idx}章 {chapter_title}\n\n{narration}\n"


async def _export_storyteller_outputs(
    *,
    repo: SQLAlchemyRepo,
    book_id: int,
    output_dir: Path,
    book_title: str | None,
) -> ExportResult:
    narrations = await repo.list_narrations_by_book(book_id)
    chapters = await repo.list_chapters(book_id)
    chapter_title_map = {chapter.id: chapter.title for chapter in chapters}

    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    merged_parts: list[str] = []
    for row in narrations:
        chapter_title = chapter_title_map.get(row.chapter_id, f"第{row.chapter_idx}章")
        chapter_content = _render_chapter_narration(row.chapter_idx, chapter_title, row.narration_text)
        chapter_filename = f"{int(row.chapter_idx):03d}_{_safe_filename(chapter_title)}.md"
        (chapters_dir / chapter_filename).write_text(chapter_content, encoding="utf-8")
        merged_parts.append(chapter_content)

    full_story_text = "\n\n".join(merged_parts) if merged_parts else "# 全书说书稿\n\n暂无说书稿数据。\n"
    full_story_path = output_dir / "full_story.md"
    full_story_path.write_text(full_story_text, encoding="utf-8")

    story_path = output_dir / "story.md"
    story_path.write_text(_render_story(full_story_text if merged_parts else ""), encoding="utf-8")

    characters_rows = await repo.list_character_states(book_id)
    items_rows = await repo.list_item_states(book_id)
    plot_events_rows = await repo.list_plot_events_by_book(book_id)
    world_facts_rows = await repo.list_world_facts(book_id=book_id, limit=1000)

    characters = [asdict(row) for row in characters_rows]
    items = [asdict(row) for row in items_rows]
    events = [asdict(row) for row in plot_events_rows]
    world_facts = [asdict(row) for row in world_facts_rows]

    characters_path = output_dir / "characters.md"
    timeline_path = output_dir / "timeline.md"
    book_summary_path = output_dir / "book_summary.md"
    world_state_path = output_dir / "world_state.json"

    characters_path.write_text(_render_characters(characters), encoding="utf-8")
    timeline_path.write_text(_render_timeline(events), encoding="utf-8")
    summary_text = f"共导出 {len(narrations)} 章说书稿。"
    book_summary_path.write_text(_render_book_summary(book_title, summary_text), encoding="utf-8")
    world_state_path.write_text(
        orjson.dumps(
            {
                "book_id": book_id,
                "book_title": book_title,
                "narrations": [asdict(row) for row in narrations],
                "characters": characters,
                "items": items,
                "plot_events": events,
                "world_facts": world_facts,
            },
            option=orjson.OPT_INDENT_2,
        ).decode("utf-8"),
        encoding="utf-8",
    )

    return ExportResult(
        output_dir=output_dir,
        book_summary_path=book_summary_path,
        characters_path=characters_path,
        timeline_path=timeline_path,
        story_path=story_path,
        mode="storyteller",
        chapters_dir=chapters_dir,
        full_story_path=full_story_path,
        world_state_path=world_state_path,
    )


async def _export_legacy_outputs(*, repo: SQLAlchemyRepo, book_id: int, output_dir: Path, book_title: str | None) -> ExportResult:
    book_summary_row = await repo.get_latest_summary("book", book_id, "book_summary")
    characters_row = await repo.get_latest_summary("book", book_id, "characters")
    timeline_row = await repo.get_latest_summary("book", book_id, "timeline")
    story_row = await repo.get_latest_summary("book", book_id, "story")

    if not book_summary_row:
        raise ValueError("No storyteller narrations or legacy book summary found. Run storytell or summarize first.")

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

    book_summary_path.write_text(_render_book_summary(book_title, summary_text), encoding="utf-8")
    characters_path.write_text(_render_characters(characters), encoding="utf-8")
    timeline_path.write_text(_render_timeline(events), encoding="utf-8")
    story_path.write_text(_render_story(story_text), encoding="utf-8")

    return ExportResult(
        output_dir=output_dir,
        book_summary_path=book_summary_path,
        characters_path=characters_path,
        timeline_path=timeline_path,
        story_path=story_path,
        mode="legacy",
    )


async def export_book_markdown(book_id: int, config: AppConfigRoot) -> ExportResult:
    output_root = config.app.output_dir
    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        book = await repo.get_book(book_id)
        output_dir = (output_root / book.book_hash).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        narrations = await repo.list_narrations_by_book(book_id)
        if narrations:
            return await _export_storyteller_outputs(
                repo=repo,
                book_id=book_id,
                output_dir=output_dir,
                book_title=book.title,
            )

        return await _export_legacy_outputs(
            repo=repo,
            book_id=book_id,
            output_dir=output_dir,
            book_title=book.title,
        )
