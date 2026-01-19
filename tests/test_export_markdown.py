from __future__ import annotations

from novel_summarizer.export.markdown import (
    _render_book_summary,
    _render_characters,
    _render_story,
    _render_timeline,
    _safe_load_json,
)


def test_safe_load_json_extracts_embedded_object() -> None:
    payload = "prefix {\"summary\": \"ok\"} suffix"
    assert _safe_load_json(payload) == {"summary": "ok"}


def test_safe_load_json_handles_list() -> None:
    payload = "[1, 2, 3]"
    assert _safe_load_json(payload) == [1, 2, 3]


def test_render_book_summary_default_title() -> None:
    rendered = _render_book_summary(None, "内容")
    assert rendered.startswith("# (未命名)")


def test_render_characters_empty() -> None:
    rendered = _render_characters([])
    assert "暂无人物数据" in rendered


def test_render_timeline_includes_chapter_and_impact() -> None:
    rendered = _render_timeline(
        [
            {"chapter_idx": 2, "event": "冲突爆发", "impact": "主角受伤"},
            {"event": "小结"},
        ]
    )

    assert "[第2章] 冲突爆发（影响：主角受伤）" in rendered
    assert "2. 小结" in rendered


def test_render_story_empty() -> None:
    rendered = _render_story("")
    assert "暂无说书稿数据" in rendered
