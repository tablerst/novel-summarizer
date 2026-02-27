from __future__ import annotations

import asyncio
from pathlib import Path

from novel_summarizer.export.markdown import (
    _export_storyteller_outputs,
    _render_book_summary,
    _render_legacy_characters,
    _render_storyteller_characters,
    _safe_filename,
    _render_story,
    _render_legacy_timeline,
    _render_storyteller_timeline,
    _safe_load_json,
)
from novel_summarizer.storage.types import CharacterRow, ChapterRow, ItemRow, NarrationRow, PlotEventRow, WorldFactRow


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
    rendered = _render_storyteller_characters([])
    assert "暂无人物数据" in rendered


def test_render_timeline_includes_chapter_and_impact() -> None:
    rendered = _render_legacy_timeline(
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


def test_render_characters_world_state_shape() -> None:
    rendered = _render_storyteller_characters(
        [
            {
                "canonical_name": "韩立",
                "aliases_json": '["韩跑跑"]',
                "relationships_json": "同门:厉飞雨",
                "motivation": "求长生",
                "status": "active",
            }
        ]
    )
    assert "韩立" in rendered
    assert "韩跑跑" in rendered
    assert "同门:厉飞雨" in rendered
    assert "active" in rendered


def test_render_timeline_supports_event_summary_field() -> None:
    rendered = _render_storyteller_timeline(
        [
            {"chapter_idx": 3, "event_summary": "秘境开启", "impact": "势力重排"},
        ]
    )
    assert "[第3章] 秘境开启（影响：势力重排）" in rendered


def test_render_legacy_characters_shape() -> None:
    rendered = _render_legacy_characters(
        [
            {
                "name": "韩立",
                "aliases": ["韩跑跑"],
                "relationships": "同门:厉飞雨",
                "motivation": "求长生",
                "changes": "境界突破",
            }
        ]
    )
    assert "韩立" in rendered
    assert "韩跑跑" in rendered
    assert "同门:厉飞雨" in rendered
    assert "境界突破" in rendered


def test_safe_filename_replaces_invalid_chars() -> None:
    assert _safe_filename('第1章: /危机?*') == "第1章___危机_"


def test_export_storyteller_outputs_uses_latest_narrations_only(tmp_path: Path) -> None:
    class _FakeRepo:
        def __init__(self) -> None:
            self.called_latest = False

        async def list_latest_narrations_by_book(self, book_id: int):
            _ = book_id
            self.called_latest = True
            return [
                NarrationRow(
                    id=2,
                    book_id=1,
                    chapter_id=101,
                    chapter_idx=1,
                    narration_text="第二版正文",
                    key_events_json=None,
                    prompt_version="v1",
                    model="m",
                    input_hash="h2",
                )
            ]

        async def list_narrations_by_book(self, book_id: int):  # pragma: no cover
            _ = book_id
            raise AssertionError("export should use latest narrations list")

        async def list_chapters(self, book_id: int):
            _ = book_id
            return [ChapterRow(id=101, idx=1, title="第1章 风起")]

        async def list_character_states(self, book_id: int):
            _ = book_id
            return [
                CharacterRow(
                    id=1,
                    book_id=1,
                    canonical_name="韩立",
                    aliases_json='["韩跑跑"]',
                    first_chapter_idx=1,
                    last_chapter_idx=1,
                    status="active",
                    location="天南",
                    abilities_json=None,
                    relationships_json=None,
                    motivation=None,
                    notes=None,
                )
            ]

        async def list_item_states(self, book_id: int):
            _ = book_id
            return [
                ItemRow(
                    id=1,
                    book_id=1,
                    name="掌天瓶",
                    owner_name="韩立",
                    first_chapter_idx=1,
                    last_chapter_idx=1,
                    description="神秘小瓶",
                    status="active",
                )
            ]

        async def list_plot_events_by_book(self, book_id: int):
            _ = book_id
            return [
                PlotEventRow(
                    id=1,
                    book_id=1,
                    chapter_idx=1,
                    event_summary="初入修仙界",
                    involved_characters_json='["韩立"]',
                    event_type="narration_draft",
                    impact="踏上修行",
                )
            ]

        async def list_world_facts(self, book_id: int, limit: int = 1000):
            _ = (book_id, limit)
            return [
                WorldFactRow(
                    id=1,
                    book_id=1,
                    fact_key="character:韩立:status",
                    fact_value="active",
                    confidence=0.9,
                    source_chapter_idx=1,
                    source_excerpt="韩立存活",
                )
            ]

    repo = _FakeRepo()
    result = asyncio.run(
        _export_storyteller_outputs(
            repo=repo,
            book_id=1,
            output_dir=tmp_path,
            book_title="测试书",
        )
    )

    assert repo.called_latest is True
    assert result.full_story_path is not None
    full_story = result.full_story_path.read_text(encoding="utf-8")
    assert full_story.count("# 第1章") == 1
    assert "第二版正文" in full_story

