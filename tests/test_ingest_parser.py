from __future__ import annotations

from novel_summarizer.config.schema import IngestCleanupConfig
from novel_summarizer.ingest.parser import normalize_text, parse_chapters


def test_normalize_text_strips_blank_lines_and_fullwidth() -> None:
    cleanup = IngestCleanupConfig(strip_blank_lines=True, normalize_fullwidth=True)
    text = "ＡＢＣ\r\n\r\n　\n１２３"

    normalized = normalize_text(text, cleanup)

    assert normalized == "ABC\n123"


def test_parse_chapters_with_regex_and_preface() -> None:
    text = "序言内容\n第1章 开始\n内容一\n第2章 继续\n内容二"
    chapters = parse_chapters(text, r"^第[0-9]+章.*$")

    assert len(chapters) == 3
    assert chapters[0].title == "序章"
    assert chapters[0].text == "序言内容"
    assert chapters[1].title == "第1章 开始"
    assert chapters[1].text == "内容一"
    assert chapters[2].title == "第2章 继续"
    assert chapters[2].text == "内容二"


def test_parse_chapters_fallback_split() -> None:
    text = "abcdefghij"
    chapters = parse_chapters(text, None, fallback_chapter_chars=4)

    assert len(chapters) == 3
    assert chapters[0].title == "第1章"
    assert chapters[1].title == "第2章"
    assert chapters[2].title == "第3章"
