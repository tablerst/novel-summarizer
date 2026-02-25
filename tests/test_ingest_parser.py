from __future__ import annotations

from novel_summarizer.config.schema import IngestCleanupConfig
from novel_summarizer.ingest.parser import load_text_auto, normalize_text, parse_chapters


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


def test_load_text_auto_detects_utf8(tmp_path) -> None:
    content = "序章\n第1章 开始\n韩立登场。"
    file_path = tmp_path / "utf8_novel.txt"
    file_path.write_bytes(content.encode("utf-8"))

    result = load_text_auto(file_path, "auto", chapter_regex=r"^第[0-9]+章.*$")

    assert result.encoding in {"utf-8", "utf-8-sig"}
    assert "韩立" in result.text
    assert result.autodetected is True
    assert result.used_replace_fallback is False


def test_load_text_auto_detects_gb18030(tmp_path) -> None:
    content = "序章\n第一章山边小村\n韩立出门。"
    file_path = tmp_path / "gb_novel.txt"
    file_path.write_bytes(content.encode("gb18030"))

    result = load_text_auto(file_path, "auto", chapter_regex=r"^第[0-9一二三四五六七八九十百千]+章.*$")

    assert result.encoding == "gb18030"
    assert "韩立" in result.text
    assert result.autodetected is True
    assert result.used_replace_fallback is False
