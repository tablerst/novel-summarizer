from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from novel_summarizer.config.schema import IngestCleanupConfig


@dataclass
class Chapter:
    idx: int
    title: str
    text: str
    start_pos: int
    end_pos: int


def load_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="replace")


def normalize_text(text: str, cleanup: IngestCleanupConfig) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if cleanup.normalize_fullwidth:
        normalized = unicodedata.normalize("NFKC", normalized)
    if cleanup.strip_blank_lines:
        lines = [line.rstrip() for line in normalized.split("\n") if line.strip()]
        normalized = "\n".join(lines)
    return normalized.strip()


def _fallback_split(text: str, max_chars: int) -> list[Chapter]:
    chapters: list[Chapter] = []
    if not text:
        return chapters
    length = len(text)
    idx = 1
    for start in range(0, length, max_chars):
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        title = f"第{idx}章"
        chapters.append(Chapter(idx=idx, title=title, text=chunk, start_pos=start, end_pos=end))
        idx += 1
    return chapters


def parse_chapters(text: str, chapter_regex: str | None, fallback_chapter_chars: int = 20000) -> list[Chapter]:
    if not text:
        return []

    if not chapter_regex:
        return _fallback_split(text, fallback_chapter_chars)

    pattern = re.compile(chapter_regex, re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return _fallback_split(text, fallback_chapter_chars)

    chapters: list[Chapter] = []
    idx = 1

    if matches[0].start() > 0:
        preface_text = text[: matches[0].start()].strip()
        if preface_text:
            chapters.append(
                Chapter(idx=idx, title="序章", text=preface_text, start_pos=0, end_pos=matches[0].start())
            )
            idx += 1

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        title = match.group(0).strip()

        block_lines = block.splitlines()
        if block_lines and block_lines[0].strip() == title:
            content = "\n".join(block_lines[1:]).strip()
        else:
            content = block

        if not content:
            content = block

        chapters.append(Chapter(idx=idx, title=title, text=content, start_pos=start, end_pos=end))
        idx += 1

    return chapters
