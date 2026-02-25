from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import NamedTuple

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


class TextLoadResult(NamedTuple):
    text: str
    encoding: str
    autodetected: bool
    confidence: float
    used_replace_fallback: bool


_AUTO_CANDIDATE_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "big5",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
)


def _is_cjk(codepoint: int) -> bool:
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
    )


def _is_cjk_punctuation(codepoint: int) -> bool:
    return 0x3000 <= codepoint <= 0x303F or 0xFF00 <= codepoint <= 0xFFEF


def _is_expected_text_char(ch: str) -> bool:
    if ch in "\n\r\t":
        return True
    if ch.isascii() and ch.isprintable():
        return True
    codepoint = ord(ch)
    if _is_cjk(codepoint) or _is_cjk_punctuation(codepoint):
        return True
    return False


def _score_decoded_text(text: str, chapter_regex: str | None) -> float:
    if not text:
        return -1e9

    sample = text[:120000]
    total = len(sample)
    expected_count = 0
    cjk_count = 0
    control_count = 0

    for ch in sample:
        if _is_expected_text_char(ch):
            expected_count += 1
            if _is_cjk(ord(ch)):
                cjk_count += 1
            continue
        if not ch.isprintable():
            control_count += 1

    expected_ratio = expected_count / total
    cjk_ratio = cjk_count / total
    control_ratio = control_count / total

    chapter_hits = 0
    pattern = chapter_regex or r"^第[0-9一二三四五六七八九十百千]+章.*$"
    try:
        chapter_hits = len(re.findall(pattern, sample, flags=re.MULTILINE))
    except re.error:
        chapter_hits = 0

    return expected_ratio * 100 + cjk_ratio * 20 + min(chapter_hits, 300) * 0.5 - control_ratio * 200


def load_text_auto(path: Path, encoding: str, chapter_regex: str | None = None) -> TextLoadResult:
    normalized_encoding = (encoding or "auto").strip().lower()
    if normalized_encoding != "auto":
        text = load_text(path, encoding)
        return TextLoadResult(
            text=text,
            encoding=encoding,
            autodetected=False,
            confidence=1.0,
            used_replace_fallback=("\ufffd" in text),
        )

    raw_bytes = path.read_bytes()
    candidates: list[tuple[float, str, str]] = []
    for candidate in _AUTO_CANDIDATE_ENCODINGS:
        try:
            decoded = raw_bytes.decode(candidate, errors="strict")
        except UnicodeDecodeError:
            continue
        score = _score_decoded_text(decoded, chapter_regex)
        candidates.append((score, candidate, decoded))

    if not candidates:
        fallback_text = raw_bytes.decode("utf-8", errors="replace")
        return TextLoadResult(
            text=fallback_text,
            encoding="utf-8",
            autodetected=True,
            confidence=0.0,
            used_replace_fallback=True,
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_encoding, best_text = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else best_score
    confidence = 1.0 if len(candidates) == 1 else max(0.0, min(1.0, (best_score - second_score) / 30.0))

    return TextLoadResult(
        text=best_text,
        encoding=best_encoding,
        autodetected=True,
        confidence=confidence,
        used_replace_fallback=False,
    )


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
