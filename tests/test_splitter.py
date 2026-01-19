from __future__ import annotations

from novel_summarizer.ingest.splitter import split_text


def test_split_text_empty_returns_empty() -> None:
    assert split_text("", 4, 1, 2) == []


def test_split_text_short_text_single_chunk() -> None:
    chunks = split_text("abcd", 10, 2, 2)

    assert len(chunks) == 1
    assert chunks[0].text == "abcd"
    assert chunks[0].start_pos == 0
    assert chunks[0].end_pos == 4


def test_split_text_overlap_and_merge_min_chunk() -> None:
    chunks = split_text("abcdefghi", chunk_size_tokens=4, chunk_overlap_tokens=1, min_chunk_tokens=4)

    assert len(chunks) == 2
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "defgghi"
    assert chunks[1].start_pos == 3
    assert chunks[1].end_pos == 9
