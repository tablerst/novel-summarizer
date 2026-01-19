from __future__ import annotations

from novel_summarizer.domain.hashing import book_hash, chapter_hash, chunk_hash, sha256_text


def test_sha256_text_deterministic() -> None:
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")
    assert len(sha256_text("hello")) == 64


def test_domain_hashes_are_distinct() -> None:
    book = book_hash("normalized")
    chapter = chapter_hash(book, "title", "text")
    chunk = chunk_hash(chapter, "chunk", "params")

    assert book != chapter
    assert chapter != chunk
