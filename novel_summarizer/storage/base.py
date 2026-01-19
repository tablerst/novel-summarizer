from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def import_all_models() -> None:
    from novel_summarizer.storage.books.base import Book
    from novel_summarizer.storage.chapters.base import Chapter
    from novel_summarizer.storage.chunks.base import Chunk
    from novel_summarizer.storage.summaries.base import Summary

    _ = (Book, Chapter, Chunk, Summary)
