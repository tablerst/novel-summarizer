"""Deprecated: use per-table models in storage/<table>/base.py."""

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.books.base import Book
from novel_summarizer.storage.chapters.base import Chapter
from novel_summarizer.storage.chunks.base import Chunk
from novel_summarizer.storage.summaries.base import Summary

__all__ = ["Base", "Book", "Chapter", "Chunk", "Summary"]
