"""Chapter storage models and CRUD helpers."""

from novel_summarizer.storage.chapters.base import Chapter
from novel_summarizer.storage.chapters import crud

__all__ = ["Chapter", "crud"]
