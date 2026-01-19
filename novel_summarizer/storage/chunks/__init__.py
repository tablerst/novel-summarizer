"""Chunk storage models and CRUD helpers."""

from novel_summarizer.storage.chunks.base import Chunk
from novel_summarizer.storage.chunks import crud

__all__ = ["Chunk", "crud"]
