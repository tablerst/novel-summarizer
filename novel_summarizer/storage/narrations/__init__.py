"""Narration storage models and CRUD helpers."""

from novel_summarizer.storage.narrations.base import Narration
from novel_summarizer.storage.narrations import crud

__all__ = ["Narration", "crud"]