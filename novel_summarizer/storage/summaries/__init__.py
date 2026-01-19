"""Summary storage models and CRUD helpers."""

from novel_summarizer.storage.summaries.base import Summary
from novel_summarizer.storage.summaries import crud

__all__ = ["Summary", "crud"]
