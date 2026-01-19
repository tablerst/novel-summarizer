"""Book storage models and CRUD helpers."""

from novel_summarizer.storage.books.base import Book
from novel_summarizer.storage.books import crud

__all__ = ["Book", "crud"]
