from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.books.base import Book
from novel_summarizer.storage.types import BookRow, InsertResult


async def get_or_create_book(
    session: AsyncSession,
    title: str | None,
    author: str | None,
    book_hash: str,
    source_path: str,
) -> InsertResult:
    stmt = (
        sqlite_insert(Book)
        .values(title=title, author=author, book_hash=book_hash, source_path=source_path)
        .on_conflict_do_nothing(index_elements=[Book.book_hash])
    )
    result = await session.execute(stmt)
    inserted = result.rowcount == 1
    if inserted and result.lastrowid is not None:
        book_id = result.lastrowid
    else:
        id_result = await session.execute(select(Book.id).where(Book.book_hash == book_hash))
        book_id = id_result.scalar_one()
    return InsertResult(id=int(book_id), inserted=inserted)


async def get_book(session: AsyncSession, book_id: int) -> BookRow:
    result = await session.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if book is None:
        raise ValueError("Book not found")
    return BookRow(
        id=int(book.id),
        title=book.title,
        author=book.author,
        book_hash=str(book.book_hash),
        source_path=book.source_path,
    )
