from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.chapters.base import Chapter
from novel_summarizer.storage.types import ChapterRow, InsertResult


async def upsert_chapter(
    session: AsyncSession,
    book_id: int,
    idx: int,
    title: str,
    chapter_hash: str,
    start_pos: int,
    end_pos: int,
) -> InsertResult:
    stmt = (
        sqlite_insert(Chapter)
        .values(
            book_id=book_id,
            idx=idx,
            title=title,
            chapter_hash=chapter_hash,
            start_pos=start_pos,
            end_pos=end_pos,
        )
        .on_conflict_do_nothing(index_elements=[Chapter.chapter_hash])
    )
    result = await session.execute(stmt)
    inserted = result.rowcount == 1
    if inserted and result.lastrowid is not None:
        chapter_id = result.lastrowid
    else:
        id_result = await session.execute(select(Chapter.id).where(Chapter.chapter_hash == chapter_hash))
        chapter_id = id_result.scalar_one()
    return InsertResult(id=int(chapter_id), inserted=inserted)


async def list_chapters(session: AsyncSession, book_id: int) -> list[ChapterRow]:
    result = await session.execute(
        select(Chapter.id, Chapter.idx, Chapter.title)
        .where(Chapter.book_id == book_id)
        .order_by(Chapter.idx)
    )
    rows = result.all()
    return [ChapterRow(id=int(row[0]), idx=int(row[1]), title=str(row[2])) for row in rows]
