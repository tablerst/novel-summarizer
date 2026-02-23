from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.narrations.base import Narration
from novel_summarizer.storage.types import InsertResult, NarrationRow, SearchHitRow


def _to_row(row: tuple) -> NarrationRow:
    return NarrationRow(
        id=int(row[0]),
        book_id=int(row[1]),
        chapter_id=int(row[2]),
        chapter_idx=int(row[3]),
        narration_text=str(row[4]),
        key_events_json=row[5],
        prompt_version=str(row[6]),
        model=str(row[7]),
        input_hash=str(row[8]),
    )


async def get_narration(
    session: AsyncSession,
    chapter_id: int,
    prompt_version: str,
    model: str,
    input_hash: str,
) -> NarrationRow | None:
    result = await session.execute(
        select(
            Narration.id,
            Narration.book_id,
            Narration.chapter_id,
            Narration.chapter_idx,
            Narration.narration_text,
            Narration.key_events_json,
            Narration.prompt_version,
            Narration.model,
            Narration.input_hash,
        ).where(
            Narration.chapter_id == chapter_id,
            Narration.prompt_version == prompt_version,
            Narration.model == model,
            Narration.input_hash == input_hash,
        )
    )
    row = result.first()
    return _to_row(row) if row else None


async def get_latest_narration(session: AsyncSession, chapter_id: int) -> NarrationRow | None:
    result = await session.execute(
        select(
            Narration.id,
            Narration.book_id,
            Narration.chapter_id,
            Narration.chapter_idx,
            Narration.narration_text,
            Narration.key_events_json,
            Narration.prompt_version,
            Narration.model,
            Narration.input_hash,
        )
        .where(Narration.chapter_id == chapter_id)
        .order_by(desc(Narration.created_at))
        .limit(1)
    )
    row = result.first()
    return _to_row(row) if row else None


async def list_narrations_by_book(session: AsyncSession, book_id: int) -> list[NarrationRow]:
    result = await session.execute(
        select(
            Narration.id,
            Narration.book_id,
            Narration.chapter_id,
            Narration.chapter_idx,
            Narration.narration_text,
            Narration.key_events_json,
            Narration.prompt_version,
            Narration.model,
            Narration.input_hash,
        )
        .where(Narration.book_id == book_id)
        .order_by(Narration.chapter_idx)
    )
    rows = result.all()
    return [_to_row(row) for row in rows]


async def upsert_narration(
    session: AsyncSession,
    book_id: int,
    chapter_id: int,
    chapter_idx: int,
    narration_text: str,
    key_events_json: str | None,
    prompt_version: str,
    model: str,
    input_hash: str,
) -> InsertResult:
    stmt = (
        sqlite_insert(Narration)
        .values(
            book_id=book_id,
            chapter_id=chapter_id,
            chapter_idx=chapter_idx,
            narration_text=narration_text,
            key_events_json=key_events_json,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
        )
        .on_conflict_do_nothing(
            index_elements=[
                Narration.chapter_id,
                Narration.prompt_version,
                Narration.model,
                Narration.input_hash,
            ]
        )
    )
    result = await session.execute(stmt)
    inserted = result.rowcount == 1
    if inserted and result.lastrowid is not None:
        narration_id = result.lastrowid
    else:
        id_result = await session.execute(
            select(Narration.id).where(
                Narration.chapter_id == chapter_id,
                Narration.prompt_version == prompt_version,
                Narration.model == model,
                Narration.input_hash == input_hash,
            )
        )
        narration_id = id_result.scalar_one()
    return InsertResult(id=int(narration_id), inserted=inserted)


async def rebuild_narrations_fts_for_book(session: AsyncSession, book_id: int) -> int:
    await session.execute(sa_text("DELETE FROM narrations_fts WHERE book_id = :book_id"), {"book_id": str(book_id)})
    await session.execute(
        sa_text(
            """
            INSERT INTO narrations_fts (narration_id, book_id, chapter_idx, chapter_title, text)
            SELECT n.id, n.book_id, n.chapter_idx, ch.title, n.narration_text
            FROM narrations n
            JOIN chapters ch ON ch.id = n.chapter_id
            WHERE n.book_id = :book_id
            """
        ),
        {"book_id": book_id},
    )
    count_result = await session.execute(
        sa_text("SELECT COUNT(*) FROM narrations_fts WHERE book_id = :book_id"), {"book_id": str(book_id)}
    )
    return int(count_result.scalar_one())


async def search_narrations_fts(
    session: AsyncSession,
    *,
    book_id: int,
    query: str,
    before_chapter_idx: int | None,
    limit: int,
) -> list[SearchHitRow]:
    if limit <= 0 or not query.strip():
        return []

    result = await session.execute(
        sa_text(
            """
            SELECT
                CAST(narration_id AS INTEGER) AS source_id,
                CAST(chapter_idx AS INTEGER) AS chapter_idx,
                chapter_title,
                text,
                bm25(narrations_fts) AS score
            FROM narrations_fts
            WHERE narrations_fts MATCH :query
              AND book_id = :book_id
              AND (:before_chapter_idx IS NULL OR CAST(chapter_idx AS INTEGER) < :before_chapter_idx)
            ORDER BY score ASC
            LIMIT :limit
            """
        ),
        {
            "query": query,
            "book_id": str(book_id),
            "before_chapter_idx": before_chapter_idx,
            "limit": limit,
        },
    )
    rows = result.all()
    return [
        SearchHitRow(
            source_type="narration",
            source_id=int(row[0]),
            chapter_idx=int(row[1]),
            chapter_title=str(row[2] or ""),
            text=str(row[3] or ""),
            score=float(row[4]) if row[4] is not None else None,
        )
        for row in rows
    ]