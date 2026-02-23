from __future__ import annotations

from sqlalchemy import select, text as sa_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.chunks.base import Chunk
from novel_summarizer.storage.types import ChunkRow, InsertResult, SearchHitRow


async def upsert_chunk(
    session: AsyncSession,
    chapter_id: int,
    idx: int,
    chunk_hash: str,
    text: str,
    token_count: int,
    start_pos: int,
    end_pos: int,
    meta_json: str | None = None,
) -> InsertResult:
    stmt = (
        sqlite_insert(Chunk)
        .values(
            chapter_id=chapter_id,
            idx=idx,
            chunk_hash=chunk_hash,
            text=text,
            token_count=token_count,
            start_pos=start_pos,
            end_pos=end_pos,
            meta_json=meta_json,
        )
        .on_conflict_do_nothing(index_elements=[Chunk.chunk_hash])
    )
    result = await session.execute(stmt)
    inserted = result.rowcount == 1
    if inserted and result.lastrowid is not None:
        chunk_id = result.lastrowid
    else:
        id_result = await session.execute(select(Chunk.id).where(Chunk.chunk_hash == chunk_hash))
        chunk_id = id_result.scalar_one()
    return InsertResult(id=int(chunk_id), inserted=inserted)


async def list_chunks(session: AsyncSession, chapter_id: int) -> list[ChunkRow]:
    result = await session.execute(
        select(Chunk.id, Chunk.idx, Chunk.text, Chunk.chunk_hash)
        .where(Chunk.chapter_id == chapter_id)
        .order_by(Chunk.idx)
    )
    rows = result.all()
    return [ChunkRow(id=int(row[0]), idx=int(row[1]), text=str(row[2]), chunk_hash=str(row[3])) for row in rows]


async def rebuild_chunks_fts_for_book(session: AsyncSession, book_id: int) -> int:
    await session.execute(sa_text("DELETE FROM chunks_fts WHERE book_id = :book_id"), {"book_id": str(book_id)})
    await session.execute(
        sa_text(
            """
            INSERT INTO chunks_fts (chunk_id, book_id, chapter_idx, chapter_title, text)
            SELECT c.id, ch.book_id, ch.idx, ch.title, c.text
            FROM chunks c
            JOIN chapters ch ON ch.id = c.chapter_id
            WHERE ch.book_id = :book_id
            """
        ),
        {"book_id": book_id},
    )
    count_result = await session.execute(
        sa_text("SELECT COUNT(*) FROM chunks_fts WHERE book_id = :book_id"), {"book_id": str(book_id)}
    )
    return int(count_result.scalar_one())


async def search_chunks_fts(
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
                CAST(chunk_id AS INTEGER) AS source_id,
                CAST(chapter_idx AS INTEGER) AS chapter_idx,
                chapter_title,
                text,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH :query
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
            source_type="chunk",
            source_id=int(row[0]),
            chapter_idx=int(row[1]),
            chapter_title=str(row[2] or ""),
            text=str(row[3] or ""),
            score=float(row[4]) if row[4] is not None else None,
        )
        for row in rows
    ]
