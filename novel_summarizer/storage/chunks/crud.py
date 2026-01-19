from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.chunks.base import Chunk
from novel_summarizer.storage.types import ChunkRow, InsertResult


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
