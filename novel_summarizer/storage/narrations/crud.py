from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.narrations.base import Narration
from novel_summarizer.storage.types import InsertResult, NarrationRow


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