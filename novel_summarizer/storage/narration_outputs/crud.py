from __future__ import annotations

from sqlalchemy import desc, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.narration_outputs.base import NarrationOutput
from novel_summarizer.storage.types import InsertResult, NarrationOutputRow


def _to_row(row: tuple) -> NarrationOutputRow:
    return NarrationOutputRow(
        id=int(row[0]),
        narration_id=int(row[1]),
        book_id=int(row[2]),
        chapter_id=int(row[3]),
        chapter_idx=int(row[4]),
        prompt_version=str(row[5]),
        model=str(row[6]),
        input_hash=str(row[7]),
        payload_json=str(row[8]),
    )


async def get_narration_output(session: AsyncSession, narration_id: int) -> NarrationOutputRow | None:
    result = await session.execute(
        select(
            NarrationOutput.id,
            NarrationOutput.narration_id,
            NarrationOutput.book_id,
            NarrationOutput.chapter_id,
            NarrationOutput.chapter_idx,
            NarrationOutput.prompt_version,
            NarrationOutput.model,
            NarrationOutput.input_hash,
            NarrationOutput.payload_json,
        ).where(NarrationOutput.narration_id == narration_id)
    )
    row = result.first()
    return _to_row(tuple(row)) if row else None


async def get_latest_narration_output_for_chapter(session: AsyncSession, chapter_id: int) -> NarrationOutputRow | None:
    ranked = (
        select(
            NarrationOutput.id.label("id"),
            NarrationOutput.narration_id.label("narration_id"),
            NarrationOutput.book_id.label("book_id"),
            NarrationOutput.chapter_id.label("chapter_id"),
            NarrationOutput.chapter_idx.label("chapter_idx"),
            NarrationOutput.prompt_version.label("prompt_version"),
            NarrationOutput.model.label("model"),
            NarrationOutput.input_hash.label("input_hash"),
            NarrationOutput.payload_json.label("payload_json"),
            func.row_number()
            .over(
                partition_by=NarrationOutput.chapter_id,
                order_by=(NarrationOutput.updated_at.desc(), NarrationOutput.id.desc()),
            )
            .label("row_num"),
        )
        .where(NarrationOutput.chapter_id == chapter_id)
        .subquery()
    )

    result = await session.execute(
        select(
            ranked.c.id,
            ranked.c.narration_id,
            ranked.c.book_id,
            ranked.c.chapter_id,
            ranked.c.chapter_idx,
            ranked.c.prompt_version,
            ranked.c.model,
            ranked.c.input_hash,
            ranked.c.payload_json,
        )
        .where(ranked.c.row_num == 1)
        .order_by(desc(ranked.c.id))
        .limit(1)
    )

    row = result.first()
    return _to_row(tuple(row)) if row else None


async def upsert_narration_output(
    session: AsyncSession,
    *,
    narration_id: int,
    book_id: int,
    chapter_id: int,
    chapter_idx: int,
    prompt_version: str,
    model: str,
    input_hash: str,
    payload_json: str,
) -> InsertResult:
    stmt = (
        sqlite_insert(NarrationOutput)
        .values(
            narration_id=narration_id,
            book_id=book_id,
            chapter_id=chapter_id,
            chapter_idx=chapter_idx,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
            payload_json=payload_json,
        )
        .on_conflict_do_update(
            index_elements=[NarrationOutput.narration_id],
            set_={
                "payload_json": payload_json,
                "prompt_version": prompt_version,
                "model": model,
                "input_hash": input_hash,
            },
        )
    )

    existing = await get_narration_output(session, narration_id)
    await session.execute(stmt)
    lookup = await session.execute(select(NarrationOutput.id).where(NarrationOutput.narration_id == narration_id))
    output_id = int(lookup.scalar_one())
    return InsertResult(id=output_id, inserted=(existing is None))
