from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.summaries.base import Summary
from novel_summarizer.storage.types import InsertResult, SummaryRow


async def get_summary(
    session: AsyncSession,
    scope: str,
    ref_id: int,
    summary_type: str,
    prompt_version: str,
    model: str,
    input_hash: str,
) -> SummaryRow | None:
    result = await session.execute(
        select(
            Summary.id,
            Summary.scope,
            Summary.ref_id,
            Summary.summary_type,
            Summary.prompt_version,
            Summary.model,
            Summary.input_hash,
            Summary.content,
        ).where(
            Summary.scope == scope,
            Summary.ref_id == ref_id,
            Summary.summary_type == summary_type,
            Summary.prompt_version == prompt_version,
            Summary.model == model,
            Summary.input_hash == input_hash,
        )
    )
    row = result.first()
    if not row:
        return None
    return SummaryRow(
        id=int(row[0]),
        scope=str(row[1]),
        ref_id=int(row[2]),
        summary_type=str(row[3]),
        prompt_version=str(row[4]),
        model=str(row[5]),
        input_hash=str(row[6]),
        content=str(row[7]),
    )


async def get_latest_summary(session: AsyncSession, scope: str, ref_id: int, summary_type: str) -> SummaryRow | None:
    result = await session.execute(
        select(
            Summary.id,
            Summary.scope,
            Summary.ref_id,
            Summary.summary_type,
            Summary.prompt_version,
            Summary.model,
            Summary.input_hash,
            Summary.content,
        )
        .where(
            Summary.scope == scope,
            Summary.ref_id == ref_id,
            Summary.summary_type == summary_type,
        )
        .order_by(desc(Summary.created_at))
        .limit(1)
    )
    row = result.first()
    if not row:
        return None
    return SummaryRow(
        id=int(row[0]),
        scope=str(row[1]),
        ref_id=int(row[2]),
        summary_type=str(row[3]),
        prompt_version=str(row[4]),
        model=str(row[5]),
        input_hash=str(row[6]),
        content=str(row[7]),
    )


async def upsert_summary(
    session: AsyncSession,
    scope: str,
    ref_id: int,
    summary_type: str,
    prompt_version: str,
    model: str,
    input_hash: str,
    content: str,
    params_json: str | None = None,
) -> InsertResult:
    stmt = (
        sqlite_insert(Summary)
        .values(
            scope=scope,
            ref_id=ref_id,
            summary_type=summary_type,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
            params_json=params_json,
            content=content,
        )
        .on_conflict_do_nothing(
            index_elements=[
                Summary.scope,
                Summary.ref_id,
                Summary.summary_type,
                Summary.prompt_version,
                Summary.model,
                Summary.input_hash,
            ]
        )
    )
    result = await session.execute(stmt)
    inserted = result.rowcount == 1
    if inserted and result.lastrowid is not None:
        summary_id = result.lastrowid
    else:
        id_result = await session.execute(
            select(Summary.id).where(
                Summary.scope == scope,
                Summary.ref_id == ref_id,
                Summary.summary_type == summary_type,
                Summary.prompt_version == prompt_version,
                Summary.model == model,
                Summary.input_hash == input_hash,
            )
        )
        summary_id = id_result.scalar_one()
    return InsertResult(id=int(summary_id), inserted=inserted)
