from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, select, update, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.types import InsertResult, WorldFactRow


class WorldFact(Base):
    __tablename__ = "world_facts"
    __table_args__ = (
        UniqueConstraint("book_id", "fact_key", name="uq_world_facts_book_key"),
        Index("idx_world_facts_book_id", "book_id"),
        Index("idx_world_facts_source_chapter", "book_id", "source_chapter_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    fact_key: Mapped[str] = mapped_column(String(255), nullable=False)
    fact_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False, default=0.8, server_default="0.8")
    source_chapter_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
        onupdate=sa_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def _to_row(row: tuple) -> WorldFactRow:
    return WorldFactRow(
        id=int(row[0]),
        book_id=int(row[1]),
        fact_key=str(row[2]),
        fact_value=str(row[3]),
        confidence=float(row[4]),
        source_chapter_idx=row[5],
        source_excerpt=row[6],
    )


async def list_world_facts(session: AsyncSession, book_id: int, limit: int = 500) -> list[WorldFactRow]:
    result = await session.execute(
        select(
            WorldFact.id,
            WorldFact.book_id,
            WorldFact.fact_key,
            WorldFact.fact_value,
            WorldFact.confidence,
            WorldFact.source_chapter_idx,
            WorldFact.source_excerpt,
        )
        .where(WorldFact.book_id == book_id)
        .order_by(WorldFact.fact_key)
        .limit(limit)
    )
    rows = result.all()
    return [_to_row(row) for row in rows]


async def upsert_world_fact(
    session: AsyncSession,
    *,
    book_id: int,
    fact_key: str,
    fact_value: str,
    confidence: float = 0.8,
    source_chapter_idx: int | None = None,
    source_excerpt: str | None = None,
) -> InsertResult:
    existing = await session.execute(
        select(WorldFact.id).where(WorldFact.book_id == book_id, WorldFact.fact_key == fact_key)
    )
    existing_id = existing.scalar_one_or_none()

    if existing_id is None:
        stmt = WorldFact.__table__.insert().values(
            book_id=book_id,
            fact_key=fact_key,
            fact_value=fact_value,
            confidence=confidence,
            source_chapter_idx=source_chapter_idx,
            source_excerpt=source_excerpt,
        )
        result = await session.execute(stmt)
        if result.lastrowid is None:
            lookup = await session.execute(
                select(WorldFact.id).where(WorldFact.book_id == book_id, WorldFact.fact_key == fact_key)
            )
            world_fact_id = int(lookup.scalar_one())
        else:
            world_fact_id = int(result.lastrowid)
        return InsertResult(id=world_fact_id, inserted=True)

    await session.execute(
        update(WorldFact)
        .where(WorldFact.id == existing_id)
        .values(
            fact_value=fact_value,
            confidence=confidence,
            source_chapter_idx=source_chapter_idx,
            source_excerpt=source_excerpt,
        )
    )
    return InsertResult(id=int(existing_id), inserted=False)