from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.types import InsertResult, PlotEventRow


class PlotEvent(Base):
    __tablename__ = "plot_events"
    __table_args__ = (
        Index("idx_plot_events_book_id", "book_id"),
        Index("idx_plot_events_book_chapter", "book_id", "chapter_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    event_summary: Mapped[str] = mapped_column(Text, nullable=False)
    involved_characters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    impact: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )


def _to_row(row: tuple) -> PlotEventRow:
    return PlotEventRow(
        id=int(row[0]),
        book_id=int(row[1]),
        chapter_idx=int(row[2]),
        event_summary=str(row[3]),
        involved_characters_json=row[4],
        event_type=row[5],
        impact=row[6],
    )


async def list_recent_plot_events(
    session: AsyncSession,
    book_id: int,
    chapter_idx: int | None = None,
    window: int = 5,
    limit: int = 20,
) -> list[PlotEventRow]:
    stmt = select(
        PlotEvent.id,
        PlotEvent.book_id,
        PlotEvent.chapter_idx,
        PlotEvent.event_summary,
        PlotEvent.involved_characters_json,
        PlotEvent.event_type,
        PlotEvent.impact,
    ).where(PlotEvent.book_id == book_id)

    if chapter_idx is not None:
        min_chapter_idx = max(1, chapter_idx - max(window, 1))
        stmt = stmt.where(PlotEvent.chapter_idx < chapter_idx, PlotEvent.chapter_idx >= min_chapter_idx)

    result = await session.execute(stmt.order_by(PlotEvent.chapter_idx.desc(), PlotEvent.id.desc()).limit(limit))
    rows = result.all()
    return [_to_row(row) for row in rows]


async def list_plot_events_by_book(session: AsyncSession, book_id: int) -> list[PlotEventRow]:
    result = await session.execute(
        select(
            PlotEvent.id,
            PlotEvent.book_id,
            PlotEvent.chapter_idx,
            PlotEvent.event_summary,
            PlotEvent.involved_characters_json,
            PlotEvent.event_type,
            PlotEvent.impact,
        )
        .where(PlotEvent.book_id == book_id)
        .order_by(PlotEvent.chapter_idx, PlotEvent.id)
    )
    rows = result.all()
    return [_to_row(row) for row in rows]


async def insert_plot_event(
    session: AsyncSession,
    book_id: int,
    chapter_idx: int,
    event_summary: str,
    involved_characters_json: str | None = None,
    event_type: str | None = None,
    impact: str | None = None,
) -> InsertResult:
    stmt = PlotEvent.__table__.insert().values(
        book_id=book_id,
        chapter_idx=chapter_idx,
        event_summary=event_summary,
        involved_characters_json=involved_characters_json,
        event_type=event_type,
        impact=impact,
    )
    result = await session.execute(stmt)
    if result.lastrowid is None:
        lookup = await session.execute(select(PlotEvent.id).order_by(PlotEvent.id.desc()).limit(1))
        event_id = int(lookup.scalar_one())
    else:
        event_id = int(result.lastrowid)
    return InsertResult(id=event_id, inserted=True)