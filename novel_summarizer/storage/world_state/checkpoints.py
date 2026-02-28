from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, select, text as sa_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.types import InsertResult, WorldStateCheckpointRow


class WorldStateCheckpoint(Base):
    __tablename__ = "world_state_checkpoints"
    __table_args__ = (
        UniqueConstraint("book_id", "chapter_idx", "step_size", name="uq_world_state_checkpoints_identity"),
        Index("idx_world_state_checkpoints_book_id", "book_id"),
        Index("idx_world_state_checkpoints_book_chapter", "book_id", "chapter_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    step_size: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
        onupdate=sa_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def _to_row(row: tuple) -> WorldStateCheckpointRow:
    return WorldStateCheckpointRow(
        id=int(row[0]),
        book_id=int(row[1]),
        chapter_idx=int(row[2]),
        step_size=int(row[3]),
        snapshot_json=str(row[4]),
        snapshot_hash=str(row[5]),
        created_at=row[6],
        updated_at=row[7],
    )


async def get_latest_checkpoint_at_or_before(
    session: AsyncSession,
    *,
    book_id: int,
    chapter_idx: int,
) -> WorldStateCheckpointRow | None:
    result = await session.execute(
        select(
            WorldStateCheckpoint.id,
            WorldStateCheckpoint.book_id,
            WorldStateCheckpoint.chapter_idx,
            WorldStateCheckpoint.step_size,
            WorldStateCheckpoint.snapshot_json,
            WorldStateCheckpoint.snapshot_hash,
            WorldStateCheckpoint.created_at,
            WorldStateCheckpoint.updated_at,
        )
        .where(WorldStateCheckpoint.book_id == book_id, WorldStateCheckpoint.chapter_idx <= chapter_idx)
        .order_by(WorldStateCheckpoint.chapter_idx.desc(), WorldStateCheckpoint.created_at.desc(), WorldStateCheckpoint.id.desc())
        .limit(1)
    )

    row = result.first()
    return _to_row(tuple(row)) if row else None


async def upsert_checkpoint(
    session: AsyncSession,
    *,
    book_id: int,
    chapter_idx: int,
    step_size: int,
    snapshot_json: str,
    snapshot_hash: str,
) -> InsertResult:
    existing = await session.execute(
        select(WorldStateCheckpoint.id).where(
            WorldStateCheckpoint.book_id == book_id,
            WorldStateCheckpoint.chapter_idx == chapter_idx,
            WorldStateCheckpoint.step_size == step_size,
        )
    )
    existing_id = existing.scalar_one_or_none()

    stmt = (
        sqlite_insert(WorldStateCheckpoint)
        .values(
            book_id=book_id,
            chapter_idx=chapter_idx,
            step_size=step_size,
            snapshot_json=snapshot_json,
            snapshot_hash=snapshot_hash,
        )
        .on_conflict_do_update(
            index_elements=[WorldStateCheckpoint.book_id, WorldStateCheckpoint.chapter_idx, WorldStateCheckpoint.step_size],
            set_={
                "snapshot_json": snapshot_json,
                "snapshot_hash": snapshot_hash,
                "updated_at": sa_text("CURRENT_TIMESTAMP"),
            },
        )
    )
    result = await session.execute(stmt)

    _ = result
    lookup = await session.execute(
        select(WorldStateCheckpoint.id).where(
            WorldStateCheckpoint.book_id == book_id,
            WorldStateCheckpoint.chapter_idx == chapter_idx,
            WorldStateCheckpoint.step_size == step_size,
        )
    )
    checkpoint_id = int(lookup.scalar_one())

    return InsertResult(id=checkpoint_id, inserted=(existing_id is None))
