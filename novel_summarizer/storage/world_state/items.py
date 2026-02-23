from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, select, update, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.types import InsertResult, ItemRow


class ItemState(Base):
    __tablename__ = "items"
    __table_args__ = (
        UniqueConstraint("book_id", "name", name="uq_items_book_name"),
        Index("idx_items_book_id", "book_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_chapter_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_chapter_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active", server_default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
        onupdate=sa_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def _to_row(row: tuple) -> ItemRow:
    return ItemRow(
        id=int(row[0]),
        book_id=int(row[1]),
        name=str(row[2]),
        owner_name=row[3],
        first_chapter_idx=row[4],
        last_chapter_idx=row[5],
        description=row[6],
        status=str(row[7]),
    )


async def list_item_states(session: AsyncSession, book_id: int, names: list[str] | None = None) -> list[ItemRow]:
    stmt = select(
        ItemState.id,
        ItemState.book_id,
        ItemState.name,
        ItemState.owner_name,
        ItemState.first_chapter_idx,
        ItemState.last_chapter_idx,
        ItemState.description,
        ItemState.status,
    ).where(ItemState.book_id == book_id)
    if names:
        stmt = stmt.where(ItemState.name.in_(names))

    result = await session.execute(stmt.order_by(ItemState.name))
    rows = result.all()
    return [_to_row(row) for row in rows]


async def upsert_item_state(
    session: AsyncSession,
    book_id: int,
    name: str,
    owner_name: str | None = None,
    first_chapter_idx: int | None = None,
    last_chapter_idx: int | None = None,
    description: str | None = None,
    status: str = "active",
) -> InsertResult:
    existing = await session.execute(
        select(ItemState.id).where(
            ItemState.book_id == book_id,
            ItemState.name == name,
        )
    )
    existing_id = existing.scalar_one_or_none()

    if existing_id is None:
        insert_stmt = ItemState.__table__.insert().values(
            book_id=book_id,
            name=name,
            owner_name=owner_name,
            first_chapter_idx=first_chapter_idx,
            last_chapter_idx=last_chapter_idx,
            description=description,
            status=status,
        )
        result = await session.execute(insert_stmt)
        if result.lastrowid is None:
            lookup = await session.execute(
                select(ItemState.id).where(
                    ItemState.book_id == book_id,
                    ItemState.name == name,
                )
            )
            item_id = int(lookup.scalar_one())
        else:
            item_id = int(result.lastrowid)
        return InsertResult(id=item_id, inserted=True)

    await session.execute(
        update(ItemState)
        .where(ItemState.id == existing_id)
        .values(
            owner_name=owner_name,
            first_chapter_idx=first_chapter_idx,
            last_chapter_idx=last_chapter_idx,
            description=description,
            status=status,
        )
    )
    return InsertResult(id=int(existing_id), inserted=False)