from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, select, update, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.types import CharacterRow, InsertResult


class CharacterState(Base):
    __tablename__ = "characters"
    __table_args__ = (
        UniqueConstraint("book_id", "canonical_name", name="uq_characters_book_name"),
        Index("idx_characters_book_id", "book_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]", server_default="[]")
    first_chapter_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_chapter_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active", server_default="active")
    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    abilities_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    relationships_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    motivation: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
        onupdate=sa_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def _to_row(row: tuple) -> CharacterRow:
    return CharacterRow(
        id=int(row[0]),
        book_id=int(row[1]),
        canonical_name=str(row[2]),
        aliases_json=str(row[3]),
        first_chapter_idx=row[4],
        last_chapter_idx=row[5],
        status=str(row[6]),
        location=row[7],
        abilities_json=row[8],
        relationships_json=row[9],
        motivation=row[10],
        notes=row[11],
    )


async def list_character_states(
    session: AsyncSession,
    book_id: int,
    canonical_names: list[str] | None = None,
) -> list[CharacterRow]:
    stmt = select(
        CharacterState.id,
        CharacterState.book_id,
        CharacterState.canonical_name,
        CharacterState.aliases_json,
        CharacterState.first_chapter_idx,
        CharacterState.last_chapter_idx,
        CharacterState.status,
        CharacterState.location,
        CharacterState.abilities_json,
        CharacterState.relationships_json,
        CharacterState.motivation,
        CharacterState.notes,
    ).where(CharacterState.book_id == book_id)
    if canonical_names:
        stmt = stmt.where(CharacterState.canonical_name.in_(canonical_names))

    result = await session.execute(stmt.order_by(CharacterState.canonical_name))
    rows = result.all()
    return [_to_row(row) for row in rows]


async def upsert_character_state(
    session: AsyncSession,
    book_id: int,
    canonical_name: str,
    aliases_json: str = "[]",
    first_chapter_idx: int | None = None,
    last_chapter_idx: int | None = None,
    status: str = "active",
    location: str | None = None,
    abilities_json: str | None = None,
    relationships_json: str | None = None,
    motivation: str | None = None,
    notes: str | None = None,
) -> InsertResult:
    existing = await session.execute(
        select(CharacterState.id).where(
            CharacterState.book_id == book_id,
            CharacterState.canonical_name == canonical_name,
        )
    )
    existing_id = existing.scalar_one_or_none()

    if existing_id is None:
        insert_stmt = CharacterState.__table__.insert().values(
            book_id=book_id,
            canonical_name=canonical_name,
            aliases_json=aliases_json,
            first_chapter_idx=first_chapter_idx,
            last_chapter_idx=last_chapter_idx,
            status=status,
            location=location,
            abilities_json=abilities_json,
            relationships_json=relationships_json,
            motivation=motivation,
            notes=notes,
        )
        result = await session.execute(insert_stmt)
        if result.lastrowid is None:
            lookup = await session.execute(
                select(CharacterState.id).where(
                    CharacterState.book_id == book_id,
                    CharacterState.canonical_name == canonical_name,
                )
            )
            character_id = int(lookup.scalar_one())
        else:
            character_id = int(result.lastrowid)
        return InsertResult(id=character_id, inserted=True)

    await session.execute(
        update(CharacterState)
        .where(CharacterState.id == existing_id)
        .values(
            aliases_json=aliases_json,
            first_chapter_idx=first_chapter_idx,
            last_chapter_idx=last_chapter_idx,
            status=status,
            location=location,
            abilities_json=abilities_json,
            relationships_json=relationships_json,
            motivation=motivation,
            notes=notes,
        )
    )
    return InsertResult(id=int(existing_id), inserted=False)