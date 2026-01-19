from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from novel_summarizer.storage.base import Base


class Chapter(Base):
    __tablename__ = "chapters"
    __table_args__ = (
        UniqueConstraint("book_id", "idx", name="uq_chapters_book_id_idx"),
        Index("idx_chapters_book_id", "book_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    chapter_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    start_pos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_pos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )

    book: Mapped["Book"] = relationship("Book", back_populates="chapters")
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="chapter",
        cascade="all, delete-orphan",
    )
