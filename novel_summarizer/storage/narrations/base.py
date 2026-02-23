from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from novel_summarizer.storage.base import Base


class Narration(Base):
    __tablename__ = "narrations"
    __table_args__ = (
        UniqueConstraint(
            "chapter_id",
            "prompt_version",
            "model",
            "input_hash",
            name="uq_narrations_identity",
        ),
        Index("idx_narrations_book_id", "book_id"),
        Index("idx_narrations_chapter_idx", "chapter_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_id: Mapped[int] = mapped_column(ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False)
    chapter_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    narration_text: Mapped[str] = mapped_column(Text, nullable=False)
    key_events_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )

    book: Mapped["Book"] = relationship("Book", back_populates="narrations")
    chapter: Mapped["Chapter"] = relationship("Chapter", back_populates="narrations")