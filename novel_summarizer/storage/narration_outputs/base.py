from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base


class NarrationOutput(Base):
    __tablename__ = "narration_outputs"
    __table_args__ = (
        UniqueConstraint("narration_id", name="uq_narration_outputs_narration_id"),
        Index("idx_narration_outputs_book_id", "book_id"),
        Index("idx_narration_outputs_chapter_id", "chapter_id"),
        Index("idx_narration_outputs_chapter_idx", "chapter_idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    narration_id: Mapped[int] = mapped_column(ForeignKey("narrations.id", ondelete="CASCADE"), nullable=False)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_id: Mapped[int] = mapped_column(ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False)
    chapter_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
        onupdate=sa_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
