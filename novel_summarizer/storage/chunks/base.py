from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from novel_summarizer.storage.base import Base


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("chapter_id", "idx", name="uq_chunks_chapter_id_idx"),
        Index("idx_chunks_chapter_id", "chapter_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chapter_id: Mapped[int] = mapped_column(ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    start_pos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_pos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )

    chapter: Mapped["Chapter"] = relationship("Chapter", back_populates="chunks")
