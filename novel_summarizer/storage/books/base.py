from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from novel_summarizer.storage.base import Base


class Book(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    book_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )

    chapters: Mapped[list["Chapter"]] = relationship(
        "Chapter",
        back_populates="book",
        cascade="all, delete-orphan",
    )
