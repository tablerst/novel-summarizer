from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column

from novel_summarizer.storage.base import Base


class Summary(Base):
    __tablename__ = "summaries"
    __table_args__ = (
        UniqueConstraint(
            "scope",
            "ref_id",
            "summary_type",
            "prompt_version",
            "model",
            "input_hash",
            name="uq_summaries_identity",
        ),
        Index("idx_summaries_ref_id", "ref_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    ref_id: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_type: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), nullable=False
    )
