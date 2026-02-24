from __future__ import annotations

import sys
from loguru import logger


_DEFAULT_CONTEXT = {
    "trace_id": "-",
    "node": "-",
    "chapter_id": "-",
    "chapter_idx": "-",
    "attempt": "-",
    "cache_key": "-",
    "input_hash": "-",
}


def _inject_default_context(record: dict) -> None:
    extra = record["extra"]
    for key, value in _DEFAULT_CONTEXT.items():
        extra.setdefault(key, value)


def setup_logging(level: str) -> None:
    """Configure loguru logging for CLI runs."""
    logger.remove()
    logger.configure(patcher=_inject_default_context)
    logger.add(
        sys.stderr,
        level=level,
        backtrace=True,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "| <level>{level:<8}</level> "
            "| trace={extra[trace_id]} node={extra[node]} chapter={extra[chapter_id]} idx={extra[chapter_idx]} "
            "attempt={extra[attempt]} cache={extra[cache_key]} hash={extra[input_hash]} "
            "| {message}"
        ),
    )
