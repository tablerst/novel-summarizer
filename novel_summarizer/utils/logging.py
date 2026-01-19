from __future__ import annotations

import sys
from loguru import logger


def setup_logging(level: str) -> None:
    """Configure loguru logging for CLI runs."""
    logger.remove()
    logger.add(sys.stderr, level=level)
