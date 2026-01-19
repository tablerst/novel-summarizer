"""Configuration loading and schema."""

from novel_summarizer.config.loader import load_config
from novel_summarizer.config.schema import AppConfig

__all__ = ["AppConfig", "load_config"]
