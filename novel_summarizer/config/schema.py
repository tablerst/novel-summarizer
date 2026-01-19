from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./output"))
    log_level: str = Field(default="INFO")


class IngestCleanupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strip_blank_lines: bool = True
    normalize_fullwidth: bool = True


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: str = "utf-8"
    chapter_regex: str | None = None
    cleanup: IngestCleanupConfig = IngestCleanupConfig()


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size_tokens: int = 1200
    chunk_overlap_tokens: int = 120
    min_chunk_tokens: int = 200

    @field_validator("chunk_size_tokens", "chunk_overlap_tokens", "min_chunk_tokens")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("split config values must be positive")
        return value

    @model_validator(mode="after")
    def _validate_overlap(self) -> "SplitConfig":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be less than chunk_size_tokens")
        return self


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai"] = "openai"
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.3
    timeout_s: int = 60
    max_concurrency: int = 6
    retries: int = 3
    base_url: str | None = None

    @field_validator("temperature")
    @classmethod
    def _temperature_range(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value


class WithCitationsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    top_k: int = 6


class SummarizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "zh"
    style: str = "清晰、克制、不剧透关键反转"
    chapter_summary_words: tuple[int, int] = (200, 500)
    book_summary_words: tuple[int, int] = (800, 1500)
    story_words: tuple[int, int] | None = None
    include_quotes: bool = False
    with_citations: WithCitationsConfig = WithCitationsConfig()


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sqlite_path: Path = Field(default=Path("./data/novel.db"))
    lancedb_dir: Path = Field(default=Path("./data/lancedb"))


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: str = "sqlite"
    ttl_seconds: int = 2_592_000


class AppConfigRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig = AppConfig()
    ingest: IngestConfig = IngestConfig()
    split: SplitConfig = SplitConfig()
    llm: LLMConfig = LLMConfig()
    summarize: SummarizeConfig = SummarizeConfig()
    storage: StorageConfig = StorageConfig()
    cache: CacheConfig = CacheConfig()


def resolve_paths(config: AppConfigRoot, base_dir: Path) -> AppConfigRoot:
    def _resolve(path_value: Path) -> Path:
        return path_value if path_value.is_absolute() else (base_dir / path_value).resolve()

    config.app.data_dir = _resolve(config.app.data_dir)
    config.app.output_dir = _resolve(config.app.output_dir)
    config.storage.sqlite_path = _resolve(config.storage.sqlite_path)
    config.storage.lancedb_dir = _resolve(config.storage.lancedb_dir)
    return config

