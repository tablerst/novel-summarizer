from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


NARRATION_PRESET_RATIOS: dict[str, tuple[float, float]] = {
    "short": (0.2, 0.3),
    "medium": (0.4, 0.5),
    "long": (0.65, 0.8),
}


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

    encoding: str = "auto"
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


class LLMProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["openai_compatible", "ollama"] = "openai_compatible"
    base_url: str | None = None
    api_key_env: str | None = None


class ChatEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = 0.3
    timeout_s: int = 60
    max_concurrency: int = 6
    retries: int = 3
    max_tokens: int | None = None

    @field_validator("temperature")
    @classmethod
    def _temperature_range(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("timeout_s", "max_concurrency", "retries")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("endpoint integer settings must be non-negative")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _positive_optional_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_tokens must be positive when provided")
        return value


class EmbeddingEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    timeout_s: int = 60
    max_concurrency: int = 6
    retries: int = 3

    @field_validator("timeout_s", "max_concurrency", "retries")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("endpoint integer settings must be non-negative")
        return value


class LLMRoutesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summarize_chat: str | None = None
    storyteller_chat: str = "storyteller_default"
    storyteller_entity_chat: str | None = None
    storyteller_narration_chat: str | None = None
    storyteller_refine_chat: str | None = None
    embedding: str = "embedding_default"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, LLMProviderConfig]
    chat_endpoints: dict[str, ChatEndpointConfig]
    embedding_endpoints: dict[str, EmbeddingEndpointConfig]
    routes: LLMRoutesConfig = LLMRoutesConfig()

    @model_validator(mode="after")
    def _validate_references(self) -> "LLMConfig":
        if not self.providers:
            raise ValueError("llm.providers cannot be empty")
        if not self.chat_endpoints:
            raise ValueError("llm.chat_endpoints cannot be empty")
        if not self.embedding_endpoints:
            raise ValueError("llm.embedding_endpoints cannot be empty")

        for endpoint_name, endpoint in self.chat_endpoints.items():
            if endpoint.provider not in self.providers:
                raise ValueError(
                    f"chat endpoint '{endpoint_name}' references unknown provider '{endpoint.provider}'"
                )

        for endpoint_name, endpoint in self.embedding_endpoints.items():
            if endpoint.provider not in self.providers:
                raise ValueError(
                    f"embedding endpoint '{endpoint_name}' references unknown provider '{endpoint.provider}'"
                )

        if self.routes.summarize_chat and self.routes.summarize_chat not in self.chat_endpoints:
            raise ValueError(f"llm.routes.summarize_chat not found: {self.routes.summarize_chat}")
        if self.routes.storyteller_chat not in self.chat_endpoints:
            raise ValueError(f"llm.routes.storyteller_chat not found: {self.routes.storyteller_chat}")
        if self.routes.storyteller_entity_chat and self.routes.storyteller_entity_chat not in self.chat_endpoints:
            raise ValueError(
                f"llm.routes.storyteller_entity_chat not found: {self.routes.storyteller_entity_chat}"
            )
        if self.routes.storyteller_narration_chat and self.routes.storyteller_narration_chat not in self.chat_endpoints:
            raise ValueError(
                f"llm.routes.storyteller_narration_chat not found: {self.routes.storyteller_narration_chat}"
            )
        if self.routes.storyteller_refine_chat and self.routes.storyteller_refine_chat not in self.chat_endpoints:
            raise ValueError(
                f"llm.routes.storyteller_refine_chat not found: {self.routes.storyteller_refine_chat}"
            )
        if self.routes.embedding not in self.embedding_endpoints:
            raise ValueError(f"llm.routes.embedding not found: {self.routes.embedding}")

        return self

    def resolve_chat_route(
        self,
        route: Literal["summarize", "storyteller", "storyteller_entity", "storyteller_narration", "storyteller_refine"],
    ) -> tuple[str, ChatEndpointConfig, LLMProviderConfig]:
        if route == "summarize":
            endpoint_name = self.routes.summarize_chat or self.routes.storyteller_chat
        elif route == "storyteller":
            endpoint_name = self.routes.storyteller_chat
        elif route == "storyteller_entity":
            endpoint_name = self.routes.storyteller_entity_chat or self.routes.storyteller_chat
        elif route == "storyteller_narration":
            endpoint_name = self.routes.storyteller_narration_chat or self.routes.storyteller_chat
        else:
            endpoint_name = (
                self.routes.storyteller_refine_chat
                or self.routes.storyteller_narration_chat
                or self.routes.storyteller_chat
            )

        endpoint = self.chat_endpoints[endpoint_name]
        provider = self.providers[endpoint.provider]
        return endpoint_name, endpoint, provider

    def resolve_embedding_route(self) -> tuple[str, EmbeddingEndpointConfig, LLMProviderConfig]:
        endpoint_name = self.routes.embedding
        endpoint = self.embedding_endpoints[endpoint_name]
        provider = self.providers[endpoint.provider]
        return endpoint_name, endpoint, provider


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


class StorytellerTierProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narration_ratio: tuple[float, float]
    memory_top_k: int
    include_key_dialogue: bool
    include_inner_thoughts: bool
    refine_enabled: bool
    entity_extract_mode: Literal["llm", "fallback"] = "llm"

    @field_validator("narration_ratio")
    @classmethod
    def _validate_ratio(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not (0 < low < 1 and 0 < high < 1):
            raise ValueError("narration_ratio values must be in range (0, 1)")
        if low >= high:
            raise ValueError("narration_ratio[0] must be less than narration_ratio[1]")
        return value

    @field_validator("memory_top_k")
    @classmethod
    def _non_negative_memory_top_k(cls, value: int) -> int:
        if value < 0:
            raise ValueError("memory_top_k must be non-negative")
        return value


class StorytellerTieringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_tier: Literal["short", "medium", "long"] = "medium"
    long_every_n: int = 0
    long_min_chars: int = 0
    long_keyword_triggers: list[str] = Field(default_factory=list)
    short: StorytellerTierProfileConfig = StorytellerTierProfileConfig(
        narration_ratio=(0.12, 0.2),
        memory_top_k=0,
        include_key_dialogue=False,
        include_inner_thoughts=False,
        refine_enabled=False,
        entity_extract_mode="fallback",
    )
    medium: StorytellerTierProfileConfig = StorytellerTierProfileConfig(
        narration_ratio=(0.2, 0.3),
        memory_top_k=4,
        include_key_dialogue=True,
        include_inner_thoughts=False,
        refine_enabled=False,
        entity_extract_mode="llm",
    )
    long: StorytellerTierProfileConfig = StorytellerTierProfileConfig(
        narration_ratio=(0.4, 0.5),
        memory_top_k=8,
        include_key_dialogue=True,
        include_inner_thoughts=True,
        refine_enabled=True,
        entity_extract_mode="llm",
    )

    @field_validator("long_every_n", "long_min_chars")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("tiering integer settings must be non-negative")
        return value


class StorytellerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "zh"
    style: str = "说书人/评书艺人风格，沉浸感强，保留关键对白和心理博弈"
    narration_preset: Literal["short", "medium", "long"] = "medium"
    narration_ratio: tuple[float, float] = (0.4, 0.5)
    narration_temperature: float = 0.45
    entity_temperature: float = 0.1
    entity_extract_mode: Literal["llm", "fallback"] = "llm"
    state_temperature: float = 0.1
    memory_top_k: int = 8
    recent_events_window: int = 5
    include_key_dialogue: bool = True
    include_inner_thoughts: bool = True
    refine_enabled: bool = True
    refine_temperature: float = 0.35
    evidence_min_support_score: float = 0.18
    evidence_max_snippets: int = 3
    prefetch_window: int = 0
    tiering: StorytellerTieringConfig = StorytellerTieringConfig()

    # Step execution (multi-chapter batching) controls. Defaults keep legacy behavior.
    step_size: int = 1
    step_align: Literal["auto", "off"] = "auto"
    step_checkpoint_enabled: bool = True
    step_resume_mode: Literal["continue", "restore"] = "restore"
    step_memory_mode: Literal["per_chapter", "per_step_shared", "off"] = "per_chapter"
    step_refine_mode: Literal["inline", "off"] = "inline"
    step_retrieval_concurrency: int = 6

    @field_validator("narration_temperature", "entity_temperature", "state_temperature", "refine_temperature")
    @classmethod
    def _temperature_range(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("memory_top_k", "recent_events_window")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("storyteller integer config values must be positive")
        return value

    @field_validator("prefetch_window")
    @classmethod
    def _non_negative_prefetch_window(cls, value: int) -> int:
        if value < 0:
            raise ValueError("prefetch_window must be non-negative")
        return value

    @field_validator("step_size")
    @classmethod
    def _validate_step_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("step_size must be a positive integer")
        return value

    @field_validator("step_retrieval_concurrency")
    @classmethod
    def _validate_step_retrieval_concurrency(cls, value: int) -> int:
        if value < 0:
            raise ValueError("step_retrieval_concurrency must be non-negative")
        return value

    @field_validator("evidence_max_snippets")
    @classmethod
    def _positive_snippets(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("evidence_max_snippets must be positive")
        return value

    @field_validator("evidence_min_support_score")
    @classmethod
    def _support_score_range(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("evidence_min_support_score must be between 0 and 1")
        return value

    @model_validator(mode="before")
    @classmethod
    def _apply_narration_preset(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        # ratio has higher priority than preset when both are provided.
        if payload.get("narration_ratio") is not None:
            return payload

        preset = str(payload.get("narration_preset", "medium"))
        if preset in NARRATION_PRESET_RATIOS:
            payload["narration_ratio"] = NARRATION_PRESET_RATIOS[preset]
        return payload

    @field_validator("narration_ratio")
    @classmethod
    def _validate_ratio(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not (0 < low < 1 and 0 < high < 1):
            raise ValueError("narration_ratio values must be in range (0, 1)")
        if low >= high:
            raise ValueError("narration_ratio[0] must be less than narration_ratio[1]")
        return value


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sqlite_path: Path = Field(default=Path("./data/novel.db"))
    lancedb_dir: Path = Field(default=Path("./data/lancedb"))


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: str = "sqlite"
    ttl_seconds: int = 2_592_000


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_json_error_payload: bool = True
    json_error_payload_max_chars: int = 0
    log_retry_attempts: bool = True

    @field_validator("json_error_payload_max_chars")
    @classmethod
    def _non_negative_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("json_error_payload_max_chars must be non-negative")
        return value


def default_llm_config() -> LLMConfig:
    return LLMConfig.model_validate(
        {
            "providers": {
                "default": {
                    "kind": "openai_compatible",
                    "base_url": None,
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
            "chat_endpoints": {
                "storyteller_default": {
                    "provider": "default",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.45,
                    "timeout_s": 60,
                    "max_concurrency": 4,
                    "retries": 3,
                },
            },
            "embedding_endpoints": {
                "embedding_default": {
                    "provider": "default",
                    "model": "text-embedding-3-large",
                    "timeout_s": 60,
                    "max_concurrency": 6,
                    "retries": 3,
                }
            },
            "routes": {
                "storyteller_chat": "storyteller_default",
                "embedding": "embedding_default",
            },
        }
    )


class AppConfigRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig = AppConfig()
    ingest: IngestConfig = IngestConfig()
    split: SplitConfig = SplitConfig()
    llm: LLMConfig = Field(default_factory=default_llm_config)
    summarize: SummarizeConfig = SummarizeConfig()
    storyteller: StorytellerConfig = StorytellerConfig()
    storage: StorageConfig = StorageConfig()
    cache: CacheConfig = CacheConfig()
    observability: ObservabilityConfig = ObservabilityConfig()


def resolve_paths(config: AppConfigRoot, base_dir: Path) -> AppConfigRoot:
    def _resolve(path_value: Path) -> Path:
        return path_value if path_value.is_absolute() else (base_dir / path_value).resolve()

    config.app.data_dir = _resolve(config.app.data_dir)
    config.app.output_dir = _resolve(config.app.output_dir)
    config.storage.sqlite_path = _resolve(config.storage.sqlite_path)
    config.storage.lancedb_dir = _resolve(config.storage.lancedb_dir)
    return config

