from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from langchain_openai import OpenAIEmbeddings

from novel_summarizer.config.schema import AppConfigRoot


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]


@dataclass(frozen=True)
class ResolvedEmbeddingRuntime:
    endpoint_name: str
    provider_name: str
    model: str
    timeout_s: int
    max_concurrency: int
    retries: int
    base_url: str | None
    api_key_env: str | None
    api_key: str | None


def resolve_embedding_runtime(config: AppConfigRoot) -> ResolvedEmbeddingRuntime:
    endpoint_name, endpoint, provider = config.llm.resolve_embedding_route()

    api_key = None
    if provider.api_key_env:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing required API key env for embedding route: {provider.api_key_env}"
            )

    return ResolvedEmbeddingRuntime(
        endpoint_name=endpoint_name,
        provider_name=endpoint.provider,
        model=endpoint.model,
        timeout_s=endpoint.timeout_s,
        max_concurrency=endpoint.max_concurrency,
        retries=endpoint.retries,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        api_key=api_key,
    )


def _build_embeddings_model(runtime: ResolvedEmbeddingRuntime) -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {
        "model": runtime.model,
        "max_retries": runtime.retries,
    }

    if runtime.base_url:
        kwargs["base_url"] = runtime.base_url
    if runtime.api_key:
        kwargs["api_key"] = runtime.api_key

    try:
        return OpenAIEmbeddings(**kwargs)
    except TypeError:
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        kwargs.pop("max_retries", None)
        return OpenAIEmbeddings(**kwargs)


class OpenAIEmbeddingClient:
    def __init__(self, config: AppConfigRoot):
        self.config = config
        self.runtime = resolve_embedding_runtime(config)
        self.model = _build_embeddings_model(self.runtime)
        self.model_identifier = f"{self.runtime.provider_name}/{self.runtime.endpoint_name}/{self.runtime.model}"

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        vectors = self.model.embed_documents(texts)
        return EmbeddingResult(vectors=vectors)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
