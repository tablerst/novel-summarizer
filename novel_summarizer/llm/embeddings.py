from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Any

import httpx
from langchain_openai import OpenAIEmbeddings

from novel_summarizer.config.schema import AppConfigRoot


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]


@dataclass(frozen=True)
class ResolvedEmbeddingRuntime:
    endpoint_name: str
    provider_name: str
    provider_kind: str
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
        provider_kind=provider.kind,
        model=endpoint.model,
        timeout_s=endpoint.timeout_s,
        max_concurrency=endpoint.max_concurrency,
        retries=endpoint.retries,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        api_key=api_key,
    )


class OllamaEmbeddingsModel:
    def __init__(self, runtime: ResolvedEmbeddingRuntime):
        base_url = (runtime.base_url or "http://127.0.0.1:11434").rstrip("/")
        if base_url.endswith("/api"):
            self.embed_url = f"{base_url}/embed"
        else:
            self.embed_url = f"{base_url}/api/embed"
        self.model = runtime.model
        self.timeout_s = runtime.timeout_s
        self.retries = runtime.retries

    def _extract_embeddings(self, payload: dict[str, Any]) -> list[list[float]]:
        if "embeddings" in payload and isinstance(payload["embeddings"], list):
            values = payload["embeddings"]
            if values and isinstance(values[0], (int, float)):
                return [list(values)]
            return [list(item) for item in values]

        if "embedding" in payload and isinstance(payload["embedding"], list):
            return [list(payload["embedding"])]

        raise ValueError("Invalid Ollama embedding response: missing 'embeddings' or 'embedding'")

    def _embed(self, input_payload: str | list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        body = {"model": self.model, "input": input_payload}

        last_exc: Exception | None = None
        attempts = max(1, self.retries + 1)
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(self.embed_url, headers=headers, json=body)
                    response.raise_for_status()
                    payload = response.json()
                return self._extract_embeddings(payload)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(min(0.3 * (2**attempt), 2.0))

        raise RuntimeError("Ollama embedding request failed after retries") from last_exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._embed(text)
        if not embeddings:
            raise ValueError("Ollama embedding response is empty")
        return embeddings[0]


def _extract_batch_limit(error_text: str) -> int | None:
    match = re.search(r"larger than\s+(\d+)", error_text)
    if not match:
        return None
    try:
        limit = int(match.group(1))
    except ValueError:
        return None
    return limit if limit > 0 else None


class OpenAICompatibleEmbeddingsModel:
    def __init__(self, model: OpenAIEmbeddings, default_max_batch_size: int | None = None):
        self._model = model
        self._default_max_batch_size = default_max_batch_size

    def _embed_in_batches(self, texts: list[str], max_batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max_batch_size):
            chunk = texts[start : start + max_batch_size]
            vectors.extend(self._model.embed_documents(chunk))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._default_max_batch_size and len(texts) > self._default_max_batch_size:
            return self._embed_in_batches(texts, self._default_max_batch_size)

        try:
            return self._model.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            inferred_limit = _extract_batch_limit(str(exc))
            if inferred_limit is None or len(texts) <= inferred_limit:
                raise
            return self._embed_in_batches(texts, inferred_limit)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def _default_embedding_batch_limit(runtime: ResolvedEmbeddingRuntime) -> int | None:
    base = (runtime.base_url or "").lower()
    if "dashscope.aliyuncs.com" in base:
        return 10
    return None


def _build_openai_embeddings_model(runtime: ResolvedEmbeddingRuntime) -> OpenAICompatibleEmbeddingsModel:
    kwargs: dict[str, Any] = {
        "model": runtime.model,
        "max_retries": runtime.retries,
        # Some OpenAI-compatible providers (for example DashScope compatible mode)
        # reject token-array embedding inputs and only accept raw strings.
        # Disable local token-length preprocessing to keep request inputs as strings.
        "check_embedding_ctx_length": False,
        "tiktoken_enabled": False,
    }

    if runtime.base_url:
        kwargs["base_url"] = runtime.base_url
    if runtime.api_key:
        kwargs["api_key"] = runtime.api_key

    try:
        model = OpenAIEmbeddings(**kwargs)
    except TypeError:
        kwargs.pop("check_embedding_ctx_length", None)
        kwargs.pop("tiktoken_enabled", None)
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        kwargs.pop("max_retries", None)
        model = OpenAIEmbeddings(**kwargs)

    return OpenAICompatibleEmbeddingsModel(model, default_max_batch_size=_default_embedding_batch_limit(runtime))


def _build_embeddings_model(runtime: ResolvedEmbeddingRuntime):
    if runtime.provider_kind == "ollama":
        return OllamaEmbeddingsModel(runtime)
    return _build_openai_embeddings_model(runtime)


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
