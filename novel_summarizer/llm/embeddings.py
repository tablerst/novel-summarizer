from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import OpenAIEmbeddings

from novel_summarizer.config.schema import AppConfigRoot


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]


def _build_embeddings_model(config: AppConfigRoot) -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {
        "model": config.llm.embedding_model,
    }

    if config.llm.base_url:
        kwargs["base_url"] = config.llm.base_url

    try:
        return OpenAIEmbeddings(**kwargs)
    except TypeError:
        kwargs.pop("base_url", None)
        return OpenAIEmbeddings(**kwargs)


class OpenAIEmbeddingClient:
    def __init__(self, config: AppConfigRoot):
        self.config = config
        self.model = _build_embeddings_model(config)

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        vectors = self.model.embed_documents(texts)
        return EmbeddingResult(vectors=vectors)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
