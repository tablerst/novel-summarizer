from __future__ import annotations

import pytest

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.llm.embeddings import OpenAIEmbeddingClient


def _base_llm_config(base_url: str) -> dict:
    return {
        "providers": {
            "chat_provider": {
                "kind": "openai_compatible",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": None,
            },
            "local_embedding": {
                "kind": "ollama",
                "base_url": base_url,
                "api_key_env": None,
            },
        },
        "chat_endpoints": {
            "summarize_default": {
                "provider": "chat_provider",
                "model": "deepseek-v3.2",
                "temperature": 0.1,
                "timeout_s": 60,
                "max_concurrency": 6,
                "retries": 3,
            },
            "storyteller_default": {
                "provider": "chat_provider",
                "model": "deepseek-v3.2",
                "temperature": 0.45,
                "timeout_s": 60,
                "max_concurrency": 4,
                "retries": 3,
            },
        },
        "embedding_endpoints": {
            "embedding_default": {
                "provider": "local_embedding",
                "model": "qwen3-embedding:4b",
                "timeout_s": 60,
                "max_concurrency": 6,
                "retries": 3,
            }
        },
        "routes": {
            "summarize_chat": "summarize_default",
            "storyteller_chat": "storyteller_default",
            "embedding": "embedding_default",
        },
    }


@pytest.mark.parametrize(
    ("base_url", "expected_embed_url"),
    [
        ("http://localhost:11434", "http://localhost:11434/api/embed"),
        ("http://localhost:11434/api", "http://localhost:11434/api/embed"),
    ],
)
def test_ollama_embedding_url_normalization(base_url: str, expected_embed_url: str) -> None:
    config = AppConfigRoot.model_validate({"llm": _base_llm_config(base_url)})

    client = OpenAIEmbeddingClient(config)

    assert client.runtime.provider_kind == "ollama"
    assert getattr(client.model, "embed_url") == expected_embed_url
