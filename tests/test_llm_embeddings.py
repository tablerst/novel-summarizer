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


def test_openai_compatible_embedding_disables_token_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict = {}

    class _DummyEmbeddings:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    monkeypatch.setattr("novel_summarizer.llm.embeddings.OpenAIEmbeddings", _DummyEmbeddings)

    config = AppConfigRoot.model_validate(
        {
            "llm": {
                "providers": {
                    "embedding_provider": {
                        "kind": "openai_compatible",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": None,
                    },
                    "chat_provider": {
                        "kind": "openai_compatible",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": None,
                    },
                },
                "chat_endpoints": {
                    "storyteller_default": {
                        "provider": "chat_provider",
                        "model": "deepseek-chat",
                        "temperature": 0.3,
                        "timeout_s": 60,
                        "max_concurrency": 2,
                        "retries": 1,
                    }
                },
                "embedding_endpoints": {
                    "embedding_default": {
                        "provider": "embedding_provider",
                        "model": "text-embedding-v4",
                        "timeout_s": 60,
                        "max_concurrency": 2,
                        "retries": 1,
                    }
                },
                "routes": {
                    "storyteller_chat": "storyteller_default",
                    "embedding": "embedding_default",
                },
            }
        }
    )

    _ = OpenAIEmbeddingClient(config)

    assert captured_kwargs["check_embedding_ctx_length"] is False
    assert captured_kwargs["tiktoken_enabled"] is False


def test_openai_compatible_embedding_auto_batches_for_dashscope(monkeypatch: pytest.MonkeyPatch) -> None:
    call_sizes: list[int] = []

    class _DummyEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            call_sizes.append(len(texts))
            return [[0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    monkeypatch.setattr("novel_summarizer.llm.embeddings.OpenAIEmbeddings", _DummyEmbeddings)

    config = AppConfigRoot.model_validate(
        {
            "llm": {
                "providers": {
                    "embedding_provider": {
                        "kind": "openai_compatible",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": None,
                    },
                    "chat_provider": {
                        "kind": "openai_compatible",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": None,
                    },
                },
                "chat_endpoints": {
                    "storyteller_default": {
                        "provider": "chat_provider",
                        "model": "deepseek-chat",
                        "temperature": 0.3,
                        "timeout_s": 60,
                        "max_concurrency": 2,
                        "retries": 1,
                    }
                },
                "embedding_endpoints": {
                    "embedding_default": {
                        "provider": "embedding_provider",
                        "model": "text-embedding-v4",
                        "timeout_s": 60,
                        "max_concurrency": 2,
                        "retries": 1,
                    }
                },
                "routes": {
                    "storyteller_chat": "storyteller_default",
                    "embedding": "embedding_default",
                },
            }
        }
    )

    client = OpenAIEmbeddingClient(config)
    vectors = client.model.embed_documents([f"chunk-{idx}" for idx in range(21)])

    assert len(vectors) == 21
    assert call_sizes == [10, 10, 1]
