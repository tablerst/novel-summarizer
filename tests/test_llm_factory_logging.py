from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from loguru import logger
import pytest

import novel_summarizer.llm.factory as llm_factory
from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.llm.cache import SimpleCache
from novel_summarizer.storyteller.json_utils import safe_load_json_dict


class _FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke(self, messages):
        _ = messages
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[index])


def _make_config(*, retries: int, log_payload: bool = True, payload_max_chars: int = 0) -> AppConfigRoot:
    return AppConfigRoot.model_validate(
        {
            "llm": {
                "providers": {
                    "fake": {
                        "kind": "openai_compatible",
                        "base_url": "https://api.example.com/v1",
                        "api_key_env": None,
                    }
                },
                "chat_endpoints": {
                    "storyteller_default": {
                        "provider": "fake",
                        "model": "fake-chat",
                        "temperature": 0.3,
                        "timeout_s": 30,
                        "max_concurrency": 1,
                        "retries": retries,
                    }
                },
                "embedding_endpoints": {
                    "embedding_default": {
                        "provider": "fake",
                        "model": "fake-embedding",
                        "timeout_s": 30,
                        "max_concurrency": 1,
                        "retries": 0,
                    }
                },
                "routes": {
                    "storyteller_chat": "storyteller_default",
                    "embedding": "embedding_default",
                },
            },
            "observability": {
                "log_json_error_payload": log_payload,
                "json_error_payload_max_chars": payload_max_chars,
                "log_retry_attempts": True,
            },
        }
    )


def test_complete_json_logs_raw_payload_on_parse_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_model = _FakeModel(["not json at all"])
    monkeypatch.setattr(llm_factory, "_build_chat_model", lambda runtime: fake_model)

    config = _make_config(retries=0, log_payload=True)
    cache = SimpleCache(False, "sqlite", tmp_path, ttl_seconds=0)
    client = llm_factory.OpenAIChatClient(config=config, cache=cache, route="storyteller_narration")

    records: list[Any] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        with pytest.raises(RuntimeError, match="LLM call failed after retries"):
            client.complete_json(
                "system",
                "user",
                "cache-key-001",
                safe_load_json_dict,
                context={
                    "node": "storyteller_generate",
                    "chapter_id": 7,
                    "chapter_idx": 3,
                    "input_hash": "abc123456789",
                },
            )
    finally:
        logger.remove(sink_id)
        cache.close()

    assert any("JSON parse failed source=llm_response" in record["message"] for record in records)
    assert any("JSON parse raw_response=not json at all" in record["message"] for record in records)
    assert any(
        record["extra"].get("node") == "storyteller_generate"
        and record["extra"].get("chapter_id") == 7
        and record["extra"].get("chapter_idx") == 3
        for record in records
    )


def test_complete_json_logs_and_deletes_invalid_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_model = _FakeModel(['{"narration":"ok"}'])
    monkeypatch.setattr(llm_factory, "_build_chat_model", lambda runtime: fake_model)

    config = _make_config(retries=0, log_payload=True)
    cache = SimpleCache(True, "sqlite", tmp_path, ttl_seconds=3600)
    cache_key = "cache-key-002"
    cache.set(cache_key, "cached value not json")

    client = llm_factory.OpenAIChatClient(config=config, cache=cache, route="storyteller_narration")

    records: list[Any] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        response, payload = client.complete_json("system", "user", cache_key, safe_load_json_dict)
    finally:
        logger.remove(sink_id)
        cache.close()

    assert response.cached is False
    assert payload["narration"] == "ok"
    assert fake_model.calls == 1
    assert any("JSON parse failed source=cache" in record["message"] for record in records)
    assert any("Deleting invalid cached LLM response" in record["message"] for record in records)


def test_json_error_payload_respects_truncation_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    invalid_payload = "x" * 80
    fake_model = _FakeModel([invalid_payload])
    monkeypatch.setattr(llm_factory, "_build_chat_model", lambda runtime: fake_model)

    config = _make_config(retries=0, log_payload=True, payload_max_chars=20)
    cache = SimpleCache(False, "sqlite", tmp_path, ttl_seconds=0)
    client = llm_factory.OpenAIChatClient(config=config, cache=cache, route="storyteller_narration")

    records: list[Any] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        with pytest.raises(RuntimeError):
            client.complete_json("system", "user", "cache-key-003", safe_load_json_dict)
    finally:
        logger.remove(sink_id)
        cache.close()

    assert any("JSON parse raw_response=" in record["message"] for record in records)
    assert any("[truncated" in record["message"] for record in records)
