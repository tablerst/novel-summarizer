from __future__ import annotations

from pathlib import Path
import textwrap

import pytest
from pydantic import ValidationError

from novel_summarizer.config.loader import load_config
from novel_summarizer.config.schema import AppConfigRoot, ChatEndpointConfig, LLMConfig, SplitConfig, resolve_paths


def test_resolve_paths_makes_absolute(tmp_path: Path) -> None:
    config = AppConfigRoot()
    config.app.data_dir = Path("data")
    config.app.output_dir = Path("output")
    config.storage.sqlite_path = Path("data/novel.db")
    config.storage.lancedb_dir = Path("data/lancedb")

    resolved = resolve_paths(config, tmp_path)

    assert resolved.app.data_dir == (tmp_path / "data").resolve()
    assert resolved.app.output_dir == (tmp_path / "output").resolve()
    assert resolved.storage.sqlite_path == (tmp_path / "data/novel.db").resolve()
    assert resolved.storage.lancedb_dir == (tmp_path / "data/lancedb").resolve()


def test_split_config_validates_overlap_and_positive() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(chunk_size_tokens=0, chunk_overlap_tokens=1, min_chunk_tokens=1)

    with pytest.raises(ValidationError):
        SplitConfig(chunk_size_tokens=10, chunk_overlap_tokens=10, min_chunk_tokens=1)


def test_chat_endpoint_validates_temperature() -> None:
    with pytest.raises(ValidationError):
        ChatEndpointConfig(provider="p", model="m", temperature=2.5)


def test_llm_config_validates_endpoint_provider_reference() -> None:
    with pytest.raises(ValidationError):
        LLMConfig.model_validate(
            {
                "providers": {
                    "p1": {"kind": "openai_compatible", "base_url": "https://x", "api_key_env": "KEY"},
                },
                "chat_endpoints": {
                    "summarize_default": {
                        "provider": "missing_provider",
                        "model": "m",
                        "temperature": 0.3,
                        "timeout_s": 60,
                        "max_concurrency": 6,
                        "retries": 3,
                    },
                    "storyteller_default": {
                        "provider": "p1",
                        "model": "m",
                        "temperature": 0.4,
                        "timeout_s": 60,
                        "max_concurrency": 4,
                        "retries": 3,
                    },
                },
                "embedding_endpoints": {
                    "embedding_default": {
                        "provider": "p1",
                        "model": "e",
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
        )


def test_load_config_merge_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs_dir = tmp_path / "configs"
    profiles_dir = configs_dir / "profiles"
    profiles_dir.mkdir(parents=True)

    (configs_dir / "default.yaml").write_text(
        textwrap.dedent(
            """
            app:
              data_dir: "./data-default"
              output_dir: "./out-default"
            llm:
                            providers:
                                openai:
                                    kind: "openai_compatible"
                                    base_url: "https://default-llm.example/v1"
                                    api_key_env: "OPENAI_API_KEY"
                            chat_endpoints:
                                summarize_default:
                                    provider: "openai"
                                    model: "gpt-default"
                                    temperature: 0.3
                                    timeout_s: 60
                                    max_concurrency: 6
                                    retries: 3
                                storyteller_default:
                                    provider: "openai"
                                    model: "gpt-story"
                                    temperature: 0.4
                                    timeout_s: 60
                                    max_concurrency: 4
                                    retries: 3
                            embedding_endpoints:
                                embedding_default:
                                    provider: "openai"
                                    model: "embed-default"
                                    timeout_s: 60
                                    max_concurrency: 6
                                    retries: 3
                            routes:
                                summarize_chat: "summarize_default"
                                storyteller_chat: "storyteller_default"
                                embedding: "embedding_default"
            """
        ).strip(),
        encoding="utf-8",
    )

    (profiles_dir / "fast.yaml").write_text(
        textwrap.dedent(
            """
            app:
              output_dir: "./out-profile"
            llm:
                            chat_endpoints:
                                summarize_default:
                                    model: "gpt-profile"
            """
        ).strip(),
        encoding="utf-8",
    )

    (configs_dir / "custom.yaml").write_text(
        textwrap.dedent(
            """
            app:
              output_dir: "./out-custom"
            """
        ).strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_SUMMARIZER_DATA_DIR", str(tmp_path / "env-data"))
    monkeypatch.setenv("NOVEL_SUMMARIZER_LLM_PROVIDER_OPENAI_BASE_URL", "https://env-llm.example/v1")

    config = load_config(
        config_path=configs_dir / "custom.yaml",
        profile="fast",
        overrides={"app": {"output_dir": "./out-override"}},
    )

    assert config.app.data_dir == (tmp_path / "env-data").resolve()
    assert config.app.output_dir == (tmp_path / "out-override").resolve()
    assert config.llm.chat_endpoints["summarize_default"].model == "gpt-profile"
    assert config.llm.providers["openai"].base_url == "https://env-llm.example/v1"
