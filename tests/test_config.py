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


def test_llm_route_storyteller_node_specific_fallback() -> None:
    config = AppConfigRoot()

    endpoint_name_entity, _, _ = config.llm.resolve_chat_route("storyteller_entity")
    endpoint_name_narration, _, _ = config.llm.resolve_chat_route("storyteller_narration")
    endpoint_name_refine, _, _ = config.llm.resolve_chat_route("storyteller_refine")

    assert endpoint_name_entity == config.llm.routes.storyteller_chat
    assert endpoint_name_narration == config.llm.routes.storyteller_chat
    assert endpoint_name_refine == config.llm.routes.storyteller_chat


def test_observability_config_defaults_and_validation() -> None:
    config = AppConfigRoot()
    assert config.observability.log_json_error_payload is True
    assert config.observability.json_error_payload_max_chars == 0
    assert config.observability.log_retry_attempts is True

    with pytest.raises(ValidationError):
        AppConfigRoot.model_validate(
            {
                "observability": {
                    "json_error_payload_max_chars": -1,
                }
            }
        )


def test_default_llm_config_is_v2_focused() -> None:
    config = AppConfigRoot()

    assert "storyteller_default" in config.llm.chat_endpoints
    assert "summarize_default" not in config.llm.chat_endpoints
    assert config.llm.routes.summarize_chat is None


def test_default_ingest_encoding_is_auto() -> None:
    config = AppConfigRoot()

    assert config.ingest.encoding == "auto"


def test_storyteller_narration_preset_defaults_to_medium_ratio() -> None:
    config = AppConfigRoot()

    assert config.storyteller.narration_preset == "medium"
    assert config.storyteller.narration_ratio == (0.4, 0.5)
    assert config.storyteller.prefetch_window == 0


@pytest.mark.parametrize(
    ("preset", "expected_ratio"),
    [
        ("short", (0.2, 0.3)),
        ("medium", (0.4, 0.5)),
        ("long", (0.65, 0.8)),
    ],
)
def test_storyteller_narration_preset_applies_ratio(
    preset: str,
    expected_ratio: tuple[float, float],
) -> None:
    config = AppConfigRoot.model_validate({"storyteller": {"narration_preset": preset}})

    assert config.storyteller.narration_ratio == expected_ratio


def test_storyteller_narration_ratio_overrides_preset() -> None:
    config = AppConfigRoot.model_validate(
        {
            "storyteller": {
                "narration_preset": "long",
                "narration_ratio": (0.22, 0.28),
            }
        }
    )

    assert config.storyteller.narration_preset == "long"
    assert config.storyteller.narration_ratio == (0.22, 0.28)


def test_llm_route_summarize_falls_back_to_storyteller() -> None:
    config = AppConfigRoot()

    endpoint_name, endpoint, _ = config.llm.resolve_chat_route("summarize")

    assert endpoint_name == config.llm.routes.storyteller_chat
    assert endpoint.model == config.llm.chat_endpoints[config.llm.routes.storyteller_chat].model


def test_storyteller_prefetch_window_validation() -> None:
    with pytest.raises(ValidationError):
        AppConfigRoot.model_validate({"storyteller": {"prefetch_window": -1}})


def test_llm_route_storyteller_node_specific_override() -> None:
    custom = AppConfigRoot.model_validate(
        {
            "llm": {
                "providers": {
                    "p": {
                        "kind": "openai_compatible",
                        "base_url": "https://api.example.com/v1",
                        "api_key_env": None,
                    }
                },
                "chat_endpoints": {
                    "summarize_default": {
                        "provider": "p",
                        "model": "s",
                    },
                    "storyteller_default": {
                        "provider": "p",
                        "model": "g",
                    },
                    "storyteller_entity_fast": {
                        "provider": "p",
                        "model": "e-fast",
                    },
                    "storyteller_narration_quality": {
                        "provider": "p",
                        "model": "n-quality",
                    },
                    "storyteller_refine_quality": {
                        "provider": "p",
                        "model": "r-quality",
                    },
                },
                "embedding_endpoints": {
                    "embedding_default": {
                        "provider": "p",
                        "model": "emb",
                    }
                },
                "routes": {
                    "summarize_chat": "summarize_default",
                    "storyteller_chat": "storyteller_default",
                    "storyteller_entity_chat": "storyteller_entity_fast",
                    "storyteller_narration_chat": "storyteller_narration_quality",
                    "storyteller_refine_chat": "storyteller_refine_quality",
                    "embedding": "embedding_default",
                },
            }
        }
    )

    endpoint_name_entity, endpoint_entity, _ = custom.llm.resolve_chat_route("storyteller_entity")
    endpoint_name_narration, endpoint_narration, _ = custom.llm.resolve_chat_route("storyteller_narration")
    endpoint_name_refine, endpoint_refine, _ = custom.llm.resolve_chat_route("storyteller_refine")

    assert endpoint_name_entity == "storyteller_entity_fast"
    assert endpoint_entity.model == "e-fast"
    assert endpoint_name_narration == "storyteller_narration_quality"
    assert endpoint_narration.model == "n-quality"
    assert endpoint_name_refine == "storyteller_refine_quality"
    assert endpoint_refine.model == "r-quality"
