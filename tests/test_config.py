from __future__ import annotations

from pathlib import Path
import textwrap

import pytest
from pydantic import ValidationError

from novel_summarizer.config.loader import load_config
from novel_summarizer.config.schema import AppConfigRoot, LLMConfig, SplitConfig, resolve_paths


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


def test_llm_config_validates_temperature() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(temperature=2.5)


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
              chat_model: "gpt-default"
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
              chat_model: "gpt-profile"
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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NOVEL_SUMMARIZER_DATA_DIR", str(tmp_path / "env-data"))

    config = load_config(
        config_path=configs_dir / "custom.yaml",
        profile="fast",
        overrides={"app": {"output_dir": "./out-override"}},
        require_api_key=True,
    )

    assert config.app.data_dir == (tmp_path / "env-data").resolve()
    assert config.app.output_dir == (tmp_path / "out-override").resolve()
    assert config.llm.chat_model == "gpt-profile"
