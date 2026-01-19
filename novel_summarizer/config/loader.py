from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot, resolve_paths


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"")
        os.environ.setdefault(key, value)


def _apply_env(config_data: dict[str, Any]) -> dict[str, Any]:
    llm = config_data.setdefault("llm", {})
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        llm["base_url"] = base_url

    data_dir = os.getenv("NOVEL_SUMMARIZER_DATA_DIR")
    if data_dir:
        app = config_data.setdefault("app", {})
        app["data_dir"] = data_dir
    return config_data


def load_config(
    config_path: Path | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
    require_api_key: bool = True,
) -> AppConfigRoot:
    base_dir = Path.cwd()
    _load_dotenv(base_dir / ".env")

    config_data: dict[str, Any] = {}
    config_data = _deep_merge(config_data, _read_yaml(base_dir / "configs" / "default.yaml"))

    if profile:
        profile_path = base_dir / "configs" / "profiles" / f"{profile}.yaml"
        config_data = _deep_merge(config_data, _read_yaml(profile_path))

    if config_path:
        config_data = _deep_merge(config_data, _read_yaml(config_path))

    if overrides:
        config_data = _deep_merge(config_data, overrides)

    config_data = _apply_env(config_data)

    config = AppConfigRoot.model_validate(config_data)
    config = resolve_paths(config, base_dir)

    if require_api_key and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY in environment or .env file")

    logger.debug("Loaded config from %s", base_dir)
    return config


def masked_env_snapshot() -> dict[str, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    data_dir = os.getenv("NOVEL_SUMMARIZER_DATA_DIR")
    return {
        "OPENAI_API_KEY": "***" if api_key else None,
        "OPENAI_BASE_URL": base_url,
        "NOVEL_SUMMARIZER_DATA_DIR": data_dir,
    }
