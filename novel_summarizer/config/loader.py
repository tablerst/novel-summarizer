from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re

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


def _provider_base_url_override_var(provider_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]", "_", provider_name).upper()
    return f"NOVEL_SUMMARIZER_LLM_PROVIDER_{normalized}_BASE_URL"


def _apply_env(config_data: dict[str, Any]) -> dict[str, Any]:
    llm = config_data.setdefault("llm", {})
    providers = llm.setdefault("providers", {})
    for provider_name, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict):
            continue
        base_url_override = os.getenv(_provider_base_url_override_var(provider_name))
        if base_url_override:
            provider_cfg["base_url"] = base_url_override

    data_dir = os.getenv("NOVEL_SUMMARIZER_DATA_DIR")
    if data_dir:
        app = config_data.setdefault("app", {})
        app["data_dir"] = data_dir
    return config_data


def load_config(
    config_path: Path | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
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

    logger.debug("Loaded config from %s", base_dir)
    return config


def masked_env_snapshot(config: AppConfigRoot | None = None) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    data_dir = os.getenv("NOVEL_SUMMARIZER_DATA_DIR")
    snapshot["NOVEL_SUMMARIZER_DATA_DIR"] = data_dir

    if config is None:
        return snapshot

    for provider_name, provider in config.llm.providers.items():
        override_var = _provider_base_url_override_var(provider_name)
        snapshot[override_var] = os.getenv(override_var)
        snapshot[f"llm.providers.{provider_name}.base_url"] = provider.base_url

        if provider.api_key_env:
            snapshot[provider.api_key_env] = "***" if os.getenv(provider.api_key_env) else None

    return snapshot
