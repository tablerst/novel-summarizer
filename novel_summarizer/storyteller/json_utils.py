from __future__ import annotations

from typing import Any
import re

import orjson


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _sanitize_json_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def safe_load_json_dict(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty JSON text")

    candidate = _sanitize_json_text(_strip_code_fence(text))
    try:
        payload = orjson.loads(candidate)
    except orjson.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = orjson.loads(candidate[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def safe_load_json_list(text: str) -> list[Any]:
    if not text or not text.strip():
        raise ValueError("Empty JSON text")

    candidate = _sanitize_json_text(_strip_code_fence(text))
    try:
        payload = orjson.loads(candidate)
    except orjson.JSONDecodeError:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = orjson.loads(candidate[start : end + 1])

    if not isinstance(payload, list):
        raise ValueError("Expected JSON array")
    return payload