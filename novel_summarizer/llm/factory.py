from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, TypeVar

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.llm.cache import SimpleCache


@dataclass
class LLMResponse:
    text: str
    cached: bool


T = TypeVar("T")


def _build_chat_model(config: AppConfigRoot) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": config.llm.chat_model,
        "temperature": config.llm.temperature,
        "timeout": config.llm.timeout_s,
        "max_retries": config.llm.retries,
    }

    if config.llm.base_url:
        kwargs["base_url"] = config.llm.base_url

    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        kwargs.pop("base_url", None)
        return ChatOpenAI(**kwargs)


def make_cache_key(*parts: str) -> str:
    joined = "::".join(parts)
    return sha256_text(joined)


class OpenAIChatClient:
    def __init__(self, config: AppConfigRoot, cache: SimpleCache):
        self.config = config
        self.cache = cache
        self.model = _build_chat_model(config)

    def complete(self, system_prompt: str, user_prompt: str, cache_key: str) -> LLMResponse:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            return LLMResponse(text=cached.value, cached=True)

        text = self._invoke_with_retry(system_prompt, user_prompt)
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        parser: Callable[[str], T],
    ) -> tuple[LLMResponse, T]:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            try:
                parsed = parser(cached.value)
                return LLMResponse(text=cached.value, cached=True), parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cached LLM response invalid; cache_key={} error={}", cache_key, exc)
                self.cache.delete(cache_key)

        text, parsed = self._invoke_with_retry(system_prompt, user_prompt, parser)
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False), parsed

    def _invoke_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[str], T] | None = None,
    ) -> tuple[str, T] | str:
        attempts = max(1, self.config.llm.retries + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
                text = str(response.content).strip()
                if not text:
                    raise ValueError("Empty LLM response")
                if parser is None:
                    return text
                parsed = parser(text)
                return text, parsed
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "LLM call failed (attempt {}/{}): {}",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt < attempts - 1:
                    time.sleep(min(0.5 * (2**attempt), 4.0))

        raise RuntimeError("LLM call failed after retries") from last_exc
