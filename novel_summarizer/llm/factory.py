from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Literal, TypeVar

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


@dataclass(frozen=True)
class ResolvedChatRuntime:
    route: str
    endpoint_name: str
    provider_name: str
    model: str
    temperature: float
    timeout_s: int
    max_concurrency: int
    retries: int
    base_url: str | None
    api_key_env: str | None
    api_key: str | None


T = TypeVar("T")


def resolve_chat_runtime(
    config: AppConfigRoot,
    route: Literal["summarize", "storyteller", "storyteller_entity", "storyteller_narration", "storyteller_refine"],
) -> ResolvedChatRuntime:
    endpoint_name, endpoint, provider = config.llm.resolve_chat_route(route)
    api_key = None
    if provider.api_key_env:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing required API key env for route '{route}': {provider.api_key_env}"
            )

    return ResolvedChatRuntime(
        route=route,
        endpoint_name=endpoint_name,
        provider_name=endpoint.provider,
        model=endpoint.model,
        temperature=endpoint.temperature,
        timeout_s=endpoint.timeout_s,
        max_concurrency=endpoint.max_concurrency,
        retries=endpoint.retries,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        api_key=api_key,
    )


def _build_chat_model(runtime: ResolvedChatRuntime) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": runtime.model,
        "temperature": runtime.temperature,
        "timeout": runtime.timeout_s,
        "max_retries": runtime.retries,
    }

    if runtime.base_url:
        kwargs["base_url"] = runtime.base_url
    if runtime.api_key:
        kwargs["api_key"] = runtime.api_key

    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        return ChatOpenAI(**kwargs)


def make_cache_key(*parts: str) -> str:
    joined = "::".join(parts)
    return sha256_text(joined)


class OpenAIChatClient:
    def __init__(
        self,
        config: AppConfigRoot,
        cache: SimpleCache,
        route: Literal[
            "summarize",
            "storyteller",
            "storyteller_entity",
            "storyteller_narration",
            "storyteller_refine",
        ] = "summarize",
    ):
        self.config = config
        self.cache = cache
        self.runtime = resolve_chat_runtime(config, route)
        self.model = _build_chat_model(self.runtime)
        self.model_identifier = f"{self.runtime.provider_name}/{self.runtime.endpoint_name}/{self.runtime.model}"

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
        attempts = max(1, self.runtime.retries + 1)
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
