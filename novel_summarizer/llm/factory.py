from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Literal, Mapping, TypeVar

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

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
S = TypeVar("S", bound=BaseModel)


def _short_key(value: str | None, length: int = 12) -> str:
    if not value:
        return "-"
    return value[:length]


def _coerce_payload_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    return str(payload)


def _extract_json_error_location(exc: Exception) -> str | None:
    lineno = getattr(exc, "lineno", None)
    colno = getattr(exc, "colno", None)
    pos = getattr(exc, "pos", None)

    parts: list[str] = []
    if isinstance(lineno, int):
        parts.append(f"line={lineno}")
    if isinstance(colno, int):
        parts.append(f"column={colno}")
    if isinstance(pos, int):
        parts.append(f"pos={pos}")

    if not parts:
        return None
    return ", ".join(parts)


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
        # Retry is handled explicitly in OpenAIChatClient to keep attempt count predictable.
        # Keeping SDK retries enabled here multiplies retries (SDK * wrapper) and can
        # dramatically increase latency under provider instability.
        "max_retries": 0,
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
        ] = "storyteller",
    ):
        self.config = config
        self.cache = cache
        self.runtime = resolve_chat_runtime(config, route)
        self.model = _build_chat_model(self.runtime)
        self.model_identifier = f"{self.runtime.provider_name}/{self.runtime.endpoint_name}/{self.runtime.model}"
        self._async_semaphore = asyncio.Semaphore(max(1, self.runtime.max_concurrency))

    def _build_log_context(
        self,
        *,
        cache_key: str | None = None,
        attempt: int | None = None,
        attempts_total: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {
            "route": self.runtime.route,
            "provider": self.runtime.provider_name,
            "endpoint": self.runtime.endpoint_name,
            "model": self.runtime.model,
        }
        if context:
            for key, value in context.items():
                if value is not None:
                    merged[key] = value
        if cache_key:
            merged["cache_key"] = _short_key(cache_key)
        if "input_hash" in merged:
            merged["input_hash"] = _short_key(str(merged["input_hash"]))
        if attempt is not None and attempts_total is not None:
            merged["attempt"] = f"{attempt}/{attempts_total}"
        return merged

    def _format_payload_for_log(self, payload: str) -> str:
        max_chars = int(self.config.observability.json_error_payload_max_chars)
        if max_chars <= 0 or len(payload) <= max_chars:
            return payload

        head = max_chars // 2
        tail = max_chars - head
        omitted = max(0, len(payload) - max_chars)
        if head <= 0 or tail <= 0:
            return payload[:max_chars]

        return f"{payload[:head]}\n...[truncated {omitted} chars]...\n{payload[-tail:]}"

    def _log_json_parse_failure(
        self,
        *,
        source: str,
        raw_text: str,
        exc: Exception,
        cache_key: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        log = logger.bind(**self._build_log_context(cache_key=cache_key, context=context))
        location = _extract_json_error_location(exc)
        raw_hash = sha256_text(raw_text)

        log.warning(
            "JSON parse failed source={} error_type={} error={} location={} raw_len={} raw_hash={}",
            source,
            type(exc).__name__,
            exc,
            location or "-",
            len(raw_text),
            raw_hash,
        )

        if self.config.observability.log_json_error_payload:
            payload_to_log = self._format_payload_for_log(raw_text)
            log.warning("JSON parse raw_response={}", payload_to_log)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            return LLMResponse(text=cached.value, cached=True)

        text = self._invoke_with_retry(system_prompt, user_prompt, cache_key=cache_key, context=context)
        if not isinstance(text, str):
            raise RuntimeError("Unexpected non-text response from LLM")
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False)

    async def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            return LLMResponse(text=cached.value, cached=True)

        text = await self._ainvoke_with_retry(system_prompt, user_prompt, cache_key=cache_key, context=context)
        if not isinstance(text, str):
            raise RuntimeError("Unexpected non-text response from LLM")
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        parser: Callable[[str], T],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[LLMResponse, T]:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            try:
                parsed = parser(cached.value)
                return LLMResponse(text=cached.value, cached=True), parsed
            except Exception as exc:  # noqa: BLE001
                self._log_json_parse_failure(
                    source="cache",
                    raw_text=cached.value,
                    exc=exc,
                    cache_key=cache_key,
                    context=context,
                )
                logger.bind(**self._build_log_context(cache_key=cache_key, context=context)).warning(
                    "Deleting invalid cached LLM response"
                )
                self.cache.delete(cache_key)

        result = self._invoke_with_retry(system_prompt, user_prompt, parser, cache_key=cache_key, context=context)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Unexpected non-JSON response from LLM")
        text, parsed = result
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False), parsed

    async def complete_json_async(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        parser: Callable[[str], T],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[LLMResponse, T]:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            try:
                parsed = parser(cached.value)
                return LLMResponse(text=cached.value, cached=True), parsed
            except Exception as exc:  # noqa: BLE001
                self._log_json_parse_failure(
                    source="cache",
                    raw_text=cached.value,
                    exc=exc,
                    cache_key=cache_key,
                    context=context,
                )
                logger.bind(**self._build_log_context(cache_key=cache_key, context=context)).warning(
                    "Deleting invalid cached LLM response"
                )
                self.cache.delete(cache_key)

        result = await self._ainvoke_with_retry(system_prompt, user_prompt, parser, cache_key=cache_key, context=context)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Unexpected non-JSON response from LLM")
        text, parsed = result
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False), parsed

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        schema: type[S],
        *,
        method: Literal["function_calling", "json_schema", "json_mode"] = "function_calling",
        context: Mapping[str, Any] | None = None,
    ) -> tuple[LLMResponse, S]:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            try:
                parsed = schema.model_validate_json(cached.value)
                return LLMResponse(text=cached.value, cached=True), parsed
            except Exception as exc:  # noqa: BLE001
                self._log_json_parse_failure(
                    source="cache_structured",
                    raw_text=cached.value,
                    exc=exc,
                    cache_key=cache_key,
                    context=context,
                )
                logger.bind(**self._build_log_context(cache_key=cache_key, context=context)).warning(
                    "Deleting invalid cached structured response"
                )
                self.cache.delete(cache_key)

        parsed = self._invoke_structured_with_retry(
            system_prompt,
            user_prompt,
            schema=schema,
            method=method,
            cache_key=cache_key,
            context=context,
        )
        text = parsed.model_dump_json()
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False), parsed

    async def complete_structured_async(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: str,
        schema: type[S],
        *,
        method: Literal["function_calling", "json_schema", "json_mode"] = "function_calling",
        context: Mapping[str, Any] | None = None,
    ) -> tuple[LLMResponse, S]:
        cached = self.cache.get(cache_key)
        if cached.hit and cached.value is not None:
            try:
                parsed = schema.model_validate_json(cached.value)
                return LLMResponse(text=cached.value, cached=True), parsed
            except Exception as exc:  # noqa: BLE001
                self._log_json_parse_failure(
                    source="cache_structured",
                    raw_text=cached.value,
                    exc=exc,
                    cache_key=cache_key,
                    context=context,
                )
                logger.bind(**self._build_log_context(cache_key=cache_key, context=context)).warning(
                    "Deleting invalid cached structured response"
                )
                self.cache.delete(cache_key)

        parsed = await self._ainvoke_structured_with_retry(
            system_prompt,
            user_prompt,
            schema=schema,
            method=method,
            cache_key=cache_key,
            context=context,
        )
        text = parsed.model_dump_json()
        self.cache.set(cache_key, text)
        return LLMResponse(text=text, cached=False), parsed

    async def _invoke_on_worker(self, fn: Callable[..., Any], *args: Any) -> Any:
        async with self._async_semaphore:
            return await asyncio.to_thread(fn, *args)

    def _invoke_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[str], T] | None = None,
        *,
        cache_key: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[str, T] | str:
        attempts = max(1, self.runtime.retries + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            attempt_started = time.perf_counter()
            try:
                response = self.model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
                text = str(response.content).strip()
                if not text:
                    raise ValueError("Empty LLM response")
                if parser is None:
                    return text
                try:
                    parsed = parser(text)
                except Exception as parse_exc:  # noqa: BLE001
                    self._log_json_parse_failure(
                        source="llm_response",
                        raw_text=text,
                        exc=parse_exc,
                        cache_key=cache_key or "-",
                        context=self._build_log_context(
                            attempt=attempt + 1,
                            attempts_total=attempts,
                            context=context,
                        ),
                    )
                    raise
                return text, parsed
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                log = logger.bind(
                    **self._build_log_context(
                        cache_key=cache_key,
                        attempt=attempt + 1,
                        attempts_total=attempts,
                        context=context,
                    )
                )
                if self.config.observability.log_retry_attempts:
                    log.warning(
                        "LLM call failed elapsed_ms={} error_type={} error={}",
                        elapsed_ms,
                        type(exc).__name__,
                        exc,
                    )
                if attempt == attempts - 1:
                    log.exception("LLM call failed on final attempt")
                if attempt < attempts - 1:
                    time.sleep(min(0.5 * (2**attempt), 4.0))

        raise RuntimeError("LLM call failed after retries") from last_exc

    async def _ainvoke_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[str], T] | None = None,
        *,
        cache_key: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[str, T] | str:
        attempts = max(1, self.runtime.retries + 1)
        last_exc: Exception | None = None
        messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

        for attempt in range(attempts):
            attempt_started = time.perf_counter()
            try:
                response = await self._invoke_on_worker(self.model.invoke, messages)
                text = str(response.content).strip()
                if not text:
                    raise ValueError("Empty LLM response")
                if parser is None:
                    return text
                try:
                    parsed = parser(text)
                except Exception as parse_exc:  # noqa: BLE001
                    self._log_json_parse_failure(
                        source="llm_response",
                        raw_text=text,
                        exc=parse_exc,
                        cache_key=cache_key or "-",
                        context=self._build_log_context(
                            attempt=attempt + 1,
                            attempts_total=attempts,
                            context=context,
                        ),
                    )
                    raise
                return text, parsed
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                log = logger.bind(
                    **self._build_log_context(
                        cache_key=cache_key,
                        attempt=attempt + 1,
                        attempts_total=attempts,
                        context=context,
                    )
                )
                if self.config.observability.log_retry_attempts:
                    log.warning(
                        "LLM call failed elapsed_ms={} error_type={} error= {}",
                        elapsed_ms,
                        type(exc).__name__,
                        exc,
                    )
                if attempt == attempts - 1:
                    log.exception("LLM call failed on final attempt")
                if attempt < attempts - 1:
                    await asyncio.sleep(min(0.5 * (2**attempt), 4.0))

        raise RuntimeError("LLM call failed after retries") from last_exc

    def _invoke_structured_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: type[S],
        method: Literal["function_calling", "json_schema", "json_mode"],
        cache_key: str,
        context: Mapping[str, Any] | None = None,
    ) -> S:
        attempts = max(1, self.runtime.retries + 1)
        last_exc: Exception | None = None

        structured_model = None
        for builder in (
            lambda: self.model.with_structured_output(schema, method=method, include_raw=True),
            lambda: self.model.with_structured_output(schema, method=method),
            lambda: self.model.with_structured_output(schema, include_raw=True),
            lambda: self.model.with_structured_output(schema),
        ):
            try:
                structured_model = builder()
                break
            except TypeError:
                continue

        if structured_model is None:
            raise RuntimeError("Structured output is not supported by current model client")

        for attempt in range(attempts):
            attempt_started = time.perf_counter()
            try:
                response = structured_model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])

                parsed_payload: Any = response
                if isinstance(response, dict) and "parsed" in response:
                    parsing_error = response.get("parsing_error")
                    if parsing_error:
                        raw_payload = _coerce_payload_text(response.get("raw"))
                        if raw_payload:
                            self._log_json_parse_failure(
                                source="structured_response",
                                raw_text=raw_payload,
                                exc=ValueError(str(parsing_error)),
                                cache_key=cache_key,
                                context=self._build_log_context(
                                    attempt=attempt + 1,
                                    attempts_total=attempts,
                                    context=context,
                                ),
                            )
                        raise ValueError(f"Structured output parsing failed: {parsing_error}")
                    parsed_payload = response.get("parsed")

                if parsed_payload is None:
                    if isinstance(response, dict):
                        raw_payload = _coerce_payload_text(response.get("raw"))
                        if raw_payload:
                            self._log_json_parse_failure(
                                source="structured_response_empty_parsed",
                                raw_text=raw_payload,
                                exc=ValueError("Empty structured output response"),
                                cache_key=cache_key,
                                context=self._build_log_context(
                                    attempt=attempt + 1,
                                    attempts_total=attempts,
                                    context=context,
                                ),
                            )
                    raise ValueError("Empty structured output response")

                if isinstance(parsed_payload, schema):
                    return parsed_payload
                return schema.model_validate(parsed_payload)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                log = logger.bind(
                    **self._build_log_context(
                        cache_key=cache_key,
                        attempt=attempt + 1,
                        attempts_total=attempts,
                        context=context,
                    )
                )
                if self.config.observability.log_retry_attempts:
                    log.warning(
                        "LLM structured call failed elapsed_ms={} error_type={} error={}",
                        elapsed_ms,
                        type(exc).__name__,
                        exc,
                    )
                if attempt == attempts - 1:
                    log.exception("LLM structured call failed on final attempt")
                if attempt < attempts - 1:
                    time.sleep(min(0.5 * (2**attempt), 4.0))

        raise RuntimeError("LLM structured call failed after retries") from last_exc

    async def _ainvoke_structured_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: type[S],
        method: Literal["function_calling", "json_schema", "json_mode"],
        cache_key: str,
        context: Mapping[str, Any] | None = None,
    ) -> S:
        attempts = max(1, self.runtime.retries + 1)
        last_exc: Exception | None = None

        structured_model = None
        for builder in (
            lambda: self.model.with_structured_output(schema, method=method, include_raw=True),
            lambda: self.model.with_structured_output(schema, method=method),
            lambda: self.model.with_structured_output(schema, include_raw=True),
            lambda: self.model.with_structured_output(schema),
        ):
            try:
                structured_model = builder()
                break
            except TypeError:
                continue

        if structured_model is None:
            raise RuntimeError("Structured output is not supported by current model client")

        messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

        for attempt in range(attempts):
            attempt_started = time.perf_counter()
            try:
                response = await self._invoke_on_worker(structured_model.invoke, messages)

                parsed_payload: Any = response
                if isinstance(response, dict) and "parsed" in response:
                    parsing_error = response.get("parsing_error")
                    if parsing_error:
                        raw_payload = _coerce_payload_text(response.get("raw"))
                        if raw_payload:
                            self._log_json_parse_failure(
                                source="structured_response",
                                raw_text=raw_payload,
                                exc=ValueError(str(parsing_error)),
                                cache_key=cache_key,
                                context=self._build_log_context(
                                    attempt=attempt + 1,
                                    attempts_total=attempts,
                                    context=context,
                                ),
                            )
                        raise ValueError(f"Structured output parsing failed: {parsing_error}")
                    parsed_payload = response.get("parsed")

                if parsed_payload is None:
                    if isinstance(response, dict):
                        raw_payload = _coerce_payload_text(response.get("raw"))
                        if raw_payload:
                            self._log_json_parse_failure(
                                source="structured_response_empty_parsed",
                                raw_text=raw_payload,
                                exc=ValueError("Empty structured output response"),
                                cache_key=cache_key,
                                context=self._build_log_context(
                                    attempt=attempt + 1,
                                    attempts_total=attempts,
                                    context=context,
                                ),
                            )
                    raise ValueError("Empty structured output response")

                if isinstance(parsed_payload, schema):
                    return parsed_payload
                return schema.model_validate(parsed_payload)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                log = logger.bind(
                    **self._build_log_context(
                        cache_key=cache_key,
                        attempt=attempt + 1,
                        attempts_total=attempts,
                        context=context,
                    )
                )
                if self.config.observability.log_retry_attempts:
                    log.warning(
                        "LLM structured call failed elapsed_ms={} error_type={} error={} ",
                        elapsed_ms,
                        type(exc).__name__,
                        exc,
                    )
                if attempt == attempts - 1:
                    log.exception("LLM structured call failed on final attempt")
                if attempt < attempts - 1:
                    await asyncio.sleep(min(0.5 * (2**attempt), 4.0))

        raise RuntimeError("LLM structured call failed after retries") from last_exc
