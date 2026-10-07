"""LLM access layer: OpenRouter (OpenAI-compatible) via LangChain's ChatOpenAI.

* get_llm(role)          - configured chat model; model names come from env vars
* invoke_llm(...)        - invoke with retry/backoff on rate limits and transient errors,
                           and write a 'llm' trace record
* invoke_structured(...) - force a Pydantic-shaped answer, validate it, log any
                           validation failure, attempt one repair, never raise

A test hook (set_llm_factory) lets offline tests swap in a fake model.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'OpenRouter ChatOpenAI factory with
# tenacity retries and a validate-and-repair structured output helper', Date: 2026-10-07
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import get_settings
from .observability.tracer import trace_call, tracer

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

try:  # openai is a dependency of langchain-openai
    import openai

    RETRYABLE: tuple[type[BaseException], ...] = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
except Exception:  # pragma: no cover
    RETRYABLE = (TimeoutError, ConnectionError)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def _default_factory(role: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    s = get_settings()
    model = s.sentiment_model if role == "sentiment" else s.model
    return ChatOpenAI(
        model=model,
        api_key=s.require_api_key(),
        base_url=s.openrouter_base_url,
        temperature=s.temperature,
        max_tokens=s.max_output_tokens,
        timeout=s.request_timeout,
        max_retries=0,  # retries are handled by tenacity in invoke_llm
        default_headers={"HTTP-Referer": s.app_url, "X-Title": s.app_name},
    )


_factory: Callable[[str], BaseChatModel] = _default_factory


def set_llm_factory(factory: Callable[[str], BaseChatModel] | None) -> None:
    """Override how chat models are created (used by offline tests)."""
    global _factory
    _factory = factory or _default_factory


def get_llm(role: str = "agent") -> BaseChatModel:
    """role: 'agent' (default) or 'sentiment'. Model names come from env vars."""
    return _factory(role)


def model_name(role: str = "agent") -> str:
    s = get_settings()
    return s.sentiment_model if role == "sentiment" else s.model


# --------------------------------------------------------------------------- #
# Invocation with retry + tracing
# --------------------------------------------------------------------------- #
def _summarise(msg: Any) -> str:
    if isinstance(msg, AIMessage):
        calls = [f"{c['name']}({json.dumps(c.get('args', {}), default=str)})" for c in msg.tool_calls]
        text = (msg.content or "") if isinstance(msg.content, str) else str(msg.content)
        return (text + (" | tool_calls: " + "; ".join(calls) if calls else "")).strip()
    return str(msg)


def invoke_llm(runnable: Any, messages: Sequence[BaseMessage], label: str,
               role: str = "agent") -> AIMessage:
    """Invoke a chat model (optionally tool-bound) with exponential backoff."""
    s = get_settings()
    retrying = Retrying(
        stop=stop_after_attempt(s.llm_max_attempts),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )
    with trace_call(label, {"model": model_name(role), "n_messages": len(messages)}, kind="llm") as span:
        for attempt in retrying:
            with attempt:
                result = runnable.invoke(list(messages))
        span.finish(_summarise(result))
    return result


# --------------------------------------------------------------------------- #
# Structured output with validation, logging and one repair attempt
# --------------------------------------------------------------------------- #
@dataclass
class StructuredResult(Generic[T]):
    ok: bool
    value: T | None
    error: str | None
    attempts: int


def _extract_args(msg: AIMessage, tool_name: str) -> dict | None:
    for call in msg.tool_calls or []:
        if call.get("name") == tool_name:
            return call.get("args") or {}
    if msg.tool_calls:
        return msg.tool_calls[0].get("args") or {}
    # Some models answer in plain JSON text instead of a tool call.
    text = msg.content if isinstance(msg.content, str) else ""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def invoke_structured(schema: type[T], messages: Sequence[BaseMessage], label: str,
                      role: str = "agent", max_attempts: int = 2) -> StructuredResult[T]:
    """Ask the model for `schema` via a forced tool call and validate with Pydantic.

    Validation failures are written to the trace (kind='validation'), the model is
    shown the error and asked once more, and the caller always gets a result object
    instead of an exception.
    """
    name = schema.__name__
    conversation: list[BaseMessage] = list(messages)
    last_error = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm(role).bind_tools([schema], tool_choice=name)
            msg = invoke_llm(llm, conversation, label=f"{label}:{name}", role=role)
        except RETRYABLE as exc:
            last_error = f"LLM unavailable after retries: {type(exc).__name__}: {exc}"
            break
        except (RetryError, Exception) as exc:  # auth errors, bad request, missing key ...
            last_error = f"{type(exc).__name__}: {exc}"
            if type(exc).__name__ == "MissingAPIKeyError":
                raise
            break

        args = _extract_args(msg, name)
        if args is None:
            last_error = "model returned no structured output"
        else:
            try:
                return StructuredResult(True, schema.model_validate(args), None, attempt)
            except ValidationError as exc:
                last_error = exc.json(include_url=False)[:1500]

        tracer.log(kind="validation", tool=name, args={"label": label, "attempt": attempt},
                   output=last_error, duration_ms=0.0, status="validation_error")
        log.warning("Structured output for %s failed validation (attempt %d): %s", name, attempt, last_error[:300])
        conversation = list(messages) + [
            HumanMessage(
                content=(
                    f"Your previous {name} output failed validation with these errors:\n"
                    f"{last_error}\n\nPrevious output:\n{json.dumps(args, default=str)[:3000]}\n\n"
                    f"Return a corrected {name} by calling the {name} tool. Fix every listed error."
                )
            )
        ]
    return StructuredResult(False, None, last_error, max_attempts)
