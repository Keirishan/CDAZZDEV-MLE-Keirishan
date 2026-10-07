"""Tool infrastructure: ToolResult envelope, safe execution, failure injection.

`run_tool` is the single place where a tool's core function is executed. It
  * applies failure injection (for the error-handling demo),
  * converts every exception into ToolResult(ok=False, error=...),
  * writes one trace record with args, truncated output and duration.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Safe tool runner returning a uniform
# result envelope with tracing and configurable failure injection', Date: 2026-10-07
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from ..config import get_settings
from ..observability.tracer import trace_call
from ..schemas import ToolResult

_injected: set[str] = set(get_settings().fail_tools)


class ToolError(Exception):
    """An expected failure with a message that is safe to show to the agent."""


def inject_failures(names: Iterable[str]) -> None:
    _injected.update(names)


def clear_failures() -> None:
    _injected.clear()


def injected_failures() -> set[str]:
    return set(_injected)


@contextmanager
def failing(*names: str) -> Iterator[None]:
    """Temporarily force the named tools to fail: `with failing("get_news"): ...`"""
    before = set(_injected)
    _injected.update(names)
    try:
        yield
    finally:
        _injected.clear()
        _injected.update(before)


def clean(obj: Any, ndigits: int = 4) -> Any:
    """Make data JSON-safe: numpy -> python, NaN/inf -> None, round floats."""
    if isinstance(obj, dict):
        return {str(k): clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, ndigits) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, ndigits)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_tool(name: str, fn: Callable[..., tuple[Any, str]], args: dict) -> ToolResult:
    """Execute a tool core function. Never raises."""
    with trace_call(name, args, kind="tool") as span:
        if name in _injected:
            result = ToolResult(ok=False, tool=name,
                                error="Simulated outage (failure injection via FAIL_TOOLS). "
                                      "Try an alternative tool or approach.")
            span.finish(result.model_dump(), status="injected_failure")
            return result
        try:
            data, source = fn(**args)
            result = ToolResult(ok=True, tool=name, data=clean(data), source=source)
            span.finish(result.data, status="ok")
        except ToolError as exc:
            result = ToolResult(ok=False, tool=name, error=str(exc))
            span.finish(result.error, status="error")
        except Exception as exc:  # unexpected: still returned, never raised
            result = ToolResult(ok=False, tool=name, error=f"{type(exc).__name__}: {exc}")
            span.finish(result.error, status="error")
        return result


def to_message_content(result: ToolResult | dict) -> str:
    payload = result.model_dump() if isinstance(result, ToolResult) else result
    return json.dumps(payload, default=str, ensure_ascii=False)
