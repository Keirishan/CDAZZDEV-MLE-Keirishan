"""Structured tracing to agent_trace.jsonl.

Every tool call (and, as extra observability, every LLM call, guard decision,
validation failure and cache lookup) is written as one JSON line:

    {"ts", "run_id", "session", "agent", "kind", "tool", "args",
     "output" (truncated to 200 chars), "duration_ms", "status"}

The current agent / run / session are held in context variables so tools do not
need extra parameters. Listeners (e.g. the notebook printer) are notified of each
record as it is written.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Thread-safe JSONL tracer with
# contextvars for agent/run/session and a timing context manager', Date: 2026-10-07
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import get_settings

OUTPUT_PREVIEW_CHARS = 200  # required by the assessment brief

_agent: ContextVar[str] = ContextVar("trace_agent", default="system")
_run_id: ContextVar[str] = ContextVar("trace_run_id", default="-")
_session: ContextVar[str] = ContextVar("trace_session", default="-")


def truncate(value: Any, limit: int = OUTPUT_PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())  # collapse newlines so each record stays one line
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj, default=str)
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


@contextmanager
def trace_context(agent: str | None = None, run_id: str | None = None,
                  session: str | None = None) -> Iterator[None]:
    """Set the agent / run / session recorded on every trace line inside the block."""
    tokens = []
    if agent is not None:
        tokens.append((_agent, _agent.set(agent)))
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    if session is not None:
        tokens.append((_session, _session.set(session)))
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)


def current_agent() -> str:
    return _agent.get()


class Tracer:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []

    @property
    def path(self) -> Path:
        return self._path or get_settings().trace_path

    def set_path(self, path: Path | str | None) -> None:
        self._path = Path(path) if path else None

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[dict], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def log(self, *, kind: str, tool: str, args: dict | None, output: Any,
            duration_ms: float, status: str, agent: str | None = None) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": _run_id.get(),
            "session": _session.get(),
            "agent": agent or _agent.get(),
            "kind": kind,
            "tool": tool,
            "args": _json_safe(args or {}),
            "output": truncate(output),
            "duration_ms": round(duration_ms, 1),
            "status": status,
        }
        path = self.path
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
        for fn in list(self._listeners):
            try:
                fn(record)
            except Exception:
                pass  # a broken listener must never break the agent
        return record

    # ---- reading helpers ------------------------------------------------- #
    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def count(self, kind: str | None = "tool", run_id: str | None = None,
              session: str | None = None) -> int:
        return sum(
            1 for r in self.read()
            if (kind is None or r.get("kind") == kind)
            and (run_id is None or r.get("run_id") == run_id)
            and (session is None or r.get("session") == session)
        )

    def clear(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()


tracer = Tracer()


@dataclass
class _Span:
    kind: str
    tool: str
    args: dict
    start: float = field(default_factory=time.perf_counter)
    output: Any = None
    status: str = "ok"

    def finish(self, output: Any, status: str = "ok") -> None:
        self.output, self.status = output, status


@contextmanager
def trace_call(tool: str, args: dict | None = None, kind: str = "tool") -> Iterator[_Span]:
    """Time a call and write one trace record when the block exits."""
    span = _Span(kind=kind, tool=tool, args=args or {})
    try:
        yield span
    except Exception as exc:  # record, then re-raise for the caller to handle
        span.finish(f"{type(exc).__name__}: {exc}", status="exception")
        raise
    finally:
        tracer.log(
            kind=span.kind, tool=span.tool, args=span.args, output=span.output,
            duration_ms=(time.perf_counter() - span.start) * 1000, status=span.status,
        )
