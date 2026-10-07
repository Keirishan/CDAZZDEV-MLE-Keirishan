"""Human-readable live trace for the notebook and CLI.

Prints, as they happen:
  [AGENT]  Thought: ...                 (the model's stated reasoning)
  [AGENT]  -> CALL tool(args)           (the action it chose)
  [AGENT]     ok  tool  312 ms -> {...} (the observation, from the trace stream)
  [HANDOFF analyst -> writer] DataBrief {...}

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Console printer that subscribes to
# JSONL trace records and prints agent thoughts, tool calls and handoffs', Date: 2026-10-07
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from .tracer import tracer

_COLORS = {
    "analyst": "\033[36m", "writer": "\033[35m", "research_agent": "\033[34m",
    "system": "\033[90m", "guard": "\033[31m", "handoff": "\033[33m",
}
_RESET, _DIM, _BOLD, _RED, _GREEN = "\033[0m", "\033[2m", "\033[1m", "\033[31m", "\033[32m"


def _use_color() -> bool:
    return os.getenv("NO_COLOR") is None


class Console:
    def __init__(self) -> None:
        self.enabled = True
        self.show_llm_records = False  # LLM calls are in the JSONL file; hidden on screen by default

    # ---- formatting helpers ---------------------------------------------- #
    def _c(self, key: str, text: str) -> str:
        if not _use_color():
            return text
        return f"{_COLORS.get(key, '')}{text}{_RESET}"

    def _tag(self, agent: str) -> str:
        return self._c(agent, f"[{agent.upper():<14}]")

    def _out(self, text: str) -> None:
        if self.enabled:
            print(text, file=sys.stdout, flush=True)

    # ---- public API -------------------------------------------------------- #
    def section(self, title: str) -> None:
        line = "=" * 78
        self._out(f"\n{line}\n{_BOLD if _use_color() else ''}{title}{_RESET if _use_color() else ''}\n{line}")

    def info(self, text: str, agent: str = "system") -> None:
        self._out(f"{self._tag(agent)} {text}")

    def thought(self, agent: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            for i, line in enumerate(text.splitlines()):
                self._out(f"{self._tag(agent) if i == 0 else ' ' * 16} {line}")

    def action(self, agent: str, name: str, args: dict) -> None:
        arg_text = ", ".join(f"{k}={json.dumps(v, default=str)[:80]}" for k, v in (args or {}).items())
        self._out(f"{self._tag(agent)} -> CALL {name}({arg_text})")

    def handoff(self, src: str, dst: str, schema: str, payload: Any, max_lines: int = 40) -> None:
        body = json.dumps(payload, indent=2, default=str, ensure_ascii=False).splitlines()
        if len(body) > max_lines:
            body = body[:max_lines] + [f"  ... ({len(body) - max_lines} more lines)"]
        head = self._c("handoff", f"[HANDOFF {src} -> {dst}] {schema}")
        self._out(head + "\n" + "\n".join("    " + b for b in body))

    def on_trace(self, rec: dict) -> None:
        kind, status = rec.get("kind"), rec.get("status")
        if kind == "llm" and not self.show_llm_records:
            return
        ok = status == "ok"
        mark = (_GREEN + "ok " + _RESET) if ok and _use_color() else ("ok " if ok else "")
        if not ok:
            mark = (_RED + f"{status.upper()} " + _RESET) if _use_color() else f"{status.upper()} "
        self._out(f"{self._tag(rec.get('agent', 'system'))}    {mark}{rec['tool']}  "
                  f"{rec['duration_ms']:.0f} ms -> {_DIM if _use_color() else ''}{rec['output']}"
                  f"{_RESET if _use_color() else ''}")


console = Console()
tracer.add_listener(console.on_trace)
