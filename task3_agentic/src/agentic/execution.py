"""Tool execution node with an enforced per-agent allow-list.

The LLM only *sees* the tools bound to it, but tool restriction is also enforced
here: a call to any tool outside the agent's allow-list is blocked, logged to the
trace with status 'blocked', and answered with an error ToolMessage.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph tool-executor node that
# enforces an allow-list, traces blocked calls, records observations and counts calls',
# Date: 2026-10-07
"""
from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from .observability.tracer import trace_context, tracer
from .schemas import ToolResult
from .tools.base import to_message_content


def check_permission(agent: str, tool_name: str, allowed: set[str]) -> ToolResult | None:
    """Return a 'blocked' ToolResult if the tool is not allowed, else None."""
    if tool_name in allowed:
        return None
    result = ToolResult(
        ok=False, tool=tool_name,
        error=(f"Tool '{tool_name}' is not permitted for agent '{agent}'. "
               f"Allowed tools: {sorted(allowed)}."),
    )
    tracer.log(kind="guard", tool=tool_name, args={"agent": agent}, output=result.error,
               duration_ms=0.0, status="blocked", agent=agent)
    return result


class ToolExecutor:
    """Callable LangGraph node: runs the tool calls in the last AI message."""

    def __init__(self, agent: str, tools: Sequence[BaseTool], messages_key: str = "messages",
                 phase: str = "main") -> None:
        self.agent = agent
        self.phase = phase
        self.messages_key = messages_key
        self.tools = {t.name: t for t in tools}

    @property
    def allowed(self) -> set[str]:
        return set(self.tools)

    def execute_calls(self, tool_calls: list[dict]) -> tuple[list[ToolMessage], list[dict], int]:
        messages: list[ToolMessage] = []
        observations: list[dict] = []
        executed = 0
        for call in tool_calls:
            name, args, call_id = call.get("name", ""), call.get("args") or {}, call.get("id") or name
            blocked = check_permission(self.agent, name, self.allowed)
            if blocked is not None:
                result = blocked.model_dump()
            else:
                try:
                    result = self.tools[name].invoke(args)
                except ValidationError as exc:  # the LLM produced invalid arguments
                    err = f"Invalid arguments for {name}: {exc.errors(include_url=False)}"
                    tracer.log(kind="tool", tool=name, args=args, output=err, duration_ms=0.0,
                               status="invalid_args", agent=self.agent)
                    result = ToolResult(ok=False, tool=name, error=err).model_dump()
                except Exception as exc:  # defensive: tools should never raise
                    err = f"{type(exc).__name__}: {exc}"
                    tracer.log(kind="tool", tool=name, args=args, output=err, duration_ms=0.0,
                               status="error", agent=self.agent)
                    result = ToolResult(ok=False, tool=name, error=err).model_dump()
                executed += 1
            messages.append(ToolMessage(content=to_message_content(result), tool_call_id=call_id, name=name))
            observations.append({"agent": self.agent, "phase": self.phase, "tool": name,
                                 "args": args, "result": result, "executed": blocked is None})
        return messages, observations, executed

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        last = state[self.messages_key][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        with trace_context(agent=self.agent, run_id=state.get("run_id"), session=state.get("session")):
            messages, observations, executed = self.execute_calls(calls)
        return {self.messages_key: messages, "observations": observations, "tool_calls": executed}


def close_dangling_tool_calls(messages: list, note: str = "Skipped: tool-call budget reached.") -> list[ToolMessage]:
    """If the last AI message has unanswered tool calls, answer them so history stays valid."""
    if not messages:
        return []
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return [ToolMessage(content=to_message_content(ToolResult(ok=False, tool=c["name"], error=note)),
                            tool_call_id=c.get("id") or c["name"], name=c["name"]) for c in last.tool_calls]
    return []
