"""Agent B, the Research Writer (Task 3B).

Tools: web_search, get_news. No access to price or volatility tools.

Steps:
  * research - ReAct loop over web_search / get_news, informed by the DataBrief
  * review   - critique the evidence; the first review must send one specific
               ClarificationRequest to Agent A (later reviews are optional, capped)
  * finalize - write the FinalReport, incorporating Agent A's clarification answers

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Writer agent nodes for a LangGraph
# two-agent pipeline with a guaranteed first-round critique loop and validated final
# report', Date: 2026-10-07
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .. import prompts
from ..config import get_settings
from ..execution import ToolExecutor, close_dangling_tool_calls
from ..llm import invoke_structured
from ..observability.printer import console
from ..observability.tracer import trace_context
from ..schemas import ClarificationRequest, FinalReport, WriterReview
from ..tools import WRITER_TOOLS
from .analyst import brief_json_for_prompt
from .common import calls_by, collected_headlines, digest, fallback_report, react_turn

AGENT = "writer"

writer_tools_node = ToolExecutor(AGENT, WRITER_TOOLS, messages_key="writer_messages", phase="research")

DEFAULT_REQUEST = ClarificationRequest(
    question=("Please score the sentiment of the headlines I collected, and give 90-day annualised "
              "volatility, downside volatility and maximum drawdown so I can size the hedge."),
    requested_metrics=["sentiment_overall_score", "vol_90d", "downside_vol_90d", "max_drawdown_90d"],
    reason="Risk evidence needs sentiment; hedge sizing needs 90-day volatility.",
)


def writer_agent(state: dict) -> dict:
    return {"writer_messages": [react_turn(AGENT, WRITER_TOOLS, state["writer_messages"], state)]}


def route_writer(state: dict) -> str:
    last = state["writer_messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and \
            calls_by(state.get("observations", []), AGENT, "research") < get_settings().max_tool_calls:
        return "writer_tools"
    return "writer_review"


def _history_json(state: dict) -> str:
    hist = [h["response"] for h in state.get("clarification_history", [])]
    return json.dumps(hist, indent=1, default=str) if hist else "[]"


def writer_review(state: dict) -> dict:
    s = get_settings()
    rounds = state.get("clarification_rounds", 0)
    dangling = close_dangling_tool_calls(state["writer_messages"])
    if rounds >= s.max_clarification_rounds:
        console.info(f"Review: clarification cap ({s.max_clarification_rounds}) reached, finalising.", AGENT)
        return {"clarification_request": None, "writer_messages": dangling}

    requirement = prompts.REVIEW_REQUIRED if rounds == 0 else prompts.REVIEW_OPTIONAL.format(rounds=rounds)
    obs = state.get("observations", [])
    with trace_context(agent=AGENT, run_id=state.get("run_id"), session=state.get("session")):
        res = invoke_structured(WriterReview, [
            SystemMessage(content=prompts.REVIEW_SYSTEM.format(requirement=requirement)),
            HumanMessage(content=prompts.REVIEW_USER.format(
                query=state["query"], brief=brief_json_for_prompt(state.get("data_brief") or {}),
                research=digest(obs, agent=AGENT), clarifications=_history_json(state))),
        ], label=f"writer_review_r{rounds}")

    review = res.value if res.ok else None
    request: ClarificationRequest | None = None
    if review and review.needs_clarification and review.request:
        request = review.request
    elif rounds == 0:
        # The brief requires at least one critique round; fall back to a sensible default.
        reason = "review returned no request" if review else f"review failed validation: {res.error}"
        console.info(f"Review: {reason}; sending default clarification request.", AGENT)
        request = DEFAULT_REQUEST

    if review:
        console.thought(AGENT, f"Review: {review.assessment}")
    if request is None:
        console.info("Review: evidence sufficient, no further clarification needed.", AGENT)
        return {"review": review.model_dump() if review else None, "clarification_request": None,
                "writer_messages": dangling}

    request = request.model_copy(update={"headlines": collected_headlines(obs, AGENT)})
    console.handoff("writer", "analyst", "ClarificationRequest", request.model_dump(mode="json"))
    return {"review": review.model_dump() if review else None,
            "clarification_request": request.model_dump(mode="json"), "writer_messages": dangling}


def route_review(state: dict) -> str:
    return "analyst_clarify_start" if state.get("clarification_request") else "writer_finalize"


def writer_finalize(state: dict) -> dict:
    obs = state.get("observations", [])
    history = state.get("clarification_history", [])
    brief = state.get("data_brief") or {}
    with trace_context(agent=AGENT, run_id=state.get("run_id"), session=state.get("session")):
        res = invoke_structured(FinalReport, [
            SystemMessage(content=prompts.FINAL_SYSTEM),
            HumanMessage(content=prompts.FINAL_USER.format(
                query=state["query"], brief=brief_json_for_prompt(brief),
                research=digest(obs, agent=AGENT), clarifications=_history_json(state))),
        ], label="final_report")

    if res.ok and res.value is not None:
        report = res.value.model_copy(update={"generated_by": "llm"})
    else:
        console.info(f"Final report failed validation twice; using deterministic fallback ({(res.error or '')[:120]})", AGENT)
        clar_data: dict = {}
        for h in history:
            clar_data.update(h["response"].get("data", {}))
        fb = fallback_report(state["ticker"], brief.get("price"),
                             clar_data.get("calculate_volatility") or brief.get("volatility"),
                             clar_data.get("llm_sentiment") or brief.get("sentiment"), res.error or "unknown")
        report = FinalReport(**fb.model_dump(), clarification_summary="Fallback report built from tool data.")

    tool_sources = sorted({o["tool"] for o in obs if o.get("result", {}).get("ok")})
    report = report.model_copy(update={
        "ticker": state["ticker"],
        "clarification_used": bool(history),
        "sources": list(dict.fromkeys([*report.sources, *tool_sources])),
    })
    console.info(f"FinalReport ready ({report.generated_by}); clarification rounds used: {len(history)}", AGENT)
    return {"final_report": report.model_dump(mode="json")}
