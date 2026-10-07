"""Agent A, the Data Analyst (Task 3B).

Tools: get_price_data, calculate_volatility, llm_sentiment. No web or news access.

Two modes, each a ReAct loop:
  * brief   - gather quantitative data, then emit a typed DataBrief
  * clarify - answer one ClarificationRequest from Agent B, then emit a
              typed ClarificationResponse

Numbers in the DataBrief and ClarificationResponse are copied from tool results by
code; the LLM only writes the interpretation, so figures cannot be hallucinated in
the handoff.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Analyst agent nodes for a LangGraph
# two-agent pipeline: ReAct loop, typed DataBrief assembly and clarification answering',
# Date: 2026-10-07
"""
from __future__ import annotations

import json
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .. import prompts
from ..config import get_settings
from ..execution import ToolExecutor, close_dangling_tool_calls
from ..llm import invoke_structured
from ..observability.printer import console
from ..observability.tracer import trace_context
from ..schemas import (
    AnalystNotes,
    ClarificationAnswer,
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    PriceSnapshot,
    SentimentResult,
    VolatilityMetrics,
)
from ..tools import ANALYST_TOOLS
from .common import calls_by, digest, failures, latest_data, react_turn

AGENT = "analyst"
CLARIFY_BUDGET = 5  # tool calls per clarification round

brief_tools_node = ToolExecutor(AGENT, ANALYST_TOOLS, messages_key="analyst_messages", phase="brief")
clarify_tools_node = ToolExecutor(AGENT, ANALYST_TOOLS, messages_key="clarify_messages", phase="clarify")


def brief_json_for_prompt(brief: dict) -> str:
    """DataBrief without the verbose OHLCV rows, for LLM prompts."""
    b = json.loads(json.dumps(brief, default=str))
    if b.get("price"):
        b["price"].pop("recent_ohlcv", None)
    return json.dumps(b, indent=1, ensure_ascii=False)


# ---- brief mode ------------------------------------------------------------ #
def analyst_agent(state: dict) -> dict:
    return {"analyst_messages": [react_turn(AGENT, ANALYST_TOOLS, state["analyst_messages"], state)]}


def route_analyst(state: dict) -> str:
    last = state["analyst_messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and \
            calls_by(state.get("observations", []), AGENT, "brief") < get_settings().max_tool_calls:
        return "analyst_tools"
    return "analyst_brief"


def build_brief(ticker: str, observations: list[dict], notes: AnalystNotes | None) -> DataBrief:
    price = latest_data(observations, "get_price_data", agent=AGENT)
    vol = latest_data(observations, "calculate_volatility", agent=AGENT)
    sent = latest_data(observations, "llm_sentiment", agent=AGENT)
    gaps = list(notes.data_gaps) if notes else ["analyst notes unavailable (structured output failed)"]
    if price is None:
        gaps.append("price/indicator data unavailable")
    if vol is None:
        gaps.append("volatility data unavailable")
    if sent is None:
        gaps.append("news sentiment not scored yet: Agent A has no news access; headlines must come from Agent B")
    return DataBrief(
        ticker=ticker,
        as_of=(price or vol or {}).get("as_of") or date.today().isoformat(),
        price=PriceSnapshot.model_validate(price) if price else None,
        volatility=VolatilityMetrics.model_validate(vol) if vol else None,
        sentiment=SentimentResult.model_validate(sent) if sent else None,
        key_observations=list(notes.key_observations) if notes else [],
        quantitative_flags=list(notes.quantitative_flags) if notes else [],
        data_gaps=list(dict.fromkeys(gaps)),
    )


def analyst_brief(state: dict) -> dict:
    s = get_settings()
    obs = state.get("observations", [])
    ticker = state["ticker"]
    with trace_context(agent=AGENT, run_id=state.get("run_id"), session=state.get("session")):
        res = invoke_structured(AnalystNotes, [
            SystemMessage(content=prompts.ANALYST_NOTES_SYSTEM),
            HumanMessage(content=prompts.ANALYST_NOTES_USER.format(
                ticker=ticker, evidence=digest(obs, agent=AGENT), failures=failures(obs, agent=AGENT))),
        ], label="analyst_notes")
    brief = build_brief(ticker, obs, res.value if res.ok else None)
    brief_dict = brief.model_dump(mode="json")
    console.handoff("analyst", "writer", "DataBrief", json.loads(brief_json_for_prompt(brief_dict)))
    writer_init = [
        SystemMessage(content=prompts.WRITER_SYSTEM.format(budget=s.max_tool_calls)),
        HumanMessage(content=prompts.WRITER_USER.format(query=state["query"], ticker=ticker,
                                                        brief=brief_json_for_prompt(brief_dict))),
    ]
    return {"data_brief": brief_dict, "writer_messages": writer_init,
            "analyst_messages": close_dangling_tool_calls(state["analyst_messages"])}


# ---- clarify mode ----------------------------------------------------------- #
def analyst_clarify_start(state: dict) -> dict:
    s = get_settings()
    req = ClarificationRequest.model_validate(state["clarification_request"])
    request_view = req.model_dump(exclude={"headlines"})
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(req.headlines)) or "none supplied"
    new: list = []
    if not state.get("clarify_messages"):
        new.append(SystemMessage(content=prompts.CLARIFY_SYSTEM.format(budget=CLARIFY_BUDGET)))
    new.append(HumanMessage(content=prompts.CLARIFY_USER.format(
        request=json.dumps(request_view, indent=1), ticker=state["ticker"], headlines=numbered)))
    console.info(f"Clarification round {state.get('clarification_rounds', 0) + 1}: received request from writer", AGENT)
    return {"clarify_messages": new, "clarify_obs_start": len(state.get("observations", []))}


def analyst_clarify(state: dict) -> dict:
    return {"clarify_messages": [react_turn(AGENT, ANALYST_TOOLS, state["clarify_messages"], state)]}


def _round_obs(state: dict) -> list[dict]:
    start = state.get("clarify_obs_start", 0)
    return [o for o in state.get("observations", [])[start:] if o.get("agent") == AGENT and o.get("phase") == "clarify"]


def route_clarify(state: dict) -> str:
    last = state["clarify_messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and \
            sum(1 for o in _round_obs(state) if o.get("executed", True)) < CLARIFY_BUDGET:
        return "analyst_clarify_tools"
    return "clarify_respond"


def clarify_respond(state: dict) -> dict:
    req = ClarificationRequest.model_validate(state["clarification_request"])
    round_obs = _round_obs(state)
    with trace_context(agent=AGENT, run_id=state.get("run_id"), session=state.get("session")):
        res = invoke_structured(ClarificationAnswer, [
            SystemMessage(content=prompts.CLARIFY_ANSWER_SYSTEM),
            HumanMessage(content=prompts.CLARIFY_ANSWER_USER.format(
                question=req.question, metrics=", ".join(req.requested_metrics),
                evidence=digest(round_obs), failures=failures(round_obs))),
        ], label="clarify_answer")
    answer = res.value.answer if res.ok and res.value else f"No written answer could be produced ({res.error}). See data."

    data: dict = {}
    for o in round_obs:
        if o["result"].get("ok"):
            d = o["result"]["data"]
            if o["tool"] == "get_price_data" and isinstance(d, dict):
                d = {k: v for k, v in d.items() if k != "recent_ohlcv"}
            key = o["tool"] if o["tool"] not in data else f"{o['tool']}_{len(data)}"
            data[key] = d
    response = ClarificationResponse(question=req.question, answer=answer, data=data,
                                     tools_used=[o["tool"] for o in round_obs])
    console.handoff("analyst", "writer", "ClarificationResponse", response.model_dump(mode="json"))

    update: dict = {
        "clarification_history": [{"request": req.model_dump(mode="json"), "response": response.model_dump(mode="json")}],
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        "clarification_request": None,
        "clarify_messages": close_dangling_tool_calls(state["clarify_messages"]),
    }
    # Sentiment scored during clarification is added to the DataBrief so it is cached too.
    sent = latest_data(round_obs, "llm_sentiment")
    if sent and state.get("data_brief") and not state["data_brief"].get("sentiment"):
        brief = dict(state["data_brief"])
        brief["sentiment"] = sent
        brief["data_gaps"] = [g for g in brief.get("data_gaps", []) if "sentiment" not in g]
        update["data_brief"] = brief
    return update
