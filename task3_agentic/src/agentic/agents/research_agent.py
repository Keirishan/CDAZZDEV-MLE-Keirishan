r"""Task 3A (+ 3C short-term memory): a single tool-using research agent in LangGraph.

Graph:
    START -> agent --(tool calls, within budget)--> tools -> agent   (observe -> replan loop)
                  \--(no tool calls / budget hit)--> finalize -> END   (research mode)
                  \--(no tool calls)--------------> END               (follow-up mode)

The agent's prompt states goals and rules only; the order of tool calls is chosen by
the model from what it has observed. The graph is compiled with a checkpointer, so
a follow-up on the same thread_id sees every earlier message and tool result.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph ReAct research agent with
# custom tool executor, budget-aware routing, structured final report, MemorySaver
# follow-ups', Date: 2026-10-07
"""
from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .. import prompts
from ..config import get_settings
from ..data.market import normalise_ticker
from ..execution import ToolExecutor, close_dangling_tool_calls
from ..llm import invoke_structured
from ..observability.printer import console
from ..observability.tracer import trace_context, tracer
from ..schemas import ResearchReport
from ..tools import ALL_TOOLS
from .common import digest, failures, fallback_report, last_ai_text, latest_data, react_turn

AGENT = "research_agent"
FOLLOWUP_EXTRA_BUDGET = 3  # tool calls allowed in a follow-up turn if data is truly missing


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    observations: Annotated[list[dict], operator.add]
    tool_calls: Annotated[int, operator.add]
    ticker: str
    query: str
    run_id: str
    session: str
    mode: str          # "research" | "followup"
    budget: int        # cumulative tool-call ceiling for the current invocation
    report: dict | None


# ---- nodes ---------------------------------------------------------------- #
def agent_node(state: ResearchState) -> dict:
    return {"messages": [react_turn(AGENT, ALL_TOOLS, state["messages"], state)]}


tools_node = ToolExecutor(AGENT, ALL_TOOLS, messages_key="messages", phase="research")


def finalize_node(state: ResearchState) -> dict:
    obs = state.get("observations", [])
    ticker = state["ticker"]
    with trace_context(agent=AGENT, run_id=state.get("run_id"), session=state.get("session")):
        dangling = close_dangling_tool_calls(state["messages"])
        messages = [
            SystemMessage(content=prompts.REPORT_SYSTEM),
            HumanMessage(content=prompts.REPORT_USER.format(
                query=state.get("query", ""), notes=last_ai_text(state["messages"]) or "none",
                evidence=digest(obs), failures=failures(obs))),
        ]
        res = invoke_structured(ResearchReport, messages, label="research_report")
    if res.ok and res.value is not None:
        report = res.value.model_copy(update={"ticker": ticker, "generated_by": "llm"})
    else:
        console.info(f"Report failed validation twice; using deterministic fallback ({res.error[:120]})", AGENT)
        report = fallback_report(ticker, latest_data(obs, "get_price_data"),
                                 latest_data(obs, "calculate_volatility"),
                                 latest_data(obs, "llm_sentiment"), res.error or "unknown")
    console.info(f"ResearchReport ready ({report.generated_by}): 3 risks, hedge = {report.hedge_strategy.strategy}", AGENT)
    summary = AIMessage(content="Final report (ResearchReport JSON):\n" + report.model_dump_json())
    return {"messages": dangling + [summary], "report": report.model_dump()}


def wrap_up_node(state: ResearchState) -> dict:
    """Follow-up turn hit its budget: answer pending calls so the thread stays valid."""
    return {"messages": close_dangling_tool_calls(state["messages"])}


# ---- routing -------------------------------------------------------------- #
def route_after_agent(state: ResearchState) -> str:
    last = state["messages"][-1]
    wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
    if wants_tools and state.get("tool_calls", 0) < state.get("budget", 0):
        return "tools"
    if state.get("mode") == "followup":
        return "wrap_up" if wants_tools else END
    return "finalize"


def build_research_graph() -> StateGraph:
    g = StateGraph(ResearchState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)
    g.add_node("wrap_up", wrap_up_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "finalize": "finalize", "wrap_up": "wrap_up", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    g.add_edge("wrap_up", END)
    return g


# ---- public API ------------------------------------------------------------ #
@dataclass
class ResearchRunResult:
    report: ResearchReport
    tool_calls: int
    session: str
    run_id: str
    observations: list[dict] = field(default_factory=list)

    @property
    def tool_sequence(self) -> list[str]:
        return [o["tool"] for o in self.observations]


@dataclass
class FollowupResult:
    question: str
    answer: str
    tool_calls_before: int
    tool_calls_after: int
    trace_tool_lines_before: int
    trace_tool_lines_after: int

    @property
    def answered_from_memory(self) -> bool:
        return self.tool_calls_after == self.tool_calls_before and \
            self.trace_tool_lines_after == self.trace_tool_lines_before


class ResearchAgent:
    """Single research agent. One instance keeps short-term memory for its sessions."""

    def __init__(self, checkpointer: Any | None = None) -> None:
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_research_graph().compile(checkpointer=self.checkpointer)

    def _config(self, session: str) -> dict:
        return {"configurable": {"thread_id": session}, "recursion_limit": get_settings().recursion_limit}

    def run(self, ticker: str | None = None, query: str | None = None,
            session: str | None = None) -> ResearchRunResult:
        s = get_settings()
        t = normalise_ticker(ticker or s.default_ticker)
        query = query or prompts.RESEARCH_AGENT_USER.format(ticker=t)
        session = session or f"{t.lower()}-{uuid.uuid4().hex[:6]}"
        run_id = uuid.uuid4().hex[:12]
        console.section(f"Task 3A research agent | ticker={t} | session={session} | run={run_id}")
        console.info(f"Query: {query}", AGENT)
        state = self.graph.invoke(
            {
                "messages": [SystemMessage(content=prompts.RESEARCH_AGENT_SYSTEM.format(budget=s.max_tool_calls)),
                             HumanMessage(content=query)],
                "ticker": t, "query": query, "run_id": run_id, "session": session,
                "mode": "research", "budget": s.max_tool_calls,
            },
            config=self._config(session),
        )
        return ResearchRunResult(report=ResearchReport.model_validate(state["report"]),
                                 tool_calls=state.get("tool_calls", 0), session=session, run_id=run_id,
                                 observations=state.get("observations", []))

    def ask(self, question: str, session: str) -> FollowupResult:
        """Follow-up in an existing session. Tools stay available; memory should make them unnecessary."""
        cfg = self._config(session)
        snapshot = self.graph.get_state(cfg)
        if not snapshot or not snapshot.values:
            raise ValueError(f"No saved session '{session}'. Run the agent first.")
        before = snapshot.values.get("tool_calls", 0)
        trace_before = tracer.count("tool", session=session)
        console.section(f"Task 3C follow-up | session={session}")
        console.info(f"Question: {question}", AGENT)
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=question)], "mode": "followup",
             "budget": before + FOLLOWUP_EXTRA_BUDGET, "run_id": uuid.uuid4().hex[:12]},
            config=cfg,
        )
        after = state.get("tool_calls", 0)
        result = FollowupResult(question=question, answer=last_ai_text(state["messages"]),
                                tool_calls_before=before, tool_calls_after=after,
                                trace_tool_lines_before=trace_before,
                                trace_tool_lines_after=tracer.count("tool", session=session))
        console.info(f"Tool calls in session before/after follow-up: {before} -> {after}; "
                     f"trace tool lines: {result.trace_tool_lines_before} -> {result.trace_tool_lines_after}", AGENT)
        return result
