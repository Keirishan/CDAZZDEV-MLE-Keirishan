r"""Task 3B + 3C: the two-agent pipeline with critique loop, persistent cache and memory.

    START -> cache_check --hit--> load_cached_brief -> END
                         \-miss-> analyst <-> analyst_tools          (Agent A, brief mode)
                                  -> analyst_brief                   (typed DataBrief handoff)
                                  -> writer <-> writer_tools         (Agent B, research)
                                  -> writer_review --request--> analyst_clarify_start
                                                                     -> analyst_clarify <-> analyst_clarify_tools
                                                                     -> clarify_respond -> writer_review
                                                   --done-----> writer_finalize -> save_cache -> END

Runs from query to report with no manual step. Compiled with a MemorySaver
checkpointer so each session's state is kept under its thread_id.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph orchestrator for analyst and
# writer agents with cache_check entry node, critique loop and save_cache', Date: 2026-10-07
"""
from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import prompts
from .agents import analyst, writer
from .config import get_settings
from .data.market import normalise_ticker
from .llm import model_name
from .memory.cache import cache_path, load_cached, save_cached
from .observability.printer import console
from .observability.tracer import trace_context, tracer
from .schemas import CachedBrief, DataBrief, FinalReport


class PipelineState(TypedDict, total=False):
    query: str
    ticker: str
    run_id: str
    session: str
    force_refresh: bool
    analyst_messages: Annotated[list[AnyMessage], add_messages]
    writer_messages: Annotated[list[AnyMessage], add_messages]
    clarify_messages: Annotated[list[AnyMessage], add_messages]
    observations: Annotated[list[dict], operator.add]
    tool_calls: Annotated[int, operator.add]
    data_brief: dict | None
    review: dict | None
    clarification_request: dict | None
    clarification_history: Annotated[list[dict], operator.add]
    clarification_rounds: int
    clarify_obs_start: int
    final_report: dict | None
    cache_hit: bool
    cache_path: str | None


# ---- cache nodes ------------------------------------------------------------ #
def cache_check(state: PipelineState) -> dict:
    path = cache_path(state["ticker"])
    if state.get("force_refresh"):
        console.info(f"cache_check: force_refresh=True, ignoring {path.name}")
        return {"cache_hit": False, "cache_path": str(path)}
    with trace_context(agent="system", run_id=state.get("run_id"), session=state.get("session")):
        cached = load_cached(state["ticker"])
    console.info(f"cache_check: {'HIT' if cached else 'MISS'} ({path.name})")
    return {"cache_hit": cached is not None, "cache_path": str(path)}


def route_cache(state: PipelineState) -> str:
    return "load_cached_brief" if state.get("cache_hit") else "analyst"


def load_cached_brief(state: PipelineState) -> dict:
    with trace_context(agent="system", run_id=state.get("run_id"), session=state.get("session")):
        cached = load_cached(state["ticker"])
    if cached is None:  # file vanished between check and load
        raise RuntimeError("Cached brief disappeared; rerun with force_refresh=True.")
    console.info(f"Loaded cached brief from {cached.date} (original run {cached.run_id}, "
                 f"{cached.tool_call_count} tool calls). No agents or tools run.")
    return {"final_report": cached.final_report.model_dump(mode="json"),
            "data_brief": cached.data_brief.model_dump(mode="json") if cached.data_brief else None}


def save_cache(state: PipelineState) -> dict:
    brief = CachedBrief(
        ticker=state["ticker"], date=date.today().isoformat(), model=model_name(),
        run_id=state["run_id"], tool_call_count=state.get("tool_calls", 0),
        data_brief=DataBrief.model_validate(state["data_brief"]) if state.get("data_brief") else None,
        final_report=FinalReport.model_validate(state["final_report"]),
    )
    with trace_context(agent="system", run_id=state.get("run_id"), session=state.get("session")):
        path = save_cached(brief)
    console.info(f"save_cache: wrote {path.name}")
    return {"cache_path": str(path)}


# ---- graph ------------------------------------------------------------------ #
def build_pipeline_graph() -> StateGraph:
    g = StateGraph(PipelineState)
    g.add_node("cache_check", cache_check)
    g.add_node("load_cached_brief", load_cached_brief)
    g.add_node("analyst", analyst.analyst_agent)
    g.add_node("analyst_tools", analyst.brief_tools_node)
    g.add_node("analyst_brief", analyst.analyst_brief)
    g.add_node("writer", writer.writer_agent)
    g.add_node("writer_tools", writer.writer_tools_node)
    g.add_node("writer_review", writer.writer_review)
    g.add_node("analyst_clarify_start", analyst.analyst_clarify_start)
    g.add_node("analyst_clarify", analyst.analyst_clarify)
    g.add_node("analyst_clarify_tools", analyst.clarify_tools_node)
    g.add_node("clarify_respond", analyst.clarify_respond)
    g.add_node("writer_finalize", writer.writer_finalize)
    g.add_node("save_cache", save_cache)

    g.add_edge(START, "cache_check")
    g.add_conditional_edges("cache_check", route_cache,
                            {"load_cached_brief": "load_cached_brief", "analyst": "analyst"})
    g.add_edge("load_cached_brief", END)
    g.add_conditional_edges("analyst", analyst.route_analyst,
                            {"analyst_tools": "analyst_tools", "analyst_brief": "analyst_brief"})
    g.add_edge("analyst_tools", "analyst")
    g.add_edge("analyst_brief", "writer")
    g.add_conditional_edges("writer", writer.route_writer,
                            {"writer_tools": "writer_tools", "writer_review": "writer_review"})
    g.add_edge("writer_tools", "writer")
    g.add_conditional_edges("writer_review", writer.route_review,
                            {"analyst_clarify_start": "analyst_clarify_start", "writer_finalize": "writer_finalize"})
    g.add_edge("analyst_clarify_start", "analyst_clarify")
    g.add_conditional_edges("analyst_clarify", analyst.route_clarify,
                            {"analyst_clarify_tools": "analyst_clarify_tools", "clarify_respond": "clarify_respond"})
    g.add_edge("analyst_clarify_tools", "analyst_clarify")
    g.add_edge("clarify_respond", "writer_review")
    g.add_edge("writer_finalize", "save_cache")
    g.add_edge("save_cache", END)
    return g


@dataclass
class PipelineResult:
    final_report: FinalReport
    data_brief: DataBrief | None
    cache_hit: bool
    cache_path: str | None
    tool_calls: int
    run_id: str
    session: str
    clarifications: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)

    @property
    def trace_tool_calls(self) -> int:
        return tracer.count("tool", run_id=self.run_id)


class ResearchPipeline:
    """Two-agent pipeline. One instance keeps session memory in its checkpointer."""

    def __init__(self, checkpointer: Any | None = None) -> None:
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_pipeline_graph().compile(checkpointer=self.checkpointer)

    def run(self, ticker: str | None = None, query: str | None = None, session: str | None = None,
            force_refresh: bool = False) -> PipelineResult:
        s = get_settings()
        t = normalise_ticker(ticker or s.default_ticker)
        query = query or prompts.RESEARCH_AGENT_USER.format(ticker=t)
        session = session or f"pipeline-{t.lower()}-{uuid.uuid4().hex[:6]}"
        run_id = uuid.uuid4().hex[:12]
        console.section(f"Task 3B pipeline | ticker={t} | session={session} | run={run_id}")
        state = self.graph.invoke(
            {
                "query": query, "ticker": t, "run_id": run_id, "session": session,
                "force_refresh": force_refresh, "clarification_rounds": 0,
                "analyst_messages": [
                    SystemMessage(content=prompts.ANALYST_SYSTEM.format(budget=s.max_tool_calls)),
                    HumanMessage(content=prompts.ANALYST_USER.format(query=query, ticker=t)),
                ],
            },
            config={"configurable": {"thread_id": session}, "recursion_limit": s.recursion_limit},
        )
        return PipelineResult(
            final_report=FinalReport.model_validate(state["final_report"]),
            data_brief=DataBrief.model_validate(state["data_brief"]) if state.get("data_brief") else None,
            cache_hit=bool(state.get("cache_hit")), cache_path=state.get("cache_path"),
            tool_calls=state.get("tool_calls", 0), run_id=run_id, session=session,
            clarifications=state.get("clarification_history", []),
            observations=state.get("observations", []),
        )


def run_pipeline(ticker: str | None = None, query: str | None = None, session: str | None = None,
                 force_refresh: bool = False) -> PipelineResult:
    """One-call entry point: query in, validated FinalReport out, no manual steps."""
    return ResearchPipeline().run(ticker=ticker, query=query, session=session, force_refresh=force_refresh)
