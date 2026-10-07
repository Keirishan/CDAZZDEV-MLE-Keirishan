"""Helpers shared by the agent graphs.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Helpers for LangGraph agents: ReAct
# turn with printed thoughts, observation digests, budget counting and fallback report',
# Date: 2026-10-07
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from ..config import MissingAPIKeyError
from ..llm import get_llm, invoke_llm
from ..observability.printer import console
from ..observability.tracer import trace_context
from ..schemas import HedgeStrategy, ResearchReport, Risk

EVIDENCE_CHARS_PER_ITEM = 3500   # keeps prompts compact
SKIP_KEYS = {"recent_ohlcv"}     # verbose fields left out of LLM digests


def react_turn(agent: str, tools: Sequence[BaseTool], messages: Sequence[BaseMessage],
               state: dict[str, Any]) -> AIMessage:
    """One reasoning step: the LLM sees the conversation and its tools and decides."""
    llm = get_llm("agent").bind_tools(list(tools))
    with trace_context(agent=agent, run_id=state.get("run_id"), session=state.get("session")):
        try:
            msg = invoke_llm(llm, messages, label=f"{agent}_turn")
        except MissingAPIKeyError:
            raise  # configuration error: fail fast with the setup message
        except Exception as exc:  # provider error after retries: stop tool use, let the graph finish
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            console.info(f"LLM call failed ({err}); continuing without further tool calls.", agent)
            return AIMessage(content=f"Thought: the language model call failed ({err}). Stopping tool use.")
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    console.thought(agent, content)
    for call in msg.tool_calls:
        console.action(agent, call["name"], call.get("args", {}))
    return msg


def calls_by(observations: Iterable[dict], agent: str, phase: str | None = None) -> int:
    """Tool calls actually executed (blocked calls excluded) by an agent in a phase."""
    return sum(1 for o in observations
               if o.get("agent") == agent and o.get("executed", True)
               and (phase is None or o.get("phase") == phase))


def _strip(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _strip(v) for k, v in data.items() if k not in SKIP_KEYS}
    if isinstance(data, list):
        return [_strip(v) for v in data]
    return data


def digest(observations: Iterable[dict], agent: str | None = None, phase: str | None = None,
           tools: set[str] | None = None) -> str:
    """Successful tool results as compact JSON for an LLM prompt."""
    rows = []
    for o in observations:
        if agent and o.get("agent") != agent:
            continue
        if phase and o.get("phase") != phase:
            continue
        if tools and o.get("tool") not in tools:
            continue
        r = o.get("result", {})
        if not r.get("ok"):
            continue
        text = json.dumps({"tool": o["tool"], "args": o.get("args"), "data": _strip(r.get("data"))},
                          default=str, ensure_ascii=False)
        rows.append(text[:EVIDENCE_CHARS_PER_ITEM])
    return "[\n" + ",\n".join(rows) + "\n]" if rows else "[]"


def failures(observations: Iterable[dict], agent: str | None = None, phase: str | None = None) -> str:
    rows = [f"- {o['tool']}({json.dumps(o.get('args'), default=str)[:120]}): {o['result'].get('error')}"
            for o in observations
            if (not agent or o.get("agent") == agent) and (not phase or o.get("phase") == phase)
            and not o.get("result", {}).get("ok")]
    return "\n".join(rows) or "none"


def latest_data(observations: Iterable[dict], tool: str, agent: str | None = None,
                phase: str | None = None) -> dict | None:
    found = None
    for o in observations:
        if o.get("tool") != tool or (agent and o.get("agent") != agent) or (phase and o.get("phase") != phase):
            continue
        if o.get("result", {}).get("ok"):
            found = o["result"]["data"]
    return found


def collected_headlines(observations: Iterable[dict], agent: str = "writer", limit: int = 15) -> list[str]:
    """Headlines gathered by an agent via get_news (preferred) and web_search titles."""
    out: list[str] = []
    for tool_name, key, title_key in (("get_news", "items", "title"), ("web_search", "results", "title")):
        for o in observations:
            if o.get("agent") != agent or o.get("tool") != tool_name or not o.get("result", {}).get("ok"):
                continue
            for item in (o["result"].get("data") or {}).get(key, []):
                t = (item.get(title_key) or "").strip()
                if t and t not in out:
                    out.append(t)
    return out[:limit]


def last_ai_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""


def _fmt(x: float | None, pct: bool = False, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{digits}f}"


def fallback_report(ticker: str, price: dict | None, vol: dict | None, sentiment: dict | None,
                    reason: str) -> ResearchReport:
    """Deterministic report built from tool data when the LLM report fails validation twice.

    Keeps the pipeline completing with evidence-backed content instead of crashing.
    """
    p, v, s = price or {}, vol or {}, sentiment or {}
    close = p.get("last_close")
    move = v.get("expected_move_90d_1sigma")
    strike = (close - move) if (close and move) else None
    risks = [
        Risk(title="Trend extension and momentum reversal",
             description="The share price's position against its long-term averages and momentum indicators "
                         "suggests the trend could reverse within 90 days.",
             evidence=[f"Last close {_fmt(close)} vs SMA200 {_fmt(p.get('sma200'))}",
                       f"RSI14 {_fmt(p.get('rsi14'), digits=1)}, MACD histogram {_fmt(p.get('macd_hist'), digits=3)}"],
             sources=["get_price_data"], severity="medium"),
        Risk(title="Elevated volatility and drawdown risk",
             description="Realised volatility implies a wide range of outcomes over the next quarter.",
             evidence=[f"Annualised volatility {_fmt(v.get('annualised_vol'), pct=True)} "
                       f"({v.get('window', 'n/a')}-day window), regime: {v.get('vol_regime', 'n/a')}",
                       f"Max drawdown in window {_fmt(v.get('max_drawdown'), pct=True)}"],
             sources=["calculate_volatility"], severity="high"),
        Risk(title="News-driven sentiment shift",
             description="Headline sentiment can move the price sharply around news and events.",
             evidence=[f"Overall sentiment score {s.get('overall_score', 'n/a')} ({s.get('overall_label', 'n/a')})",
                       f"Headline counts: {s.get('counts', 'n/a')}"],
             sources=["llm_sentiment"], severity="medium"),
    ]
    hedge = HedgeStrategy(
        strategy="Protective put",
        instruments="Buy ~90-day put options about one standard deviation below the current price.",
        rationale="Limits downside from the volatility and reversal risks while keeping upside.",
        sizing_math=(f"1-sigma 90-day move = {_fmt(move)}; strike ~ {_fmt(close)} - {_fmt(move)} = {_fmt(strike)}"
                     if strike else "Volatility data unavailable; strike could not be computed."),
        data_used=[f"last_close={_fmt(close)}", f"expected_move_90d_1sigma={_fmt(move)}"],
    )
    return ResearchReport(
        ticker=ticker,
        financial_health_summary=(f"Automatic fallback summary (LLM report failed validation: {reason[:160]}). "
                                  f"Momentum signal: {p.get('momentum_signal', 'n/a')}; "
                                  f"YTD return {_fmt(p.get('ytd_return_pct'), digits=1)}%; "
                                  f"trailing P/E {_fmt((p.get('fundamentals') or {}).get('trailing_pe'), digits=1)}."),
        top_risks=risks, hedge_strategy=hedge, generated_by="fallback",
    )
