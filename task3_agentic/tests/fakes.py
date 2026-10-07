"""Offline test doubles: a scripted chat model and synthetic market data.

The scripted model behaves like a tool-calling LLM: it decides tool calls from the
conversation and returns valid (or deliberately invalid) structured outputs, so the
LangGraph wiring can be tested end to end without network access or an API key.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Scripted fake BaseChatModel supporting
# bind_tools and forced tool_choice for offline LangGraph tests', Date: 2026-10-07
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def synthetic_prices(n: int = 520, seed: int = 7, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0008, 0.02, n)
    close = start * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({
        "Open": close * (1 - 0.003), "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


FAKE_HEADLINES = [
    {"title": f"Headline {i}: chipmaker {'beats' if i % 3 else 'faces probe over'} expectations",
     "publisher": "Wire", "published": "2026-10-06", "url": f"https://example.com/n{i}", "source": "yfinance"}
    for i in range(12)
]

FAKE_SEARCH = [
    {"title": "Analysts warn on export rules", "snippet": "Export restrictions may cut revenue.",
     "url": "https://example.com/s1", "date": None},
    {"title": "Earnings date set for November", "snippet": "Company to report in 6 weeks.",
     "url": "https://example.com/s2", "date": None},
]


def _call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def _risk(title: str) -> dict:
    return {"title": title, "description": "Risk description.", "evidence": ["RSI 71", "vol 52%"],
            "sources": ["get_price_data"], "severity": "medium"}


HEDGE = {"strategy": "Protective put", "instruments": "Buy 90-day put 10% OTM",
         "rationale": "Caps downside", "sizing_math": "100 x 0.5 x sqrt(62/252) = 24.8",
         "data_used": ["vol_30d=0.5"]}


class Brain:
    """Decides what the fake model says. Keeps per-agent plans and counts calls."""

    def __init__(self, invalid_first_report: bool = True) -> None:
        self.invalid_first_report = invalid_first_report
        self.structured_calls: dict[str, int] = {}
        self.turns: dict[str, int] = {}
        self.review_round = 0

    # -- agent turns ------------------------------------------------------- #
    @staticmethod
    def _identify(messages: list[BaseMessage], tools: list[str]) -> str:
        sys = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        if "answering a specific" in sys:
            return "clarify"
        if set(tools) == {"web_search", "get_news"}:
            return "writer"
        if set(tools) == {"get_price_data", "calculate_volatility", "llm_sentiment"}:
            return "analyst"
        return "research"

    @staticmethod
    def _done_tools(messages: list[BaseMessage]) -> list[str]:
        # Tool messages since the latest human message (one "task" at a time).
        last_h = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        return [m.name for m in messages[last_h + 1:] if isinstance(m, ToolMessage)]

    def agent_turn(self, messages: list[BaseMessage], tools: list[str]) -> AIMessage:
        who = self._identify(messages, tools)
        self.turns[who] = self.turns.get(who, 0) + 1
        done = self._done_tools(messages)
        last_h = next(m for m in reversed(messages) if isinstance(m, HumanMessage))

        if who == "research":
            if "earlier" in last_h.content:  # follow-up question: answer from memory
                return AIMessage(content="RSI(14) was retrieved earlier by get_price_data; no new tool call needed.")
            plan = [("get_price_data", {"ticker": "NVDA", "period": "1y"}),
                    ("calculate_volatility", {"ticker": "NVDA", "window": 30}),
                    ("get_news", {"ticker": "NVDA", "n": 10}),
                    ("web_search", {"query": "NVDA analyst news"}),
                    ("llm_sentiment", {"headlines": [h["title"] for h in FAKE_SEARCH]})]
        elif who == "analyst":
            plan = [("get_price_data", {"ticker": "NVDA", "period": "1y"}),
                    ("calculate_volatility", {"ticker": "NVDA", "window": 30})]
        elif who == "clarify":
            heads = re.findall(r"^\d+\. (.+)$", last_h.content, flags=re.M)
            plan = [("calculate_volatility", {"ticker": "NVDA", "window": 90}),
                    ("llm_sentiment", {"headlines": heads[:5]})]
        else:  # writer
            plan = [("get_news", {"ticker": "NVDA", "n": 10}),
                    ("web_search", {"query": "NVDA risks next quarter"})]

        if len(done) < len(plan):
            name, args = plan[len(done)]
            return AIMessage(content=f"Thought: next I need {name}.", tool_calls=[_call(name, args)])
        return AIMessage(content="Thought: I have enough evidence.\n- summary")

    # -- structured outputs --------------------------------------------- #
    def structured(self, name: str, messages: list[BaseMessage]) -> dict:
        n = self.structured_calls[name] = self.structured_calls.get(name, 0) + 1
        if name == "SentimentBatch":
            heads = re.findall(r"^\d+\. (.+)$", messages[-1].content, flags=re.M)
            return {"items": [{"headline": h, "sentiment": ["positive", "negative", "neutral"][i % 3],
                               "confidence": 0.8, "brief_reason": "test"} for i, h in enumerate(heads)]}
        if name == "ResearchReport":
            risks = [_risk("A"), _risk("B")] if (self.invalid_first_report and n == 1) else [_risk("A"), _risk("B"), _risk("C")]
            return {"ticker": "NVDA", "financial_health_summary": "Healthy.", "top_risks": risks, "hedge_strategy": HEDGE}
        if name == "AnalystNotes":
            return {"key_observations": ["Price above SMA200 with RSI high"], "quantitative_flags": ["overbought"], "data_gaps": []}
        if name == "WriterReview":
            self.review_round += 1
            if self.review_round == 1:
                return {"assessment": "Need 90-day vol and sentiment.", "needs_clarification": True,
                        "request": {"question": "What is 90-day vol and headline sentiment?",
                                    "requested_metrics": ["vol_90d", "sentiment"], "reason": "hedge sizing"}}
            return {"assessment": "Sufficient.", "needs_clarification": False, "request": None}
        if name == "ClarificationAnswer":
            return {"answer": "90-day vol is X; sentiment is Y."}
        if name == "FinalReport":
            return {"ticker": "NVDA", "financial_health_summary": "Healthy.",
                    "top_risks": [_risk("A"), _risk("B"), _risk("C")], "hedge_strategy": HEDGE,
                    "clarification_used": True, "clarification_summary": "Used 90-day vol.",
                    "sources": ["https://example.com/s1"]}
        raise AssertionError(f"unexpected schema {name}")


class ScriptedChatModel(BaseChatModel):
    brain: Any
    bound_tools: list[str] = []
    tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: list, tool_choice: str | None = None, **kwargs: Any) -> "ScriptedChatModel":
        names = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        return self.model_copy(update={"bound_tools": names, "tool_choice": tool_choice})

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        if self.tool_choice:
            msg = AIMessage(content="", tool_calls=[_call(self.tool_choice, self.brain.structured(self.tool_choice, messages))])
        else:
            msg = self.brain.agent_turn(messages, self.bound_tools)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def factory_for(brain: Brain) -> Callable[[str], ScriptedChatModel]:
    return lambda role: ScriptedChatModel(brain=brain)
