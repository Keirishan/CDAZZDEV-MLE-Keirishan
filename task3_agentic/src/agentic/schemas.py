"""Pydantic schemas for every structured object in the system.

Three groups:
  * Tool outputs   - what each tool returns inside a ToolResult envelope
  * LLM outputs    - what the LLM must return (validated before use)
  * Agent messages - the typed handoff between Agent A and Agent B

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic v2 schemas for tool results,
# DataBrief handoff, clarification messages and the final equity research report', Date: 2026-10-07
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

DISCLAIMER = (
    "This report is generated automatically for an engineering assessment. It is not "
    "investment advice. Figures come from free public data sources that may be delayed "
    "or incomplete. Consult a licensed financial adviser before making investment decisions."
)


# --------------------------------------------------------------------------- #
# Tool envelope
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    """Uniform envelope every tool returns. Tools never raise; failures set ok=False."""

    ok: bool
    tool: str
    data: Any | None = None
    error: str | None = None
    source: str | None = None


# --------------------------------------------------------------------------- #
# Tool outputs
# --------------------------------------------------------------------------- #
class OHLCVRow(BaseModel):
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


class Fundamentals(BaseModel):
    name: str | None = None
    sector: str | None = None
    currency: str | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    return_on_equity: float | None = None
    free_cash_flow: float | None = None
    beta: float | None = None


class PriceSnapshot(BaseModel):
    ticker: str
    as_of: str
    period: str
    rows_analysed: int
    last_close: float | None
    period_return_pct: float | None
    ytd_return_pct: float | None
    high_52w: float | None
    low_52w: float | None
    pct_from_52w_high: float | None
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    macd_hist_5d_ago: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_percent_b: float | None
    avg_volume_20d: float | None
    volume_ratio_20d: float | None
    momentum_score: int
    momentum_signal: Literal["bullish", "bearish", "neutral"]
    flags: list[str]
    fundamentals: Fundamentals | None = None
    recent_ohlcv: list[OHLCVRow] = Field(default_factory=list)


class VolatilityMetrics(BaseModel):
    ticker: str
    as_of: str
    window: int
    annualised_vol: float | None
    vol_30d: float | None
    vol_60d: float | None
    vol_90d: float | None
    vol_1y: float | None
    downside_vol: float | None
    max_drawdown: float | None
    last_close: float | None
    expected_move_90d_1sigma: float | None
    expected_move_90d_pct: float | None
    vol_regime: Literal["elevated", "normal", "subdued", "unknown"]


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    published: str | None = None
    url: str | None = None
    source: str


class NewsResult(BaseModel):
    ticker: str
    count: int
    items: list[NewsItem]
    sources_used: list[str]


class SearchHit(BaseModel):
    title: str
    snippet: str | None = None
    url: str | None = None
    date: str | None = None


class SearchResult(BaseModel):
    query: str
    count: int
    backend: str
    results: list[SearchHit]


class HeadlineSentiment(BaseModel):
    """Sentiment for one headline."""

    headline: str = Field(description="The headline text, copied exactly.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Likely effect of this news on the company's share price."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1.")
    brief_reason: str = Field(description="One short sentence explaining the label.")


class SentimentBatch(BaseModel):
    """Sentiment labels for a list of headlines, one item per headline, in the same order."""

    items: list[HeadlineSentiment]


class SentimentResult(BaseModel):
    count: int
    overall_score: float          # confidence-weighted, range -1 (negative) .. +1 (positive)
    overall_label: Literal["positive", "negative", "neutral"]
    counts: dict[str, int]
    items: list[HeadlineSentiment]


# --------------------------------------------------------------------------- #
# Report schemas (LLM output, validated)
# --------------------------------------------------------------------------- #
class Risk(BaseModel):
    """One risk to the share price over the next 90 days."""

    title: str = Field(description="Short name of the risk.")
    description: str = Field(description="Two or three sentences explaining the risk and why it matters within 90 days.")
    evidence: list[str] = Field(
        min_length=2,
        description="At least two specific pieces of evidence, each with a number or a named event.",
    )
    sources: list[str] = Field(
        min_length=1, description="Tool names and/or URLs the evidence came from."
    )
    severity: Literal["high", "medium", "low"]


class HedgeStrategy(BaseModel):
    """One data-driven hedge recommendation."""

    strategy: str = Field(description="Name of the hedge, e.g. 'Protective put' or 'Zero-cost collar'.")
    instruments: str = Field(description="What to buy/sell, with strike and expiry guidance.")
    rationale: str = Field(description="Why this hedge fits the identified risks.")
    sizing_math: str = Field(description="The arithmetic used to size or place the hedge, using volatility figures.")
    data_used: list[str] = Field(min_length=1, description="The specific figures the hedge relies on.")


class ResearchReport(BaseModel):
    """Structured equity research report with exactly three risks."""

    ticker: str
    financial_health_summary: str = Field(
        description="One paragraph combining fundamentals, trend, indicators and sentiment."
    )
    top_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy
    # Set by code, hidden from the LLM tool schema:
    disclaimer: SkipJsonSchema[str] = DISCLAIMER
    generated_by: SkipJsonSchema[Literal["llm", "fallback"]] = "llm"

    @field_validator("disclaimer", mode="before")
    @classmethod
    def _force_disclaimer(cls, _v: Any) -> str:  # the disclaimer is mandatory and fixed
        return DISCLAIMER


class FinalReport(ResearchReport):
    """Final report written by Agent B after the critique loop."""

    clarification_used: bool = Field(
        default=False, description="True if data from Agent A's clarification answer is used."
    )
    clarification_summary: str | None = Field(
        default=None, description="How the clarification answer changed the report."
    )
    sources: list[str] = Field(default_factory=list, description="URLs and tool names cited.")


# --------------------------------------------------------------------------- #
# Agent-to-agent messages
# --------------------------------------------------------------------------- #
class AnalystNotes(BaseModel):
    """Agent A's interpretation of its own quantitative tool results."""

    key_observations: list[str] = Field(
        description="3-6 observations that combine indicators (not single values restated)."
    )
    quantitative_flags: list[str] = Field(
        description="Short machine-friendly flags such as 'overbought', 'elevated_volatility'."
    )
    data_gaps: list[str] = Field(default_factory=list, description="Data that is missing or unreliable.")


class DataBrief(BaseModel):
    """Typed handoff from Agent A (Data Analyst) to Agent B (Research Writer).

    Numeric fields are copied from tool results by code, never typed by the LLM.
    """

    ticker: str
    as_of: str
    price: PriceSnapshot | None = None
    volatility: VolatilityMetrics | None = None
    sentiment: SentimentResult | None = None
    key_observations: list[str] = Field(default_factory=list)
    quantitative_flags: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """A specific request from Agent B to Agent A for additional quantitative data."""

    question: str = Field(description="One specific question for the Data Analyst.")
    requested_metrics: list[str] = Field(
        min_length=1, description="The exact metrics needed, e.g. 'vol_90d', 'max_drawdown_90d'."
    )
    reason: str = Field(description="Which part of the report needs this and why.")
    # Attached by code from Agent B's get_news / web_search results (hidden from the LLM schema):
    headlines: SkipJsonSchema[list[str]] = Field(default_factory=list)


class WriterReview(BaseModel):
    """Agent B's critique of the evidence collected so far."""

    assessment: str = Field(description="What the evidence supports well and what is weak or missing.")
    needs_clarification: bool
    request: ClarificationRequest | None = None


class ClarificationAnswer(BaseModel):
    """Agent A's written answer to a clarification request."""

    answer: str = Field(description="A direct answer with the requested numbers.")


class ClarificationResponse(BaseModel):
    """Typed response from Agent A back to Agent B."""

    question: str
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Persistence and tracing
# --------------------------------------------------------------------------- #
class CachedBrief(BaseModel):
    ticker: str
    date: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str
    run_id: str
    tool_call_count: int
    data_brief: DataBrief | None
    final_report: FinalReport


class TraceRecord(BaseModel):
    ts: str
    run_id: str
    session: str
    agent: str
    kind: Literal["tool", "llm", "guard", "validation", "cache"]
    tool: str
    args: dict[str, Any]
    output: str
    duration_ms: float
    status: str
