"""llm_sentiment(headlines): LLM headline sentiment with validated structured output.

Aggregation: each headline maps to +1 / 0 / -1 (positive / neutral / negative) and the
overall score is the confidence-weighted mean, in [-1, 1].

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool that scores headlines
# with an LLM via a Pydantic schema and aggregates a confidence-weighted score', Date: 2026-10-07
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .. import prompts
from ..llm import invoke_structured
from ..schemas import HeadlineSentiment, SentimentBatch, SentimentResult
from .base import ToolError, run_tool

MAX_HEADLINES = 25
SENTIMENT_VALUE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
LABEL_THRESHOLD = 0.15  # |score| below this is labelled neutral


class SentimentArgs(BaseModel):
    headlines: list[str] = Field(description="Headline strings to score (max 25).")
    context: str | None = Field(default=None, description="Optional company or ticker context, e.g. 'NVDA (Nvidia)'.")


def aggregate(items: list[HeadlineSentiment]) -> SentimentResult:
    weight = sum(i.confidence for i in items)
    score = (sum(SENTIMENT_VALUE[i.sentiment] * i.confidence for i in items) / weight) if weight else 0.0
    label = "positive" if score > LABEL_THRESHOLD else "negative" if score < -LABEL_THRESHOLD else "neutral"
    counts = {k: sum(1 for i in items if i.sentiment == k) for k in SENTIMENT_VALUE}
    return SentimentResult(count=len(items), overall_score=round(score, 4), overall_label=label,
                           counts=counts, items=items)


def _llm_sentiment(headlines: list[str], context: str | None = None):
    clean = [h.strip() for h in (headlines or []) if isinstance(h, str) and h.strip()][:MAX_HEADLINES]
    if not clean:
        raise ToolError("No headlines supplied. Retrieve headlines first (get_news or web_search).")
    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(clean))
    messages = [
        SystemMessage(content=prompts.SENTIMENT_SYSTEM),
        HumanMessage(content=prompts.SENTIMENT_USER.format(context=context or "not given", headlines=numbered)),
    ]
    res = invoke_structured(SentimentBatch, messages, label="llm_sentiment", role="sentiment")
    if not res.ok or res.value is None:
        raise ToolError(f"Sentiment model output failed validation: {res.error}")
    items = res.value.items
    if len(items) != len(clean):
        # Re-align by position; keep only as many as both lists have.
        items = items[: len(clean)]
    # Copy original headline text so the output always matches the input.
    items = [it.model_copy(update={"headline": clean[i]}) for i, it in enumerate(items)]
    return aggregate(items).model_dump(), "llm"


@tool("llm_sentiment", args_schema=SentimentArgs)
def llm_sentiment(headlines: list[str], context: str | None = None) -> dict:
    """Score a list of news headlines with an LLM. Returns per-headline sentiment
    (positive/negative/neutral), confidence (0-1) and a brief reason, plus a
    confidence-weighted overall score from -1 to 1. Returns a ToolResult dict."""
    return run_tool("llm_sentiment", _llm_sentiment, {"headlines": headlines, "context": context}).model_dump()
