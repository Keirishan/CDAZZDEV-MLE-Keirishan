"""get_news(ticker, n): recent headlines as a structured list.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool returning recent
# headlines from yfinance with RSS fallbacks as a validated list', Date: 2026-10-07
"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..data import market, news_sources
from ..schemas import NewsItem, NewsResult
from .base import ToolError, run_tool

MAX_HEADLINES = 25


class NewsArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'NVDA'.")
    n: int = Field(default=10, ge=1, le=MAX_HEADLINES, description="Number of headlines to return.")


def _get_news(ticker: str, n: int = 10):
    try:
        t = market.normalise_ticker(ticker)
    except market.DataError as exc:
        raise ToolError(str(exc)) from exc
    items, used, errors = news_sources.collect_headlines(t, int(n))
    if not items:
        raise ToolError(f"No headlines found for {t}. Sources tried: {'; '.join(errors) or 'none'}")
    result = NewsResult(ticker=t, count=len(items),
                        items=[NewsItem(**it) for it in items], sources_used=used)
    return result.model_dump(), "+".join(used)


@tool("get_news", args_schema=NewsArgs)
def get_news(ticker: str, n: int = 10) -> dict:
    """Get the n most recent news headlines for a ticker, each with title, publisher,
    published time, url and source. Uses yfinance news first, then Yahoo Finance and
    Google News RSS feeds as fallbacks. Returns a ToolResult dict."""
    return run_tool("get_news", _get_news, {"ticker": ticker, "n": n}).model_dump()
