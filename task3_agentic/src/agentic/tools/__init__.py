"""The five research tools and the per-agent allow-lists.

Agent A (Data Analyst) and Agent B (Research Writer) get disjoint tool sets, as the
brief requires. The allow-lists are enforced again at execution time by ToolExecutor.
"""
from .base import ToolError, clear_failures, failing, inject_failures, injected_failures
from .news import get_news
from .price import get_price_data
from .search import web_search
from .sentiment import llm_sentiment
from .volatility import calculate_volatility

ALL_TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
ANALYST_TOOLS = [get_price_data, calculate_volatility, llm_sentiment]
WRITER_TOOLS = [web_search, get_news]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS", "ANALYST_TOOLS", "WRITER_TOOLS", "TOOLS_BY_NAME",
    "get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search",
    "ToolError", "inject_failures", "clear_failures", "failing", "injected_failures",
]
