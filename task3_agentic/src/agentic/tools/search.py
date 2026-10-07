"""web_search(query): analyst commentary and context via DuckDuckGo (ddgs).

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain web search tool over ddgs
# returning a validated list of title/snippet/url hits', Date: 2026-10-07
"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..data import search_backend
from ..schemas import SearchHit, SearchResult
from .base import ToolError, run_tool


class SearchArgs(BaseModel):
    query: str = Field(description="Search query, e.g. 'NVDA analyst downgrade risk 2026'.")
    max_results: int = Field(default=6, ge=1, le=10, description="Number of results.")


def _web_search(query: str, max_results: int = 6):
    q = (query or "").strip()
    if not q:
        raise ToolError("Empty search query.")
    hits, backend, errors = search_backend.search(q, int(max_results))
    if not hits:
        raise ToolError(f"Web search returned no results. {'; '.join(errors)}")
    result = SearchResult(query=q, count=len(hits), backend=backend, results=[SearchHit(**h) for h in hits])
    return result.model_dump(), backend


@tool("web_search", args_schema=SearchArgs)
def web_search(query: str, max_results: int = 6) -> dict:
    """Search the web (DuckDuckGo) for analyst commentary, upcoming events, regulation,
    competition or macro news. Returns titles, snippets and URLs in a ToolResult dict."""
    return run_tool("web_search", _web_search, {"query": query, "max_results": max_results}).model_dump()
