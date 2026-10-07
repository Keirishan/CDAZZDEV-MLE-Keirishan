"""Web search via the `ddgs` package (the renamed duckduckgo-search library).

Text search first; on rate limits we back off and retry, then fall back to the
news vertical.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'DuckDuckGo search wrapper using ddgs
# with tenacity backoff on RatelimitException and a news-search fallback', Date: 2026-10-07
"""
from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException, TimeoutException
except ImportError:  # older package name
    from duckduckgo_search import DDGS  # type: ignore
    from duckduckgo_search.exceptions import RatelimitException, TimeoutException  # type: ignore


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=12),
       retry=retry_if_exception_type((RatelimitException, TimeoutException)), reraise=True)
def _text(query: str, max_results: int) -> list[dict]:
    return DDGS().text(query, max_results=max_results) or []


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=8),
       retry=retry_if_exception_type((RatelimitException, TimeoutException)), reraise=True)
def _news(query: str, max_results: int) -> list[dict]:
    return DDGS().news(query, max_results=max_results) or []


def search(query: str, max_results: int) -> tuple[list[dict], str, list[str]]:
    """Returns (hits, backend_used, errors). hits: {title, snippet, url, date}."""
    errors: list[str] = []
    for backend, fn in (("ddgs_text", _text), ("ddgs_news", _news)):
        try:
            raw = fn(query, max_results)
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")
            continue
        hits = [
            {
                "title": (r.get("title") or "").strip(),
                "snippet": (r.get("body") or r.get("excerpt") or "").strip()[:400],
                "url": r.get("href") or r.get("url"),
                "date": r.get("date"),
            }
            for r in raw
            if (r.get("title") or "").strip()
        ]
        if hits:
            return hits, backend, errors
        errors.append(f"{backend}: no results")
    return [], "none", errors
