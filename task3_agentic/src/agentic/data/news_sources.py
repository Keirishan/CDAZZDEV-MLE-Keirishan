"""News headline sources: yfinance first, then free RSS feeds as fallbacks.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Parse yfinance news in both the old flat
# and new nested content formats, with Yahoo and Google News RSS fallbacks', Date: 2026-10-07
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
USER_AGENT = "Mozilla/5.0 (compatible; cdazzdev-task3-research/1.0)"


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def fetch_yfinance(ticker: str, n: int) -> list[dict]:
    import yfinance as yf

    raw = yf.Ticker(ticker).get_news(count=max(n, 10)) or []
    items: list[dict] = []
    for entry in raw:
        content = entry.get("content") if isinstance(entry.get("content"), dict) else None
        if content:  # newer yfinance format
            title = content.get("title")
            publisher = (content.get("provider") or {}).get("displayName")
            published = content.get("pubDate") or content.get("displayTime")
            url = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url")
        else:  # older flat format
            title = entry.get("title")
            publisher = entry.get("publisher")
            ts = entry.get("providerPublishTime")
            published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
            url = entry.get("link")
        if _clean(title):
            items.append({"title": _clean(title), "publisher": publisher, "published": published,
                          "url": url, "source": "yfinance"})
    return items


def _fetch_rss(url: str, source: str) -> list[dict]:
    import feedparser

    feed = feedparser.parse(url, agent=USER_AGENT)
    items = []
    for e in feed.entries or []:
        title = _clean(getattr(e, "title", ""))
        if not title:
            continue
        publisher = None
        src = getattr(e, "source", None)
        if isinstance(src, dict):
            publisher = src.get("title")
        items.append({"title": title, "publisher": publisher,
                      "published": getattr(e, "published", None),
                      "url": getattr(e, "link", None), "source": source})
    return items


def fetch_yahoo_rss(ticker: str) -> list[dict]:
    return _fetch_rss(YAHOO_RSS.format(ticker=quote_plus(ticker)), "yahoo_rss")


def fetch_google_news_rss(ticker: str) -> list[dict]:
    return _fetch_rss(GOOGLE_NEWS_RSS.format(query=quote_plus(f"{ticker} stock")), "google_news_rss")


SOURCES = (
    ("yfinance", fetch_yfinance),
    ("yahoo_rss", lambda t, n: fetch_yahoo_rss(t)),
    ("google_news_rss", lambda t, n: fetch_google_news_rss(t)),
)


def collect_headlines(ticker: str, n: int) -> tuple[list[dict], list[str], list[str]]:
    """Try each source in order until `n` unique headlines are collected.

    Returns (items, sources_used, errors).
    """
    seen: set[str] = set()
    items: list[dict] = []
    used: list[str] = []
    errors: list[str] = []
    for name, fn in SOURCES:
        if len(items) >= n:
            break
        try:
            batch = fn(ticker, n)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        added = 0
        for it in batch:
            key = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:80]
            if key and key not in seen:
                seen.add(key)
                items.append(it)
                added += 1
                if len(items) >= n:
                    break
        if added:
            used.append(name)
        else:
            errors.append(f"{name}: no new headlines")
    return items, used, errors
