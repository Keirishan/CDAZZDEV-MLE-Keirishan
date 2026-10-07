"""Test configuration: isolate trace/cache files and replace all network access."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FAKE_HEADLINES, FAKE_SEARCH, Brain, factory_for, synthetic_prices  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Each test gets its own trace file, cache dir, and offline data sources."""
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "agent_trace.jsonl"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("NO_COLOR", "1")

    from agentic import config
    from agentic.data import market, news_sources, search_backend
    from agentic.observability.tracer import tracer
    from agentic.tools import base

    config.reset_settings()
    tracer.set_path(None)
    base.clear_failures()
    market.clear_caches()

    df = synthetic_prices()
    monkeypatch.setattr(market, "load_history", lambda ticker, period="2y": df.copy())
    monkeypatch.setattr(market, "load_fundamentals", lambda ticker: {"name": "Test Corp", "trailing_pe": 45.2})
    monkeypatch.setattr(news_sources, "collect_headlines",
                        lambda ticker, n: (FAKE_HEADLINES[:n], ["yfinance"], []))
    monkeypatch.setattr(search_backend, "search", lambda q, k: (FAKE_SEARCH[:k], "ddgs_text", []))
    yield
    base.clear_failures()
    config.reset_settings()


@pytest.fixture
def brain():
    from agentic import llm

    b = Brain()
    llm.set_llm_factory(factory_for(b))
    yield b
    llm.set_llm_factory(None)
