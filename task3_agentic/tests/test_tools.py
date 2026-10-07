from agentic.schemas import NewsResult, PriceSnapshot, SearchResult, SentimentResult, ToolResult, VolatilityMetrics
from agentic.tools import (calculate_volatility, failing, get_news, get_price_data, llm_sentiment,
                           web_search)


def test_get_price_data_returns_validated_snapshot():
    out = ToolResult.model_validate(get_price_data.invoke({"ticker": "nvda", "period": "1y"}))
    assert out.ok and out.tool == "get_price_data"
    snap = PriceSnapshot.model_validate(out.data)
    assert snap.ticker == "NVDA" and snap.sma200 is not None and 0 <= snap.rsi14 <= 100
    assert snap.fundamentals and snap.fundamentals.trailing_pe == 45.2
    assert len(snap.recent_ohlcv) == 5


def test_bad_ticker_is_an_error_not_an_exception():
    out = get_price_data.invoke({"ticker": "not a ticker!", "period": "1y"})
    assert out["ok"] is False and "valid ticker" in out["error"]


def test_volatility_metrics():
    out = calculate_volatility.invoke({"ticker": "NVDA", "window": 30})
    v = VolatilityMetrics.model_validate(out["data"])
    assert 0 < v.annualised_vol < 2 and v.max_drawdown <= 0 and v.expected_move_90d_1sigma > 0


def test_news_and_search():
    n = NewsResult.model_validate(get_news.invoke({"ticker": "NVDA", "n": 10})["data"])
    assert n.count == 10
    s = SearchResult.model_validate(web_search.invoke({"query": "NVDA risk"})["data"])
    assert s.count >= 1


def test_failure_injection():
    with failing("get_news"):
        out = get_news.invoke({"ticker": "NVDA", "n": 5})
    assert out["ok"] is False and "Simulated outage" in out["error"]
    assert get_news.invoke({"ticker": "NVDA", "n": 5})["ok"] is True


def test_sentiment_tool(brain):
    out = llm_sentiment.invoke({"headlines": ["Chipmaker beats", "Probe launched", "Conference held"]})
    res = SentimentResult.model_validate(out["data"])
    assert res.count == 3 and -1 <= res.overall_score <= 1
    assert set(res.counts) == {"positive", "neutral", "negative"}


def test_sentiment_without_headlines_is_handled(brain):
    out = llm_sentiment.invoke({"headlines": []})
    assert out["ok"] is False and "No headlines" in out["error"]
