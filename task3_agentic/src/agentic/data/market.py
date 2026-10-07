"""Market data access (yfinance) with an in-process cache and one retry.

Kept separate from the tools so tests can replace it with synthetic data.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yfinance history and info loaders with
# validation, retry and a same-day in-memory cache', Date: 2026-10-07
"""
from __future__ import annotations

import math
import re
import time
from datetime import date
from typing import Any

import pandas as pd

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")
VALID_PERIODS = ("6mo", "1y", "2y", "5y")
# Trading days represented by each period (used to slice returns)
PERIOD_TRADING_DAYS = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
# Always load at least 2 years so the 200-day SMA has enough warm-up data
MIN_HISTORY_PERIOD = "2y"


class DataError(Exception):
    """Expected, explainable data problem (bad ticker, empty response...)."""


_history_cache: dict[tuple[str, str, date], pd.DataFrame] = {}
_info_cache: dict[tuple[str, date], dict] = {}


def normalise_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not TICKER_RE.match(t):
        raise DataError(f"'{ticker}' is not a valid ticker symbol.")
    return t


def history_period_for(period: str) -> str:
    if period not in VALID_PERIODS:
        raise DataError(f"period must be one of {VALID_PERIODS}, got '{period}'.")
    order = list(VALID_PERIODS)
    return period if order.index(period) >= order.index(MIN_HISTORY_PERIOD) else MIN_HISTORY_PERIOD


def load_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily OHLCV, auto-adjusted. Raises DataError on empty data."""
    import yfinance as yf

    t = normalise_ticker(ticker)
    key = (t, period, date.today())
    if key in _history_cache:
        return _history_cache[key].copy()

    last_exc: Exception | None = None
    df = pd.DataFrame()
    for attempt in range(2):
        try:
            df = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                break
        except Exception as exc:  # network / parsing issues inside yfinance
            last_exc = exc
        time.sleep(1.5 * (attempt + 1))

    if df is None or df.empty:
        detail = f" ({type(last_exc).__name__}: {last_exc})" if last_exc else ""
        raise DataError(f"No price data returned for '{t}'{detail}. Check the ticker symbol.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[~df["Close"].isna()]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    _history_cache[key] = df
    return df.copy()


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def load_fundamentals(ticker: str) -> dict:
    """Best-effort fundamentals from yfinance .info. Returns {} if unavailable."""
    import yfinance as yf

    t = normalise_ticker(ticker)
    key = (t, date.today())
    if key in _info_cache:
        return dict(_info_cache[key])
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}
    out = {
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "currency": info.get("currency"),
        "market_cap": _num(info.get("marketCap")),
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "profit_margin": _num(info.get("profitMargins")),
        "revenue_growth": _num(info.get("revenueGrowth")),
        "earnings_growth": _num(info.get("earningsGrowth")),
        "debt_to_equity": _num(info.get("debtToEquity")),
        "current_ratio": _num(info.get("currentRatio")),
        "return_on_equity": _num(info.get("returnOnEquity")),
        "free_cash_flow": _num(info.get("freeCashflow")),
        "beta": _num(info.get("beta")),
    }
    _info_cache[key] = out
    return dict(out)


def clear_caches() -> None:
    _history_cache.clear()
    _info_cache.clear()
