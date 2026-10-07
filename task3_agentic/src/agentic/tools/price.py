"""get_price_data(ticker, period): OHLCV + indicators + fundamentals snapshot.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool wrapping yfinance that
# returns a compact price snapshot with SMA/RSI/MACD/Bollinger, momentum flags and
# fundamentals', Date: 2026-10-07
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .. import indicators as ind
from ..data import market
from ..schemas import Fundamentals, OHLCVRow, PriceSnapshot
from .base import ToolError, run_tool

RECENT_ROWS = 5  # OHLCV rows returned verbatim; the rest is summarised to save tokens


class PriceArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'NVDA'.")
    period: Literal["6mo", "1y", "2y", "5y"] = Field(
        default="1y", description="Look-back period for returns and the OHLCV summary."
    )


def _last(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0


def compute_snapshot(df: pd.DataFrame, ticker: str, period: str) -> PriceSnapshot:
    close = df["Close"].astype(float)
    if len(close) < ind.MACD_SLOW + ind.MACD_SIGNAL:
        raise ToolError(f"Only {len(close)} trading days of data for {ticker}; not enough for indicators.")

    sma50, sma200 = ind.sma(close, ind.SMA_SHORT), ind.sma(close, ind.SMA_LONG)
    rsi = ind.rsi_wilder(close, ind.RSI_PERIOD)
    m = ind.macd(close)
    bb = ind.bollinger(close)

    last_close = _last(close)
    window_days = market.PERIOD_TRADING_DAYS[period]
    period_close = close.tail(window_days + 1)
    year = close.tail(ind.TRADING_DAYS_PER_YEAR)
    this_year = close[close.index.year == close.index[-1].year]
    prev_year_end = close[close.index.year < close.index[-1].year]
    ytd_base = float(prev_year_end.iloc[-1]) if len(prev_year_end) else (float(this_year.iloc[0]) if len(this_year) else None)

    v = {
        "sma50": _last(sma50), "sma200": _last(sma200), "rsi14": _last(rsi),
        "macd": _last(m["macd"]), "macd_signal": _last(m["signal"]), "macd_hist": _last(m["hist"]),
    }
    hist = m["hist"].dropna()
    hist_5 = float(hist.iloc[-6]) if len(hist) >= 6 else None
    vol20 = df["Volume"].astype(float).tail(20).mean()
    score = ind.momentum_score(last_close, v["sma50"], v["sma200"], v["macd"], v["macd_signal"], v["rsi14"])

    flags: list[str] = []
    if v["sma200"] is not None and last_close is not None:
        flags.append("above_sma200" if last_close > v["sma200"] else "below_sma200")
    if v["sma50"] is not None and v["sma200"] is not None:
        flags.append("sma50_above_sma200" if v["sma50"] > v["sma200"] else "sma50_below_sma200")
    if v["rsi14"] is not None:
        if v["rsi14"] >= ind.RSI_OVERBOUGHT:
            flags.append("rsi_overbought")
        elif v["rsi14"] <= ind.RSI_OVERSOLD:
            flags.append("rsi_oversold")
    if v["macd"] is not None and v["macd_signal"] is not None:
        flags.append("macd_above_signal" if v["macd"] > v["macd_signal"] else "macd_below_signal")
    if v["macd_hist"] is not None and hist_5 is not None:
        flags.append("macd_momentum_rising" if v["macd_hist"] > hist_5 else "macd_momentum_fading")
    pb = _last(bb["percent_b"])
    if pb is not None:
        if pb > 1:
            flags.append("above_upper_bollinger")
        elif pb < 0:
            flags.append("below_lower_bollinger")

    recent = [
        OHLCVRow(date=idx.strftime("%Y-%m-%d"), open=row.Open, high=row.High, low=row.Low,
                 close=row.Close, volume=row.Volume)
        for idx, row in df.tail(RECENT_ROWS).iterrows()
    ]
    return PriceSnapshot(
        ticker=ticker,
        as_of=close.index[-1].strftime("%Y-%m-%d"),
        period=period,
        rows_analysed=len(period_close),
        last_close=last_close,
        period_return_pct=_pct(last_close, float(period_close.iloc[0])),
        ytd_return_pct=_pct(last_close, ytd_base),
        high_52w=float(df["High"].tail(ind.TRADING_DAYS_PER_YEAR).max()),
        low_52w=float(df["Low"].tail(ind.TRADING_DAYS_PER_YEAR).min()),
        pct_from_52w_high=_pct(last_close, float(year.max())),
        **v,
        macd_hist_5d_ago=hist_5,
        bb_upper=_last(bb["upper"]), bb_middle=_last(bb["middle"]), bb_lower=_last(bb["lower"]),
        bb_percent_b=pb,
        avg_volume_20d=float(vol20) if pd.notna(vol20) else None,
        volume_ratio_20d=(float(df["Volume"].iloc[-1]) / float(vol20)) if vol20 else None,
        momentum_score=score,
        momentum_signal=ind.momentum_signal(score),
        flags=flags,
        recent_ohlcv=recent,
    )


def _get_price_data(ticker: str, period: str = "1y", include_fundamentals: bool = True):
    try:
        t = market.normalise_ticker(ticker)
        df = market.load_history(t, market.history_period_for(period))
    except market.DataError as exc:
        raise ToolError(str(exc)) from exc
    snap = compute_snapshot(df, t, period)
    if include_fundamentals:
        try:
            snap.fundamentals = Fundamentals(**market.load_fundamentals(t))
        except Exception:
            snap.fundamentals = None  # fundamentals are best-effort
    return snap.model_dump(), "yfinance"


@tool("get_price_data", args_schema=PriceArgs)
def get_price_data(ticker: str, period: str = "1y") -> dict:
    """Get daily price data for a stock with computed indicators: last close, period and
    YTD return, 52-week range, SMA50, SMA200, RSI14 (Wilder), MACD(12,26,9), Bollinger
    Bands(20,2), a momentum signal, trend flags, key fundamentals (P/E, margins, growth,
    debt/equity) and the last 5 OHLCV rows. Returns a ToolResult dict."""
    return run_tool("get_price_data", _get_price_data, {"ticker": ticker, "period": period}).model_dump()
