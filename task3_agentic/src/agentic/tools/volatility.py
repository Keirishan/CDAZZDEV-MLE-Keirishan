"""calculate_volatility(ticker, window): annualised historical volatility and risk metrics.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool computing annualised
# log-return volatility for a window plus 30/60/90/252-day vols, downside vol, max
# drawdown and the expected 90-day 1-sigma move', Date: 2026-10-07
"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .. import indicators as ind
from ..data import market
from ..schemas import VolatilityMetrics
from .base import ToolError, run_tool

HORIZON_DAYS = 90  # the research question's horizon (calendar days)


class VolatilityArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'NVDA'.")
    window: int = Field(default=30, ge=5, le=252,
                        description="Number of trading days for the main volatility estimate.")


def compute_volatility(close, ticker: str, window: int) -> VolatilityMetrics:
    if len(close.dropna()) < window + 1:
        raise ToolError(f"Need at least {window + 1} prices for a {window}-day window; got {len(close)}.")
    main = ind.annualised_vol(close, window)
    v1y = ind.annualised_vol(close, ind.TRADING_DAYS_PER_YEAR)
    last = float(close.dropna().iloc[-1])
    regime = "unknown"
    if main is not None and v1y:
        ratio = main / v1y
        regime = ("elevated" if ratio > ind.VOL_ELEVATED_RATIO
                  else "subdued" if ratio < ind.VOL_SUBDUED_RATIO else "normal")
    move = ind.expected_move(last, main, HORIZON_DAYS) if main is not None else None
    return VolatilityMetrics(
        ticker=ticker,
        as_of=close.index[-1].strftime("%Y-%m-%d"),
        window=window,
        annualised_vol=main,
        vol_30d=ind.annualised_vol(close, 30),
        vol_60d=ind.annualised_vol(close, 60),
        vol_90d=ind.annualised_vol(close, 90),
        vol_1y=v1y,
        downside_vol=ind.downside_vol(close, window),
        max_drawdown=ind.max_drawdown(close, window),
        last_close=last,
        expected_move_90d_1sigma=move,
        expected_move_90d_pct=(move / last * 100.0) if move is not None else None,
        vol_regime=regime,
    )


def _calculate_volatility(ticker: str, window: int = 30):
    try:
        t = market.normalise_ticker(ticker)
        df = market.load_history(t, "2y")
    except market.DataError as exc:
        raise ToolError(str(exc)) from exc
    return compute_volatility(df["Close"].astype(float), t, int(window)).model_dump(), "yfinance"


@tool("calculate_volatility", args_schema=VolatilityArgs)
def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Compute annualised historical volatility (std of daily log returns x sqrt(252)) over
    `window` trading days, plus 30/60/90-day and 1-year volatility, downside volatility,
    maximum drawdown over the window, a volatility regime label, and the expected
    one-sigma price move over the next 90 days. Returns a ToolResult dict."""
    return run_tool("calculate_volatility", _calculate_volatility,
                    {"ticker": ticker, "window": window}).model_dump()
