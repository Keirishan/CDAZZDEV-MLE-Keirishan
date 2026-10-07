"""Technical indicators and risk metrics computed from first principles (no TA-Lib).

All functions take pandas Series and return Series (or floats) aligned to the input.
Warm-up periods are NaN rather than misleading partial values.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pandas implementations of SMA, RSI with
# Wilder smoothing, MACD(12,26,9), Bollinger Bands(20,2), annualised volatility,
# downside volatility and max drawdown', Date: 2026-10-07
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Indicator parameters required by the brief
SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2.0

# Interpretation thresholds
RSI_OVERBOUGHT, RSI_OVERSOLD = 70.0, 30.0
VOL_ELEVATED_RATIO, VOL_SUBDUED_RATIO = 1.2, 0.8  # window vol vs 1-year vol


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the standard alpha = 2 / (span + 1)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing.

    Seed: simple average of the first `period` gains/losses.
    Then: avg_t = (avg_{t-1} * (period - 1) + value_t) / period.
    """
    values = close.astype(float).to_numpy()
    n = len(values)
    out = np.full(n, np.nan)
    if n <= period:
        return pd.Series(out, index=close.index, name=f"rsi{period}")

    delta = np.diff(values)                     # length n-1, delta[i] = v[i+1] - v[i]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    def _rsi(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else 50.0
        rs = g / l
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _rsi(avg_gain, avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name=f"rsi{period}")


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line})


def bollinger(close: pd.Series, window: int = BB_WINDOW, n_std: float = BB_STD) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)  # population std
    upper, lower = mid + n_std * std, mid - n_std * std
    width = upper - lower
    percent_b = (close - lower) / width.replace(0, np.nan)
    return pd.DataFrame({"upper": upper, "middle": mid, "lower": lower, "percent_b": percent_b})


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def annualised_vol(close: pd.Series, window: int) -> float | None:
    """Std-dev of the last `window` daily log returns, scaled by sqrt(252)."""
    r = log_returns(close)
    if len(r) < max(window, 2):
        return None
    return float(r.tail(window).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def downside_vol(close: pd.Series, window: int) -> float | None:
    """Annualised semi-deviation: only negative returns contribute."""
    r = log_returns(close)
    if len(r) < max(window, 2):
        return None
    tail = r.tail(window)
    semi = np.sqrt(np.mean(np.minimum(tail.to_numpy(), 0.0) ** 2))
    return float(semi * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(close: pd.Series, window: int | None = None) -> float | None:
    """Largest peak-to-trough fall (negative fraction) over the last `window` rows."""
    s = close.dropna()
    if window:
        s = s.tail(window + 1)
    if len(s) < 2:
        return None
    return float((s / s.cummax() - 1.0).min())


def expected_move(price: float, annual_vol: float, horizon_days: int = 90) -> float:
    """One-standard-deviation price move over `horizon_days` calendar days.

    Uses trading-day scaling: sigma * sqrt(horizon_trading_days / 252), where
    horizon_trading_days is approximated as horizon_days * 252 / 365.
    """
    trading_days = horizon_days * TRADING_DAYS_PER_YEAR / 365.0
    return float(price * annual_vol * math.sqrt(trading_days / TRADING_DAYS_PER_YEAR))


def momentum_score(close: float | None, sma50: float | None, sma200: float | None,
                   macd_v: float | None, macd_sig: float | None, rsi_v: float | None) -> int:
    """Simple composite: +1/-1 for each of four trend conditions (range -4 .. +4)."""
    score = 0
    pairs = [(close, sma50), (close, sma200), (macd_v, macd_sig), (rsi_v, 50.0)]
    for a, b in pairs:
        if a is None or b is None or any(map(lambda x: isinstance(x, float) and math.isnan(x), (a, b))):
            continue
        score += 1 if a > b else -1
    return score


def momentum_signal(score: int) -> str:
    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"
