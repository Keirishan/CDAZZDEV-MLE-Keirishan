import math

import numpy as np
import pandas as pd
import pytest

from agentic import indicators as ind


def test_sma_simple():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi_wilder(s, 14).dropna().eq(100.0).all()


def test_rsi_alternating_is_near_50():
    s = pd.Series([100 + (1 if i % 2 else -1) for i in range(200)], dtype=float)
    assert ind.rsi_wilder(s, 14).iloc[-1] == pytest.approx(50.0, abs=3.0)


def test_rsi_matches_manual_wilder():
    s = pd.Series([44, 44.3, 44.1, 43.6, 44.3, 44.8, 45.1, 45.4, 45.8, 46.1, 45.9, 46.2, 45.6, 46.3, 46.3, 46.0], dtype=float)
    d = np.diff(s.to_numpy())
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:14].mean(), l[:14].mean()
    expected_14 = 100 - 100 / (1 + ag / al)
    ag2, al2 = (ag * 13 + g[14]) / 14, (al * 13 + l[14]) / 14
    expected_15 = 100 - 100 / (1 + ag2 / al2)
    r = ind.rsi_wilder(s, 14)
    assert r.iloc[14] == pytest.approx(expected_14)
    assert r.iloc[15] == pytest.approx(expected_15)
    assert r.iloc[:14].isna().all()


def test_macd_identity_and_ema_alpha():
    s = pd.Series(np.linspace(10, 30, 120) + np.sin(np.arange(120)))
    m = ind.macd(s)
    pd.testing.assert_series_equal(m["hist"], m["macd"] - m["signal"], check_names=False)
    manual = s.ewm(alpha=2 / 13, adjust=False).mean() - s.ewm(alpha=2 / 27, adjust=False).mean()
    assert m["macd"].iloc[-1] == pytest.approx(manual.iloc[-1])


def test_bollinger_uses_population_std():
    s = pd.Series(np.arange(1, 41, dtype=float))
    bb = ind.bollinger(s, 20, 2)
    window = s.iloc[-20:]
    assert bb["upper"].iloc[-1] == pytest.approx(window.mean() + 2 * window.std(ddof=0))
    assert 0.5 < bb["percent_b"].iloc[-1] <= 1.0


def test_annualised_vol_recovers_known_sigma():
    rng = np.random.default_rng(0)
    daily = 0.02
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, daily, 5000))))
    assert ind.annualised_vol(close, 4000) == pytest.approx(daily * math.sqrt(252), rel=0.05)


def test_max_drawdown_and_expected_move():
    s = pd.Series([100, 120, 90, 110], dtype=float)
    assert ind.max_drawdown(s) == pytest.approx(90 / 120 - 1)
    assert ind.expected_move(100, 0.5, 90) == pytest.approx(100 * 0.5 * math.sqrt(90 / 365))


def test_momentum_score_bounds():
    assert ind.momentum_score(110, 100, 90, 1, 0, 60) == 4
    assert ind.momentum_signal(4) == "bullish"
    assert ind.momentum_score(80, 100, 90, -1, 0, 40) == -4
    assert ind.momentum_score(None, None, None, None, None, None) == 0
