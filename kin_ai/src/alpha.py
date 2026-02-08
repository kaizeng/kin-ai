"""Alpha model — multi-factor scoring for ETF selection.

Computes a composite alpha score for each ETF in the universe based on
multiple signals.  Higher score = more attractive.

Factors
-------
1. **Momentum (40 %)** — 6-month price return, z-scored across universe.
2. **Trend (20 %)** — distance of price above/below its 200-day SMA,
   expressed as a percentage, then z-scored.
3. **Low Volatility (15 %)** — inverse of 60-day annualised volatility,
   z-scored (lower vol → higher score).
4. **Mean Reversion (10 %)** — 14-day RSI inverted: assets near oversold
   (RSI < 30) score higher, overbought (RSI > 70) score lower.
5. **Dividend Yield (15 %)** — trailing 12-month dividend yield, z-scored.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ── Factor weights (must sum to 1.0) ───────────────────────────
FACTOR_WEIGHTS: Dict[str, float] = {
    "momentum":       0.40,
    "trend":          0.20,
    "low_vol":        0.15,
    "mean_reversion": 0.10,
    "dividend_yield": 0.15,
}


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score (NaN-safe)."""
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or np.isnan(sigma):
        return s * 0.0
    return (s - mu) / sigma


def momentum_score(prices: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """6-month total return, z-scored across tickers."""
    if len(prices) < lookback:
        lookback = max(len(prices) - 1, 1)
    ret = prices.iloc[-1] / prices.iloc[-lookback] - 1
    return _zscore(ret)


def trend_score(prices: pd.DataFrame, window: int = 200) -> pd.Series:
    """Distance from 200-day SMA as %, z-scored."""
    if len(prices) < window:
        window = max(len(prices) - 1, 1)
    sma = prices.rolling(window).mean().iloc[-1]
    dist = (prices.iloc[-1] - sma) / sma
    return _zscore(dist)


def low_vol_score(prices: pd.DataFrame, window: int = 60) -> pd.Series:
    """Inverse annualised volatility over last 60 days, z-scored."""
    if len(prices) < window:
        window = max(len(prices) - 1, 2)
    vol = prices.pct_change().iloc[-window:].std() * np.sqrt(252)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    return _zscore(inv_vol.fillna(0.0))


def mean_reversion_score(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI-based mean-reversion signal, z-scored.

    Inverted: low RSI (oversold) → high score.
    """
    if len(prices) < period + 1:
        return pd.Series(0.0, index=prices.columns)

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    # Invert: lower RSI = higher score
    inverted = 100 - rsi
    return _zscore(inverted)


def dividend_yield_score(
    prices: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Trailing 12-month dividend yield, z-scored."""
    if dividends is None or dividends.empty:
        return pd.Series(0.0, index=prices.columns)
    # Align dividends to prices date range
    common = prices.columns.intersection(dividends.columns)
    if common.empty:
        return pd.Series(0.0, index=prices.columns)
    last_252 = dividends[common].iloc[-252:] if len(dividends) >= 252 else dividends[common]
    annual_div = last_252.sum()
    current_price = prices[common].iloc[-1]
    yld = annual_div / current_price.replace(0, np.nan)
    yld = yld.fillna(0.0)
    # Extend to all tickers in prices
    full = pd.Series(0.0, index=prices.columns)
    full.update(yld)
    return _zscore(full)


def compute_alpha(
    prices: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute composite alpha score for all tickers.

    Parameters
    ----------
    prices : DataFrame
        Daily close prices (columns = tickers).
    dividends : DataFrame, optional
        Daily per-share dividends (columns = tickers).
    weights : dict, optional
        Override default factor weights.

    Returns
    -------
    DataFrame with columns: ticker, momentum, trend, low_vol, mean_reversion,
    dividend_yield, alpha (composite).  Sorted descending by alpha.
    """
    fw = weights or FACTOR_WEIGHTS

    factors = pd.DataFrame(index=prices.columns)
    factors["momentum"] = momentum_score(prices)
    factors["trend"] = trend_score(prices)
    factors["low_vol"] = low_vol_score(prices)
    factors["mean_reversion"] = mean_reversion_score(prices)
    factors["dividend_yield"] = dividend_yield_score(prices, dividends)

    # Composite
    factors["alpha"] = sum(
        factors[f] * fw.get(f, 0.0) for f in fw
    )
    factors.index.name = "ticker"
    return factors.sort_values("alpha", ascending=False)


def rank_etfs(
    prices: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> list[str]:
    """Return the top-N tickers by composite alpha score."""
    scores = compute_alpha(prices, dividends)
    return scores.head(top_n).index.tolist()
