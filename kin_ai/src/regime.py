from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def detect_market_regime(
    price: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
    vol_window: int = 20,
) -> Tuple[str, str]:
    """Detect market regime using trend + volatility.

    Returns (trend_regime, vol_regime) as strings.
    """
    short_ma = price.rolling(short_window).mean()
    long_ma = price.rolling(long_window).mean()
    trend = "bull" if short_ma.iloc[-1] > long_ma.iloc[-1] else "bear"

    vol = price.pct_change().rolling(vol_window).std()
    vol_level = "high" if vol.iloc[-1] > vol.quantile(0.75) else "low"
    return trend, vol_level


def infer_economic_cycle(trend_regime: str, vol_regime: str) -> str:
    """Infer a simple economic cycle from regime signals."""
    if trend_regime == "bull" and vol_regime == "low":
        return "expansion"
    if trend_regime == "bull" and vol_regime == "high":
        return "slowdown"
    if trend_regime == "bear" and vol_regime == "high":
        return "contraction"
    return "recovery"
