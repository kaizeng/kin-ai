from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def risk_parity_weights(returns: pd.DataFrame, min_weight: float = 0.0) -> Dict[str, float]:
    """Simple inverse-volatility weights (risk parity proxy)."""
    vol = returns.std().replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.fillna(0.0)
    if inv.sum() == 0:
        w = np.ones(len(inv)) / len(inv)
        return dict(zip(returns.columns, w))
    weights = inv / inv.sum()
    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()
    return weights.to_dict()


def calibrate_etf_plan(prices: pd.DataFrame, method: str = "risk_parity") -> Dict[str, float]:
    """Calibrate a long-term ETF plan from price history.

    Returns a dict of target weights by ticker.
    """
    returns = prices.pct_change().dropna()
    if method == "risk_parity":
        return risk_parity_weights(returns)
    if method == "equal_weight":
        w = 1.0 / len(prices.columns)
        return {t: w for t in prices.columns}
    raise ValueError(f"Unknown method: {method}")
