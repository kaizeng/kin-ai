from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def backtest_lump_sum(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    initial_cash: float = 10_000.0,
    rebalance_freq: str = "QE",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Backtest a lump-sum investment with periodic rebalancing.

    Returns (portfolio_value_series, weights_over_time_dataframe).
    """
    prices = prices.dropna()
    weights = {k: v for k, v in weights.items() if k in prices.columns}
    if not weights:
        raise ValueError("weights must overlap price columns")

    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    w = w / w.sum()

    rebal_dates = prices.resample(rebalance_freq).last().index
    rebal_dates = set(rebal_dates)

    holdings = pd.Series(0.0, index=prices.columns)
    portfolio_values = []
    weight_records = []

    for dt, row in prices.iterrows():
        if dt in rebal_dates:
            target_value = initial_cash if holdings.sum() == 0 else (holdings * row).sum()
            holdings = (target_value * w) / row
        port_val = (holdings * row).sum()
        portfolio_values.append(port_val)
        # Record actual weight of each asset
        if port_val > 0:
            weight_records.append((holdings * row / port_val).to_dict())
        else:
            weight_records.append({c: 0.0 for c in prices.columns})

    pv = pd.Series(portfolio_values, index=prices.index, name="portfolio_value")
    wt = pd.DataFrame(weight_records, index=prices.index)
    return pv, wt
