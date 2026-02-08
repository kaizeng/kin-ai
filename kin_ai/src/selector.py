"""ETF selector — pick the best 3-5 ETFs at each rebalance date.

Uses the alpha model to score every ETF in the universe, then selects
the top N and computes risk-parity weights among them.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kin_ai.src.alpha import compute_alpha
from kin_ai.src.strategy import risk_parity_weights, max_sharpe_min_dd_weights


def select_etfs(
    prices: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
    top_n: int = 4,
    min_history: int = 200,
    max_weight: float = 0.40,
    weight_method: str = "max_sharpe",
) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    """Score the universe and pick the best ETFs.

    Parameters
    ----------
    prices : DataFrame
        Full universe daily prices up to the evaluation date.
    dividends : DataFrame, optional
        Full universe dividends.
    top_n : int
        Number of ETFs to select.
    min_history : int
        Minimum trading days a ticker must have to be eligible.

    Returns
    -------
    (selected_tickers, weights_dict, alpha_scores_df)
    """
    # Filter tickers with enough history
    valid = [c for c in prices.columns if prices[c].dropna().shape[0] >= min_history]
    if len(valid) < top_n:
        valid = list(prices.columns)  # fall back to whatever is available

    p = prices[valid].copy()

    # Align dividends
    d = None
    if dividends is not None and not dividends.empty:
        common = [c for c in valid if c in dividends.columns]
        if common:
            d = dividends[common]

    scores = compute_alpha(p, d)

    selected = scores.head(top_n).index.tolist()

    # Compute portfolio weights among selected ETFs
    sel_returns = p[selected].pct_change().dropna()
    if len(sel_returns) < 2:
        w = {t: 1.0 / len(selected) for t in selected}
    elif weight_method == "max_sharpe":
        w = max_sharpe_min_dd_weights(sel_returns, max_weight=max_weight)
    else:
        w = risk_parity_weights(sel_returns, max_weight=max_weight)

    return selected, w, scores


def build_selector_fn(
    universe_prices: pd.DataFrame,
    universe_dividends: Optional[pd.DataFrame] = None,
    top_n: int = 4,
    min_history: int = 200,
    max_weight: float = 0.40,
    weight_method: str = "max_sharpe",
) -> Callable[[pd.Timestamp], Tuple[List[str], Dict[str, float]]]:
    """Build a callable that the backtester can invoke on each rebalance date.

    Parameters
    ----------
    universe_prices : DataFrame
        Full universe daily prices for the entire backtest period.
    universe_dividends : DataFrame, optional
        Full universe dividends.
    top_n : int
        How many ETFs to select.
    min_history : int
        Minimum trading days required.

    Returns
    -------
    A function ``selector(date) -> (selected_tickers, weights_dict)``
    that uses only data up to the given date (look-ahead-free).
    """
    def _selector(dt: pd.Timestamp) -> Tuple[List[str], Dict[str, float]]:
        # Only use data up to (and including) the rebalance date
        hist = universe_prices.loc[:dt]
        div_hist = None
        if universe_dividends is not None and not universe_dividends.empty:
            div_hist = universe_dividends.loc[:dt]
        tickers, weights, _ = select_etfs(
            hist, div_hist, top_n, min_history, max_weight, weight_method,
        )
        return tickers, weights

    return _selector
