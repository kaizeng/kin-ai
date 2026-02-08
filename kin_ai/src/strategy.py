from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def risk_parity_weights(
    returns: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Dict[str, float]:
    """Simple inverse-volatility weights (risk parity proxy).

    Parameters
    ----------
    max_weight : float
        Maximum weight for any single asset (concentration cap).
        Excess weight is redistributed proportionally to other assets.
    """
    vol = returns.std().replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.fillna(0.0)
    if inv.sum() == 0:
        w = np.ones(len(inv)) / len(inv)
        return dict(zip(returns.columns, w))
    weights = inv / inv.sum()
    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()

    # Iterative cap: clip at max_weight, redistribute excess
    for _ in range(10):  # converges in 2-3 iterations
        capped = weights.clip(upper=max_weight)
        excess = weights.sum() - capped.sum()
        if excess < 1e-9:
            break
        # Redistribute excess proportionally to uncapped assets
        uncapped_mask = capped < max_weight
        if uncapped_mask.sum() == 0:
            break
        uncapped_total = capped[uncapped_mask].sum()
        if uncapped_total > 0:
            capped[uncapped_mask] += excess * capped[uncapped_mask] / uncapped_total
        else:
            capped[uncapped_mask] += excess / uncapped_mask.sum()
        weights = capped
    weights = weights / weights.sum()
    return weights.to_dict()


def max_sharpe_min_dd_weights(
    returns: pd.DataFrame,
    max_weight: float = 1.0,
    rf: float = 0.045,
    dd_penalty: float = 2.0,
    n_portfolios: int = 12_000,
) -> Dict[str, float]:
    """Monte-Carlo optimisation targeting max Sharpe with drawdown penalty.

    Generates ``n_portfolios`` random weight vectors, scores each by:
        score = Sharpe - dd_penalty * |MaxDrawdown|
    and returns the weights of the best portfolio.

    Parameters
    ----------
    returns : DataFrame
        Daily returns (columns = tickers).
    max_weight : float
        Maximum weight for any single asset.
    rf : float
        Annualised risk-free rate.
    dd_penalty : float
        How much to penalise drawdown relative to Sharpe.
    n_portfolios : int
        Number of random portfolios to sample.
    """
    n = len(returns.columns)
    if n == 0:
        return {}
    ret_matrix = returns.values  # (T, N)

    best_score = -np.inf
    best_w = np.ones(n) / n

    for _ in range(n_portfolios):
        # Random Dirichlet weights, then cap
        raw = np.random.dirichlet(np.ones(n))
        raw = np.minimum(raw, max_weight)
        raw /= raw.sum()

        # Portfolio daily returns
        port_ret = ret_matrix @ raw
        mu = port_ret.mean() * 252
        sigma = port_ret.std() * np.sqrt(252)
        sharpe = (mu - rf) / sigma if sigma > 1e-9 else 0.0

        # Max drawdown
        cum = np.cumprod(1 + port_ret)
        peak = np.maximum.accumulate(cum)
        dd = ((cum - peak) / peak).min()  # negative

        score = sharpe - dd_penalty * abs(dd)

        if score > best_score:
            best_score = score
            best_w = raw.copy()

    # Iterative cap (same as risk_parity_weights)
    w_series = pd.Series(best_w, index=returns.columns)
    for _ in range(10):
        capped = w_series.clip(upper=max_weight)
        excess = w_series.sum() - capped.sum()
        if excess < 1e-9:
            break
        uncapped = capped < max_weight
        if uncapped.sum() == 0:
            break
        ut = capped[uncapped].sum()
        if ut > 0:
            capped[uncapped] += excess * capped[uncapped] / ut
        else:
            capped[uncapped] += excess / uncapped.sum()
        w_series = capped
    w_series = w_series / w_series.sum()
    return w_series.to_dict()


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
    if method == "max_sharpe":
        return max_sharpe_min_dd_weights(returns)
    raise ValueError(f"Unknown method: {method}")
