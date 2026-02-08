from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd


class BacktestResult(NamedTuple):
    """Container for backtest outputs."""
    portfolio: pd.Series            # total portfolio value (positions + cash)
    weights: pd.DataFrame           # actual asset weights over time
    total_dividends: float          # cumulative dividends received
    dividend_cash: pd.Series        # running cash balance from dividends (cash-out mode)
    cumulative_dividends: pd.Series # running total of all dividends received
    selection_log: list             # list of (date, selected_tickers) on each rebalance


def backtest_lump_sum(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    initial_cash: float = 10_000.0,
    rebalance_freq: str = "QE",
    dividends: Optional[pd.DataFrame] = None,
    reinvest_dividends: bool = True,
    selector: Optional[Callable] = None,
) -> BacktestResult:
    """Backtest a lump-sum investment with periodic rebalancing.

    Parameters
    ----------
    prices : DataFrame
        Daily close prices (columns = tickers).
    weights : dict
        Target allocation weights (used for first deployment and when
        no selector is provided).
    initial_cash : float
        Starting capital.
    rebalance_freq : str
        Pandas offset alias for rebalance schedule (e.g. "QE").
    dividends : DataFrame or None
        Per-share dividend amounts indexed by date (columns = tickers).
        If None, dividends are ignored (prices assumed adjusted).
    reinvest_dividends : bool
        If True, dividends buy additional shares on the ex-date.
        If False, dividends accumulate as a separate cash balance.
    selector : callable or None
        If provided, called as ``selector(date)`` on each rebalance date.
        Must return ``(selected_tickers, weights_dict)``.
        This enables dynamic ETF selection (auto-pick mode).

    Returns
    -------
    BacktestResult
        Named tuple with portfolio, weights, dividends, selection_log.
    """
    prices = prices.dropna()
    weights = {k: v for k, v in weights.items() if k in prices.columns}
    if not weights:
        raise ValueError("weights must overlap price columns")

    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    w = w / w.sum()

    rebal_dates = set(prices.resample(rebalance_freq).last().index)

    # Align dividend data to prices index
    has_divs = dividends is not None and not dividends.empty
    if has_divs:
        dividends = dividends.reindex(prices.index).fillna(0.0)
        # Keep only columns present in prices
        common = prices.columns.intersection(dividends.columns)
        div_aligned = dividends[common].reindex(columns=prices.columns, fill_value=0.0)
    else:
        div_aligned = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    holdings = pd.Series(0.0, index=prices.columns)  # shares held
    cash_balance = 0.0          # cash from dividends (cash-out mode)
    total_divs_received = 0.0   # lifetime dividends received

    portfolio_values = []
    weight_records = []
    cash_values = []
    cum_div_values = []
    selection_log: list = []

    first_day = True
    # Track current target weights as a Series aligned to prices.columns
    w_current = w.copy()

    for dt, row in prices.iterrows():
        # ── Collect dividends for today ──────────────────────
        if holdings.sum() > 0:
            div_today = div_aligned.loc[dt]
            div_income = (holdings * div_today).sum()
            total_divs_received += div_income

            if div_income > 0:
                if reinvest_dividends:
                    # Buy more shares at today's price
                    new_shares = (div_income * w_current) / row
                    # Guard against zero-price division
                    new_shares = new_shares.fillna(0.0).replace([np.inf, -np.inf], 0.0)
                    holdings = holdings + new_shares
                else:
                    cash_balance += div_income

        # ── Rebalance (always deploy on first day) ───────────
        if first_day or dt in rebal_dates:
            # Dynamic selection: ask the selector for new picks
            if selector is not None:
                try:
                    sel_tickers, sel_weights = selector(dt)
                    # Update w_current to reflect new selection
                    w_current = pd.Series(0.0, index=prices.columns)
                    for t_sel, w_sel in sel_weights.items():
                        if t_sel in w_current.index:
                            w_current[t_sel] = w_sel
                    if w_current.sum() > 0:
                        w_current = w_current / w_current.sum()
                    selection_log.append((dt, sel_tickers))
                except Exception:
                    pass  # keep previous weights on failure

            if holdings.sum() == 0:
                target_value = initial_cash
            else:
                target_value = (holdings * row).sum()

            # Zero out holdings in tickers we no longer want
            for col in prices.columns:
                if w_current[col] == 0:
                    holdings[col] = 0.0

            # Allocate to tickers with non-zero weight
            active = w_current[w_current > 0]
            if not active.empty:
                for t_a in active.index:
                    if row[t_a] > 0:
                        holdings[t_a] = (target_value * w_current[t_a]) / row[t_a]
            first_day = False

        # ── Mark to market ───────────────────────────────────
        invested_value = (holdings * row).sum()
        port_val = invested_value + (0.0 if reinvest_dividends else cash_balance)
        portfolio_values.append(port_val)
        cash_values.append(cash_balance)
        cum_div_values.append(total_divs_received)

        # Record actual asset weights (excluding cash)
        if invested_value > 0:
            weight_records.append((holdings * row / invested_value).to_dict())
        else:
            weight_records.append({c: 0.0 for c in prices.columns})

    pv = pd.Series(portfolio_values, index=prices.index, name="portfolio_value")
    wt = pd.DataFrame(weight_records, index=prices.index)
    cv = pd.Series(cash_values, index=prices.index, name="dividend_cash")
    cd = pd.Series(cum_div_values, index=prices.index, name="cumulative_dividends")

    return BacktestResult(
        portfolio=pv,
        weights=wt,
        total_dividends=total_divs_received,
        dividend_cash=cv,
        cumulative_dividends=cd,
        selection_log=selection_log,
    )
