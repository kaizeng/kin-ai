from __future__ import annotations

import sys
from typing import Iterable, List

import pandas as pd
import yfinance as yf

from kin_ai.src.cache import (
    init_db,
    read_prices,
    write_prices,
    read_dividends,
    write_dividends,
)


def _download_prices(
    tickers: List[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Raw Yahoo Finance download (no cache)."""
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
    )

    if data.empty:
        raise ValueError("No data downloaded from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        closes = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].copy()
            df = df.rename(columns=str.title)
            close_col = "Close" if "Close" in df.columns else "Adj Close"
            closes.append(df[close_col].rename(t))
        prices = pd.concat(closes, axis=1)
    else:
        close_col = "Close" if "Close" in data.columns else "Adj Close"
        prices = data[[close_col]].rename(columns={close_col: tickers[0]})

    prices = prices.ffill().bfill()
    return prices


def get_price_data(
    tickers: Iterable[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily close prices — uses SQLite cache when possible.

    Returns a DataFrame indexed by date with columns = tickers.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")

    # Try cache first
    if use_cache:
        cached = read_prices(tickers, start, end)
        if cached is not None:
            print("      (using cached prices)", file=sys.stderr)
            return cached

    # Download from Yahoo Finance
    prices = _download_prices(tickers, start, end, auto_adjust)

    # Persist to cache
    if use_cache:
        try:
            write_prices(prices, tickers)
        except Exception:
            pass  # cache write failure is non-fatal

    return prices


def _download_dividends(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Raw Yahoo Finance dividend download (no cache)."""
    div_frames = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            divs = tk.dividends
            if divs is None or divs.empty:
                div_frames.append(pd.Series(dtype=float, name=t))
            else:
                if divs.index.tz is not None:
                    divs = divs.tz_localize(None)
                divs = divs[(divs.index >= start) & (divs.index <= end)]
                divs = divs.rename(t)
                div_frames.append(divs)
        except Exception:
            div_frames.append(pd.Series(dtype=float, name=t))

    dividends = pd.concat(div_frames, axis=1).fillna(0.0)
    return dividends


def get_dividend_data(
    tickers: Iterable[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch per-share dividends — uses SQLite cache when possible.

    Returns a DataFrame indexed by date with columns = tickers.
    Values are 0.0 on non-ex-dividend dates.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")

    if use_cache:
        cached = read_dividends(tickers, start, end)
        if cached is not None:
            print("      (using cached dividends)", file=sys.stderr)
            return cached

    dividends = _download_dividends(tickers, start, end)

    if use_cache:
        try:
            write_dividends(dividends, tickers, start, end)
        except Exception:
            pass

    return dividends
