from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
import yfinance as yf


def get_price_data(
    tickers: Iterable[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch daily close prices from Yahoo Finance.

    Returns a DataFrame indexed by date with columns = tickers.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")

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


def get_dividend_data(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch daily per-share dividend payments from Yahoo Finance.

    Returns a DataFrame indexed by date with columns = tickers.
    Values are 0.0 on non-ex-dividend dates.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")

    div_frames = []
    for t in tickers:
        tk = yf.Ticker(t)
        divs = tk.dividends
        if divs.empty:
            div_frames.append(pd.Series(dtype=float, name=t))
        else:
            # Localised tz → naive for alignment
            if divs.index.tz is not None:
                divs = divs.tz_localize(None)
            divs = divs[(divs.index >= start) & (divs.index <= end)]
            divs = divs.rename(t)
            div_frames.append(divs)

    dividends = pd.concat(div_frames, axis=1).fillna(0.0)
    return dividends
