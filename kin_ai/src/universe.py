"""ETF universe definition and download.

A broad universe of liquid ETFs spanning major asset classes, sectors,
and geographies.  Each ETF is tagged with an asset_class, sector, and
short description so the alpha model and selector can filter by category.
"""

from __future__ import annotations

from typing import List

from kin_ai.src.cache import write_universe, read_universe

# ── Master list ─────────────────────────────────────────────────
# fmt: off
ETF_UNIVERSE: List[dict] = [
    # ── US Broad Equity ──────────────────────────────────────
    {"ticker": "SPY",  "name": "SPDR S&P 500",              "asset_class": "US Equity",    "sector": "Broad",        "description": "S&P 500 tracker"},
    {"ticker": "IVV",  "name": "iShares Core S&P 500",      "asset_class": "US Equity",    "sector": "Broad",        "description": "S&P 500 tracker"},
    {"ticker": "VOO",  "name": "Vanguard S&P 500",          "asset_class": "US Equity",    "sector": "Broad",        "description": "S&P 500 tracker"},
    {"ticker": "VTI",  "name": "Vanguard Total Stock Mkt",  "asset_class": "US Equity",    "sector": "Broad",        "description": "Total US market"},
    {"ticker": "QQQ",  "name": "Invesco Nasdaq 100",        "asset_class": "US Equity",    "sector": "Growth",       "description": "Nasdaq 100 tracker"},
    {"ticker": "IWM",  "name": "iShares Russell 2000",      "asset_class": "US Equity",    "sector": "Small Cap",    "description": "US small-cap"},
    {"ticker": "IWF",  "name": "iShares Russell 1000 Growth","asset_class": "US Equity",   "sector": "Growth",       "description": "US large-cap growth"},
    {"ticker": "IWD",  "name": "iShares Russell 1000 Value", "asset_class": "US Equity",   "sector": "Value",        "description": "US large-cap value"},
    {"ticker": "MDY",  "name": "SPDR S&P MidCap 400",       "asset_class": "US Equity",    "sector": "Mid Cap",      "description": "US mid-cap"},
    {"ticker": "RSP",  "name": "Invesco S&P 500 Equal Wt",  "asset_class": "US Equity",    "sector": "Broad",        "description": "Equal-weight S&P 500"},
    {"ticker": "MTUM", "name": "iShares MSCI USA Momentum",  "asset_class": "US Equity",   "sector": "Factor",       "description": "US momentum factor"},
    {"ticker": "QUAL", "name": "iShares MSCI USA Quality",   "asset_class": "US Equity",   "sector": "Factor",       "description": "US quality factor"},
    {"ticker": "USMV", "name": "iShares MSCI USA Min Vol",   "asset_class": "US Equity",   "sector": "Factor",       "description": "US min-volatility"},
    {"ticker": "VLUE", "name": "iShares MSCI USA Value",     "asset_class": "US Equity",   "sector": "Factor",       "description": "US value factor"},

    # ── US Sector ────────────────────────────────────────────
    {"ticker": "XLK",  "name": "Technology Select SPDR",     "asset_class": "US Equity",    "sector": "Technology",   "description": "US tech sector"},
    {"ticker": "XLF",  "name": "Financial Select SPDR",      "asset_class": "US Equity",    "sector": "Financials",   "description": "US financials"},
    {"ticker": "XLV",  "name": "Health Care Select SPDR",    "asset_class": "US Equity",    "sector": "Healthcare",   "description": "US healthcare"},
    {"ticker": "XLE",  "name": "Energy Select SPDR",         "asset_class": "US Equity",    "sector": "Energy",       "description": "US energy sector"},
    {"ticker": "XLI",  "name": "Industrial Select SPDR",     "asset_class": "US Equity",    "sector": "Industrials",  "description": "US industrials"},
    {"ticker": "XLY",  "name": "Consumer Disc Select SPDR",  "asset_class": "US Equity",    "sector": "Cons Disc",    "description": "US consumer discretionary"},
    {"ticker": "XLP",  "name": "Consumer Staples Select SPDR","asset_class": "US Equity",   "sector": "Cons Staples", "description": "US consumer staples"},
    {"ticker": "XLU",  "name": "Utilities Select SPDR",      "asset_class": "US Equity",    "sector": "Utilities",    "description": "US utilities"},
    {"ticker": "XLB",  "name": "Materials Select SPDR",      "asset_class": "US Equity",    "sector": "Materials",    "description": "US materials"},
    {"ticker": "XLRE", "name": "Real Estate Select SPDR",    "asset_class": "US Equity",    "sector": "Real Estate",  "description": "US real estate sector"},
    {"ticker": "XLC",  "name": "Communication Svcs SPDR",    "asset_class": "US Equity",    "sector": "Comm Svcs",    "description": "US communication services"},

    # ── International Equity ─────────────────────────────────
    {"ticker": "EFA",  "name": "iShares MSCI EAFE",          "asset_class": "Intl Equity",  "sector": "Developed",    "description": "Developed ex-US"},
    {"ticker": "VEA",  "name": "Vanguard FTSE Developed",    "asset_class": "Intl Equity",  "sector": "Developed",    "description": "Developed ex-US"},
    {"ticker": "EEM",  "name": "iShares MSCI Emerging Mkts", "asset_class": "Intl Equity",  "sector": "Emerging",     "description": "Emerging markets"},
    {"ticker": "VWO",  "name": "Vanguard FTSE Emerging Mkts","asset_class": "Intl Equity",  "sector": "Emerging",     "description": "Emerging markets"},
    {"ticker": "IEFA", "name": "iShares Core MSCI EAFE",     "asset_class": "Intl Equity",  "sector": "Developed",    "description": "Developed ex-US core"},
    {"ticker": "INDA", "name": "iShares MSCI India",          "asset_class": "Intl Equity",  "sector": "Emerging",     "description": "India equities"},
    {"ticker": "MCHI", "name": "iShares MSCI China",          "asset_class": "Intl Equity",  "sector": "Emerging",     "description": "China equities"},
    {"ticker": "EWJ",  "name": "iShares MSCI Japan",          "asset_class": "Intl Equity",  "sector": "Developed",    "description": "Japan equities"},
    {"ticker": "EWG",  "name": "iShares MSCI Germany",        "asset_class": "Intl Equity",  "sector": "Developed",    "description": "Germany equities"},

    # ── US Fixed Income ──────────────────────────────────────
    {"ticker": "AGG",  "name": "iShares Core US Aggregate",  "asset_class": "Fixed Income", "sector": "Aggregate",    "description": "US aggregate bond"},
    {"ticker": "BND",  "name": "Vanguard Total Bond Market", "asset_class": "Fixed Income", "sector": "Aggregate",    "description": "Total US bond market"},
    {"ticker": "TLT",  "name": "iShares 20+ Year Treasury",  "asset_class": "Fixed Income", "sector": "Long Treasury","description": "Long-term treasuries"},
    {"ticker": "TLH",  "name": "iShares 10-20 Year Treasury","asset_class": "Fixed Income", "sector": "Treasury",     "description": "Intermediate-long treasuries"},
    {"ticker": "IEF",  "name": "iShares 7-10 Year Treasury", "asset_class": "Fixed Income", "sector": "Treasury",     "description": "Intermediate treasuries"},
    {"ticker": "SHY",  "name": "iShares 1-3 Year Treasury",  "asset_class": "Fixed Income", "sector": "Short Treasury","description": "Short-term treasuries"},
    {"ticker": "GOVT", "name": "iShares US Treasury Bond",   "asset_class": "Fixed Income", "sector": "Treasury",     "description": "All-maturity treasuries"},
    {"ticker": "TIP",  "name": "iShares TIPS Bond",          "asset_class": "Fixed Income", "sector": "TIPS",         "description": "Inflation-protected"},
    {"ticker": "LQD",  "name": "iShares IG Corporate Bond",  "asset_class": "Fixed Income", "sector": "Corp IG",      "description": "Investment-grade corporate"},
    {"ticker": "HYG",  "name": "iShares High Yield Corp",    "asset_class": "Fixed Income", "sector": "Corp HY",      "description": "High-yield corporate"},
    {"ticker": "JNK",  "name": "SPDR High Yield Bond",       "asset_class": "Fixed Income", "sector": "Corp HY",      "description": "High-yield corporate"},
    {"ticker": "EMB",  "name": "iShares JP Morgan EM Bond",  "asset_class": "Fixed Income", "sector": "EM Debt",      "description": "Emerging-market bonds"},
    {"ticker": "MBB",  "name": "iShares MBS",                "asset_class": "Fixed Income", "sector": "MBS",          "description": "Mortgage-backed securities"},
    {"ticker": "VCSH", "name": "Vanguard Short Corp Bond",   "asset_class": "Fixed Income", "sector": "Corp Short",   "description": "Short-term corporate"},

    # ── Income / Covered Call ────────────────────────────────
    {"ticker": "JEPQ", "name": "JPM Nasdaq Equity Premium",  "asset_class": "Income",       "sector": "Covered Call", "description": "Nasdaq covered-call income"},
    {"ticker": "JEPI", "name": "JPM Equity Premium Income",  "asset_class": "Income",       "sector": "Covered Call", "description": "S&P 500 covered-call income"},
    {"ticker": "XYLD", "name": "Global X S&P 500 Cov Call",  "asset_class": "Income",       "sector": "Covered Call", "description": "S&P 500 covered call"},
    {"ticker": "QYLD", "name": "Global X Nasdaq Cov Call",   "asset_class": "Income",       "sector": "Covered Call", "description": "Nasdaq 100 covered call"},
    {"ticker": "DIVO", "name": "Amplify CWP Enhanced Div",   "asset_class": "Income",       "sector": "Dividend",     "description": "Enhanced dividend income"},

    # ── Dividend Equity ──────────────────────────────────────
    {"ticker": "VYM",  "name": "Vanguard High Dividend Yld", "asset_class": "US Equity",    "sector": "Dividend",     "description": "High dividend yield"},
    {"ticker": "DVY",  "name": "iShares Select Dividend",    "asset_class": "US Equity",    "sector": "Dividend",     "description": "Select high-dividend"},
    {"ticker": "SCHD", "name": "Schwab US Dividend Equity",  "asset_class": "US Equity",    "sector": "Dividend",     "description": "US dividend equity"},
    {"ticker": "HDV",  "name": "iShares Core High Dividend", "asset_class": "US Equity",    "sector": "Dividend",     "description": "Core high-dividend US"},

    # ── Commodities ──────────────────────────────────────────
    {"ticker": "GLD",  "name": "SPDR Gold Shares",           "asset_class": "Commodity",    "sector": "Gold",         "description": "Physical gold"},
    {"ticker": "IAU",  "name": "iShares Gold Trust",         "asset_class": "Commodity",    "sector": "Gold",         "description": "Physical gold"},
    {"ticker": "SLV",  "name": "iShares Silver Trust",       "asset_class": "Commodity",    "sector": "Silver",       "description": "Physical silver"},
    {"ticker": "DBC",  "name": "Invesco DB Commodity",       "asset_class": "Commodity",    "sector": "Broad",        "description": "Diversified commodity"},
    {"ticker": "USO",  "name": "United States Oil Fund",     "asset_class": "Commodity",    "sector": "Energy",       "description": "Crude oil futures"},
    {"ticker": "PDBC", "name": "Invesco Optimum Yield Div",  "asset_class": "Commodity",    "sector": "Broad",        "description": "Diversified commodity"},

    # ── Real Estate ──────────────────────────────────────────
    {"ticker": "VNQ",  "name": "Vanguard Real Estate",       "asset_class": "Real Estate",  "sector": "Broad REIT",   "description": "US REITs"},
    {"ticker": "IYR",  "name": "iShares US Real Estate",     "asset_class": "Real Estate",  "sector": "Broad REIT",   "description": "US real estate"},
    {"ticker": "REM",  "name": "iShares Mortgage Real Est",  "asset_class": "Real Estate",  "sector": "Mortgage REIT","description": "Mortgage REITs"},
    {"ticker": "VNQI", "name": "Vanguard Global ex-US RE",   "asset_class": "Real Estate",  "sector": "Intl REIT",    "description": "International REITs"},

    # ── Alternatives / Volatility ────────────────────────────
    {"ticker": "GDX",  "name": "VanEck Gold Miners",         "asset_class": "Alternative",  "sector": "Gold Miners",  "description": "Gold mining equities"},
    {"ticker": "GDXJ", "name": "VanEck Junior Gold Miners",  "asset_class": "Alternative",  "sector": "Gold Miners",  "description": "Junior gold miners"},
    {"ticker": "ARKK", "name": "ARK Innovation",             "asset_class": "Alternative",  "sector": "Innovation",   "description": "Disruptive innovation"},
    {"ticker": "BITQ", "name": "Bitwise Crypto Industry",    "asset_class": "Alternative",  "sector": "Crypto",       "description": "Crypto-related equities"},
]
# fmt: on


def get_universe() -> list[dict]:
    """Return the master ETF universe list."""
    return ETF_UNIVERSE


def save_universe_to_db() -> int:
    """Persist the universe to the SQLite database. Returns count."""
    write_universe(ETF_UNIVERSE)
    return len(ETF_UNIVERSE)


def load_universe_from_db():
    """Load universe from DB as a DataFrame."""
    return read_universe()


def get_tickers_by_class(asset_class: str) -> list[str]:
    """Filter universe tickers by asset class."""
    return [e["ticker"] for e in ETF_UNIVERSE if e["asset_class"] == asset_class]


def get_all_tickers() -> list[str]:
    """Return all tickers in the universe."""
    return [e["ticker"] for e in ETF_UNIVERSE]
