# Kin-AI

Personal investment research pipeline for long-term ETF portfolio construction with **alpha-driven automatic ETF selection**, backtesting, and tactical allocation using market regime detection.

---

## Sample Output

![Portfolio Analytics Chart](portfolio.png)

---

## Features

- **Max-Sharpe / min-drawdown optimisation** — Monte-Carlo weight optimizer that maximises Sharpe ratio while penalising drawdown, with a 40 % single-asset concentration cap
- **Alpha-driven ETF selection** — multi-factor model automatically picks the best 3–4 ETFs at each rebalance from a universe of ~71 ETFs
- **SQLite cache** — prices, dividends, and universe data are cached locally for fast reruns
- **Broad ETF universe** — ~71 ETFs across 8 asset classes (US equity, intl equity, fixed income, commodities, real estate, alternatives, income, dividend)
- **Yahoo Finance data access** — automated daily close-price download via `yfinance`
- **Dividend tracking** — fetches per-share dividends with option to reinvest or cash out
- **Lump-sum backtesting** — periodic rebalancing with dynamic portfolio rotation and full weight-drift tracking
- **Market regime detection** — trend + volatility classification via moving averages
- **Tactical allocation advice** — risk-on / risk-off tilts based on regime & cycle
- **Bloomberg-style analytics chart** — portfolio value, allocation weights with per-rebalance annotations, drawdown, dividends, key stats

---

## Pipeline Overview

The pipeline runs eight steps in sequence:

```
[1] Universe / Prices →  Load ETF universe, download daily prices (cache-first)
[2] Dividends         →  Fetch per-share dividend history for all tickers
[3] Alpha Scoring     →  Run multi-factor alpha model, pick top N ETFs
[4] Backtest          →  Simulate lump-sum investment with periodic rebalancing
[5] Regime            →  Detect current market regime (trend + volatility)
[6] Advice            →  Generate tactical allocation tilt
[7] Chart             →  Render Bloomberg-style portfolio analytics chart
[8] Summary           →  Print final results
```

### Operating Modes

| Mode | Description |
|------|-------------|
| **Auto-Select** (`AUTO_SELECT = True`) | Alpha model scores all ETFs in the universe, picks the top `TOP_N` at each rebalance, and allocates using max-Sharpe optimised weights. Different ETFs may be held in different periods. |
| **Manual** (`AUTO_SELECT = False`) | Uses a fixed list of tickers (`TICKERS`) with weights from `calibrate_etf_plan()`. |

### Default Parameters

| Parameter             | Value                              |
|-----------------------|------------------------------------|
| Mode                  | Manual                             |
| Tickers (manual mode) | HYG, TLH, JEPQ, JEPI, QQQ         |
| Top N (auto mode)     | 4 ETFs per rebalance               |
| Max weight            | 40 % (concentration cap)           |
| Min history           | 200 trading days                   |
| Period                | 2020-01-01 → 2026-02-07           |
| Initial capital       | $10,000                            |
| Rebalance freq        | Semi-annual (6ME)                  |
| Weight method         | Max Sharpe / min drawdown          |
| Reinvest dividends    | True (reinvest by default)         |
| Risk-free rate        | 4.5 % annualised                   |

---

## SQLite Cache

All downloaded data is cached in a local SQLite database at `data/kin_ai.db`. The cache stores:

| Table | Contents |
|-------|----------|
| `prices` | Daily close prices per ticker |
| `dividends` | Dividend amounts per ticker |
| `cache_meta` | Timestamps and date ranges for cache validity |
| `universe` | ETF metadata (ticker, name, asset class, sector) |

Cache entries expire after **12 hours** (`CACHE_EXPIRY_HOURS`). On subsequent runs, prices are served from the local database, making reruns near-instant.

---

## ETF Universe

The investment universe (`universe.py`) contains ~71 ETFs organised by asset class:

| Asset Class | Examples | Count |
|-------------|----------|-------|
| US Equity — Broad | SPY, QQQ, IWM, VTI, DIA | 14 |
| US Equity — Sector | XLF, XLE, XLK, XLV, XBI | 11 |
| US Equity — Factor | MTUM, QUAL, VLUE, USMV | 4 |
| International Equity | EFA, EEM, VWO, FXI, EWJ | 9 |
| Fixed Income | AGG, TLT, HYG, LQD, TIP | 12 |
| Income / Covered Call | JEPQ, JEPI, XYLD, QYLD, DIVO | 5 |
| Dividend | VYM, SCHD, DVY, HDV | 4 |
| Commodity | GLD, SLV, GDX, USO, DBC, PDBC | 6 |
| Real Estate | VNQ, IYR, XLRE, RWR | 4 |
| Alternative | BITO, ARKK, TAN, ICLN | 4 |

The universe is saved to the SQLite cache on first run and reloaded from the DB on subsequent runs.

---

## Alpha Model

The alpha model (`alpha.py`) uses five cross-sectional factors to score and rank every ETF in the universe:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Momentum** | 40% | 6-month total return (126 trading days) |
| **Trend** | 20% | Percentage distance from the 200-day SMA |
| **Low Volatility** | 15% | Inverse of 60-day rolling return volatility |
| **Mean Reversion** | 10% | Inverted RSI-14 (favours oversold ETFs) |
| **Dividend Yield** | 15% | Trailing 12-month dividend yield |

### Scoring Process

1. Each factor is computed for all eligible ETFs (those with ≥ `MIN_HISTORY` trading days).
2. Factors are **z-scored cross-sectionally** (zero mean, unit variance).
3. The composite alpha is a weighted sum: $\alpha_i = \sum_k w_k \cdot z_{i,k}$
4. ETFs are ranked by alpha — the top `TOP_N` are selected.

### Look-Ahead-Free Selection

The selector (`selector.py`) builds a callable `selector(date)` that, on each rebalance date, uses **only data up to that date** to compute alpha scores and select ETFs. This prevents any look-ahead bias in the backtest.

---

## Strategy: Weight Optimisation

Three methods are supported (set via `METHOD`):

### Max Sharpe / Min Drawdown (default)

A **Monte-Carlo optimisation** that samples 12,000 random weight vectors and scores each by:

$$
\text{score} = \text{Sharpe} \;-\; \lambda \times |\text{MaxDrawdown}|
$$

where $\lambda = 2.0$ (drawdown penalty). The weight vector with the highest score is selected. A **40 % concentration cap** is enforced — any single ETF is capped at `MAX_WEIGHT` with excess redistributed proportionally.

### Risk Parity

Each asset's weight is proportional to the inverse of its historical return volatility:

$$
w_i = \frac{1 / \sigma_i}{\sum_{j} 1 / \sigma_j}
$$

### Equal Weight

Every asset receives $w_i = 1/N$ regardless of risk.

---

## Rebalancing Logic

The backtester (`backtest.py`) simulates a **lump-sum** investment with **periodic rebalancing**:

1. On day one the full `initial_cash` amount is deployed across all assets according to the target weights.
2. Between rebalance dates, holdings are left untouched — asset weights **drift** as prices move.
3. On each rebalance date (semi-annual by default), the portfolio is marked-to-market and all positions are **reset to the target weights** by selling/buying at the closing price.
4. **In auto-select mode**, the selector is called on each rebalance date to pick new ETFs. Holdings in deselected ETFs are sold and proceeds are redeployed into the new picks.
5. The system records both the portfolio value and the **actual weight of each asset** on every trading day, producing a full weight-drift time series visible in the chart.

Rebalance dates are determined by the pandas offset alias (e.g. `6ME` = semi-annual, `QE` = quarter-end, `ME` = month-end, `YE` = year-end).

---

## Dividend Handling

Dividends are fetched separately from prices using `yfinance.Ticker.dividends` for each asset. The system supports two modes controlled by the `REINVEST_DIVIDENDS` parameter:

### Reinvest Mode (`REINVEST_DIVIDENDS = True`)

On each ex-dividend date, the cash dividend income (shares held × per-share dividend) is immediately used to **purchase additional shares** at that day's closing price, distributed across assets according to the target weights. This compounds returns over time.

### Cash-Out Mode (`REINVEST_DIVIDENDS = False`)

Dividend income is **accumulated as a separate cash balance** and is _not_ reinvested into the market. The total portfolio value shown on the chart includes this cash balance, but the cash does not participate in market risk. This is useful for modelling income-oriented strategies.

### Dividend Statistics

The chart sidebar displays:

| Stat              | Description                                                |
|-------------------|------------------------------------------------------------|
| **Dividends**     | Total lifetime dividends received ($)                      |
| **Div Yield (cost)** | Average annual dividend yield on the initial investment |
| **Div Mode**      | Current mode — REINVEST or CASH OUT                        |
| **Div Cash Bal**  | Cash balance from dividends (shown only in cash-out mode)  |

---

## Performance Statistics

### Annualised Volatility

Computed from daily portfolio returns $r_t$:

$$
\sigma_\text{ann} = \text{std}(r_t) \times \sqrt{252}
$$

### Sharpe Ratio

Uses the standard **ex-post Sharpe ratio** formulation:

$$
\text{Sharpe} = \frac{\bar{r} \times 252 \;-\; R_f}{\text{std}(r_t) \times \sqrt{252}}
$$

where $\bar{r}$ is the mean daily return, $R_f$ is the annualised risk-free rate (default 4.5 %), and $\text{std}(r_t)$ is the daily return standard deviation. Both numerator and denominator are annualised from daily figures, keeping units consistent.

### Other Metrics

| Metric      | Formula                                                      |
|-------------|--------------------------------------------------------------|
| **CAGR**    | $(V_T / V_0)^{1/Y} - 1$ where $Y$ = years                  |
| **Max DD**  | Largest peak-to-trough decline                               |
| **Calmar**  | CAGR / \|Max Drawdown\|                                      |
| **Win Rate**| Fraction of positive-return trading days                     |

---

## Market Regime Detection

The regime detector (`regime.py`) classifies the current market state using two independent signals measured on a reference index (SPY by default):

### 1. Trend Regime

A **dual moving-average crossover**:

| Condition              | Regime   |
|------------------------|----------|
| SMA(50) > SMA(200)     | **Bull** |
| SMA(50) ≤ SMA(200)     | **Bear** |

This is the classic "golden cross / death cross" framework.

### 2. Volatility Regime

The 20-day rolling standard deviation of daily returns is compared against its own historical 75th percentile:

| Condition                             | Regime        |
|---------------------------------------|---------------|
| Current vol > 75th percentile         | **High vol**  |
| Current vol ≤ 75th percentile         | **Low vol**   |

### 3. Economic Cycle Inference

The trend and volatility regimes are combined into a simplified economic cycle label:

| Trend | Volatility | Cycle            |
|-------|------------|------------------|
| Bull  | Low        | **Expansion**    |
| Bull  | High       | **Slowdown**     |
| Bear  | High       | **Contraction**  |
| Bear  | Low        | **Recovery**     |

---

## Tactical Allocation Advice

Based on the detected regime, the advisor (`advice.py`) applies a **regime-based tilt** to the base weights:

- **Bull → Risk-on tilt**: overweight equities, underweight bonds and gold
- **Bear → Risk-off tilt**: overweight bonds and gold, underweight equities

The tilt is capped at `max_tilt = 10%` total, distributed evenly across the assets in each group. Weights are renormalised to sum to 100% after the tilt.

---

## Output

The pipeline produces:

- **Console output** — all statistics, alpha scores, weights, regime, dividends, and advice printed step by step
- **`portfolio.png`** — Bloomberg-style dark-theme chart with five panels:
  - Portfolio value over time (with start/end labels)
  - Cumulative dividends chart
  - Stacked-area allocation weights with **per-rebalance ETF annotations** centred in each band
  - Drawdown chart (worst drawdown annotated)
  - Key statistics sidebar (CAGR, Sharpe, max DD, Calmar, win rate, dividends, regime, cycle, action)

---

## Project Structure

```
kin-ai/
├── main.py                   # Root entry point (delegates to kin_ai.main)
├── requirements.txt          # numpy, pandas, yfinance, matplotlib
├── pyproject.toml
├── .gitignore
├── portfolio.png             # Generated chart (sample above)
├── kin_ai/
│   ├── __init__.py
│   ├── main.py               # Pipeline orchestrator + Bloomberg chart
│   └── src/
│       ├── __init__.py
│       ├── cache.py           # SQLite cache (prices, dividends, universe)
│       ├── universe.py        # ETF universe definition (~71 ETFs)
│       ├── alpha.py           # Multi-factor alpha scoring model
│       ├── selector.py        # Look-ahead-free ETF picker
│       ├── data.py            # Yahoo Finance data download (cache-first)
│       ├── strategy.py        # Weight optimisation (max-Sharpe, risk-parity, equal)
│       ├── backtest.py        # Lump-sum backtester with dynamic selection
│       ├── regime.py          # Market regime + economic cycle detection
│       ├── advice.py          # Tactical allocation advice
│       └── run_pipeline.py    # Alternate entry point
```

---

## Quick Start (WSL)

```bash
# Navigate to project
cd /mnt/c/git/kin-ai

# Create conda environment with Python 3.11
conda create -y -n kin-ai python=3.11
conda activate kin-ai

# Install dependencies (use python -m pip to ensure correct env)
python -m pip install -r requirements.txt

# Run the pipeline
python -m kin_ai.main
```

The chart is saved to `portfolio.png` in the project root. On first run, prices for ~71 ETFs are downloaded and cached to `data/kin_ai.db`. Subsequent runs load from cache in seconds.

### Auto-Select Mode

To let the alpha model pick ETFs automatically, edit `kin_ai/main.py`:

```python
AUTO_SELECT = True
TOP_N = 4
MAX_WEIGHT = 0.40
```

### Manual Mode (default)

Use a fixed set of tickers:

```python
AUTO_SELECT = False
TICKERS = ["HYG", "TLH", "JEPQ", "JEPI", "QQQ"]
```

