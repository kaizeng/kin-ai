# Kin-AI

Personal investment research pipeline for long-term ETF portfolio construction, backtesting, and tactical allocation using market regime detection.

---

## Features

- **Yahoo Finance data access** — automated daily close-price download via `yfinance`
- **Dividend tracking** — fetches per-share dividends with option to reinvest or cash out
- **Risk-parity weight calibration** — inverse-volatility target allocation
- **Lump-sum backtesting** — periodic rebalancing with full weight-drift tracking
- **Market regime detection** — trend + volatility classification via moving averages
- **Tactical allocation advice** — risk-on / risk-off tilts based on regime & cycle
- **Bloomberg-style analytics chart** — portfolio value, allocation weights, drawdown, key stats, and dividend info

---

## Pipeline Overview

The pipeline runs six steps in sequence:

```
[1] Data        →  Download daily prices from Yahoo Finance
[2] Dividends   →  Fetch per-share dividend history for all tickers
[3] Strategy    →  Calibrate target ETF weights (risk-parity or equal-weight)
[4] Backtest    →  Simulate lump-sum investment with quarterly rebalancing + dividends
[5] Regime      →  Detect current market regime (trend + volatility)
[6] Advice      →  Generate tactical allocation tilt
```

### Default Parameters

| Parameter             | Value                              |
|-----------------------|------------------------------------|
| Tickers               | SPY, QQQ, IWM, AGG, GLD           |
| Period                | 2015-01-01 → 2025-12-31           |
| Initial capital       | $10,000                            |
| Rebalance freq        | Quarterly (QE)                     |
| Weight method         | Risk parity (inverse-volatility)   |
| Reinvest dividends    | True (reinvest by default)         |
| Risk-free rate        | 4.5 % annualised                   |

---

## Strategy: Weight Calibration

Two methods are supported (set via `METHOD`):

### Risk Parity (default)

Each asset's weight is proportional to the inverse of its historical return volatility — lower-volatility assets receive higher allocations so that every asset contributes a roughly equal share of total portfolio risk.

$$
w_i = \frac{1 / \sigma_i}{\sum_{j} 1 / \sigma_j}
$$

where $\sigma_i$ is the daily return standard deviation of asset $i$.

### Equal Weight

Every asset receives $w_i = 1/N$ regardless of risk.

---

## Rebalancing Logic

The backtester (`backtest.py`) simulates a **lump-sum** investment with **periodic rebalancing**:

1. On day one the full `initial_cash` amount is deployed across all assets according to the target weights.
2. Between rebalance dates, holdings are left untouched — asset weights **drift** as prices move.
3. On each rebalance date (end of every quarter by default), the portfolio is marked-to-market and all positions are **reset to the target weights** by selling/buying at the closing price.
4. The system records both the portfolio value and the **actual weight of each asset** on every trading day, producing a full weight-drift time series visible in the chart.

Rebalance dates are determined by the pandas offset alias (e.g. `QE` = quarter-end, `ME` = month-end, `YE` = year-end).

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

- **Bull → Risk-on tilt**: overweight equities (SPY, QQQ, IWM), underweight bonds (AGG) and gold (GLD)
- **Bear → Risk-off tilt**: overweight bonds and gold, underweight equities

The tilt is capped at `max_tilt = 10%` total, distributed evenly across the assets in each group. Weights are renormalised to sum to 100% after the tilt.

---

## Output

The pipeline produces:

- **Console output** — all statistics, weights, regime, dividends, and advice printed step by step
- **`portfolio.png`** — Bloomberg-style dark-theme chart with four panels:
  - Portfolio value over time (with start/end labels)
  - Stacked-area allocation weights (showing drift between rebalance dates)
  - Drawdown chart (worst drawdown annotated)
  - Key statistics sidebar (CAGR, Sharpe, max DD, Calmar, win rate, dividends, regime, cycle, action)

---

## Project Structure

```
kin-ai/
├── main.py                  # Root entry point (delegates to kin_ai.main)
├── requirements.txt         # numpy, pandas, yfinance, matplotlib
├── pyproject.toml
├── kin_ai/
│   ├── __init__.py
│   ├── main.py              # Pipeline orchestrator + chart rendering
│   └── src/
│       ├── __init__.py
│       ├── data.py           # Yahoo Finance data download
│       ├── strategy.py       # Weight calibration (risk-parity / equal-weight)
│       ├── backtest.py       # Lump-sum backtester with rebalancing
│       ├── regime.py         # Market regime + economic cycle detection
│       ├── advice.py         # Tactical allocation advice
│       └── run_pipeline.py   # Alternate entry point
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

The chart is saved to `portfolio.png` in the project root.

