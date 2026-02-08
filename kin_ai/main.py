"""Kin-AI: investment research pipeline entry point."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

from kin_ai.src.data import get_price_data, get_dividend_data
from kin_ai.src.strategy import calibrate_etf_plan
from kin_ai.src.backtest import backtest_lump_sum
from kin_ai.src.regime import detect_market_regime, infer_economic_cycle
from kin_ai.src.advice import generate_advice

# ── Default parameters ──────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "IWM", "AGG", "GLD"]
START = "2015-01-01"
END = "2025-12-31"
INITIAL_CASH = 10_000.0
REBALANCE_FREQ = "QE"
METHOD = "risk_parity"
REINVEST_DIVIDENDS = True      # True = reinvest, False = cash out
RISK_FREE_RATE = 0.045         # annualised risk-free rate (4.5 %)


def main() -> None:
    print("=" * 60)
    print("  Kin-AI  –  Investment Research Pipeline")
    print("=" * 60)

    # 1. Fetch prices (adjusted for splits only, not dividends)
    print(f"\n[1/6] Downloading prices for {TICKERS} ({START} → {END}) …")
    prices = get_price_data(TICKERS, start=START, end=END, auto_adjust=True)
    print(f"      Got {len(prices)} trading days, {len(prices.columns)} tickers.")

    # 2. Fetch dividends
    div_mode = "reinvest" if REINVEST_DIVIDENDS else "cash out"
    print(f"\n[2/6] Downloading dividend history (mode: {div_mode}) …")
    dividends = get_dividend_data(TICKERS, start=START, end=END)
    total_div_events = (dividends > 0).sum().sum()
    print(f"      Found {total_div_events} ex-dividend events across {len(TICKERS)} tickers.")

    # 3. Calibrate strategy
    print(f"\n[3/6] Calibrating weights (method={METHOD}) …")
    weights = calibrate_etf_plan(prices, method=METHOD)
    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"      {t:>5s}  {w:6.2%}")

    # 4. Backtest
    print(f"\n[4/6] Backtesting lump-sum ${INITIAL_CASH:,.0f}, rebalance every {REBALANCE_FREQ} …")
    result = backtest_lump_sum(
        prices, weights, INITIAL_CASH, REBALANCE_FREQ,
        dividends=dividends,
        reinvest_dividends=REINVEST_DIVIDENDS,
    )
    portfolio = result.portfolio
    weight_history = result.weights
    total_divs = result.total_dividends
    dividend_cash = result.dividend_cash

    print(f"      Final value: ${portfolio.iloc[-1]:,.2f}")
    total_return = (portfolio.iloc[-1] / INITIAL_CASH - 1) * 100
    print(f"      Total return: {total_return:.1f}%")
    print(f"      Total dividends received: ${total_divs:,.2f}")
    if not REINVEST_DIVIDENDS:
        print(f"      Dividend cash balance: ${dividend_cash.iloc[-1]:,.2f}")

    # 5. Detect regime
    print("\n[5/6] Detecting market regime (SPY) …")
    trend, vol = detect_market_regime(prices["SPY"])
    cycle = infer_economic_cycle(trend, vol)
    print(f"      Trend: {trend}  |  Volatility: {vol}  |  Cycle: {cycle}")

    # 6. Generate advice
    print("\n[6/6] Generating allocation advice …")
    advice = generate_advice(trend, cycle, weights)
    print(f"      Action: {advice['action']}")
    print("      Suggested weights:")
    for t, w in sorted(advice["suggested_weights"].items(), key=lambda x: -x[1]):
        print(f"        {t:>5s}  {w:6.2%}")

    # ── Compute key stats ─────────────────────────────────────
    returns = portfolio.pct_change().dropna()
    trading_days = len(returns)
    years = trading_days / 252

    total_return = (portfolio.iloc[-1] / INITIAL_CASH - 1) * 100
    cagr = ((portfolio.iloc[-1] / INITIAL_CASH) ** (1 / years) - 1) * 100

    # Annualised volatility & Sharpe — standard ex-post formulas
    daily_mean = returns.mean()
    daily_std = returns.std()
    ann_return = daily_mean * 252          # annualised mean return (decimal)
    ann_vol_dec = daily_std * np.sqrt(252) # annualised vol (decimal)
    ann_vol = ann_vol_dec * 100            # for display (%)
    sharpe = ((ann_return - RISK_FREE_RATE) / ann_vol_dec
              if ann_vol_dec > 0 else 0.0)

    cummax = portfolio.cummax()
    drawdown = (portfolio - cummax) / cummax
    max_dd = drawdown.min() * 100
    max_dd_date = drawdown.idxmin()
    peak_date = cummax[:max_dd_date].idxmax() if max_dd_date is not None else None
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    best_day = returns.max() * 100
    worst_day = returns.min() * 100
    win_rate = (returns > 0).sum() / len(returns) * 100
    div_yield = (total_divs / INITIAL_CASH / years) * 100  # avg annual div yield on cost

    # ── Bloomberg-style chart ──────────────────────────────────
    print("\n      Saving portfolio chart to portfolio.png …")

    # Dark theme colors
    BG = "#1a1a2e"
    PANEL = "#16213e"
    ACCENT = "#0f3460"
    GREEN = "#00d26a"
    RED = "#f8312f"
    ORANGE = "#ff9500"
    BLUE = "#00aeff"
    CYAN = "#00e5ff"
    WHITE = "#e8e8e8"
    MUTED = "#6c7a89"
    GRID = "#2a2a4a"

    plt.rcParams.update({
        "font.family": "monospace",
        "font.size": 9,
        "text.color": WHITE,
        "axes.labelcolor": WHITE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })

    fig = plt.figure(figsize=(16, 13), facecolor=BG)

    # Layout: portfolio, weights, drawdown, footer + stats sidebar
    gs = fig.add_gridspec(
        4, 2,
        width_ratios=[3, 1],
        height_ratios=[3.5, 2, 1.2, 0.3],
        hspace=0.08, wspace=0.02,
        left=0.06, right=0.97, top=0.93, bottom=0.04,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    ax_wt = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_dd = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_stats = fig.add_subplot(gs[0:3, 1])
    ax_footer = fig.add_subplot(gs[3, :])

    for ax in [ax_main, ax_wt, ax_dd, ax_stats, ax_footer]:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # ── Title bar ──────────────────────────────────────────────
    fig.text(
        0.06, 0.96, "KIN-AI",
        fontsize=22, fontweight="bold", color=ORANGE,
        fontfamily="monospace",
    )
    fig.text(
        0.175, 0.965, "PORTFOLIO ANALYTICS",
        fontsize=12, color=MUTED, fontfamily="monospace",
    )
    fig.text(
        0.97, 0.965,
        f"{datetime.now().strftime('%d %b %Y  %H:%M')}  UTC",
        fontsize=9, color=MUTED, ha="right", fontfamily="monospace",
    )
    # Orange accent line under title
    fig.patches.append(plt.Rectangle(
        (0.06, 0.945), 0.91, 0.003,
        transform=fig.transFigure, facecolor=ORANGE, zorder=10,
    ))

    # ── Main portfolio chart ───────────────────────────────────
    ax_main.fill_between(
        portfolio.index, portfolio.values,
        alpha=0.15, color=CYAN, linewidth=0,
    )
    ax_main.plot(
        portfolio.index, portfolio.values,
        color=CYAN, linewidth=1.5, zorder=5,
    )
    # highlight current value
    ax_main.scatter(
        [portfolio.index[-1]], [portfolio.iloc[-1]],
        color=CYAN, s=40, zorder=6, edgecolors=WHITE, linewidths=0.8,
    )

    ax_main.set_ylabel("PORTFOLIO VALUE  (USD)", fontsize=9, labelpad=10)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_main.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax_main.tick_params(axis="x", labelbottom=False)
    ax_main.set_xlim(portfolio.index[0], portfolio.index[-1])

    # Start / End value labels
    ax_main.axhline(INITIAL_CASH, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_main.text(
        portfolio.index[0], INITIAL_CASH,
        f"  ${INITIAL_CASH:,.0f}",
        fontsize=8, color=MUTED, va="bottom",
    )
    ax_main.text(
        portfolio.index[-1], portfolio.iloc[-1],
        f"  ${portfolio.iloc[-1]:,.0f}",
        fontsize=10, fontweight="bold", color=GREEN, va="bottom",
    )

    # ── Weights time-series (stacked area) ────────────────────
    # Asset color palette
    ASSET_COLORS = {
        "SPY": "#00aeff", "QQQ": "#a855f7", "IWM": "#00d26a",
        "AGG": "#ff9500", "GLD": "#ffd700",
    }
    # Resample to weekly for smoother visual
    wt_weekly = weight_history.resample("W").last().dropna()
    # Sort columns by mean weight descending for a cleaner stack
    col_order = wt_weekly.mean().sort_values(ascending=False).index.tolist()
    wt_plot = wt_weekly[col_order] * 100  # percent

    colors = [ASSET_COLORS.get(c, MUTED) for c in col_order]
    ax_wt.stackplot(
        wt_plot.index, *[wt_plot[c].values for c in col_order],
        labels=col_order, colors=colors, alpha=0.85, linewidth=0,
    )
    # Thin white separator lines between areas for clarity
    cumulative = np.zeros(len(wt_plot))
    for c, clr in zip(col_order, colors):
        cumulative = cumulative + wt_plot[c].values
        ax_wt.plot(wt_plot.index, cumulative, color=PANEL, linewidth=0.3)

    ax_wt.set_ylabel("ALLOCATION  (%)", fontsize=9, labelpad=10)
    ax_wt.set_ylim(0, 100)
    ax_wt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_wt.grid(True, color=GRID, linewidth=0.5, alpha=0.4)
    ax_wt.tick_params(axis="x", labelbottom=False)
    ax_wt.legend(
        loc="upper center", ncol=len(col_order),
        fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 1.12),
        labelcolor=WHITE,
    )

    # ── Drawdown chart ─────────────────────────────────────────
    ax_dd.fill_between(
        drawdown.index, drawdown.values * 100,
        alpha=0.4, color=RED, linewidth=0,
    )
    ax_dd.plot(drawdown.index, drawdown.values * 100, color=RED, linewidth=0.8)
    ax_dd.set_ylabel("DRAWDOWN  (%)", fontsize=9, labelpad=10)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())

    # Mark worst drawdown
    ax_dd.scatter(
        [max_dd_date], [max_dd],
        color=RED, s=30, zorder=6, edgecolors=WHITE, linewidths=0.8,
    )
    ax_dd.annotate(
        f" {max_dd:.1f}%",
        xy=(max_dd_date, max_dd),
        fontsize=8, fontweight="bold", color=RED,
        va="top",
    )

    # ── Stats panel ────────────────────────────────────────────
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    # Panel title
    ax_stats.text(
        0.5, 0.97, "KEY STATISTICS",
        fontsize=11, fontweight="bold", color=ORANGE,
        ha="center", va="top", fontfamily="monospace",
    )
    ax_stats.axhline(y=0.945, xmin=0.05, xmax=0.95, color=ORANGE, linewidth=1)

    # Helper to draw stat rows
    y_pos = 0.92
    def _stat(label, value, color=WHITE, bar_pct=None):
        nonlocal y_pos
        ax_stats.text(0.08, y_pos, label, fontsize=9, color=MUTED, va="center")
        ax_stats.text(0.92, y_pos, value, fontsize=10, fontweight="bold",
                      color=color, va="center", ha="right", fontfamily="monospace")
        if bar_pct is not None:
            bar_w = min(max(bar_pct, 0), 1) * 0.84
            ax_stats.barh(y_pos - 0.015, bar_w, height=0.006, left=0.08,
                          color=color, alpha=0.25)
        y_pos -= 0.042
        # subtle separator
        ax_stats.axhline(y=y_pos + 0.015, xmin=0.08, xmax=0.92,
                         color=GRID, linewidth=0.5)

    ret_color = GREEN if total_return > 0 else RED
    _stat("Total Return", f"{total_return:+.1f}%", ret_color, total_return / 200)
    _stat("CAGR", f"{cagr:+.1f}%", GREEN if cagr > 0 else RED, cagr / 30)
    _stat("Annualized Vol", f"{ann_vol:.1f}%", BLUE, ann_vol / 40)
    _stat("Sharpe Ratio", f"{sharpe:.2f}", GREEN if sharpe > 1 else ORANGE if sharpe > 0.5 else RED)
    _stat("Max Drawdown", f"{max_dd:.1f}%", RED, abs(max_dd) / 50)
    _stat("Calmar Ratio", f"{calmar:.2f}", GREEN if calmar > 1 else ORANGE)
    _stat("Best Day", f"{best_day:+.2f}%", GREEN)
    _stat("Worst Day", f"{worst_day:+.2f}%", RED)
    _stat("Win Rate", f"{win_rate:.1f}%", GREEN if win_rate > 50 else RED, win_rate / 100)
    _stat("Dividends", f"${total_divs:,.0f}", CYAN)
    _stat("Div Yield (cost)", f"{div_yield:.2f}%", CYAN if div_yield > 0 else MUTED)
    _stat("Div Mode", "REINVEST" if REINVEST_DIVIDENDS else "CASH OUT",
          GREEN if REINVEST_DIVIDENDS else ORANGE)
    if not REINVEST_DIVIDENDS:
        _stat("Div Cash Bal", f"${dividend_cash.iloc[-1]:,.0f}", ORANGE)
    _stat("Trading Days", f"{trading_days:,}", WHITE)

    # ── Allocation section ─────────────────────────────────────
    y_pos -= 0.01
    ax_stats.text(0.5, y_pos, "REGIME & ALLOCATION",
                  fontsize=10, fontweight="bold", color=ORANGE,
                  ha="center", va="top", fontfamily="monospace")
    y_pos -= 0.02
    ax_stats.axhline(y=y_pos, xmin=0.05, xmax=0.95, color=ORANGE, linewidth=1)
    y_pos -= 0.035

    regime_color = GREEN if trend == "bull" else RED
    ax_stats.text(0.08, y_pos, "Regime", fontsize=9, color=MUTED, va="center")
    ax_stats.text(0.92, y_pos, f"{trend.upper()} / {vol.upper()} VOL",
                  fontsize=10, fontweight="bold", color=regime_color,
                  va="center", ha="right", fontfamily="monospace")
    y_pos -= 0.045

    ax_stats.text(0.08, y_pos, "Cycle", fontsize=9, color=MUTED, va="center")
    ax_stats.text(0.92, y_pos, cycle.upper(),
                  fontsize=10, fontweight="bold", color=CYAN,
                  va="center", ha="right", fontfamily="monospace")
    y_pos -= 0.045

    ax_stats.text(0.08, y_pos, "Action", fontsize=9, color=MUTED, va="center")
    action_color = GREEN if "on" in advice["action"].lower() else RED
    ax_stats.text(0.92, y_pos, advice["action"],
                  fontsize=10, fontweight="bold", color=action_color,
                  va="center", ha="right", fontfamily="monospace")

    # ── Footer ─────────────────────────────────────────────────
    ax_footer.axis("off")
    tickers_str = " | ".join(
        f"{t} {w:.0%}" for t, w in sorted(
            advice["suggested_weights"].items(), key=lambda x: -x[1]
        )
    )
    ax_footer.text(
        0.0, 0.5,
        f"TICKERS:  {tickers_str}     │     PERIOD: {START} → {END}"
        f"     │     INITIAL: ${INITIAL_CASH:,.0f}     │     REBAL: {REBALANCE_FREQ}"
        f"     │     DIVIDENDS: {'REINVEST' if REINVEST_DIVIDENDS else 'CASH OUT'}"
        f"     │     METHOD: {METHOD.upper().replace('_', ' ')}",
        fontsize=8, color=MUTED, va="center", fontfamily="monospace",
    )
    ax_footer.text(
        1.0, 0.5, "KIN-AI  ©  2026",
        fontsize=8, color=ORANGE, va="center", ha="right",
        fontweight="bold", fontfamily="monospace",
    )

    fig.savefig("portfolio.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print("      Done ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
