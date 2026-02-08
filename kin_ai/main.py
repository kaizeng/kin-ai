"""Kin-AI: investment research pipeline entry point."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

from kin_ai.src.data import get_price_data, get_dividend_data
from kin_ai.src.cache import init_db
from kin_ai.src.strategy import calibrate_etf_plan
from kin_ai.src.backtest import backtest_lump_sum
from kin_ai.src.regime import detect_market_regime, infer_economic_cycle
from kin_ai.src.advice import generate_advice
from kin_ai.src.universe import (
    get_all_tickers, save_universe_to_db, load_universe_from_db,
)
from kin_ai.src.alpha import compute_alpha
from kin_ai.src.selector import build_selector_fn

# ── Default parameters ──────────────────────────────────────────
TICKERS = ["HYG", "TLH", "JEPQ","JEPI","QQQ"]        # used only if AUTO_SELECT = False
START = "2020-01-01"
END = "2026-02-07"
INITIAL_CASH = 10_000.0
REBALANCE_FREQ = "6ME"
METHOD = "max_sharpe"
REINVEST_DIVIDENDS = True
RISK_FREE_RATE = 0.045

# ── Auto-selection parameters ───────────────────────────────────
AUTO_SELECT = False              # True = alpha-driven ETF selection
TOP_N = 4                       # number of ETFs to hold
MAX_WEIGHT = 0.40               # max single-ETF weight (concentration cap)
MIN_HISTORY = 200               # minimum trading days for eligibility


def main() -> None:
    print("=" * 60)
    print("  Kin-AI  –  Investment Research Pipeline")
    print("=" * 60)

    init_db()
    mode = "AUTO-SELECT" if AUTO_SELECT else "MANUAL"
    print(f"\n  Mode: {mode}")

    # ── Step 1: Universe / Prices ─────────────────────────────
    if AUTO_SELECT:
        print("\n[1/8] Loading ETF universe & downloading prices …")
        n_saved = save_universe_to_db()
        universe_df = load_universe_from_db()
        all_tickers = universe_df["ticker"].tolist()
        print(f"      Universe: {n_saved} ETFs across "
              f"{universe_df['asset_class'].nunique()} asset classes.")

        # Download full universe prices (cache speeds up reruns)
        prices_all = get_price_data(all_tickers, start=START, end=END, auto_adjust=True)
        # Drop tickers that returned no data
        prices_all = prices_all.dropna(axis=1, how="all")
        active_tickers = prices_all.columns.tolist()
        print(f"      Got {len(prices_all)} trading days, "
              f"{len(active_tickers)} tickers with data.")
    else:
        print(f"\n[1/8] Downloading prices for {TICKERS} ({START} → {END}) …")
        prices_all = get_price_data(TICKERS, start=START, end=END, auto_adjust=True)
        active_tickers = TICKERS
        print(f"      Got {len(prices_all)} trading days, "
              f"{len(prices_all.columns)} tickers.")

    # ── Step 2: Dividends ─────────────────────────────────────
    div_mode = "reinvest" if REINVEST_DIVIDENDS else "cash out"
    print(f"\n[2/8] Downloading dividend history (mode: {div_mode}) …")
    dividends_all = get_dividend_data(active_tickers, start=START, end=END)
    total_div_events = (dividends_all > 0).sum().sum()
    print(f"      Found {total_div_events} ex-dividend events.")

    # ── Step 3: Alpha scoring (auto-select only) ──────────────
    selector_fn = None
    if AUTO_SELECT:
        print(f"\n[3/8] Running alpha model (top {TOP_N} ETFs per rebalance) …")
        # Show current scores for info
        current_scores = compute_alpha(prices_all, dividends_all)
        top5_now = current_scores.head(TOP_N)
        for t in top5_now.index:
            row = top5_now.loc[t]
            print(f"      {t:>5s}  α={row['alpha']:+.2f}  "
                  f"(mom={row['momentum']:+.2f}  trend={row['trend']:+.2f}  "
                  f"vol={row['low_vol']:+.2f}  mr={row['mean_reversion']:+.2f}  "
                  f"div={row['dividend_yield']:+.2f})")
        selector_fn = build_selector_fn(
            prices_all, dividends_all,
            top_n=TOP_N, min_history=MIN_HISTORY,
            max_weight=MAX_WEIGHT, weight_method=METHOD,
        )
        # Use the current top picks as initial weights
        initial_tickers = top5_now.index.tolist()
        weights = {t: 1.0 / len(initial_tickers) for t in initial_tickers}
        print(f"      Initial picks: {initial_tickers}")
    else:
        print(f"\n[3/8] Calibrating weights (method={METHOD}) …")
        weights = calibrate_etf_plan(prices_all[TICKERS], method=METHOD)
        for t, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"      {t:>5s}  {w:6.2%}")

    # ── Step 4: Backtest ──────────────────────────────────────
    print(f"\n[4/8] Backtesting lump-sum ${INITIAL_CASH:,.0f}, "
          f"rebalance every {REBALANCE_FREQ} …")
    result = backtest_lump_sum(
        prices_all, weights, INITIAL_CASH, REBALANCE_FREQ,
        dividends=dividends_all,
        reinvest_dividends=REINVEST_DIVIDENDS,
        selector=selector_fn,
    )
    portfolio = result.portfolio
    weight_history = result.weights
    total_divs = result.total_dividends
    dividend_cash = result.dividend_cash
    cum_divs = result.cumulative_dividends
    selection_log = result.selection_log

    print(f"      Final value: ${portfolio.iloc[-1]:,.2f}")
    total_return = (portfolio.iloc[-1] / INITIAL_CASH - 1) * 100
    print(f"      Total return: {total_return:.1f}%")
    print(f"      Total dividends received: ${total_divs:,.2f}")
    if not REINVEST_DIVIDENDS:
        print(f"      Dividend cash balance: ${dividend_cash.iloc[-1]:,.2f}")

    if selection_log:
        print(f"      Rebalances with selection: {len(selection_log)}")
        # Show last selection
        last_dt, last_picks = selection_log[-1]
        print(f"      Last selection ({last_dt.strftime('%Y-%m-%d')}): {last_picks}")

    # ── Step 5: Detect regime ─────────────────────────────────
    print("\n[5/8] Detecting market regime (SPY) …")
    if "SPY" in prices_all.columns:
        spy_prices = prices_all["SPY"]
    else:
        spy_data = get_price_data(["SPY"], start=START, end=END, auto_adjust=True)
        spy_prices = spy_data["SPY"]
    trend, vol = detect_market_regime(spy_prices)
    cycle = infer_economic_cycle(trend, vol)
    print(f"      Trend: {trend}  |  Volatility: {vol}  |  Cycle: {cycle}")

    # ── Step 6: Generate advice ───────────────────────────────
    print("\n[6/8] Generating allocation advice …")
    # Use most recent weights from the backtest
    last_weights = weight_history.iloc[-1].to_dict()
    last_weights = {k: v for k, v in last_weights.items() if v > 0.001}
    advice = generate_advice(trend, cycle, last_weights)
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
    print("\n[7/8] Saving portfolio chart to portfolio.png …")

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

    # Layout: portfolio, dividends, weights, drawdown, footer + stats sidebar
    gs = fig.add_gridspec(
        5, 2,
        width_ratios=[3, 1],
        height_ratios=[3.0, 1.2, 1.8, 1.0, 0.3],
        hspace=0.08, wspace=0.02,
        left=0.06, right=0.97, top=0.93, bottom=0.04,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    ax_divs = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_wt = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_dd = fig.add_subplot(gs[3, 0], sharex=ax_main)
    ax_stats = fig.add_subplot(gs[0:4, 1])
    ax_footer = fig.add_subplot(gs[4, :])

    for ax in [ax_main, ax_divs, ax_wt, ax_dd, ax_stats, ax_footer]:
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

    # ── Cumulative dividends subplot ───────────────────────────
    GOLD = "#ffd700"
    ax_divs.fill_between(
        cum_divs.index, cum_divs.values,
        alpha=0.20, color=GOLD, linewidth=0,
    )
    ax_divs.plot(
        cum_divs.index, cum_divs.values,
        color=GOLD, linewidth=1.4, zorder=5,
    )
    # highlight final value
    ax_divs.scatter(
        [cum_divs.index[-1]], [cum_divs.iloc[-1]],
        color=GOLD, s=30, zorder=6, edgecolors=WHITE, linewidths=0.8,
    )
    ax_divs.text(
        cum_divs.index[-1], cum_divs.iloc[-1],
        f"  ${cum_divs.iloc[-1]:,.0f}",
        fontsize=9, fontweight="bold", color=GOLD, va="bottom",
    )
    ax_divs.set_ylabel("CUM. DIVIDENDS  (USD)", fontsize=9, labelpad=10)
    ax_divs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_divs.grid(True, color=GRID, linewidth=0.5, alpha=0.4)
    ax_divs.tick_params(axis="x", labelbottom=False)
    ax_divs.set_ylim(bottom=0)

    # ── Weights time-series (stacked area) ────────────────────
    # Only show tickers that were actually held at some point
    held_cols = [c for c in weight_history.columns if weight_history[c].max() > 0.001]
    wh_display = weight_history[held_cols] if held_cols else weight_history

    # Dynamic color palette — assign a distinct color to every ticker
    _COLOR_POOL = [
        "#00aeff", "#a855f7", "#00d26a", "#ff9500", "#ffd700",
        "#f8312f", "#00e5ff", "#ff6ec7", "#1abc9c", "#e74c3c",
        "#3498db", "#e67e22", "#9b59b6", "#2ecc71", "#f39c12",
        "#e84393", "#6c5ce7", "#00cec9", "#fd79a8", "#55efc4",
    ]
    ASSET_COLORS = {
        t: _COLOR_POOL[i % len(_COLOR_POOL)]
        for i, t in enumerate(sorted(wh_display.columns))
    }
    # Resample to weekly for smoother visual
    wt_weekly = wh_display.resample("W").last().dropna()
    # Sort columns by mean weight descending for a cleaner stack
    col_order = wt_weekly.mean().sort_values(ascending=False).index.tolist()
    wt_plot = wt_weekly[col_order] * 100  # percent

    colors = [ASSET_COLORS.get(c, MUTED) for c in col_order]
    ax_wt.stackplot(
        wt_plot.index, *[wt_plot[c].values for c in col_order],
        colors=colors, alpha=0.85, linewidth=0,
    )
    # Thin white separator lines between areas for clarity
    cumulative = np.zeros(len(wt_plot))
    for c, clr in zip(col_order, colors):
        cumulative = cumulative + wt_plot[c].values
        ax_wt.plot(wt_plot.index, cumulative, color=PANEL, linewidth=0.3)

    # ── Inline annotations at each rebalance showing ETF names + weights ──
    # Collect rebalance dates from selection_log (auto-select) or weight shifts
    if selection_log:
        rebal_dates_for_labels = [dt for dt, _ in selection_log]
    else:
        # Manual mode: detect rebalance dates from weight jumps
        wdiff = wh_display.diff().abs().sum(axis=1)
        rebal_dates_for_labels = wdiff[wdiff > 0.01].index.tolist()

    # Find the nearest weekly index for each rebalance date
    wt_idx = wt_plot.index
    for rdate in rebal_dates_for_labels:
        # Snap to the closest weekly sample
        dists = abs(wt_idx - rdate)
        snap_idx = dists.argmin()
        snap_date = wt_idx[snap_idx]

        # Draw a thin vertical rebalance marker
        ax_wt.axvline(snap_date, color=MUTED, linewidth=0.4, alpha=0.5, zorder=2)

        # Place each ETF label centred within its own band
        cum_bottom = 0.0
        for c in col_order:
            v = wt_plot[c].iloc[snap_idx]
            band_top = cum_bottom + v
            mid_y = (cum_bottom + band_top) / 2
            cum_bottom = band_top
            if v < 4:  # skip bands too thin to read
                continue
            ax_wt.text(
                snap_date, mid_y, f"{c}\n{v:.0f}%",
                fontsize=4.8, fontfamily="monospace", fontweight="bold",
                color=WHITE, ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=ASSET_COLORS.get(c, MUTED),
                          edgecolor="none", alpha=0.7),
            )

    ax_wt.set_ylabel("ALLOCATION  (%)", fontsize=9, labelpad=10)
    ax_wt.set_ylim(0, 100)
    ax_wt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_wt.grid(True, color=GRID, linewidth=0.5, alpha=0.4)
    ax_wt.tick_params(axis="x", labelbottom=False)

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
            last_weights.items(), key=lambda x: -x[1]
        )
    )
    mode_label = f"AUTO (top {TOP_N})" if AUTO_SELECT else METHOD.upper().replace('_', ' ')
    ax_footer.text(
        0.0, 0.5,
        f"HOLDINGS:  {tickers_str}     │     PERIOD: {START} → {END}"
        f"     │     INITIAL: ${INITIAL_CASH:,.0f}     │     REBAL: {REBALANCE_FREQ}"
        f"     │     DIVIDENDS: {'REINVEST' if REINVEST_DIVIDENDS else 'CASH OUT'}"
        f"     │     MODE: {mode_label}",
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

    # ── Step 8: Summary ───────────────────────────────────────
    print(f"\n[8/8] Summary")
    print(f"      Mode:         {mode}")
    print(f"      Final value:  ${portfolio.iloc[-1]:,.2f}")
    print(f"      Total return: {total_return:+.1f}%")
    print(f"      CAGR:         {cagr:+.1f}%")
    print(f"      Sharpe:       {sharpe:.2f}")
    print(f"      Max DD:       {max_dd:.1f}%")
    if AUTO_SELECT and selection_log:
        last_dt, last_picks = selection_log[-1]
        print(f"      Holdings:     {last_picks}")
    print(f"      Chart saved:  portfolio.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
