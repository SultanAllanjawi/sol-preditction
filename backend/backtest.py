"""
backtest.py — walk-forward P&L simulation over the model's own historical signals.

Note: the legacy Streamlit tab simulated win/loss with a random draw weighted by
the model's filtered accuracy (`np.random.random() < ens_filt`), so it never
looked at what price actually did next and gave a different answer every run.
This version walks the REAL subsequent price path (using High/Low for
intrabar TP/SL touches) to decide whether each signal actually won or lost —
deterministic and grounded in real data, which is what "good accuracy" requires.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_HOLD_BARS = 20
ATR_TP_MULT = 2.0
ATR_SL_MULT = 1.5


def _safe(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def run_backtest(cached: dict, starting_capital: float = 1000.0, trade_size_pct: float = 100.0) -> dict:
    r = cached["data"]["results"]
    te_df: pd.DataFrame = r["te_df"]
    signals = r["signals"]
    n = len(signals)

    close = te_df["Close"].values
    high = te_df["High"].values if "High" in te_df.columns else close
    low = te_df["Low"].values if "Low" in te_df.columns else close
    atr = te_df["ATR"].values if "ATR" in te_df.columns else np.full(n, np.nan)
    dates = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in te_df.index]

    cap = float(starting_capital)
    equity = [cap]
    wins = losses = 0
    peak = cap
    max_dd = 0.0
    trades = []

    for i in range(n):
        if signals[i] == 0:
            continue
        is_buy = signals[i] == 1
        entry = _safe(close[i])
        if entry <= 0:
            continue
        a = _safe(atr[i], entry * 0.02) or entry * 0.02
        tp = entry + ATR_TP_MULT * a if is_buy else entry - ATR_TP_MULT * a
        sl = entry - ATR_SL_MULT * a if is_buy else entry + ATR_SL_MULT * a

        outcome = "TIMEOUT"
        exit_price = entry
        exit_idx = min(i + MAX_HOLD_BARS, n - 1)
        for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
            hit_tp = high[j] >= tp if is_buy else low[j] <= tp
            hit_sl = low[j] <= sl if is_buy else high[j] >= sl
            if hit_tp and hit_sl:
                outcome, exit_price, exit_idx = "SL", sl, j  # conservative: assume SL first
                break
            if hit_tp:
                outcome, exit_price, exit_idx = "TP", tp, j
                break
            if hit_sl:
                outcome, exit_price, exit_idx = "SL", sl, j
                break
        else:
            exit_price = _safe(close[exit_idx], entry)

        pnl_pct = (exit_price - entry) / entry if is_buy else (entry - exit_price) / entry
        position = cap * (trade_size_pct / 100.0)
        trade_pnl = position * pnl_pct
        cap = max(cap + trade_pnl, 0.01)
        equity.append(cap)
        if trade_pnl > 0:
            wins += 1
        else:
            losses += 1
        peak = max(peak, cap)
        max_dd = max(max_dd, (peak - cap) / peak * 100 if peak > 0 else 0)

        trades.append({
            "date": dates[i],
            "signal": "BUY" if is_buy else "SELL",
            "entry": round(entry, 4),
            "exit": round(exit_price, 4),
            "outcome": outcome,
            "pnl": round(trade_pnl, 2),
            "capital": round(cap, 2),
        })

    total_trades = wins + losses
    total_return = (cap - starting_capital) / starting_capital * 100 if starting_capital else 0.0
    win_rate = wins / max(total_trades, 1) * 100

    return {
        "starting_capital": starting_capital,
        "final_capital": round(cap, 2),
        "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 2),
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
        "max_drawdown_pct": round(max_dd, 2),
        "equity_curve": [round(e, 2) for e in equity],
        "trades": list(reversed(trades)),
    }
