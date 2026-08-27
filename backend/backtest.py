"""
backtest.py — ported 1:1 from legacy-streamlit/app.py's Backtest P&L tab (lines 3264-3294).

Win/loss per trade is a single Bernoulli draw weighted by the ensemble's filtered accuracy
(np.random.random() < ens_filt), not a check against real subsequent price action. This is
the original app's exact behavior — restored deliberately after a prior "improvement" to a
deterministic walk-forward test was reverted at the user's request to match the original.
"""
from __future__ import annotations

import math

import numpy as np

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
    te_df = r["te_df"]
    signals = r["signals"]
    ens_filt = _safe(r["ensemble_filt_acc"], 0.5)
    n = len(signals)

    close = te_df["Close"].values
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

        hit = bool(np.random.random() < ens_filt)
        pnl_pct = abs(tp - entry) / entry if hit else -abs(sl - entry) / entry
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
            "exit": round(tp if hit else sl, 4),
            "outcome": "TP" if hit else "SL",
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
