"""
serialize.py — turns ModelEngine.train()'s numpy/pandas-heavy result dict into
clean JSON-safe plain dicts/lists for the API (no emoji strings, no numpy types).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _dates(index) -> list[str]:
    return [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in index]


def serialize_predict(cached: dict) -> dict:
    data = cached["data"]
    r = data["results"]
    ticker = data["ticker"]

    model_data = r["model_data"]
    models = {
        name: {
            "accuracy": _safe_float(m["acc"]),
            "f1": _safe_float(m["f1"]),
            "auc": _safe_float(m["auc"]),
        }
        for name, m in model_data.items()
    }

    te_dates = _dates(r["te_df"].index)
    ens_proba = r["ens_proba"]
    signals = r["signals"]

    signal_history = [
        {
            "date": te_dates[i] if i < len(te_dates) else None,
            "price": _safe_float(r["te_df"]["Close"].iloc[i]) if i < len(r["te_df"]) else None,
            "signal": "BUY" if signals[i] == 1 else "SELL" if signals[i] == -1 else "HOLD",
            "prob_up": _safe_float(ens_proba[i]),
            "confidence": _safe_float(max(ens_proba[i], 1 - ens_proba[i]) * 100),
        }
        for i in range(len(ens_proba))
        if signals[i] != 0
    ]
    signal_history.sort(key=lambda s: s["date"] or "", reverse=True)

    n = min(len(r["price_pred"]), len(r["y_price_te"]))
    price_dates = te_dates[-n:] if n else []
    price_prediction = {
        "rmse": _safe_float(r["rmse"]),
        "mae": _safe_float(r["mae"]),
        "dates": price_dates,
        "predicted": [_safe_float(v) for v in r["price_pred"][-n:]],
        "actual": [_safe_float(v) for v in r["y_price_te"][-n:]],
    }

    return {
        "ticker": ticker,
        "trained_at": cached["trained_at"].isoformat() if cached.get("trained_at") else None,
        "rows_used": data.get("rows"),
        "sentiment_score": _safe_float(data.get("sentiment_score", 0.0)),
        "last_signal": r["last_signal"],
        "last_confidence": _safe_float(r["last_confidence"]),
        "last_prob_up": _safe_float(r["last_prob"]),
        "thresholds": {"high": r["HIGH"], "low": r["LOW"]},
        "ensemble": {
            "accuracy": _safe_float(r["ensemble_acc"]),
            "filtered_accuracy": _safe_float(r["ensemble_filt_acc"]),
            "f1": _safe_float(r["ensemble_f1"]),
            "auc": _safe_float(r["ensemble_auc"]),
            "models_used": r["ensemble_models"],
            "models_excluded": r["excluded_models"],
            "best_model": r["best_model"],
        },
        "models": models,
        "n_signals": r["n_signals"],
        "signal_history": signal_history,
        "price_prediction": price_prediction,
    }


_CHART_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "RSI", "MACD", "MACD_sig", "BB_U", "BB_L"]


def serialize_chart(cached: dict, limit: int = 500) -> dict:
    df: pd.DataFrame = cached["data"]["df_feat"]
    df = df.tail(limit)
    r = cached["data"]["results"]

    sig_by_date: dict[str, int] = {}
    te_dates = _dates(r["te_df"].index)
    for i, s in enumerate(r["signals"]):
        if s != 0 and i < len(te_dates):
            sig_by_date[te_dates[i]] = int(s)

    dates = _dates(df.index)
    out = {"dates": dates}
    for col in _CHART_COLS:
        if col in df.columns:
            out[col.lower()] = [_safe_float(v, None) if pd.notna(v) else None for v in df[col].values]
    out["signal"] = [sig_by_date.get(d, 0) for d in dates]
    return out
