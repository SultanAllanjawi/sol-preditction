from fastapi import APIRouter

import cache as model_cache
from tickers import ALL_TICKERS, CATEGORY_OF

router = APIRouter(prefix="/api", tags=["scanner"])


@router.get("/scanner")
def scanner():
    """Signal snapshot across every ticker that's currently warm in cache.
    Tickers not yet trained are skipped rather than triggering N synchronous
    trainings on one request — call /api/predict/{ticker} once to warm one up."""
    rows = []
    for status in model_cache.cache_status():
        ticker = status["ticker"]
        if not status["ok"]:
            continue
        cached = model_cache.get_cached(ticker)
        r = cached["data"]["results"]
        rows.append({
            "ticker": ticker,
            "name": ALL_TICKERS.get(ticker, ticker),
            "category": CATEGORY_OF.get(ticker, "unknown"),
            "signal": r["last_signal"],
            "confidence": round(float(r["last_confidence"]), 1),
            "ensemble_accuracy": round(float(r["ensemble_acc"]) * 100, 1),
            "trained_at": status["trained_at"],
        })
    rows.sort(key=lambda x: x["confidence"], reverse=True)
    return {"tickers": rows}
