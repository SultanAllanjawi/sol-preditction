"""
cache.py — in-memory trained-model cache with TTL + a background scheduler that
keeps the "tracked" tickers warm, so /api/predict reads are instant instead of
retraining (RNN + RF + GB + XGBoost) on every request.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from pipeline import load_and_train
from tickers import DEFAULT_TRACKED

log = logging.getLogger("uvicorn.error")

TTL_MINUTES = int(os.environ.get("TRAIN_TTL_MINUTES", "180"))

_lock = threading.Lock()
_store: dict[str, dict] = {}   # ticker -> {"data": ..., "trained_at": datetime, "error": str|None}


def _train_and_store(ticker: str) -> dict:
    entry = _store.get(ticker, {})
    try:
        data = load_and_train(ticker)
        entry = {
            "data": data,
            "trained_at": datetime.now(timezone.utc),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        log.error("training failed for %s: %s\n%s", ticker, e, traceback.format_exc())
        entry = {
            "data": entry.get("data"),
            "trained_at": entry.get("trained_at"),
            "error": str(e),
        }
    with _lock:
        _store[ticker] = entry
    return entry


def get_cached(ticker: str, force: bool = False) -> dict:
    """Returns the cached trained result, training synchronously on a cold/stale cache."""
    entry = _store.get(ticker)
    stale = (
        entry is None
        or entry.get("data") is None
        or force
        or (datetime.now(timezone.utc) - entry["trained_at"]).total_seconds() > TTL_MINUTES * 60
    )
    if stale:
        entry = _train_and_store(ticker)
    if entry.get("data") is None:
        raise RuntimeError(entry.get("error") or f"No data available yet for {ticker}")
    return entry


def cache_status() -> list[dict]:
    with _lock:
        return [
            {
                "ticker": t,
                "trained_at": e["trained_at"].isoformat() if e.get("trained_at") else None,
                "ok": e.get("error") is None and e.get("data") is not None,
                "error": e.get("error"),
            }
            for t, e in _store.items()
        ]


def _refresh_tracked():
    for t in DEFAULT_TRACKED:
        _train_and_store(t)
        time.sleep(1)  # be polite to upstream free APIs (Binance/Yahoo/CoinGecko)


_scheduler: BackgroundScheduler | None = None


def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(_refresh_tracked, "interval", minutes=TTL_MINUTES, next_run_time=datetime.now())
    _scheduler.start()


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
