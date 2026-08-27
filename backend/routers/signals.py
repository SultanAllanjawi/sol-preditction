from fastapi import APIRouter

import cache as model_cache
import persistence as pst
from data_manager import DataManager
from tickers import normalize

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/active/{ticker}")
def active_signal(ticker: str):
    ticker = normalize(ticker)
    cached = model_cache.get_cached(ticker)
    r = cached["data"]["results"]
    live_price = DataManager.get_live_price(ticker)
    entry_price = float(r["te_df"]["Close"].iloc[-1]) if len(r["te_df"]) else None

    tp = sl = None
    if entry_price:
        atr = float(r["te_df"]["ATR"].iloc[-1]) if "ATR" in r["te_df"].columns else entry_price * 0.02
        if r["last_signal"] == "BUY":
            tp, sl = entry_price + 2 * atr, entry_price - 1.5 * atr
        elif r["last_signal"] == "SELL":
            tp, sl = entry_price - 2 * atr, entry_price + 1.5 * atr

    status = pst.update_signal_status(
        ticker=ticker, signal=r["last_signal"], entry=entry_price or 0.0,
        tp=tp, sl=sl, live_price=live_price,
    )
    return {"ticker": ticker, "live_price": live_price, **status}


@router.get("/history")
def closed_signals():
    return pst.load_closed_signals()
