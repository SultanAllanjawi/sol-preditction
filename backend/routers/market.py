from fastapi import APIRouter, HTTPException, Query

from data_manager import DataManager, is_crypto
from tickers import normalize

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/live/{ticker}")
def live(ticker: str):
    ticker = normalize(ticker)
    price = DataManager.get_live_price(ticker)
    if price is None:
        raise HTTPException(status_code=502, detail=f"No live price available for {ticker}")
    out = {"ticker": ticker, "price": price, "is_crypto": is_crypto(ticker)}
    stats = DataManager.get_24h_stats(ticker)
    if stats:
        out["stats_24h"] = stats
    if is_crypto(ticker):
        cap = DataManager.get_market_cap(ticker)
        if cap:
            out["market_cap"] = cap
    return out


@router.get("/orderbook/{ticker}")
def orderbook(ticker: str, limit: int = Query(10, ge=1, le=100)):
    ticker = normalize(ticker)
    ob = DataManager.get_order_book(ticker, limit=limit)
    if ob is None:
        raise HTTPException(status_code=502, detail=f"No order book available for {ticker}")
    return ob


@router.get("/fear-greed")
def fear_greed():
    fg = DataManager.get_fear_greed_index()
    if fg is None:
        raise HTTPException(status_code=502, detail="Fear & Greed index unavailable")
    history = DataManager.get_fear_greed_history(days=14) or []
    return {**fg, "history": history}


@router.get("/wallets/{ticker}")
def active_wallets(ticker: str):
    ticker = normalize(ticker)
    w = DataManager.get_active_wallets(ticker)
    if w is None:
        raise HTTPException(status_code=404, detail=f"No wallet data for {ticker}")
    return w
