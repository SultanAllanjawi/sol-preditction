from fastapi import APIRouter, HTTPException, Query

import cache as model_cache
from backtest import run_backtest
from serialize import serialize_chart, serialize_predict
from tickers import normalize

router = APIRouter(prefix="/api", tags=["predict"])


def _get_or_404(ticker: str, force: bool = False) -> dict:
    ticker = normalize(ticker)
    try:
        return model_cache.get_cached(ticker, force=force)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/predict/{ticker}")
def predict(ticker: str, refresh: bool = Query(False, description="Force a retrain instead of using cache")):
    cached = _get_or_404(ticker, force=refresh)
    return serialize_predict(cached)


@router.get("/ohlcv/{ticker}")
def ohlcv(ticker: str, limit: int = Query(500, ge=50, le=5000)):
    cached = _get_or_404(ticker)
    return serialize_chart(cached, limit=limit)


@router.get("/backtest/{ticker}")
def backtest(
    ticker: str,
    starting_capital: float = Query(1000.0, gt=0),
    trade_size_pct: float = Query(100.0, gt=0, le=100),
):
    cached = _get_or_404(ticker)
    return run_backtest(cached, starting_capital=starting_capital, trade_size_pct=trade_size_pct)


@router.get("/status")
def status():
    return {"cache": model_cache.cache_status()}


@router.get("/tickers")
def tickers():
    from tickers import ticker_list
    return ticker_list()
