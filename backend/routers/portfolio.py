from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import persistence as pst
from data_manager import DataManager
from tickers import normalize

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


class NewTrade(BaseModel):
    ticker: str
    side: str = Field(pattern="^(BUY|SELL)$")
    entry: float = Field(gt=0)
    size: float = Field(gt=0)
    tp: float | None = None
    sl: float | None = None
    note: str = ""


class CloseTrade(BaseModel):
    exit_price: float = Field(gt=0)


def _with_ids(trades: list) -> list:
    for i, t in enumerate(trades):
        t.setdefault("id", i)
    return trades


@router.get("")
def list_trades():
    trades = _with_ids(pst.load_portfolio())
    total_pnl = 0.0
    total_invest = 0.0
    open_count = 0
    rows = []
    for t in trades:
        live = DataManager.get_live_price(t["ticker"]) or t["entry"]
        is_open = t["status"] == "Open"
        if is_open:
            pnl_per = (live - t["entry"]) if t["side"] == "BUY" else (t["entry"] - live)
            pnl_total = pnl_per * t["size"]
            pnl_pct = (pnl_per / t["entry"]) * 100 if t["entry"] > 0 else 0
            open_count += 1
            total_invest += t["entry"] * t["size"]
        else:
            pnl_total = t.get("pnl", 0) or 0
            pnl_pct = (pnl_total / (t["entry"] * t["size"])) * 100 if t["entry"] > 0 else 0
            live = t.get("exit", t["entry"])
        total_pnl += pnl_total
        rows.append({**t, "live_price": live, "pnl": pnl_total, "pnl_pct": pnl_pct})

    roi = (total_pnl / total_invest * 100) if total_invest > 0 else 0
    return {
        "trades": rows,
        "summary": {
            "total_trades": len(trades),
            "open_positions": open_count,
            "total_pnl": total_pnl,
            "roi_pct": roi,
        },
    }


@router.post("")
def add_trade(trade: NewTrade):
    ticker = normalize(trade.ticker)
    trades = pst.load_portfolio()
    now_str = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")
    new_trade = {
        "date": now_str,
        "ticker": ticker,
        "side": trade.side,
        "entry": trade.entry,
        "size": trade.size,
        "tp": trade.tp,
        "sl": trade.sl,
        "status": "Open",
        "exit": None,
        "pnl": None,
        "note": trade.note,
    }
    trades.append(new_trade)
    pst.save_portfolio(trades)
    return {**new_trade, "id": len(trades) - 1}


@router.post("/{trade_id}/close")
def close_trade(trade_id: int, body: CloseTrade):
    trades = _with_ids(pst.load_portfolio())
    match = next((t for t in trades if t["id"] == trade_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    if match["status"] != "Open":
        raise HTTPException(status_code=400, detail="Trade already closed")

    pnl_per = (body.exit_price - match["entry"]) if match["side"] == "BUY" else (match["entry"] - body.exit_price)
    pnl = pnl_per * match["size"]
    match["status"] = "Closed"
    match["exit"] = body.exit_price
    match["pnl"] = pnl
    closed_at = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")
    match["closed_at"] = closed_at

    pst.save_portfolio(trades)
    pst.save_closed_signal(
        ticker=match["ticker"], signal=match["side"], entry=match["entry"],
        exit_price=body.exit_price, tp=match.get("tp"), sl=match.get("sl"),
        result="MANUAL_CLOSE", closed_at=closed_at,
    )
    return match


@router.delete("/{trade_id}")
def delete_trade(trade_id: int):
    trades = _with_ids(pst.load_portfolio())
    remaining = [t for t in trades if t["id"] != trade_id]
    if len(remaining) == len(trades):
        raise HTTPException(status_code=404, detail="Trade not found")
    for t in remaining:
        t.pop("id", None)
    pst.save_portfolio(remaining)
    return {"deleted": trade_id}


@router.get("/closed")
def closed_trades():
    return pst.load_closed_signals()
