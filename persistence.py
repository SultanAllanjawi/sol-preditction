"""
persistence.py — Save and load user data to data/ directory
Survives Streamlit restarts and page closes.
"""
import os, json
from datetime import datetime, timezone

DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
SIGNALS_FILE   = os.path.join(DATA_DIR, "active_signals.json")
SETTINGS_FILE  = os.path.join(DATA_DIR, "user_settings.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Portfolio ─────────────────────────────────────────────────────
def load_portfolio() -> list:
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_portfolio(trades: list):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
    except Exception:
        pass

# ── Active signals tracker ────────────────────────────────────────
def load_active_signals() -> dict:
    """Returns {ticker: {signal, entry, tp, sl, timestamp, status}}"""
    try:
        if os.path.exists(SIGNALS_FILE):
            with open(SIGNALS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_active_signals(signals: dict):
    try:
        with open(SIGNALS_FILE, 'w') as f:
            json.dump(signals, f, indent=2)
    except Exception:
        pass

def update_signal_status(ticker: str, signal: str, entry: float,
                          tp: float | None, sl: float | None,
                          live_price: float | None) -> dict:
    """
    Check if an active signal has hit TP, SL, or expired.
    Returns status dict with: status, message, pnl_pct, valid_until
    """
    signals = load_active_signals()
    now     = datetime.now(timezone.utc)

    # Record new signal if changed
    existing = signals.get(ticker, {})
    prev_sig = existing.get("signal", "HOLD")

    if signal != "HOLD" and (signal != prev_sig or not existing):
        signals[ticker] = {
            "signal"    : signal,
            "entry"     : entry,
            "tp"        : tp,
            "sl"        : sl,
            "timestamp" : now.isoformat(),
            "status"    : "ACTIVE",
        }
        save_active_signals(signals)

    sig_data = signals.get(ticker, {})
    if not sig_data or sig_data.get("signal") == "HOLD":
        return {"status": "NONE", "message": "No active signal", "pnl_pct": 0}

    _sig    = sig_data["signal"]
    _entry  = sig_data.get("entry", entry)
    _tp     = sig_data.get("tp")
    _sl     = sig_data.get("sl")
    _ts_str = sig_data.get("timestamp", now.isoformat())
    _status = sig_data.get("status", "ACTIVE")

    # Parse timestamp
    try:
        _ts = datetime.fromisoformat(_ts_str)
        if _ts.tzinfo is None:
            _ts = _ts.replace(tzinfo=timezone.utc)
        hours_old = (now - _ts).total_seconds() / 3600
    except Exception:
        hours_old = 0

    # Expiry: 72 hours for crypto, 48 hours for stocks
    _expiry_hours = 72
    _valid_until  = hours_old < _expiry_hours

    # Current P&L
    _pnl_pct = 0.0
    if live_price and _entry:
        _pnl_pct = (live_price - _entry) / _entry * 100
        if _sig == "SELL":
            _pnl_pct = -_pnl_pct

    # Check TP/SL hit
    if live_price and _tp and _sl and _status == "ACTIVE":
        if _sig == "BUY":
            if live_price >= _tp:
                _status = "HIT_TP"
                sig_data["status"] = "HIT_TP"
                save_active_signals(signals)
            elif live_price <= _sl:
                _status = "HIT_SL"
                sig_data["status"] = "HIT_SL"
                save_active_signals(signals)
        elif _sig == "SELL":
            if live_price <= _tp:
                _status = "HIT_TP"
                sig_data["status"] = "HIT_TP"
                save_active_signals(signals)
            elif live_price >= _sl:
                _status = "HIT_SL"
                sig_data["status"] = "HIT_SL"
                save_active_signals(signals)

    # Expiry check
    if not _valid_until and _status == "ACTIVE":
        _status = "EXPIRED"
        sig_data["status"] = "EXPIRED"
        save_active_signals(signals)

    # Build message
    _remaining_h = max(0, _expiry_hours - hours_old)
    msgs = {
        "ACTIVE"  : f"Signal active · {_remaining_h:.0f}h remaining · Enter if not already in",
        "HIT_TP"  : f"Target reached! Signal succeeded · P&L: +{abs(_pnl_pct):.2f}%",
        "HIT_SL"  : f"Stop loss hit · Signal failed · P&L: {_pnl_pct:.2f}%",
        "EXPIRED" : f"Signal expired (72h) · Do NOT enter · Wait for new signal",
    }

    return {
        "status"      : _status,
        "message"     : msgs.get(_status, ""),
        "pnl_pct"     : _pnl_pct,
        "hours_old"   : hours_old,
        "remaining_h" : _remaining_h,
        "valid_until" : _valid_until,
        "signal"      : _sig,
        "entry"       : _entry,
        "tp"          : _tp,
        "sl"          : _sl,
    }

# ── Settings ──────────────────────────────────────────────────────
def load_settings() -> dict:
    defaults = {
        "confidence_thresh": 0.60,
        "lookback_days"    : 90,
        "chart_tf"         : "D",
        "last_ticker"      : "SOL-USD",
    }
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
                defaults.update(saved)
    except Exception:
        pass
    return defaults

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass
