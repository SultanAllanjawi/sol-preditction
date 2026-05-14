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
        # Mark old signal as expired when new one fires
        if existing and existing.get("status") == "ACTIVE":
            existing["status"] = "EXPIRED_BY_NEW"
            signals[ticker + "_prev"] = existing
        signals[ticker] = {
            "signal"    : signal,
            "entry"     : entry,
            "tp"        : tp,
            "sl"        : sl,
            "timestamp" : now.isoformat(),
            "status"    : "ACTIVE",
        }
        save_active_signals(signals)
    elif signal == "HOLD" and existing.get("status") == "ACTIVE":
        # Model moved to HOLD = signal no longer valid
        from datetime import timedelta as _td
        _ts = existing.get("timestamp", now.isoformat())
        try:
            _age_h = (now - datetime.fromisoformat(_ts).replace(tzinfo=timezone.utc)).total_seconds()/3600
        except Exception:
            _age_h = 0
        # Expire after 24h if model goes HOLD
        if _age_h > 24:
            existing["status"] = "EXPIRED"
            signals[ticker] = existing
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

# ── Uploaded CSV storage (survives restarts) ─────────────────────
import base64

CSV_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(CSV_DIR, exist_ok=True)

def save_uploaded_csv(ticker: str, csv_bytes: bytes):
    """Save uploaded CSV bytes to disk so they survive app restarts."""
    try:
        safe = ticker.replace("/","_").replace(".","_").replace(" ","_")
        path = os.path.join(CSV_DIR, f"{safe}.csv")
        with open(path, 'wb') as f:
            f.write(csv_bytes)
        # Update the index of saved CSVs
        _update_csv_index(ticker, path)
        return True
    except Exception:
        return False

def load_uploaded_csv(ticker: str) -> bytes | None:
    """Load a previously saved CSV from disk."""
    try:
        safe = ticker.replace("/","_").replace(".","_").replace(" ","_")
        path = os.path.join(CSV_DIR, f"{safe}.csv")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
    except Exception:
        pass
    return None

def load_all_uploaded_csvs() -> dict:
    """Load all saved CSVs from disk. Returns {ticker: bytes}."""
    result = {}
    try:
        idx_path = os.path.join(CSV_DIR, "_index.json")
        if not os.path.exists(idx_path):
            return result
        with open(idx_path) as f:
            index = json.load(f)
        for ticker, path in index.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    result[ticker] = f.read()
    except Exception:
        pass
    return result

def delete_uploaded_csv(ticker: str):
    """Delete a saved CSV from disk."""
    try:
        safe = ticker.replace("/","_").replace(".","_").replace(" ","_")
        path = os.path.join(CSV_DIR, f"{safe}.csv")
        if os.path.exists(path):
            os.remove(path)
        _update_csv_index(ticker, None)
    except Exception:
        pass

def _update_csv_index(ticker: str, path: str | None):
    """Maintain an index file of ticker -> file path."""
    idx_path = os.path.join(CSV_DIR, "_index.json")
    try:
        index = {}
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                index = json.load(f)
        if path is None:
            index.pop(ticker, None)
        else:
            index[ticker] = path
        with open(idx_path, 'w') as f:
            json.dump(index, f, indent=2)
    except Exception:
        pass

# ── Closed Signals Log (TP/SL hit history) ──────────────────────
CLOSED_SIGNALS_FILE = os.path.join(DATA_DIR, "closed_signals.json")

def load_closed_signals() -> list:
    """Load history of signals that hit TP or SL."""
    try:
        if os.path.exists(CLOSED_SIGNALS_FILE):
            with open(CLOSED_SIGNALS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_closed_signal(ticker: str, signal: str, entry: float,
                        exit_price: float, tp: float, sl: float,
                        result: str, closed_at: str = None):
    """Save a signal that hit TP or SL to history."""
    try:
        history = load_closed_signals()
        from datetime import datetime, timezone, timedelta
        if closed_at is None:
            closed_at = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")
        pnl_pct = ((exit_price - entry) / entry * 100) if signal == "BUY" else ((entry - exit_price) / entry * 100)
        history.append({
            "ticker"    : ticker,
            "signal"    : signal,
            "entry"     : entry,
            "exit"      : exit_price,
            "tp"        : tp,
            "sl"        : sl,
            "result"    : result,   # "HIT_TP" or "HIT_SL"
            "pnl_pct"   : round(pnl_pct, 2),
            "closed_at" : closed_at,
        })
        # Keep last 500 closed signals
        history = history[-500:]
        with open(CLOSED_SIGNALS_FILE, "w") as f:
            json.dump(history, f, indent=2)
        return True
    except Exception:
        return False
