"""
feature_engine.py v2
────────────────────
Builds 65+ features for ML models.
New in v2:
  • Sentiment score injected as feature (if provided)
  • Hour-of-day + day-of-week features for intraday signals
  • Intraday volatility features (when hourly data available)
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def build_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Input : raw OHLCV DataFrame (index = Date or Datetime)
    Output: feature-rich DataFrame with Target column

    sentiment_score: -1.0 (very bearish) to +1.0 (very bullish), 0.0 = neutral
    """
    d = df.copy()

    # ── Savitzky-Golay denoising ───────────────────────────────────────────────
    _is_intraday = hasattr(d.index, 'hour') and hasattr(d.index, 'minute')
    _wlen = 11 if len(d) >= 11 else max(3, len(d)//3*2+1)
    _wlen = _wlen if _wlen % 2 == 1 else _wlen + 1  # must be odd
    if len(d) >= _wlen:
        d["Smooth"] = scipy_signal.savgol_filter(d["Close"].values, _wlen, 3)
    else:
        d["Smooth"] = d["Close"]

    P = d["Smooth"]
    C = d["Close"]

    # Add Change_Pct if missing (when data comes from DataManager it's already there)
    if "Change_Pct" not in d.columns:
        d["Change_Pct"] = C.pct_change() * 100

    # ── Moving Averages ────────────────────────────────────────────────────────
    _windows = [5, 10, 20, 50, 100]  # same for both daily and intraday
    for w in _windows:
        d[f"SMA{w}"] = P.rolling(w, min_periods=1).mean()
        d[f"EMA{w}"] = P.ewm(span=w, adjust=False).mean()

    d["X_5_20"]   = (d["SMA5"]  > d["SMA20"]).astype(int)
    d["X_20_50"]  = (d["SMA20"] > d["SMA50"]).astype(int)
    d["X_E12_26"] = (d["EMA10"] > d["EMA50"]).astype(int)

    # ── MACD ───────────────────────────────────────────────────────────────────
    ema12 = P.ewm(span=12, adjust=False).mean()
    ema26 = P.ewm(span=26, adjust=False).mean()
    d["MACD"]       = ema12 - ema26
    d["MACD_sig"]   = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]  = d["MACD"] - d["MACD_sig"]
    d["MACD_cross"] = (d["MACD"] > d["MACD_sig"]).astype(int)
    d["MACD_div"]   = d["MACD_hist"] - d["MACD_hist"].shift(1)

    # ── RSI ────────────────────────────────────────────────────────────────────
    delta = P.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    d["RSI"]     = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    d["RSI_OB"]  = (d["RSI"] > 70).astype(int)
    d["RSI_OS"]  = (d["RSI"] < 30).astype(int)
    d["RSI_mid"] = ((d["RSI"] >= 45) & (d["RSI"] <= 55)).astype(int)

    r_min = d["RSI"].rolling(14, min_periods=1).min()
    r_max = d["RSI"].rolling(14, min_periods=1).max()
    d["StochRSI"] = (d["RSI"] - r_min) / (r_max - r_min + 1e-9)
    d["StRSI_K"]  = d["StochRSI"].rolling(3, min_periods=1).mean()
    d["StRSI_D"]  = d["StRSI_K"].rolling(3, min_periods=1).mean()

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_std    = P.rolling(20, min_periods=1).std()
    d["BB_U"] = d["SMA20"] + 2 * bb_std
    d["BB_L"] = d["SMA20"] - 2 * bb_std
    d["BB_W"] = (d["BB_U"] - d["BB_L"]) / (d["SMA20"] + 1e-9)
    d["BB_P"] = (P - d["BB_L"]) / (d["BB_U"] - d["BB_L"] + 1e-9)
    d["BB_SQ"]= (d["BB_W"] < d["BB_W"].rolling(20, min_periods=1).mean()).astype(int)

    # ── ATR ────────────────────────────────────────────────────────────────────
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - C.shift()).abs(),
        (d["Low"]  - C.shift()).abs(),
    ], axis=1).max(axis=1)
    d["ATR"]       = tr.rolling(14, min_periods=1).mean()
    d["ATR_pct"]   = d["ATR"] / (P + 1e-9)
    d["ATR_trend"] = d["ATR"] - d["ATR"].rolling(5, min_periods=1).mean()

    # ── Returns & Lags ────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 7, 10, 14, 20]:
        d[f"Ret{lag}"]  = P.pct_change(lag) * 100
    for lag in [1, 3, 5]:
        d[f"RRaw{lag}"] = C.diff(lag)

    # ── Volume features ───────────────────────────────────────────────────────
    vol = d.get("Volume", pd.Series(1.0, index=d.index))
    d["Vol_MA20"]   = vol.rolling(20, min_periods=1).mean()
    d["Vol_ratio"]  = vol / (d["Vol_MA20"] + 1e-9)
    d["Vol_spike"]  = (d["Vol_ratio"] > 2.0).astype(int)
    d["OBV"]        = (np.sign(C.diff()) * vol).cumsum()
    d["OBV_MA"]     = d["OBV"].rolling(10, min_periods=1).mean()
    d["OBV_trend"]  = (d["OBV"] > d["OBV_MA"]).astype(int)

    # ── Candle body features ──────────────────────────────────────────────────
    d["Body"]       = (C - d["Open"]).abs()
    d["UpWick"]     = d["High"] - d[["Close","Open"]].max(axis=1)
    d["DnWick"]     = d[["Close","Open"]].min(axis=1) - d["Low"]
    d["BullCandle"] = (C > d["Open"]).astype(int)
    d["BodyRatio"]  = d["Body"] / (d["High"] - d["Low"] + 1e-9)

    # ── Market regime ────────────────────────────────────────────────────────
    d["Regime"]     = (d["SMA20"] > d["SMA50"]).astype(int)
    d["Trend_str"]  = (P - P.rolling(20, min_periods=1).mean()) / (P.rolling(20, min_periods=1).std() + 1e-9)

    # ── Noise / Efficiency ────────────────────────────────────────────────────
    d["Noise"]      = d["ATR"] / (P.diff(5).abs() + 1e-9)
    d["Efficiency"] = P.diff(10).abs() / (d["ATR"].rolling(10, min_periods=1).sum() + 1e-9)

    # ── Intraday features (hourly data only) ──────────────────────────────────
    if _is_intraday and hasattr(d.index, 'hour'):
        d["Hour"]       = d.index.hour / 23.0
        try:
            _dow2 = d.index.dayofweek
            d["DayOfWeek"] = (pd.to_numeric(_dow2, errors='coerce').fillna(0) / 6.0)
        except Exception:
            d["DayOfWeek"] = 0.5
        d["IsAsiaOpen"] = ((d.index.hour >= 0) & (d.index.hour < 8)).astype(int)
        d["IsEUOpen"]   = ((d.index.hour >= 7) & (d.index.hour < 15)).astype(int)
        d["IsUSOpen"]   = ((d.index.hour >= 13) & (d.index.hour < 21)).astype(int)
        # Intraday high/low position
        d["DayHigh"]    = d["High"].rolling(24, min_periods=1).max()
        d["DayLow"]     = d["Low"].rolling(24, min_periods=1).min()
        d["PosInDay"]   = (C - d["DayLow"]) / (d["DayHigh"] - d["DayLow"] + 1e-9)
    else:
        d["Hour"]       = 0.5
        try:
            _dow = d.index.dayofweek
            d["DayOfWeek"] = (pd.to_numeric(_dow, errors='coerce').fillna(0) / 6.0)
        except Exception:
            d["DayOfWeek"] = 0.5
        d["IsAsiaOpen"] = 0
        d["IsEUOpen"]   = 0
        d["IsUSOpen"]   = 0
        d["DayHigh"]    = d["High"].rolling(5, min_periods=1).max()
        d["DayLow"]     = d["Low"].rolling(5, min_periods=1).min()
        d["PosInDay"]   = (C - d["DayLow"]) / (d["DayHigh"] - d["DayLow"] + 1e-9)

    # ── News Sentiment Feature ────────────────────────────────────────────────
    # sentiment_score: -1.0 to +1.0 (from CryptoPanic votes)
    # Applied to all rows so model knows current sentiment context
    d["Sentiment"]      = float(sentiment_score)
    d["Sentiment_Bull"] = int(float(sentiment_score) > 0.1)
    d["Sentiment_Bear"] = int(float(sentiment_score) < -0.1)

    # ── Target ───────────────────────────────────────────────────────────────
    d["NextClose"] = C.shift(-1)
    d["Target"]    = (d["NextClose"] > C).astype(int)

    d.dropna(subset=["Target", "Close"], inplace=True)
    return d


FEATURE_COLS = [
    # Price & smoothed
    "Close","Open","High","Low","Volume","Change_Pct","Smooth",
    # Moving averages
    "SMA5","SMA10","SMA20","SMA50","SMA100",
    "EMA5","EMA10","EMA20","EMA50","EMA100",
    "X_5_20","X_20_50","X_E12_26",
    # MACD
    "MACD","MACD_sig","MACD_hist","MACD_cross","MACD_div",
    # RSI
    "RSI","RSI_OB","RSI_OS","RSI_mid","StochRSI","StRSI_K","StRSI_D",
    # Bollinger
    "BB_U","BB_L","BB_W","BB_P","BB_SQ",
    # ATR
    "ATR","ATR_pct","ATR_trend",
    # Returns
    "Ret1","Ret2","Ret3","Ret5","Ret7","Ret10","Ret14","Ret20",
    "RRaw1","RRaw3","RRaw5",
    # Volume
    "Vol_MA20","Vol_ratio","Vol_spike","OBV","OBV_MA","OBV_trend",
    # Candle
    "Body","UpWick","DnWick","BullCandle","BodyRatio",
    # Regime
    "Regime","Trend_str","Noise","Efficiency",
    # Time features
    "Hour","DayOfWeek","IsAsiaOpen","IsEUOpen","IsUSOpen","PosInDay",
    # Sentiment
    "Sentiment","Sentiment_Bull","Sentiment_Bear",
]
