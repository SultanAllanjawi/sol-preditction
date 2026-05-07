"""
feature_engine.py — builds all 60+ features used by the ML models.
Same logic as the Colab notebook so predictions are consistent.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw OHLCV DataFrame (index = Date)
    Output: feature-rich DataFrame with Target column
    """
    d = df.copy()

    # ── Savitzky-Golay denoising ───────────────────────────────────────────────
    if len(d) >= 11:
        d["Smooth"] = scipy_signal.savgol_filter(d["Close"].values, 11, 3)
    else:
        d["Smooth"] = d["Close"]

    P = d["Smooth"]
    C = d["Close"]

    # ── Moving Averages ────────────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100]:
        d[f"SMA{w}"] = P.rolling(w).mean()
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

    # ── RSI (14) ───────────────────────────────────────────────────────────────
    delta = P.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]    = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    d["RSI_OB"] = (d["RSI"] > 70).astype(int)
    d["RSI_OS"] = (d["RSI"] < 30).astype(int)
    d["RSI_mid"]= ((d["RSI"] >= 45) & (d["RSI"] <= 55)).astype(int)

    r_min, r_max   = d["RSI"].rolling(14).min(), d["RSI"].rolling(14).max()
    d["StochRSI"]  = (d["RSI"] - r_min) / (r_max - r_min + 1e-9)
    d["StRSI_K"]   = d["StochRSI"].rolling(3).mean()
    d["StRSI_D"]   = d["StRSI_K"].rolling(3).mean()

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_std   = P.rolling(20).std()
    d["BB_U"]= d["SMA20"] + 2 * bb_std
    d["BB_L"]= d["SMA20"] - 2 * bb_std
    d["BB_W"]= (d["BB_U"] - d["BB_L"]) / (d["SMA20"] + 1e-9)
    d["BB_P"]= (P - d["BB_L"]) / (d["BB_U"] - d["BB_L"] + 1e-9)
    d["BB_SQ"]= (d["BB_W"] < d["BB_W"].rolling(20).mean()).astype(int)

    # ── ATR ────────────────────────────────────────────────────────────────────
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - C.shift()).abs(),
        (d["Low"]  - C.shift()).abs(),
    ], axis=1).max(axis=1)
    d["ATR"]      = tr.rolling(14).mean()
    d["ATR_pct"]  = d["ATR"] / (P + 1e-9)
    d["ATR_trend"]= d["ATR"] - d["ATR"].rolling(5).mean()

    # ── Returns & lags ─────────────────────────────────────────────────────────
    for n in [1, 2, 3, 5, 7, 10, 14, 20]:
        d[f"Ret{n}"]  = P.pct_change(n)
        d[f"RRaw{n}"] = C.pct_change(n)

    for lag in [1, 2, 3, 5, 7, 10]:
        d[f"Lag{lag}"]  = P.shift(lag)
        d[f"LagR{lag}"] = d["Ret1"].shift(lag)

    # ── Volatility ─────────────────────────────────────────────────────────────
    d["HV5"]    = d["Ret1"].rolling(5).std()
    d["HV20"]   = d["Ret1"].rolling(20).std()
    d["VIX_like"]= d["HV20"] / (d["HV5"] + 1e-9)

    # ── Candle ─────────────────────────────────────────────────────────────────
    d["Body"]      = (C - d["Open"]).abs() / (d["ATR"] + 1e-9)
    d["HL_span"]   = (d["High"] - d["Low"]) / (C + 1e-9)
    d["Gap"]       = (d["Open"] - C.shift()) / (C.shift() + 1e-9)
    d["Bull_bar"]  = (C > d["Open"]).astype(int)
    d["Up_wick"]   = (d["High"] - d[["Close","Open"]].max(axis=1)) / (d["ATR"] + 1e-9)
    d["Down_wick"] = (d[["Close","Open"]].min(axis=1) - d["Low"])  / (d["ATR"] + 1e-9)

    # ── Volume ─────────────────────────────────────────────────────────────────
    d["VolMA5"]   = d["Volume"].rolling(5).mean()
    d["VolMA20"]  = d["Volume"].rolling(20).mean()
    d["VolRatio"] = d["Volume"] / (d["VolMA5"] + 1e-9)
    d["VolChg"]   = d["Volume"].pct_change()
    d["VolSurge"] = (d["VolRatio"] > 2.0).astype(int)
    d["OBV"]      = (np.sign(d["Ret1"]) * d["Volume"]).cumsum()
    d["OBV_sig"]  = (d["OBV"] > d["OBV"].rolling(10).mean()).astype(int)

    # ── Price position / regime ────────────────────────────────────────────────
    d["P_SMA20"]   = (P - d["SMA20"]) / (d["SMA20"] + 1e-9)
    d["P_SMA50"]   = (P - d["SMA50"]) / (d["SMA50"] + 1e-9)
    d["P_SMA100"]  = (P - d["SMA100"])/ (d["SMA100"]+ 1e-9)
    d["Regime"]    = (P > d["SMA50"]).astype(int)
    d["Regime_chg"]= d["Regime"].diff().abs()
    d["Wk_trend"]  = d["Ret5"].rolling(3).mean()
    d["Mo_trend"]  = d["Ret20"].rolling(5).mean()
    d["Trend_str"] = d["P_SMA20"] - d["P_SMA50"]

    # ── Noise feature ──────────────────────────────────────────────────────────
    d["Noise"]  = C - d["Smooth"]
    d["Noise_z"]= (d["Noise"] - d["Noise"].rolling(20).mean()) / (d["Noise"].rolling(20).std() + 1e-9)

    # ── Targets ────────────────────────────────────────────────────────────────
    d["Target"]   = (C.shift(-1) > C).astype(int)
    d["NextClose"]=  C.shift(-1)

    d.dropna(inplace=True)
    return d


# Feature columns used for training (keep in sync with model_engine.py)
FEATURE_COLS = [
    "Close","Open","High","Low","Volume","Change_Pct","Smooth",
    "SMA5","SMA10","SMA20","SMA50","SMA100",
    "EMA5","EMA10","EMA20","EMA50","EMA100",
    "X_5_20","X_20_50","X_E12_26",
    "MACD","MACD_sig","MACD_hist","MACD_cross","MACD_div",
    "RSI","RSI_OB","RSI_OS","RSI_mid","StochRSI","StRSI_K","StRSI_D",
    "BB_U","BB_L","BB_W","BB_P","BB_SQ",
    "ATR","ATR_pct","ATR_trend",
    "Ret1","Ret2","Ret3","Ret5","Ret7","Ret10","Ret14","Ret20",
    "RRaw1","RRaw3","RRaw5","RRaw10",
    "Lag1","Lag2","Lag3","Lag5","Lag7","Lag10",
    "LagR1","LagR2","LagR3","LagR5",
    "HV5","HV20","VIX_like",
    "Body","HL_span","Gap","Bull_bar","Up_wick","Down_wick",
    "VolRatio","VolChg","VolSurge","OBV_sig","VolMA20",
    "P_SMA20","P_SMA50","P_SMA100",
    "Regime","Regime_chg","Wk_trend","Mo_trend","Trend_str",
    "Noise_z",
]
