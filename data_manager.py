"""
DataManager — fetches live SOL/USD daily prices from CoinGecko.
Saves to data/sol_prices.csv and appends new rows automatically.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import streamlit as st


DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "sol_prices.csv")
META_FILE = os.path.join(DATA_DIR, "meta.json")


class DataManager:
    """
    Manages the SOL/USD price dataset.
    - First run: downloads full history from CoinGecko (max 365 days free tier)
    - Subsequent runs: only fetches new rows since last update
    - Falls back to uploaded CSV if API is unavailable
    """

    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/solana/market_chart"

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    # ── Public entry point ──────────────────────────────────────────────────────
    def get_data(self) -> pd.DataFrame:
        """
        Returns clean DataFrame with OHLCV columns.
        Auto-refreshes if data is older than 20 hours.
        """
        existing = self._load_existing()

        if existing is None or self._is_stale(existing):
            st.toast("🔄 Fetching latest SOL prices...", icon="📡")
            fresh = self._fetch_from_api()
            if fresh is not None and len(fresh) > 10:
                merged = self._merge(existing, fresh)
                self._save(merged)
                self._save_meta()
                return self._clean(merged)
            elif existing is not None:
                st.toast("⚠️ API unavailable — using cached data", icon="⚠️")
                return self._clean(existing)
            else:
                raise RuntimeError(
                    "No data available. Please upload Solana_Historical_Data.csv "
                    "to the data/ folder or check your internet connection."
                )
        return self._clean(existing)

    # ── Fetch from CoinGecko ────────────────────────────────────────────────────
    def _fetch_from_api(self, days: int = 365) -> pd.DataFrame | None:
        """
        Fetches up to `days` days of daily OHLC data from CoinGecko free API.
        Returns None if request fails.
        """
        params = {
            "vs_currency" : "usd",
            "days"        : str(days),
            "interval"    : "daily",
        }
        headers = {
            "User-Agent": "SOL-Prediction-Dashboard/1.0 (research)",
            "Accept"    : "application/json",
        }

        try:
            resp = requests.get(self.COINGECKO_URL, params=params,
                                headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # CoinGecko returns separate arrays for prices, volumes, market_caps
            prices  = data.get("prices",       [])
            volumes = data.get("total_volumes", [])

            if not prices:
                return None

            rows = []
            for (ts, price), (_, vol) in zip(prices, volumes):
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                rows.append({
                    "Date"  : dt.date(),
                    "Close" : round(price, 6),
                    "Volume": round(vol / 1e6, 4),   # convert to millions
                })

            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

            # CoinGecko daily endpoint only gives Close + Volume
            # Approximate OHLC from close (conservative estimates)
            df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
            df["High"]  = df["Close"] * 1.02   # ~2% intraday range assumption
            df["Low"]   = df["Close"] * 0.98
            df["Change_Pct"] = df["Close"].pct_change() * 100

            return df

        except Exception as e:
            return None

    # ── Load existing CSV ───────────────────────────────────────────────────────
    def _load_existing(self) -> pd.DataFrame | None:
        # Try data/sol_prices.csv first
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
            if len(df) > 30:
                return df

        # Try user-uploaded CSV (Streamlit file uploader or placed in root)
        for path in ["Solana_Historical_Data.csv", "data/Solana_Historical_Data.csv"]:
            if os.path.exists(path):
                return self._parse_uploaded_csv(path)

        return None

    def _parse_uploaded_csv(self, path: str) -> pd.DataFrame:
        """Parse the original Investing.com / manual CSV format."""
        raw = pd.read_csv(path)
        raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y", errors="coerce")
        if raw["Date"].isna().all():
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.sort_values("Date").reset_index(drop=True)

        def parse_vol(v):
            s = str(v).strip()
            if "B" in s: return float(s.replace("B","")) * 1000
            if "M" in s: return float(s.replace("M",""))
            if "K" in s: return float(s.replace("K","")) / 1000
            try:    return float(s)
            except: return np.nan

        df = pd.DataFrame()
        df["Date"]       = raw["Date"]
        df["Close"]      = pd.to_numeric(raw.get("Price", raw.get("Close", raw.iloc[:,1])), errors="coerce")
        df["Open"]       = pd.to_numeric(raw.get("Open",  df["Close"]), errors="coerce")
        df["High"]       = pd.to_numeric(raw.get("High",  df["Close"]*1.02), errors="coerce")
        df["Low"]        = pd.to_numeric(raw.get("Low",   df["Close"]*0.98), errors="coerce")
        vol_col = raw.get("Vol.", raw.get("Volume", None))
        df["Volume"]     = vol_col.apply(parse_vol) if vol_col is not None else 1.0
        df["Change_Pct"] = raw.get("Change %", pd.Series(["0%"]*len(raw))).str.replace("%","").astype(float)
        df.dropna(subset=["Close"], inplace=True)
        return df.reset_index(drop=True)

    # ── Merge old + new ─────────────────────────────────────────────────────────
    def _merge(self, existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
        if existing is None:
            return fresh
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = combined.sort_values("Date").drop_duplicates("Date", keep="last")
        return combined.reset_index(drop=True)

    # ── Save ────────────────────────────────────────────────────────────────────
    def _save(self, df: pd.DataFrame):
        df.to_csv(DATA_FILE, index=False)

    def _save_meta(self):
        meta = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(META_FILE, "w") as f:
            json.dump(meta, f)

    # ── Staleness check ─────────────────────────────────────────────────────────
    def _is_stale(self, df: pd.DataFrame) -> bool:
        if not os.path.exists(META_FILE):
            return True
        with open(META_FILE) as f:
            meta = json.load(f)
        last = datetime.fromisoformat(meta["last_updated"])
        age  = (datetime.now(timezone.utc) - last).total_seconds()
        return age > 20 * 3600   # stale if > 20 hours old

    # ── Clean ───────────────────────────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"]   = pd.to_datetime(df["Date"])
        df           = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        df["Volume"] = df["Volume"].fillna(df["Volume"].rolling(10, min_periods=1).median()).fillna(1.0)
        df.set_index("Date", inplace=True)
        return df
