"""
DataManager — fetches live SOL/USD daily prices from CoinGecko.
Gets maximum available history. Saves/updates local CSV automatically.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "sol_prices.csv")
META_FILE = os.path.join(DATA_DIR, "meta.json")


class DataManager:

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    # ── Main entry point ────────────────────────────────────────────────────────
    def get_data(self) -> pd.DataFrame:
        existing = self._load_existing()

        if existing is None or self._is_stale(existing):
            fresh = self._fetch_from_api()
            if fresh is not None and len(fresh) > 50:
                merged = self._merge(existing, fresh)
                self._save(merged)
                self._save_meta()
                return self._clean(merged)
            elif existing is not None and len(existing) >= 80:
                return self._clean(existing)
            elif existing is not None:
                return self._clean(existing)
            else:
                raise RuntimeError(
                    "No data available. Could not fetch from CoinGecko. "
                    "Please upload Solana_Historical_Data.csv to your repo in a 'data' folder."
                )

        cleaned = self._clean(existing)
        if len(cleaned) < 80:
            fresh = self._fetch_from_api(days=365)
            if fresh is not None:
                merged = self._merge(existing, fresh)
                self._save(merged)
                return self._clean(merged)

        return cleaned

    # ── Fetch from CoinGecko ────────────────────────────────────────────────────
    def _fetch_from_api(self, days: int = 365):
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SOL-Dashboard/1.0)",
            "Accept"    : "application/json",
        }

        # Method 1: market_chart
        try:
            url    = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
            params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
            resp   = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                data   = resp.json()
                prices = data.get("prices", [])
                vols   = data.get("total_volumes", [])
                if len(prices) > 10:
                    return self._parse_coingecko(prices, vols)
        except Exception:
            pass

        # Method 2: range endpoint
        try:
            now    = int(datetime.now(timezone.utc).timestamp())
            start  = now - (days * 86400)
            url    = "https://api.coingecko.com/api/v3/coins/solana/market_chart/range"
            params = {"vs_currency": "usd", "from": str(start), "to": str(now)}
            resp   = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                data   = resp.json()
                prices = data.get("prices", [])
                vols   = data.get("total_volumes", [])
                if len(prices) > 10:
                    return self._parse_coingecko(prices, vols)
        except Exception:
            pass

        return None

    def _parse_coingecko(self, prices, volumes):
        vol_map = {ts: vol for ts, vol in volumes}
        rows    = []
        for ts, price in prices:
            dt  = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
            vol = vol_map.get(ts, 0) / 1e6
            rows.append({"Date": dt, "Close": round(price, 6), "Volume": round(vol, 4)})

        df = pd.DataFrame(rows)
        df["Date"]       = pd.to_datetime(df["Date"])
        df               = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        df["Open"]       = df["Close"].shift(1).fillna(df["Close"])
        df["High"]       = df[["Close","Open"]].max(axis=1) * 1.015
        df["Low"]        = df[["Close","Open"]].min(axis=1) * 0.985
        df["Change_Pct"] = df["Close"].pct_change() * 100
        return df

    # ── Load existing ───────────────────────────────────────────────────────────
    def _load_existing(self):
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
                if len(df) > 30:
                    return df
            except Exception:
                pass

        for path in ["Solana_Historical_Data.csv",
                     "data/Solana_Historical_Data.csv"]:
            if os.path.exists(path):
                try:
                    return self._parse_uploaded_csv(path)
                except Exception:
                    continue
        return None

    def _parse_uploaded_csv(self, path: str):
        raw = pd.read_csv(path)
        for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]:
            try:
                raw["Date"] = pd.to_datetime(raw["Date"], format=fmt)
                break
            except Exception:
                continue
        else:
            raw["Date"] = pd.to_datetime(raw["Date"], infer_datetime_format=True, errors="coerce")

        raw = raw.sort_values("Date").reset_index(drop=True)

        def parse_vol(v):
            s = str(v).strip()
            if "B" in s: return float(s.replace("B","")) * 1000
            if "M" in s: return float(s.replace("M",""))
            if "K" in s: return float(s.replace("K","")) / 1000
            try:    return float(s)
            except: return 1.0

        price_col = next((c for c in ["Price","Close","price","close"] if c in raw.columns),
                          raw.columns[1])
        df = pd.DataFrame()
        df["Date"]       = raw["Date"]
        df["Close"]      = pd.to_numeric(raw[price_col], errors="coerce")
        df["Open"]       = pd.to_numeric(raw.get("Open",  df["Close"]), errors="coerce")
        df["High"]       = pd.to_numeric(raw.get("High",  df["Close"] * 1.02), errors="coerce")
        df["Low"]        = pd.to_numeric(raw.get("Low",   df["Close"] * 0.98), errors="coerce")

        vol_col = next((c for c in ["Vol.","Volume","vol"] if c in raw.columns), None)
        df["Volume"] = raw[vol_col].apply(parse_vol) if vol_col else 1.0

        chg_col = next((c for c in ["Change %","Change_Pct"] if c in raw.columns), None)
        if chg_col:
            df["Change_Pct"] = raw[chg_col].astype(str).str.replace("%","").astype(float, errors="ignore")
        else:
            df["Change_Pct"] = df["Close"].pct_change() * 100

        df.dropna(subset=["Close"], inplace=True)
        return df.reset_index(drop=True)

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _merge(self, existing, fresh):
        if existing is None:
            return fresh
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        return combined.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    def _save(self, df):
        try: df.to_csv(DATA_FILE, index=False)
        except Exception: pass

    def _save_meta(self):
        try:
            with open(META_FILE, "w") as f:
                json.dump({"last_updated": datetime.now(timezone.utc).isoformat()}, f)
        except Exception: pass

    def _is_stale(self, df) -> bool:
        if len(df) < 80: return True
        if not os.path.exists(META_FILE): return True
        try:
            with open(META_FILE) as f:
                meta = json.load(f)
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(meta["last_updated"])).total_seconds()
            return age > 20 * 3600
        except Exception: return True

    def _clean(self, df) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        df["Volume"] = pd.to_numeric(df.get("Volume", 1.0), errors="coerce")
        df["Volume"] = df["Volume"].fillna(df["Volume"].rolling(10, min_periods=1).median()).fillna(1.0)
        for col in ["Open","High","Low","Change_Pct"]:
            if col not in df.columns:
                if col == "Open":         df[col] = df["Close"].shift(1).fillna(df["Close"])
                elif col == "High":       df[col] = df["Close"] * 1.015
                elif col == "Low":        df[col] = df["Close"] * 0.985
                elif col == "Change_Pct": df[col] = df["Close"].pct_change() * 100
        df.set_index("Date", inplace=True)
        return df
