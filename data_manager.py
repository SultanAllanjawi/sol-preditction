"""
data_manager.py
Fetches OHLCV data for ANY asset:
  - Crypto  : CoinGecko API  (SOL, BTC, ETH)
  - Stocks  : Yahoo Finance  (EMAAR.DFM, AAPL, TSLA, etc.)
  - Fallback: user-uploaded CSV
Auto-saves and updates daily.
"""

import os, json, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DATA_DIR = "data"

# ── Crypto coin ID map (CoinGecko IDs) ────────────────────────────────────────
COINGECKO_IDS = {
    "SOL": "solana", "BTC": "bitcoin", "ETH": "ethereum",
    "SOL-USD": "solana", "BTC-USD": "bitcoin", "ETH-USD": "ethereum",
    "ADA": "cardano", "DOT": "polkadot", "AVAX": "avalanche-2",
    "MATIC": "matic-network", "LINK": "chainlink", "XRP": "ripple",
    "DOGE": "dogecoin", "LTC": "litecoin", "BNB": "binancecoin",
}

# ── Popular tickers info ───────────────────────────────────────────────────────
TICKER_INFO = {
    "SOL-USD"  : {"name": "Solana",          "type": "crypto"},
    "BTC-USD"  : {"name": "Bitcoin",          "type": "crypto"},
    "ETH-USD"  : {"name": "Ethereum",         "type": "crypto"},
    "EMAAR.DFM": {"name": "Emaar Properties", "type": "stock"},
    "AAPL"     : {"name": "Apple",            "type": "stock"},
    "TSLA"     : {"name": "Tesla",            "type": "stock"},
    "MSFT"     : {"name": "Microsoft",        "type": "stock"},
    "AMZN"     : {"name": "Amazon",           "type": "stock"},
    "NVDA"     : {"name": "NVIDIA",           "type": "stock"},
    "GOOGL"    : {"name": "Google",           "type": "stock"},
}


class DataManager:
    def __init__(self, ticker: str = "SOL-USD"):
        self.ticker = ticker.upper().strip()
        self.data_file = os.path.join(DATA_DIR, f"{self.ticker.replace('/', '_')}.csv")
        self.meta_file = os.path.join(DATA_DIR, f"{self.ticker.replace('/', '_')}_meta.json")
        os.makedirs(DATA_DIR, exist_ok=True)

    # ── Public entry point ──────────────────────────────────────────────────────
    def get_data(self, uploaded_file=None) -> pd.DataFrame:
        """
        Returns clean OHLCV DataFrame.
        Priority: uploaded_file > cached CSV > live API fetch
        """
        # 1. User uploaded a CSV → use it directly
        if uploaded_file is not None:
            try:
                df = self._parse_uploaded_csv(uploaded_file)
                if len(df) >= 80:
                    self._save(df)
                    self._save_meta()
                    return self._clean(df)
            except Exception as e:
                pass  # fall through to API

        # 2. Load existing cached data
        existing = self._load_cached()

        # 3. If stale or missing → fetch from API
        if existing is None or self._is_stale():
            fresh = self._fetch(days=730)  # fetch 2 years
            if fresh is not None and len(fresh) >= 80:
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
                    f"Could not fetch data for **{self.ticker}**.\n\n"
                    f"Please either:\n"
                    f"1. Check the ticker symbol is correct (e.g. EMAAR.DFM, SOL-USD, AAPL)\n"
                    f"2. Upload a CSV file using the uploader in the sidebar"
                )

        return self._clean(existing)

    # ── Fetch from APIs ─────────────────────────────────────────────────────────
    def _fetch(self, days: int = 730):
        ticker_upper = self.ticker.upper().replace("-USD", "")

        # Try CoinGecko for crypto
        if ticker_upper in COINGECKO_IDS or self.ticker in COINGECKO_IDS:
            result = self._fetch_coingecko(days)
            if result is not None:
                return result

        # Try Yahoo Finance for everything (stocks + crypto)
        result = self._fetch_yahoo(days)
        if result is not None:
            return result

        return None

    def _fetch_coingecko(self, days: int):
        coin_id = COINGECKO_IDS.get(
            self.ticker.upper().replace("-USD",""),
            COINGECKO_IDS.get(self.ticker.upper())
        )
        if not coin_id:
            return None

        headers = {"User-Agent": "Mozilla/5.0 (compatible; AssetDashboard/2.0)"}
        url     = "https://api.coingecko.com/api/v3/coins/{}/market_chart".format(coin_id)
        params  = {"vs_currency": "usd", "days": str(min(days, 365)), "interval": "daily"}

        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code != 200:
                return None
            data    = r.json()
            prices  = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            if len(prices) < 10:
                return None

            vol_map = {ts: v for ts, v in volumes}
            rows    = []
            for ts, price in prices:
                dt  = datetime.fromtimestamp(ts/1000, tz=timezone.utc).date()
                vol = vol_map.get(ts, 0) / 1e6
                rows.append({"Date": dt, "Close": price, "Volume": vol})

            df = pd.DataFrame(rows)
            df["Date"]   = pd.to_datetime(df["Date"])
            df           = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
            df["Open"]   = df["Close"].shift(1).fillna(df["Close"])
            df["High"]   = df[["Close","Open"]].max(axis=1) * 1.015
            df["Low"]    = df[["Close","Open"]].min(axis=1) * 0.985
            df["Change_Pct"] = df["Close"].pct_change() * 100
            return df
        except Exception:
            return None

    def _fetch_yahoo(self, days: int):
        """Fetch from Yahoo Finance v8 API directly (no library needed)."""
        period = f"{min(days, 730)}d"
        headers = {
            "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept"     : "application/json,text/html,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer"    : "https://finance.yahoo.com/",
        }

        urls = [
            f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}?interval=1d&range={period}",
            f"https://query2.finance.yahoo.com/v8/finance/chart/{self.ticker}?interval=1d&range={period}",
        ]

        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    continue

                chart = r.json()["chart"]["result"][0]
                ts    = chart["timestamp"]
                q     = chart["indicators"]["quote"][0]
                dates = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in ts]

                df = pd.DataFrame({
                    "Date"  : dates,
                    "Open"  : q.get("open",  [None]*len(ts)),
                    "High"  : q.get("high",  [None]*len(ts)),
                    "Low"   : q.get("low",   [None]*len(ts)),
                    "Close" : q.get("close", [None]*len(ts)),
                    "Volume": q.get("volume",[None]*len(ts)),
                })
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.dropna(subset=["Close"])
                df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

                for col in ["Open","High","Low"]:
                    df[col] = df[col].fillna(df["Close"])
                df["Volume"]     = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) / 1e6
                df["Change_Pct"] = df["Close"].pct_change() * 100

                if len(df) >= 30:
                    return df
            except Exception:
                continue

        # Fallback: try yfinance library if installed
        try:
            import yfinance as yf
            raw = yf.download(self.ticker, period=f"{days}d",
                              interval="1d", progress=False, auto_adjust=True)
            if len(raw) >= 30:
                raw = raw.reset_index()
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                raw.rename(columns={"Date":"Date","Open":"Open","High":"High",
                                    "Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
                raw["Volume"]     = raw["Volume"] / 1e6
                raw["Change_Pct"] = raw["Close"].pct_change() * 100
                return raw[["Date","Open","High","Low","Close","Volume","Change_Pct"]].reset_index(drop=True)
        except Exception:
            pass

        return None

    # ── Parse uploaded CSV ──────────────────────────────────────────────────────
    def _parse_uploaded_csv(self, file_obj) -> pd.DataFrame:
        """Parse any CSV format — Investing.com, Yahoo Finance, manual."""
        raw = pd.read_csv(file_obj)

        # Try to find Date column
        date_col = next((c for c in raw.columns
                        if c.lower() in ["date","time","datetime","timestamp"]), raw.columns[0])

        # Parse dates with multiple format attempts
        for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", None]:
            try:
                raw[date_col] = pd.to_datetime(raw[date_col],
                    format=fmt, errors="raise" if fmt else "coerce")
                break
            except Exception:
                continue

        raw = raw.sort_values(date_col).reset_index(drop=True)

        def parse_vol(v):
            s = str(v).strip().replace(",","")
            if "B" in s: return float(s.replace("B","")) * 1000
            if "M" in s: return float(s.replace("M",""))
            if "K" in s: return float(s.replace("K","")) / 1000
            try:    return float(s)
            except: return 1.0

        # Find price column
        price_col = next((c for c in raw.columns
                         if c.lower() in ["price","close","adj close","adj_close"]),
                         raw.columns[1])

        df = pd.DataFrame()
        df["Date"]  = raw[date_col]
        df["Close"] = pd.to_numeric(raw[price_col].astype(str).str.replace(",",""),
                                    errors="coerce")

        for src, dst in [("Open","Open"),("High","High"),("Low","Low")]:
            col = next((c for c in raw.columns if c.lower()==src.lower()), None)
            df[dst] = pd.to_numeric(raw[col].astype(str).str.replace(",",""),
                                    errors="coerce") if col else df["Close"]

        vol_col = next((c for c in raw.columns
                       if c.lower() in ["vol.","volume","vol"]), None)
        df["Volume"] = raw[vol_col].apply(parse_vol) if vol_col else 1.0

        chg_col = next((c for c in raw.columns
                       if c.lower() in ["change %","change_pct","chg%"]), None)
        df["Change_Pct"] = (raw[chg_col].astype(str).str.replace("%","").astype(float)
                           if chg_col else df["Close"].pct_change()*100)

        df.dropna(subset=["Close"], inplace=True)
        return df.reset_index(drop=True)

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _load_cached(self):
        if os.path.exists(self.data_file):
            try:
                df = pd.read_csv(self.data_file, parse_dates=["Date"])
                return df if len(df) >= 30 else None
            except Exception:
                return None
        return None

    def _merge(self, existing, fresh):
        if existing is None:
            return fresh
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        return (combined.sort_values("Date")
                        .drop_duplicates("Date", keep="last")
                        .reset_index(drop=True))

    def _save(self, df):
        try: df.to_csv(self.data_file, index=False)
        except Exception: pass

    def _save_meta(self):
        try:
            with open(self.meta_file, "w") as f:
                json.dump({"last_updated": datetime.now(timezone.utc).isoformat()}, f)
        except Exception: pass

    def _is_stale(self) -> bool:
        if not os.path.exists(self.meta_file):
            return True
        try:
            with open(self.meta_file) as f:
                meta = json.load(f)
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(meta["last_updated"])).total_seconds()
            return age > 20 * 3600   # stale after 20 hours
        except Exception:
            return True

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
        df["Close"]  = pd.to_numeric(df["Close"], errors="coerce")
        df["Open"]   = pd.to_numeric(df["Open"],  errors="coerce")
        df["High"]   = pd.to_numeric(df["High"],  errors="coerce")
        df["Low"]    = pd.to_numeric(df["Low"],   errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        df.set_index("Date", inplace=True)
        return df

    @staticmethod
    def get_ticker_name(ticker: str) -> str:
        return TICKER_INFO.get(ticker, {}).get("name", ticker)
