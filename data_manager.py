"""
data_manager.py v4
─────────────────
Priority chain:
  Crypto  : Binance (1h candles → 41 days intraday OR 1d → 2+ years)
  Stocks  : Yahoo Finance (1d daily)
  Fallback: CryptoCompare → CoinGecko → uploaded CSV

New in v4:
  • get_hourly()  — returns 1h OHLCV for intraday model training
  • get_daily()   — returns 1d OHLCV for longer history
  • merge logic   — uses hourly if crypto, daily for stocks
  • get_live_price() — live single price from Binance
  • CryptoPanic news sentiment fetcher
"""

import os, json, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DATA_DIR = "data"

BINANCE_MAP = {
    "SOL":"SOLUSDT","SOL-USD":"SOLUSDT",
    "BTC":"BTCUSDT","BTC-USD":"BTCUSDT",
    "ETH":"ETHUSDT","ETH-USD":"ETHUSDT",
    "ADA":"ADAUSDT","ADA-USD":"ADAUSDT",
    "BNB":"BNBUSDT","BNB-USD":"BNBUSDT",
    "XRP":"XRPUSDT","XRP-USD":"XRPUSDT",
    "DOGE":"DOGEUSDT","DOGE-USD":"DOGEUSDT",
    "AVAX":"AVAXUSDT","AVAX-USD":"AVAXUSDT",
    "MATIC":"MATICUSDT","MATIC-USD":"MATICUSDT",
    "LINK":"LINKUSDT","LINK-USD":"LINKUSDT",
    "DOT":"DOTUSDT","DOT-USD":"DOTUSDT",
    "LTC":"LTCUSDT","LTC-USD":"LTCUSDT",
}
COINGECKO_MAP = {
    "SOL":"solana","SOL-USD":"solana",
    "BTC":"bitcoin","BTC-USD":"bitcoin",
    "ETH":"ethereum","ETH-USD":"ethereum",
    "ADA":"cardano","DOGE":"dogecoin",
    "XRP":"ripple","LTC":"litecoin",
    "BNB":"binancecoin","AVAX":"avalanche-2",
}
TICKER_INFO = {
    "SOL-USD":  {"name":"Solana",           "type":"crypto"},
    "BTC-USD":  {"name":"Bitcoin",          "type":"crypto"},
    "ETH-USD":  {"name":"Ethereum",         "type":"crypto"},
    "ADA-USD":  {"name":"Cardano",          "type":"crypto"},
    "DOGE-USD": {"name":"Dogecoin",         "type":"crypto"},
    "BNB-USD":  {"name":"BNB",              "type":"crypto"},
    "EMAAR.DFM":{"name":"Emaar Properties", "type":"stock"},
    "AAPL":     {"name":"Apple",            "type":"stock"},
    "TSLA":     {"name":"Tesla",            "type":"stock"},
    "MSFT":     {"name":"Microsoft",        "type":"stock"},
    "NVDA":     {"name":"NVIDIA",           "type":"stock"},
    "AMZN":     {"name":"Amazon",           "type":"stock"},
    "GOOGL":    {"name":"Google",           "type":"stock"},
}

# ── UAE DFM/ADX → Yahoo Finance ticker translation ──────────────────
UAE_YAHOO_MAP = {
    "EMAAR.DFM" : "EMAAR.AE",
    "ENBD.DFM"  : "ENBD.AE",
    "DIB.DFM"   : "DIB.AE",
    "DU.DFM"    : "DU.AE",
    "DEWA.DFM"  : "DEWA.AE",
    "SALIK.DFM" : "SALIK.AE",
    "FAB.ADX"   : "FAB.AE",
    "ALDAR.ADX" : "ALDAR.AE",
    "ADCB.ADX"  : "ADCB.AE",
    "MASQ.DFM"  : "MASQ.AE",
}

HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}
CRYPTO_BASES = set(BINANCE_MAP.keys())


def is_crypto(ticker: str) -> bool:
    t = ticker.upper()
    return t in CRYPTO_BASES or t.replace("-USD","") in CRYPTO_BASES or t.endswith("-USD")


class DataManager:
    def __init__(self, ticker: str = "SOL-USD"):
        self.ticker = ticker.upper().strip()
        safe = self.ticker.replace("/","_").replace(".","_")
        self.data_file  = os.path.join(DATA_DIR, f"{safe}.csv")
        self.meta_file  = os.path.join(DATA_DIR, f"{safe}_meta.json")
        self.h_file     = os.path.join(DATA_DIR, f"{safe}_1h.csv")   # hourly cache
        os.makedirs(DATA_DIR, exist_ok=True)

    # ── Public entry point ──────────────────────────────────────────
    def get_data(self, uploaded_file=None, prefer_hourly: bool = True) -> pd.DataFrame:
        """
        Returns OHLCV DataFrame ready for feature engineering.
        • Crypto:  tries 1h candles first (41 days × 24 = ~984 rows)
                   then tops up with daily to get longer history
        • Stocks:  daily only
        • CSV:     merges with live API top-up
        """
        # 1. CSV upload
        if uploaded_file is not None:
            try:
                df = self._parse_csv(uploaded_file)
                if len(df) >= 80:
                    self._save(df); self._save_meta()
                    return self._clean(df)
            except Exception:
                pass

        cached = self._load_cached()

        # 2. Fetch fresh
        if self._is_stale() or cached is None:
            if is_crypto(self.ticker) and prefer_hourly:
                fresh = self._fetch_hourly_binance()      # 41 days of 1h
                if fresh is not None and len(fresh) >= 100:
                    # Also get daily for longer history (for features like SMA50/200)
                    daily = self._fetch_daily()
                    if daily is not None and len(daily) >= 30:
                        # Combine: daily for old data, hourly for recent 41 days
                        cutoff = fresh.index.min() if hasattr(fresh.index, 'min') else fresh['Date'].min()
                        fresh_clean = self._clean(fresh)
                        daily_clean = self._clean(daily)
                        # Keep daily rows older than hourly data
                        if 'Date' in daily_clean.columns:
                            old_daily = daily_clean[daily_clean['Date'] < cutoff]
                        else:
                            old_daily = daily_clean[daily_clean.index < cutoff]
                        merged = self._merge_df(old_daily, fresh_clean)
                        self._save_hourly(fresh)
                        self._save(merged.reset_index() if isinstance(merged.index, pd.DatetimeIndex) else merged)
                        self._save_meta()
                        return merged
                    self._save_hourly(fresh)
                    self._save_meta()
                    return self._clean(fresh)

            fresh = self._fetch_daily()
            if fresh is not None and len(fresh) >= 80:
                merged = self._merge(cached, fresh)
                self._save(merged); self._save_meta()
                return self._clean(merged)

        if cached is not None and len(cached) >= 80:
            return self._clean(cached)

        raise RuntimeError(
            f"❌ Could not load data for **{self.ticker}**.\n\n"
            "Please upload a CSV or check the ticker symbol."
        )

    # ── Binance 1h candles ──────────────────────────────────────────
    def _fetch_hourly_binance(self) -> pd.DataFrame | None:
        sym = BINANCE_MAP.get(self.ticker, BINANCE_MAP.get(self.ticker.replace("-USD","")))
        if not sym: return None
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                params={"symbol":sym,"interval":"1h","limit":1000},
                headers=HDR, timeout=15)
            if r.status_code != 200: return None
            rows = [{"Date": datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).replace(tzinfo=None),
                     "Open":float(k[1]),"High":float(k[2]),"Low":float(k[3]),
                     "Close":float(k[4]),"Volume":float(k[5])} for k in r.json()]
            df = pd.DataFrame(rows)
            df["Change_Pct"] = df["Close"].pct_change() * 100
            return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        except Exception:
            return None

    # ── Daily fetch (all sources) ───────────────────────────────────
    def _fetch_daily(self):
        t = self.ticker.upper()
        clean = t.replace("-USD","")
        # UAE stocks — route to dedicated multi-source fetcher
        if self.ticker in UAE_YAHOO_MAP:
            return self._yahoo_uae()
        if t in BINANCE_MAP or clean in BINANCE_MAP:
            df = self._binance_daily()
            if df is not None: return df
            df = self._cryptocompare()
            if df is not None: return df
        df = self._yahoo()
        if df is not None: return df
        return self._coingecko()

    def _binance_daily(self):
        sym = BINANCE_MAP.get(self.ticker, BINANCE_MAP.get(self.ticker.replace("-USD","")))
        if not sym: return None
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                params={"symbol":sym,"interval":"1d","limit":1000},
                headers=HDR, timeout=15)
            if r.status_code != 200: return None
            rows = [{"Date":datetime.fromtimestamp(k[0]/1000,tz=timezone.utc).date(),
                     "Open":float(k[1]),"High":float(k[2]),"Low":float(k[3]),
                     "Close":float(k[4]),"Volume":float(k[5])} for k in r.json()]
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Change_Pct"] = df["Close"].pct_change()*100
            return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        except Exception: return None

    def _cryptocompare(self):
        sym = self.ticker.replace("-USD","").upper()
        try:
            r = requests.get("https://min-api.cryptocompare.com/data/v2/histoday",
                params={"fsym":sym,"tsym":"USD","limit":2000}, headers=HDR, timeout=15)
            if r.status_code!=200: return None
            data = r.json()
            if data.get("Response")!="Success": return None
            rows=[{"Date":datetime.fromtimestamp(d["time"],tz=timezone.utc).date(),
                   "Open":float(d["open"]),"High":float(d["high"]),
                   "Low":float(d["low"]),"Close":float(d["close"]),
                   "Volume":float(d.get("volumeto",0))/1e6}
                  for d in data["Data"]["Data"] if d["close"]>0]
            df=pd.DataFrame(rows); df["Date"]=pd.to_datetime(df["Date"])
            df["Change_Pct"]=df["Close"].pct_change()*100
            return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        except Exception: return None

    def _yahoo(self):
        for base in ["https://query1.finance.yahoo.com","https://query2.finance.yahoo.com"]:
            try:
                r = requests.get(f"{base}/v8/finance/chart/{self.ticker}?interval=1d&range=5y",
                    headers=HDR, timeout=15)
                if r.status_code!=200: continue
                res=r.json()["chart"]["result"][0]; ts=res["timestamp"]
                q=res["indicators"]["quote"][0]
                dates=[datetime.fromtimestamp(t,tz=timezone.utc).date() for t in ts]
                df=pd.DataFrame({"Date":dates,"Open":q.get("open",[None]*len(ts)),
                    "High":q.get("high",[None]*len(ts)),"Low":q.get("low",[None]*len(ts)),
                    "Close":q.get("close",[None]*len(ts)),
                    "Volume":[v/1e6 if v else 0 for v in q.get("volume",[0]*len(ts))]})
                df["Date"]=pd.to_datetime(df["Date"]); df=df.dropna(subset=["Close"])
                df["Change_Pct"]=df["Close"].pct_change()*100
                df=df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
                if len(df)>=30: return df
            except Exception: continue
        return None


    def _yahoo_uae(self):
        """Fetch UAE DFM/ADX stocks using yfinance library (handles auth)."""
        yf_ticker = UAE_YAHOO_MAP.get(self.ticker, self.ticker)

        # Method 1: yfinance library (handles Yahoo cookies/crumb automatically)
        try:
            import yfinance as _yf
            _raw = _yf.download(yf_ticker, period="5y", interval="1d",
                                progress=False, auto_adjust=True)
            if _raw is not None and len(_raw) >= 30:
                _raw = _raw.reset_index()
                # Handle MultiIndex columns from yfinance
                if hasattr(_raw.columns, 'levels'):
                    _raw.columns = [c[0] if isinstance(c, tuple) else c for c in _raw.columns]
                df = pd.DataFrame()
                df["Date"]   = pd.to_datetime(_raw.get("Date", _raw.get("Datetime", _raw.index)))
                df["Open"]   = pd.to_numeric(_raw.get("Open",  _raw.get("open",  None)), errors="coerce")
                df["High"]   = pd.to_numeric(_raw.get("High",  _raw.get("high",  None)), errors="coerce")
                df["Low"]    = pd.to_numeric(_raw.get("Low",   _raw.get("low",   None)), errors="coerce")
                df["Close"]  = pd.to_numeric(_raw.get("Close", _raw.get("close", None)), errors="coerce")
                df["Volume"] = pd.to_numeric(_raw.get("Volume",_raw.get("volume",None)), errors="coerce").fillna(0) / 1e6
                df = df.dropna(subset=["Close"])
                df["Change_Pct"] = df["Close"].pct_change() * 100
                df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
                if len(df) >= 30:
                    return df
        except Exception as _e:
            pass  # fall through to raw requests

        # Method 2: Raw requests with session (fallback)
        for base in ["https://query1.finance.yahoo.com",
                     "https://query2.finance.yahoo.com"]:
            try:
                r = requests.get(
                    f"{base}/v8/finance/chart/{yf_ticker}?interval=1d&range=5y",
                    headers=HDR, timeout=15)
                if r.status_code != 200: continue
                result = r.json()["chart"]["result"][0]
                ts     = result["timestamp"]
                q      = result["indicators"]["quote"][0]
                dates  = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in ts]
                df = pd.DataFrame({
                    "Date"  : dates,
                    "Open"  : q.get("open",  [None]*len(ts)),
                    "High"  : q.get("high",  [None]*len(ts)),
                    "Low"   : q.get("low",   [None]*len(ts)),
                    "Close" : q.get("close", [None]*len(ts)),
                    "Volume": [v/1e6 if v else 0
                               for v in q.get("volume", [0]*len(ts))],
                })
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.dropna(subset=["Close"])
                df["Change_Pct"] = df["Close"].pct_change() * 100
                df = (df.sort_values("Date")
                        .drop_duplicates("Date")
                        .reset_index(drop=True))
                if len(df) >= 30:
                    return df
            except Exception:
                continue
        return None


    def _investing_com_uae(self) -> pd.DataFrame | None:
        """
        Fetch DFM/ADX stock data from Investing.com.
        Uses their public chart data endpoint — no API key needed.
        """
        inv_map = {
            "EMAAR.DFM": "2352",   # Investing.com internal ID for Emaar
            "ENBD.DFM" : "28218",
            "DIB.DFM"  : "28221",
            "DU.DFM"   : "28222",
            "DEWA.DFM" : "1192118",
            "SALIK.DFM": "1271890",
            "FAB.ADX"  : "28215",
            "ALDAR.ADX": "28216",
            "ADCB.ADX" : "28219",
            "MASQ.DFM" : "28220",
        }
        inv_id = inv_map.get(self.ticker)
        if not inv_id:
            return None

        # Investing.com chart data endpoint
        headers = {
            "User-Agent"  : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer"     : "https://www.investing.com/",
            "X-Requested-With": "XMLHttpRequest",
            "Accept"      : "application/json, text/javascript, */*; q=0.01",
        }
        try:
            import time as _t
            end_ts   = int(_t.time())
            start_ts = end_ts - 5 * 365 * 24 * 3600  # 5 years back

            r = requests.get(
                f"https://api.investing.com/api/financialdata/{inv_id}/historical/chart/",
                params={
                    "period"    : "MAX",
                    "startDate" : start_ts,
                    "endDate"   : end_ts,
                    "pointscount": 1200,
                },
                headers=headers, timeout=15
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    rows = []
                    for pt in data:
                        # pt = [timestamp_ms, open, high, low, close, volume]
                        try:
                            rows.append({
                                "Date"  : datetime.fromtimestamp(pt[0]/1000, tz=timezone.utc).date(),
                                "Open"  : float(pt[1]),
                                "High"  : float(pt[2]),
                                "Low"   : float(pt[3]),
                                "Close" : float(pt[4]),
                                "Volume": float(pt[5]) / 1e6 if len(pt) > 5 else 1.0,
                            })
                        except Exception:
                            continue
                    if rows:
                        df = pd.DataFrame(rows)
                        df["Date"] = pd.to_datetime(df["Date"])
                        df["Change_Pct"] = df["Close"].pct_change() * 100
                        return (df.sort_values("Date")
                                  .drop_duplicates("Date")
                                  .reset_index(drop=True))
        except Exception:
            pass
        return None

    def _coingecko(self):
        t=self.ticker.replace("-USD","").upper()
        coin=COINGECKO_MAP.get(self.ticker,COINGECKO_MAP.get(t))
        if not coin: return None
        try:
            r=requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                params={"vs_currency":"usd","days":"365","interval":"daily"},
                headers=HDR,timeout=15)
            if r.status_code!=200: return None
            prices=r.json().get("prices",[]); vols=r.json().get("total_volumes",[])
            vmap={ts:v/1e6 for ts,v in vols}
            rows=[{"Date":datetime.fromtimestamp(ts/1000,tz=timezone.utc).date(),
                   "Close":p,"Volume":vmap.get(ts,0)} for ts,p in prices]
            df=pd.DataFrame(rows); df["Date"]=pd.to_datetime(df["Date"])
            df["Open"]=df["Close"].shift(1).fillna(df["Close"])
            df["High"]=df[["Close","Open"]].max(axis=1)*1.015
            df["Low"]=df[["Close","Open"]].min(axis=1)*0.985
            df["Change_Pct"]=df["Close"].pct_change()*100
            return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        except Exception: return None

    # ── CSV parse ───────────────────────────────────────────────────
    def _parse_csv(self, f):
        raw=pd.read_csv(f)
        dc=next((c for c in raw.columns if c.lower() in ["date","time","datetime"]),raw.columns[0])
        for fmt in ["%m/%d/%Y","%Y-%m-%d","%d/%m/%Y",None]:
            try: raw[dc]=pd.to_datetime(raw[dc],format=fmt,errors="raise" if fmt else "coerce"); break
            except Exception: continue
        raw=raw.sort_values(dc).reset_index(drop=True)
        def pv(v):
            s=str(v).strip().replace(",","")
            if "B" in s: return float(s.replace("B",""))*1000
            if "M" in s: return float(s.replace("M",""))
            if "K" in s: return float(s.replace("K",""))/1000
            try: return float(s)
            except: return 1.0
        pc=next((c for c in raw.columns if c.lower() in ["price","close","adj close"]),raw.columns[1])
        df=pd.DataFrame(); df["Date"]=raw[dc]
        df["Close"]=pd.to_numeric(raw[pc].astype(str).str.replace(",",""),errors="coerce")
        for s in ["Open","High","Low"]:
            c=next((x for x in raw.columns if x.lower()==s.lower()),None)
            df[s]=pd.to_numeric(raw[c].astype(str).str.replace(",",""),errors="coerce") if c else df["Close"]
        vc=next((c for c in raw.columns if c.lower() in ["vol.","volume","vol"]),None)
        df["Volume"]=raw[vc].apply(pv) if vc else 1.0
        cc=next((c for c in raw.columns if c.lower() in ["change %","change_pct"]),None)
        df["Change_Pct"]=(raw[cc].astype(str).str.replace("%","").astype(float)
                         if cc else df["Close"].pct_change()*100)
        df.dropna(subset=["Close"],inplace=True)
        return df.reset_index(drop=True)

    # ── Helpers ─────────────────────────────────────────────────────
    def _load_cached(self):
        if os.path.exists(self.data_file):
            try:
                df=pd.read_csv(self.data_file,parse_dates=["Date"])
                return df if len(df)>=30 else None
            except Exception: return None
        return None

    def _merge(self, existing, fresh):
        if existing is None: return fresh
        combined=pd.concat([existing,fresh],ignore_index=True)
        combined["Date"]=pd.to_datetime(combined["Date"])
        return combined.sort_values("Date").drop_duplicates("Date",keep="last").reset_index(drop=True)

    def _merge_df(self, df1, df2):
        """Merge two cleaned DataFrames (DatetimeIndex)."""
        try:
            combined = pd.concat([df1, df2])
            combined = combined[~combined.index.duplicated(keep='last')]
            return combined.sort_index()
        except Exception:
            return df2

    def _save(self, df):
        try: df.to_csv(self.data_file, index=False)
        except Exception: pass

    def _save_hourly(self, df):
        try: df.to_csv(self.h_file, index=False)
        except Exception: pass

    def _save_meta(self):
        try:
            with open(self.meta_file,"w") as f:
                json.dump({"last_updated":datetime.now(timezone.utc).isoformat(),
                           "ticker":self.ticker},f)
        except Exception: pass

    def _is_stale(self):
        if not os.path.exists(self.meta_file): return True
        try:
            with open(self.meta_file) as f: meta=json.load(f)
            if meta.get("ticker")!=self.ticker: return True
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(meta["last_updated"])).total_seconds()
            return age > 5*60    # stale after 5 minutes
        except Exception: return True

    def _clean(self, df) -> pd.DataFrame:
        df=df.copy()
        if "Date" in df.columns:
            df["Date"]=pd.to_datetime(df["Date"])
            df=df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        df["Volume"]=pd.to_numeric(df.get("Volume",1.0),errors="coerce")
        df["Volume"]=df["Volume"].fillna(df["Volume"].rolling(10,min_periods=1).median()).fillna(1.0)
        for col in ["Open","High","Low","Change_Pct"]:
            if col not in df.columns:
                if col=="Open":          df[col]=df["Close"].shift(1).fillna(df["Close"])
                elif col=="High":        df[col]=df["Close"]*1.015
                elif col=="Low":         df[col]=df["Close"]*0.985
                elif col=="Change_Pct":  df[col]=df["Close"].pct_change()*100
        for col in ["Close","Open","High","Low"]:
            df[col]=pd.to_numeric(df[col],errors="coerce")
        df.dropna(subset=["Close"],inplace=True)
        if "Date" in df.columns:
            df.set_index("Date",inplace=True)
        return df

    # ── Static helpers ──────────────────────────────────────────────
    def get_hourly(self):
        """Fetch 1h candles from Binance. Crypto only."""
        sym = BINANCE_MAP.get(self.ticker,
              BINANCE_MAP.get(self.ticker.replace("-USD","")))
        if not sym:
            return None
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                params={"symbol":sym,"interval":"1h","limit":1000},
                headers=HDR, timeout=15)
            if r.status_code != 200:
                return None
            rows = []
            ts_list = []
            for k in r.json():
                ts_list.append(
                    datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).replace(tzinfo=None)
                )
                rows.append({
                    "Open":float(k[1]),"High":float(k[2]),
                    "Low":float(k[3]),"Close":float(k[4]),
                    "Volume":float(k[5]),
                })
            df = pd.DataFrame(rows)
            df.index = pd.DatetimeIndex(ts_list, name="Date")
            df["Change_Pct"] = df["Close"].pct_change() * 100
            df = df.sort_index().drop_duplicates()
            return df if len(df) >= 50 else None
        except Exception:
            return None

    @staticmethod
    def get_live_price(ticker: str) -> float | None:
        # UAE stocks: use Yahoo Finance .AE suffix
        if ticker in UAE_YAHOO_MAP:
            yf_ticker = UAE_YAHOO_MAP[ticker]
            # Try yfinance first (handles auth)
            try:
                import yfinance as _yf2
                _t = _yf2.Ticker(yf_ticker)
                _h = _t.history(period="5d")
                if _h is not None and len(_h) > 0:
                    return float(_h['Close'].iloc[-1])
            except Exception:
                pass
            # Fallback to raw requests
            for base in ["https://query1.finance.yahoo.com",
                         "https://query2.finance.yahoo.com"]:
                try:
                    r = requests.get(
                        f"{base}/v8/finance/chart/{yf_ticker}?interval=1d&range=5d",
                        headers=HDR, timeout=5)
                    if r.status_code == 200:
                        closes = (r.json()["chart"]["result"][0]
                                  ["indicators"]["quote"][0]["close"])
                        p = next((c for c in reversed(closes) if c), None)
                        if p: return float(p)
                except Exception:
                    pass
            return None

        sym = BINANCE_MAP.get(ticker.upper(), BINANCE_MAP.get(ticker.upper().replace("-USD","")))
        if sym:
            try:
                r=requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={sym}",
                    headers=HDR,timeout=5)
                if r.status_code==200: return float(r.json()["price"])
            except Exception: pass
        for base in ["https://query1.finance.yahoo.com","https://query2.finance.yahoo.com"]:
            try:
                r=requests.get(f"{base}/v8/finance/chart/{ticker}?interval=1d&range=1d",
                    headers=HDR,timeout=5)
                if r.status_code==200:
                    closes=r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    p=next((c for c in reversed(closes) if c),None)
                    if p: return float(p)
            except Exception: pass
        return None

    @staticmethod
    def get_ticker_name(ticker: str) -> str:
        return TICKER_INFO.get(ticker,{}).get("name", ticker)

    @staticmethod
    def get_news_sentiment(ticker: str, limit: int = 20) -> list:
        """
        Fetch latest crypto news from CryptoPanic (free, no key needed for public feed).
        Returns list of dicts: {title, url, published, sentiment, score}
        """
        # Map ticker to CryptoPanic currency code
        currency_map = {
            "SOL-USD":"SOL","BTC-USD":"BTC","ETH-USD":"ETH",
            "ADA-USD":"ADA","DOGE-USD":"DOGE","BNB-USD":"BNB",
            "XRP-USD":"XRP","AVAX-USD":"AVAX","MATIC-USD":"MATIC",
            "SOL":"SOL","BTC":"BTC","ETH":"ETH",
        }
        t = ticker.upper()
        currency = currency_map.get(t, t.replace("-USD",""))

        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": "free",
                "currencies"  : currency,
                "kind"        : "news",
                "public"      : "true",
            }
            r = requests.get(url, params=params, headers=HDR, timeout=10)
            if r.status_code != 200:
                return _get_fallback_news(currency)

            items = r.json().get("results", [])
            news  = []
            for item in items[:limit]:
                # CryptoPanic returns votes: positive, negative, important, liked, disliked
                votes = item.get("votes", {})
                pos   = votes.get("positive", 0) or 0
                neg   = votes.get("negative", 0) or 0
                total = pos + neg
                if total > 0:
                    score = (pos - neg) / total   # -1 to +1
                else:
                    score = 0.0

                if score >  0.1: sentiment = "🟢 Positive"
                elif score < -0.1: sentiment = "🔴 Negative"
                else:              sentiment = "⚪ Neutral"

                pub = item.get("published_at","")[:10]
                news.append({
                    "title"    : item.get("title",""),
                    "url"      : item.get("url",""),
                    "source"   : item.get("source",{}).get("title",""),
                    "published": pub,
                    "sentiment": sentiment,
                    "score"    : round(score, 3),
                })
            return news if news else _get_fallback_news(currency)
        except Exception:
            return _get_fallback_news(currency)


def _get_fallback_news(currency: str) -> list:
    """Fallback: fetch from CoinGecko news if CryptoPanic fails."""
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/news",
            headers=HDR, timeout=8)
        if r.status_code == 200:
            items = r.json().get("data", [])[:10]
            news = []
            for item in items:
                news.append({
                    "title"    : item.get("title",""),
                    "url"      : item.get("url",""),
                    "source"   : item.get("news_site",""),
                    "published": item.get("created_at","")[:10],
                    "sentiment": "⚪ Neutral",
                    "score"    : 0.0,
                })
            return news
    except Exception:
        pass
    return []


def get_forexfactory_calendar() -> list:
    """
    Fetch ForexFactory economic calendar for this week.
    Returns list of events sorted by impact (High first).
    Works on Streamlit Cloud with the correct Referer header.
    """
    headers = {
        "User-Agent"  : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer"     : "https://www.forexfactory.com/",
        "Accept"      : "application/json, text/plain, */*",
    }
    urls = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
    ]
    events = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data:
                impact = item.get("impact","")
                # Map impact to emoji + color class
                if impact == "High":
                    impact_icon = "🔴 High"
                elif impact == "Medium":
                    impact_icon = "🟡 Medium"
                elif impact == "Low":
                    impact_icon = "⚪ Low"
                else:
                    impact_icon = "⚫ Holiday"

                events.append({
                    "date"       : item.get("date","")[:10],
                    "time"       : item.get("date","")[11:16] if len(item.get("date","")) > 10 else "",
                    "currency"   : item.get("country",""),
                    "event"      : item.get("title",""),
                    "impact"     : impact_icon,
                    "impact_raw" : impact,
                    "forecast"   : item.get("forecast","—") or "—",
                    "previous"   : item.get("previous","—") or "—",
                    "actual"     : item.get("actual","—") or "—",
                })
        except Exception:
            continue

    # Sort: High impact first, then by date
    order = {"High":0,"Medium":1,"Low":2,"Holiday":3}
    events.sort(key=lambda x: (order.get(x["impact_raw"],4), x["date"], x["time"]))
    return events


def get_combined_news(ticker: str) -> dict:
    """
    Returns combined news from CryptoPanic + ForexFactory.
    {
        "crypto_news": [...],      # from CryptoPanic
        "forex_calendar": [...],   # from ForexFactory
        "sentiment_score": float,  # -1 to +1
    }
    """
    crypto_news  = DataManager.get_news_sentiment(ticker, limit=20)
    forex_cal    = get_forexfactory_calendar()

    # Overall sentiment
    scores = [n.get("score", 0) for n in crypto_news]
    avg_score = sum(scores) / max(len(scores), 1)

    return {
        "crypto_news"    : crypto_news,
        "forex_calendar" : forex_cal,
        "sentiment_score": avg_score,
    }
