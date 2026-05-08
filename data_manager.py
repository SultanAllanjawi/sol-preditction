"""
data_manager.py v3 — fetches live OHLCV for any crypto or stock.
Priority: Binance (crypto) → CryptoCompare → Yahoo Finance → CoinGecko → CSV upload
Refreshes every 6 hours so data is always current.
"""
import os, json, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

DATA_DIR = "data"

BINANCE_MAP = {
    "SOL":"SOLUSDT","SOL-USD":"SOLUSDT","BTC":"BTCUSDT","BTC-USD":"BTCUSDT",
    "ETH":"ETHUSDT","ETH-USD":"ETHUSDT","ADA":"ADAUSDT","ADA-USD":"ADAUSDT",
    "BNB":"BNBUSDT","BNB-USD":"BNBUSDT","XRP":"XRPUSDT","XRP-USD":"XRPUSDT",
    "DOGE":"DOGEUSDT","DOGE-USD":"DOGEUSDT","AVAX":"AVAXUSDT","AVAX-USD":"AVAXUSDT",
    "MATIC":"MATICUSDT","LINK":"LINKUSDT","DOT":"DOTUSDT","LTC":"LTCUSDT",
}
COINGECKO_MAP = {
    "SOL":"solana","SOL-USD":"solana","BTC":"bitcoin","BTC-USD":"bitcoin",
    "ETH":"ethereum","ETH-USD":"ethereum","ADA":"cardano","DOGE":"dogecoin",
    "XRP":"ripple","LTC":"litecoin","BNB":"binancecoin","AVAX":"avalanche-2",
}
TICKER_INFO = {
    "SOL-USD":{"name":"Solana","type":"crypto"},
    "BTC-USD":{"name":"Bitcoin","type":"crypto"},
    "ETH-USD":{"name":"Ethereum","type":"crypto"},
    "EMAAR.DFM":{"name":"Emaar Properties","type":"stock"},
    "AAPL":{"name":"Apple","type":"stock"},"TSLA":{"name":"Tesla","type":"stock"},
    "MSFT":{"name":"Microsoft","type":"stock"},"NVDA":{"name":"NVIDIA","type":"stock"},
}
HDR = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":"application/json","Accept-Language":"en-US,en;q=0.9",
    "Referer":"https://finance.yahoo.com/",
}


class DataManager:
    def __init__(self, ticker="SOL-USD"):
        self.ticker = ticker.upper().strip()
        safe = self.ticker.replace("/","_").replace(".","_")
        self.data_file = os.path.join(DATA_DIR, f"{safe}.csv")
        self.meta_file = os.path.join(DATA_DIR, f"{safe}_meta.json")
        os.makedirs(DATA_DIR, exist_ok=True)

    def get_data(self, uploaded_file=None):
        """
        Data loading priority:
        1. If CSV bytes uploaded → parse as historical base
        2. Try to fetch fresh data from API (appends to base)
        3. Fall back to cached data if API fails
        4. Error if nothing available
        """
        base_from_csv = None

        # ── Step 1: Parse uploaded CSV ────────────────────────────────────────
        if uploaded_file is not None:
            try:
                base_from_csv = self._parse_csv(uploaded_file)
                if len(base_from_csv) < 50:
                    base_from_csv = None  # too small, ignore
                else:
                    # Save it so subsequent runs can use it
                    self._save(base_from_csv)
            except Exception as e:
                base_from_csv = None

        # ── Step 2: Load any previously cached data ────────────────────────────
        cached = self._load_cached()

        # Use CSV base if we just parsed one, otherwise fall back to cache
        working_base = base_from_csv if base_from_csv is not None else cached

        # ── Step 3: Fetch fresh data from API (always try to top up) ──────────
        fresh = None
        if self._is_stale() or working_base is None:
            fresh = self._fetch_all()

        # ── Step 4: Merge and return ──────────────────────────────────────────
        if fresh is not None and len(fresh) >= 30:
            merged = self._merge(working_base, fresh)
            if len(merged) >= 80:
                self._save(merged)
                self._save_meta()
                return self._clean(merged)

        # API failed — use what we have
        if working_base is not None and len(working_base) >= 80:
            if fresh is None:  # only update meta if we didn't already
                pass
            return self._clean(working_base)

        # Nothing worked
        ticker_hint = self.ticker
        raise RuntimeError(
            f"❌ Could not load data for **{ticker_hint}**.\n\n"
            f"**What to do:**\n"
            f"1. If this is a Dubai stock (e.g. Emaar): upload a CSV from "
            f"[Investing.com](https://investing.com) → search {ticker_hint} → Historical Data → Download\n"
            f"2. Check the ticker symbol is correct\n"
            f"3. For crypto use format: `SOL-USD`, `BTC-USD`, `ETH-USD`\n"
            f"4. For US stocks: `AAPL`, `TSLA`, `NVDA`\n"
            f"5. For Dubai stocks: `EMAAR.DFM`, `DU.DFM`, `ENBD.DFM`"
        )

    def _fetch_all(self):
        t = self.ticker.upper()
        clean = t.replace("-USD","")
        if t in BINANCE_MAP or clean in BINANCE_MAP:
            df = self._binance(); 
            if df is not None: return df
        if t in COINGECKO_MAP or clean in COINGECKO_MAP:
            df = self._cryptocompare()
            if df is not None: return df
        df = self._yahoo()
        if df is not None: return df
        return self._coingecko()

    def _binance(self):
        t = self.ticker.upper()
        sym = BINANCE_MAP.get(t, BINANCE_MAP.get(t.replace("-USD","")))
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
                params={"fsym":sym,"tsym":"USD","limit":2000},
                headers=HDR, timeout=15)
            if r.status_code!=200: return None
            data = r.json()
            if data.get("Response")!="Success": return None
            rows=[{"Date":datetime.fromtimestamp(d["time"],tz=timezone.utc).date(),
                   "Open":float(d["open"]),"High":float(d["high"]),
                   "Low":float(d["low"]),"Close":float(d["close"]),
                   "Volume":float(d.get("volumeto",0))/1e6}
                  for d in data["Data"]["Data"] if d["close"]>0]
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Change_Pct"] = df["Close"].pct_change()*100
            return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        except Exception: return None

    def _yahoo(self):
        for base in ["https://query1.finance.yahoo.com","https://query2.finance.yahoo.com"]:
            try:
                r = requests.get(f"{base}/v8/finance/chart/{self.ticker}?interval=1d&range=5y",
                    headers=HDR, timeout=15)
                if r.status_code!=200: continue
                res = r.json()["chart"]["result"][0]
                ts  = res["timestamp"]; q=res["indicators"]["quote"][0]
                dates=[datetime.fromtimestamp(t,tz=timezone.utc).date() for t in ts]
                df = pd.DataFrame({
                    "Date":dates,"Open":q.get("open",[None]*len(ts)),
                    "High":q.get("high",[None]*len(ts)),"Low":q.get("low",[None]*len(ts)),
                    "Close":q.get("close",[None]*len(ts)),
                    "Volume":[v/1e6 if v else 0 for v in q.get("volume",[0]*len(ts))],
                })
                df["Date"]=pd.to_datetime(df["Date"])
                df=df.dropna(subset=["Close"])
                df["Change_Pct"]=df["Close"].pct_change()*100
                df=df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
                if len(df)>=30: return df
            except Exception: continue
        return None

    def _coingecko(self):
        t = self.ticker.replace("-USD","").upper()
        coin = COINGECKO_MAP.get(self.ticker, COINGECKO_MAP.get(t))
        if not coin: return None
        try:
            r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                params={"vs_currency":"usd","days":"365","interval":"daily"},
                headers=HDR, timeout=15)
            if r.status_code!=200: return None
            prices=r.json().get("prices",[])
            vols  =r.json().get("total_volumes",[])
            vmap  ={ts:v/1e6 for ts,v in vols}
            rows  =[{"Date":datetime.fromtimestamp(ts/1000,tz=timezone.utc).date(),
                     "Close":p,"Volume":vmap.get(ts,0)} for ts,p in prices]
            df=pd.DataFrame(rows)
            df["Date"]=pd.to_datetime(df["Date"])
            df["Open"]=df["Close"].shift(1).fillna(df["Close"])
            df["High"]=df[["Close","Open"]].max(axis=1)*1.015
            df["Low"] =df[["Close","Open"]].min(axis=1)*0.985
            df["Change_Pct"]=df["Close"].pct_change()*100
            return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        except Exception: return None

    def _parse_csv(self, f):
        raw=pd.read_csv(f)
        dc=next((c for c in raw.columns if c.lower() in ["date","time","datetime"]),raw.columns[0])
        for fmt in ["%m/%d/%Y","%Y-%m-%d","%d/%m/%Y",None]:
            try:
                raw[dc]=pd.to_datetime(raw[dc],format=fmt,errors="raise" if fmt else "coerce"); break
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
        df=pd.DataFrame()
        df["Date"] =raw[dc]
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

    def _load_cached(self):
        if os.path.exists(self.data_file):
            try:
                df=pd.read_csv(self.data_file,parse_dates=["Date"])
                return df if len(df)>=30 else None
            except Exception: return None
        return None

    def _merge(self,existing,fresh):
        if existing is None: return fresh
        combined=pd.concat([existing,fresh],ignore_index=True)
        combined["Date"]=pd.to_datetime(combined["Date"])
        return combined.sort_values("Date").drop_duplicates("Date",keep="last").reset_index(drop=True)

    def _save(self,df):
        try: df.to_csv(self.data_file,index=False)
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
            if meta.get("ticker") != self.ticker: return True
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(meta["last_updated"])).total_seconds()
            return age > 6 * 3600   # refresh every 6 hours
        except Exception: return True

    def _clean(self,df):
        df=df.copy()
        df["Date"]=pd.to_datetime(df["Date"])
        df=df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        df["Volume"]=pd.to_numeric(df.get("Volume",1.0),errors="coerce")
        df["Volume"]=df["Volume"].fillna(df["Volume"].rolling(10,min_periods=1).median()).fillna(1.0)
        for col in ["Open","High","Low","Change_Pct"]:
            if col not in df.columns:
                if col=="Open":         df[col]=df["Close"].shift(1).fillna(df["Close"])
                elif col=="High":       df[col]=df["Close"]*1.015
                elif col=="Low":        df[col]=df["Close"]*0.985
                elif col=="Change_Pct": df[col]=df["Close"].pct_change()*100
        for col in ["Close","Open","High","Low"]:
            df[col]=pd.to_numeric(df[col],errors="coerce")
        df.dropna(subset=["Close"],inplace=True)
        df.set_index("Date",inplace=True)
        return df

    @staticmethod
    def get_live_price(ticker):
        t=ticker.upper()
        sym=BINANCE_MAP.get(t,BINANCE_MAP.get(t.replace("-USD","")))
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
    def get_ticker_name(ticker):
        return TICKER_INFO.get(ticker,{}).get("name",ticker)
