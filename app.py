"""
SOL/USD (and any asset) Prediction Dashboard v3
- Multi-asset: type any ticker
- Live data from Binance / Yahoo Finance / CryptoCompare
- All 5 DL models (if TF available) + GB ensemble
- Proper TP / SL based on ATR
- Auto-updates every 6 hours
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_squared_error, mean_absolute_error,
)

st.set_page_config(
    page_title="Asset Prediction Dashboard",
    page_icon="🔮", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp{background:#0D1117;color:#C9D1D9}
.block-container{padding-top:1.2rem}
[data-testid="metric-container"]{background:#161B22;border:1px solid #30363D;border-radius:10px;padding:14px 18px}
[data-testid="stMetricValue"]{color:#F0F6FC;font-size:1.7rem!important}
[data-testid="stMetricLabel"]{color:#8B949E;font-size:0.82rem}
.signal-card{background:#161B22;border:1px solid #30363D;border-radius:12px;padding:18px 22px;margin:6px 0}
.buy-card{border-left:5px solid #3FB950}.sell-card{border-left:5px solid #F85149}.hold-card{border-left:5px solid #6E7681}
hr{border-color:#30363D}
[data-testid="stSidebar"]{background:#161B22;border-right:1px solid #30363D}
.stTabs [data-baseweb="tab-list"]{background:#161B22}
.stTabs [data-baseweb="tab"]{color:#8B949E}
.stTabs [aria-selected="true"]{color:#F0F6FC;border-bottom:2px solid #58A6FF}
</style>""", unsafe_allow_html=True)

C_UP='#3FB950';C_DOWN='#F85149';C_BLUE='#58A6FF'
C_GOLD='#E3B341';C_GREY='#6E7681';C_WHITE='#F0F6FC';C_DIM='#8B949E'



# ── TradingView symbol map ─────────────────────────────────────────────────────
_TV_MAP = {
    # Crypto
    "SOL-USD":"BINANCE:SOLUSDT","BTC-USD":"BINANCE:BTCUSDT",
    "ETH-USD":"BINANCE:ETHUSDT","ADA-USD":"BINANCE:ADAUSDT",
    "BNB-USD":"BINANCE:BNBUSDT","XRP-USD":"BINANCE:XRPUSDT",
    "DOGE-USD":"BINANCE:DOGEUSDT","AVAX-USD":"BINANCE:AVAXUSDT",
    "MATIC-USD":"BINANCE:MATICUSDT","LINK-USD":"BINANCE:LINKUSDT",
    # US Stocks
    "AAPL":"NASDAQ:AAPL","TSLA":"NASDAQ:TSLA","MSFT":"NASDAQ:MSFT",
    "NVDA":"NASDAQ:NVDA","AMZN":"NASDAQ:AMZN","GOOGL":"NASDAQ:GOOGL",
    # Commodities & Indices
    "GC=F":"OANDA:XAUUSD","SI=F":"OANDA:XAGUSD",
    "SPY":"AMEX:SPY","QQQ":"NASDAQ:QQQ",
    # UAE — verified TradingView symbols
    "EMAAR.DFM":"DFM:EMAAR",
    "ENBD.DFM":"DFM:EMIRATESNBD",
    "DIB.DFM":"DFM:DIB",
    "DU.DFM":"DFM:DU",
    "DEWA.DFM":"DFM:DEWA",
    "SALIK.DFM":"DFM:SALIK",
    "FAB.ADX":"ADX:FAB",
    "ALDAR.ADX":"ADX:ALDAR",
    "ADCB.ADX":"ADX:ADCB",
    "MASQ.DFM":"DFM:MASQ",
}
def get_tv_symbol(t):
    t = t.upper()
    if t in _TV_MAP: return _TV_MAP[t]
    if t.endswith("-USD"): return f"BINANCE:{t.replace('-USD','USDT')}"
    return t

from data_manager import DataManager
from model_engine import ModelEngine
from persistence import (
    load_portfolio, save_portfolio,
    load_active_signals, update_signal_status,
    load_settings, save_settings,
    save_uploaded_csv, load_uploaded_csv,
    load_all_uploaded_csvs, delete_uploaded_csv,
    load_closed_signals, save_closed_signal,
)

# ── Crypto detection (self-contained, no data_manager dependency) ──
_CRYPTO_SET = {
    "SOL","SOL-USD","BTC","BTC-USD","ETH","ETH-USD","ADA","ADA-USD",
    "BNB","BNB-USD","XRP","XRP-USD","DOGE","DOGE-USD","AVAX","AVAX-USD",
    "MATIC","MATIC-USD","LINK","LINK-USD","DOT","DOT-USD","LTC","LTC-USD",
    "UNI","ATOM","FIL","SHIB","SHIB-USD",
}
def is_crypto(t: str) -> bool:
    t = t.upper()
    return t in _CRYPTO_SET or t.replace("-USD","") in _CRYPTO_SET or t.endswith("-USD")
from feature_engine import build_features

def dark_fig():
    plt.rcParams.update({
        'figure.facecolor':'#0D1117','axes.facecolor':'#161B22',
        'axes.edgecolor':'#30363D','axes.labelcolor':'#C9D1D9',
        'xtick.color':'#8B949E','ytick.color':'#8B949E','text.color':'#C9D1D9',
        'grid.color':'#21262D','grid.linewidth':0.5,'legend.facecolor':'#161B22',
        'legend.edgecolor':'#30363D','legend.fontsize':8,'font.size':9,
    })

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
# ── Session state: init from saved settings ──────────────────────
_saved_settings = load_settings()
if "uploaded_assets" not in st.session_state:
    try:
        _saved_csvs = load_all_uploaded_csvs()
        st.session_state.uploaded_assets = _saved_csvs if _saved_csvs else {}
    except Exception:
        st.session_state.uploaded_assets = {}
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = _saved_settings.get("last_ticker","SOL-USD")
if "portfolio_trades" not in st.session_state:
    st.session_state.portfolio_trades = load_portfolio()   # load from disk
if "chart_tf" not in st.session_state:
    st.session_state.chart_tf = _saved_settings.get("chart_tf","D")
if "chart_lookback" not in st.session_state:
    st.session_state.chart_lookback = _saved_settings.get("lookback_days",90)

with st.sidebar:
    st.markdown("## 🔮 Prediction Dashboard")
    st.caption("Auto-updates every 30 minutes")
    st.divider()

    # ── Upload CSV ──────────────────────────────────────────────────
    st.subheader("📁 Upload CSV Data")

    # Upload mode selector
    _umode = st.radio("Upload for:", ["🇦🇪 UAE / DFM Stock", "📈 Other Asset"],
                      horizontal=True, key="upload_mode")

    uploaded = st.file_uploader(
        "Drop CSV here (Investing.com or Yahoo Finance format)",
        type=["csv"], key="csv_uploader"
    )

    if uploaded is not None:
        if _umode == "🇦🇪 UAE / DFM Stock":
            # Show dropdown of DFM stocks — user just picks which one
            _DFM_PICK = {
                "Emaar Properties"    : "EMAAR.DFM",
                "Emirates NBD"        : "ENBD.DFM",
                "Dubai Islamic Bank"  : "DIB.DFM",
                "du Telecom"          : "DU.DFM",
                "Dubai Electricity"   : "DEWA.DFM",
                "Salik"               : "SALIK.DFM",
                "First Abu Dhabi Bank": "FAB.ADX",
                "Aldar Properties"    : "ALDAR.ADX",
                "ADCB Bank"           : "ADCB.ADX",
                "Mashreq Bank"        : "MASQ.DFM",
            }
            _picked = st.selectbox(
                "Which stock is this CSV for?",
                list(_DFM_PICK.keys()),
                key="dfm_csv_pick"
            )
            _ticker_for_csv = _DFM_PICK[_picked]
            st.caption(f"Will be saved as: **{_ticker_for_csv}**")

            if st.button("✅ Apply to " + _picked, use_container_width=True,
                         type="primary", key="dfm_csv_add"):
                _bytes = uploaded.read()
                st.session_state.uploaded_assets[_ticker_for_csv] = _bytes
                st.session_state.selected_ticker = _ticker_for_csv
                # Save CSV to disk — persists across restarts
                save_uploaded_csv(_ticker_for_csv, _bytes)
                save_settings({**load_settings(), "last_ticker": _ticker_for_csv})
                st.success(f"✅ Saved! {_picked} ({_ticker_for_csv}) — data saved permanently 💾")
                st.rerun()

        else:
            # Other asset — user types ticker manually
            _col1, _col2 = st.columns([2, 1])
            with _col1:
                csv_ticker = st.text_input(
                    "Asset ticker",
                    placeholder="e.g. SOL-USD, AAPL, TSLA",
                    key="csv_ticker_input"
                ).upper().strip()
            with _col2:
                st.write(""); st.write("")
                if st.button("➕ Add", use_container_width=True, key="other_csv_add") and csv_ticker:
                    _obytes = uploaded.read()
                    st.session_state.uploaded_assets[csv_ticker] = _obytes
                    st.session_state.selected_ticker = csv_ticker
                    save_uploaded_csv(csv_ticker, _obytes)
                    st.success(f"✅ Saved! {csv_ticker} — data saved permanently 💾")
                    st.rerun()

    # Show what's uploaded — with persistent save indicator
    if st.session_state.uploaded_assets:
        st.caption("💾 These CSVs are saved permanently — survive app restarts")
        _ul = list(st.session_state.uploaded_assets.keys())
        for _ut in _ul:
            _uc1, _uc2 = st.columns([3,1])
            _uc1.caption(f"💾 {_ut}")
            if _uc2.button("✕", key=f"del_{_ut}", use_container_width=True):
                del st.session_state.uploaded_assets[_ut]
                delete_uploaded_csv(_ut)
                st.rerun()

    st.divider()

    # ── Asset Selection — base list + any uploaded CSVs ────────────
    st.subheader("📊 Asset Selection")

    _CRYPTO  = ["SOL-USD","BTC-USD","ETH-USD","ADA-USD","DOGE-USD","BNB-USD","AVAX-USD","XRP-USD"]
    _US      = ["AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL"]
    _INDICES = ["GC=F","SI=F","SPY","QQQ"]   # Gold, Silver, S&P500, Nasdaq
    _UAE     = ["EMAAR.DFM","ENBD.DFM","DIB.DFM","DU.DFM","DEWA.DFM",
                "SALIK.DFM","FAB.ADX","ALDAR.ADX","ADCB.ADX","MASQ.DFM"]
    _UAE_NAMES = {
        "EMAAR.DFM":"Emaar Properties","ENBD.DFM":"Emirates NBD",
        "DIB.DFM":"Dubai Islamic Bank","DU.DFM":"du Telecom",
        "DEWA.DFM":"Dubai Electricity","SALIK.DFM":"Salik",
        "FAB.ADX":"First Abu Dhabi Bank","ALDAR.ADX":"Aldar Properties",
        "ADCB.ADX":"ADCB Bank","MASQ.DFM":"Mashreq Bank",
        # Commodities & Indices
        "GC=F":"Gold (Futures)","SI=F":"Silver (Futures)",
        "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF",
    }
    BASE_ASSETS = _CRYPTO + _US + _INDICES + _UAE

    _cat = st.radio("Category",
        ["🔵 Crypto","🟢 US Stocks","📊 Commodities & Indices","🇦🇪 UAE / DFM","📁 Uploaded","✏️ Custom"],
        horizontal=False, key="asset_category")

    if   _cat == "🔵 Crypto":                _asset_list = _CRYPTO
    elif _cat == "🟢 US Stocks":             _asset_list = _US
    elif _cat == "📊 Commodities & Indices": _asset_list = _INDICES
    elif _cat == "🇦🇪 UAE / DFM":            _asset_list = _UAE
    elif _cat == "📁 Uploaded":              _asset_list = list(st.session_state.uploaded_assets.keys()) or ["(none)"]
    else:                                     _asset_list = []

    if _cat == "✏️ Custom":
        all_assets = ["✏️ Custom ticker..."]
    else:
        all_assets = _asset_list + (["✏️ Custom ticker..."] if _cat != "📁 Uploaded" else [])

    # Display names for UAE stocks
    _disp = {t: (f"{_UAE_NAMES[t]} ({t})" if t in _UAE_NAMES else t) for t in all_assets}
    _disp_opts = [_disp.get(t, t) for t in all_assets]

    # Merge uploaded into list
    all_assets_full = BASE_ASSETS + [k for k in st.session_state.uploaded_assets if k not in BASE_ASSETS]

    # Default index
    _def_idx = 0
    if st.session_state.selected_ticker in all_assets:
        _def_idx = all_assets.index(st.session_state.selected_ticker)

    if _cat == "✏️ Custom":
        ticker = st.text_input(
            "Enter ticker symbol",
            placeholder="e.g. SOL-USD, EMAAR.DFM, AAPL",
        ).upper().strip() or "SOL-USD"
    elif _cat == "📁 Uploaded" and not st.session_state.uploaded_assets:
        st.caption("No uploads yet — use the uploader above")
        ticker = st.session_state.selected_ticker
    else:
        _sel_disp = st.selectbox("Select asset", _disp_opts, index=_def_idx)
        _rev_map  = {v: k for k, v in _disp.items()}
        ticker    = _rev_map.get(_sel_disp, _sel_disp)

    # Keep session state in sync + save to disk
    if ticker != st.session_state.selected_ticker:
        st.session_state.selected_ticker = ticker
        save_settings({**_saved_settings, "last_ticker": ticker})

    # Show source badge
    if ticker in st.session_state.uploaded_assets:
        st.success(f"📂 Using uploaded CSV for **{ticker}**")
    elif ticker in _INDICES:
        _idx_names = {"GC=F":"Gold","SI=F":"Silver","SPY":"S&P 500","QQQ":"Nasdaq 100"}
        st.info(f"📡 **{_idx_names.get(ticker,ticker)}** · Yahoo Finance · Updates every 30 min")
    elif ticker in _UAE:
        st.info("📡 **UAE / DFM** · Auto-fetching from Yahoo Finance · No CSV needed")
        st.caption("Auto-updates every 6h · Sources: yfinance → Stooq → Alpha Vantage")
    else:
        _src = "Binance" if ticker.replace("-USD","").upper() in ["SOL","BTC","ETH","ADA","DOGE","BNB","AVAX","XRP","LTC"] else "Yahoo Finance"
        st.info(f"📡 Auto-fetching from **{_src}** · Updates every 30 min")

    st.divider()
    st.subheader("⚙️ Signal Settings")

    signal_mode = st.radio(
        "Signal Mode",
        ["📅 Daily", "⚡ Intraday (1h)"],
        index=0,
        horizontal=True, key="signal_mode",
        help="Daily: ~73% accuracy · 1 signal/day · recommended\nIntraday: ~68% accuracy · multiple signals/day · more noise"
    )
    if st.session_state.get("signal_mode","📅 Daily") == "⚡ Intraday (1h)":
        st.caption("⚡ Intraday: ~68% accuracy (hourly noise). Daily = ~73%")

    confidence_thresh = st.slider(
        "Signal Confidence Threshold", 0.50, 0.80, 0.60, 0.01,
        help="Higher = fewer but more reliable signals"
    )
    lookback_days = st.selectbox("Chart lookback", [30,60,90,120,180,365], index=2)

    st.divider()
    force_refresh = st.button("🔄 Force Refresh Data", use_container_width=True)
    if force_refresh:
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every 5 min · Force refresh clears all cache")
    st.divider()

    st.divider()
    st.caption("⚠️ Research only · Not financial advice")

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA + TRAIN
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)  # 30-min cache
def load_and_train(ticker, _uploaded_bytes=None, _force=False,
                   signal_mode="Daily", sentiment_score=0.0):
    import io
    dm       = DataManager(ticker)
    file_obj = io.BytesIO(_uploaded_bytes) if _uploaded_bytes else None

    if signal_mode == "⚡ Intraday (1h)" and is_crypto(ticker):
        try:
            df_raw = dm.get_hourly()
        except AttributeError:
            df_raw = None
        if df_raw is None or len(df_raw) < 100:
            df_raw = dm.get_data(uploaded_file=file_obj)
    else:
        try:
            df_raw = dm.get_data(uploaded_file=file_obj, prefer_hourly=False)
        except TypeError:
            df_raw = dm.get_data(uploaded_file=file_obj)

    try:
        df_feat = build_features(df_raw, sentiment_score=sentiment_score)
    except TypeError:
        df_feat = build_features(df_raw)
    engine   = ModelEngine(df_feat)
    results  = engine.train(verbose=False, sentiment_score=sentiment_score, is_crypto=is_crypto(ticker))
    return df_raw, df_feat, results

# Get uploaded bytes for the selected ticker (if any)
_uploaded_bytes = st.session_state.uploaded_assets.get(ticker, None)

with st.spinner(f"⏳ Loading **{ticker}** · First load ~8s · Cached for 30 min after..."):
    try:
        # Fetch sentiment for crypto assets
        _sentiment = 0.0
        if is_crypto(ticker):
            try:
                _news = DataManager.get_news_sentiment(ticker, limit=10)
                if _news:
                    _sentiment = sum(n.get("score",0) for n in _news) / len(_news)
            except Exception:
                pass

        df_raw, df_feat, results = load_and_train(
            ticker, _uploaded_bytes, force_refresh,
            signal_mode=st.session_state.get("signal_mode","📅 Daily"),
            sentiment_score=_sentiment)
        # ── Toast alert when signal changes ─────────────────────────
        _prev_sig_key = f"prev_signal_{ticker}"
        _cur_sig      = results.get("last_signal","HOLD")
        _prev_sig     = st.session_state.get(_prev_sig_key, None)
        if _prev_sig is not None and _cur_sig != _prev_sig and _cur_sig != "HOLD":
            _sig_icon = "🟢" if _cur_sig == "BUY" else "🔴"
            _conf_val = results.get("last_confidence", 0)
            st.toast(
                f"{_sig_icon} **{ticker} signal changed: {_cur_sig}** "
                f"({_conf_val:.1f}% confidence)",
                icon=_sig_icon
            )
        st.session_state[_prev_sig_key] = _cur_sig

    except Exception as e:
        _uae_list = ["EMAAR.DFM","ENBD.DFM","DIB.DFM","DU.DFM","DEWA.DFM",
                      "SALIK.DFM","FAB.ADX","ALDAR.ADX","ADCB.ADX","MASQ.DFM"]
        _is_uae_err = ticker in _uae_list
        st.markdown(f"""
<div style="background:#2D1B1B;border:2px solid #F85149;border-radius:10px;padding:20px 24px;margin:20px 0">
  <div style="color:#F85149;font-size:1.1rem;font-weight:bold;margin-bottom:8px">
    ❌ Error Loading Data for {ticker}
  </div>
  <div style="color:#F0F6FC;font-size:0.88rem">{str(e).split(chr(10))[0]}</div>
</div>""", unsafe_allow_html=True)
        if _is_uae_err:
            # Build direct download links for this specific stock
            _yf_sym  = {"EMAAR.DFM":"EMAAR.AE","ENBD.DFM":"ENBD.AE","DIB.DFM":"DIB.AE",
                        "DU.DFM":"DU.AE","DEWA.DFM":"DEWA.AE","SALIK.DFM":"SALIK.AE",
                        "FAB.ADX":"FAB.AE","ALDAR.ADX":"ALDAR.AE",
                        "ADCB.ADX":"ADCB.AE","MASQ.DFM":"MASQ.AE"}.get(ticker, ticker)
            _inv_sl  = {"EMAAR.DFM":"emaar-properties","ENBD.DFM":"emirates-nbd",
                        "DIB.DFM":"dubai-islamic-bank","DU.DFM":"emirates-integrated-telecom",
                        "DEWA.DFM":"dubai-electricity-water","SALIK.DFM":"salik-pjsc",
                        "FAB.ADX":"first-abu-dhabi-bank","ALDAR.ADX":"aldar-properties",
                        "ADCB.ADX":"abu-dhabi-commercial-bank","MASQ.DFM":"mashreqbank"}.get(ticker,"")
            _yf_csv  = f"https://query1.finance.yahoo.com/v7/finance/download/{_yf_sym}?interval=1d&range=5y&events=history"
            _inv_url = f"https://www.investing.com/equities/{_inv_sl}-historical-data"

            st.markdown(f"""
<div style="background:#1C2128;border:1px solid #E3B341;border-radius:8px;padding:16px 20px">
  <div style="color:#E3B341;font-weight:bold;margin-bottom:10px">
    🇦🇪 {ticker} — Auto-fetch failed
  </div>
  <div style="color:#C9D1D9;font-size:0.88rem;line-height:1.9">
    Auto-fetch via yfinance/Stooq is not returning data for this stock.<br>
    <b>Quick fix — download CSV manually (30 seconds):</b><br><br>
    <b>Option 1 — Yahoo Finance:</b><br>
    &nbsp;&nbsp;→ <a href="{_yf_csv}" target="_blank" 
      style="color:#58A6FF">Click to download {_yf_sym} CSV directly</a><br><br>
    <b>Option 2 — Investing.com:</b><br>
    &nbsp;&nbsp;→ <a href="{_inv_url}" target="_blank"
      style="color:#58A6FF">Open {ticker} historical data on Investing.com</a>
      → click Download<br><br>
    Then upload via sidebar → <b>🇦🇪 UAE / DFM Stock</b> → select stock → Apply
  </div>
</div>""", unsafe_allow_html=True)
            col_r1, col_r2 = st.columns(2)
            if col_r1.button("🔄 Retry auto-fetch", type="primary", key="dfm_retry"):
                st.cache_data.clear()
                st.rerun()
            col_r2.markdown(
                f'<a href="{_yf_csv}" target="_blank" style="display:inline-block;'
                f'background:#238636;color:white;border-radius:5px;padding:8px 16px;'
                f'text-decoration:none;font-size:0.85rem;margin-top:4px">'
                f'⬇️ Download CSV for {_yf_sym}</a>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
<div style="background:#1C2128;border:1px solid #30363D;border-radius:8px;padding:16px 20px">
  <div style="color:#E3B341;font-weight:bold;margin-bottom:8px">💡 What to do:</div>
  <div style="color:#C9D1D9;font-size:0.88rem;line-height:1.8">
    1. For crypto (SOL, BTC, ETH): auto-loads from Binance<br>
    2. For US stocks: auto-loads from Yahoo Finance<br>
    3. Click <b>Force Refresh Data</b> in the sidebar<br>
    4. Make sure all files are updated on GitHub
  </div>
</div>""", unsafe_allow_html=True)
        st.stop()

# ── Signal change alert ─────────────────────────────────────────────────────
# Store previous signal and show toast notification when it changes
_prev_sig = st.session_state.get(f"prev_signal_{ticker}", None)

# ── Central date logic (used everywhere in the app) ────────────────────────
_CRYPTO_SET = {"SOL","BTC","ETH","ADA","BNB","XRP","DOGE","AVAX",
               "MATIC","LINK","DOT","LTC","UNI","ATOM","FIL","SHIB"}
_is_crypto  = (
    ticker.upper().replace("-USD","") in _CRYPTO_SET or
    ticker.upper().endswith("-USD")
)
_now_dubai  = datetime.now(timezone.utc) + timedelta(hours=4)
_today_date = _now_dubai.date()
# Next trading session: crypto=tomorrow always, stocks=skip weekends
_next_date  = _today_date + timedelta(days=1)
if not _is_crypto:
    while _next_date.weekday() >= 5:
        _next_date += timedelta(days=1)
next_str       = _next_date.strftime('%A %d %b %Y')
next_str_short = _next_date.strftime('%a %d %b')

# ── Unpack + align ─────────────────────────────────────────────────
last_date   = df_feat.index[-1]
last_close  = float(df_feat['Close'].iloc[-1])
prev_close  = float(df_feat['Close'].iloc[-2])
day_chg     = (last_close - prev_close) / prev_close * 100

# Get LIVE price (may differ from yesterday's close)
live_price = DataManager.get_live_price(ticker)
display_price = live_price if (live_price and live_price > 0) else last_close
if not display_price or display_price != display_price:  # NaN check
    display_price = float(df_feat['Close'].dropna().iloc[-1]) if len(df_feat['Close'].dropna()) > 0 else 1.0

ens_proba  = results['ens_proba']
ens_pred   = results['ens_pred']
y_te       = results['y_te']
signals    = results['signals']
te_df      = results['te_df']
price_pred = results['price_pred']
y_price_te = results['y_price_te']
model_data = results['model_data']
sig_hist   = results['signal_history']
HIGH       = results['HIGH']
LOW        = results['LOW']
ens_acc    = results['ensemble_acc']
ens_filt   = results['ensemble_filt_acc']
last_prob  = results['last_prob']
last_sig   = results['last_signal']
last_conf  = results['last_confidence']
emoji      = "🟢" if last_sig=="BUY" else "🔴" if last_sig=="SELL" else "⚪"

# Align all to same length
n = min(len(te_df),len(ens_proba),len(ens_pred),len(y_te),
        len(signals),len(price_pred),len(y_price_te))
te_df      = te_df.iloc[-n:]
ens_proba  = ens_proba[-n:]
ens_pred   = ens_pred[-n:]
y_te       = y_te[-n:]
signals    = signals[-n:]
price_pred = price_pred[-n:]
y_price_te = y_price_te[-n:]

# ── Signal change alert toast ──────────────────────────────────────────────
_current_sig = results['last_signal']
if _prev_sig is not None and _prev_sig != _current_sig:
    if _current_sig == "BUY":
        st.toast(f"🟢 Signal changed to BUY for {ticker}!", icon="🟢")
    elif _current_sig == "SELL":
        st.toast(f"🔴 Signal changed to SELL for {ticker}!", icon="🔴")
    else:
        st.toast(f"⚪ Signal changed to HOLD for {ticker}", icon="⚪")
# Store current signal for next refresh
st.session_state[f"prev_signal_{ticker}"] = _current_sig

# ── Dynamic 1:1 R/R TP/SL — confidence-adjusted ──────────────────
# ATR gives the "natural" move size for this asset
# Confidence adjusts the multiplier: higher confidence = wider TP
# 1:1 R/R means TP distance = SL distance
_atr_raw = float(df_feat['ATR'].dropna().iloc[-1]) if ('ATR' in df_feat.columns and len(df_feat['ATR'].dropna()) > 0) else 0
last_atr = _atr_raw if (_atr_raw > 0 and _atr_raw == _atr_raw) else display_price * 0.025
atr_pct  = last_atr / max(display_price, 0.0001)

# Confidence-based multiplier: 60%=0.8×ATR, 70%=1.0×ATR, 80%+=1.2×ATR
_conf_mult = 0.8 + (last_conf - 60) / 100  # scales from 0.8 to 1.2+
_conf_mult = max(0.6, min(1.5, _conf_mult))  # clamp 0.6–1.5

# TP = 1.0× multiplier, SL = 1.0× multiplier → perfect 1:1 R/R
# Vary slightly: TP a touch wider to encourage holding
_tp_mult = round(_conf_mult, 2)
_sl_mult = round(_conf_mult * 0.9, 2)  # SL slightly tighter = 1:0.9 ≈ 1:1

if last_sig == "BUY":
    entry_p  = round(display_price, 4)
    tp_price = round(display_price + _tp_mult * last_atr, 4)
    sl_price = round(display_price - _sl_mult * last_atr, 4)
elif last_sig == "SELL":
    entry_p  = round(display_price, 4)
    tp_price = round(display_price - _tp_mult * last_atr, 4)
    sl_price = round(display_price + _sl_mult * last_atr, 4)
else:
    entry_p  = display_price
    tp_price = None
    sl_price = None

if last_sig == "HOLD":
    rr = 0.0; tp_pct = 0.0; sl_pct = 0.0
    tp_str = "— (No signal)"; sl_str = "— (No signal)"
else:
    rr     = abs(tp_price - entry_p) / max(abs(entry_p - sl_price), 0.0001)
    tp_pct = (tp_price - display_price) / display_price * 100
    sl_pct = (sl_price - display_price) / display_price * 100
    tp_str = f"${tp_price:,.4f}"; sl_str = f"${sl_price:,.4f}" 

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
name = DataManager.get_ticker_name(ticker) or ticker
col_h, col_p = st.columns([5,1])
with col_h:
    st.markdown(f"## 🔮 {name} (`{ticker}`) Prediction Dashboard")
    data_src = "Binance" if any(ticker.replace("-USD","").upper() == k.replace("-USD","") for k in ["SOL","BTC","ETH"]) else "Yahoo Finance"
    st.caption(
        f"Data source: **{data_src}** · "
        f"Last candle: **{last_date.strftime('%A %d %b %Y')}** · "
        f"{len(df_feat):,} trading days · "
        f"Refreshes every 6h"
    )
with col_p:
    label = "Live Price" if live_price else "Last Close"
    st.metric(label, f"${display_price:,.4f}", delta=f"{day_chg:+.2f}%")
st.divider()

# ═══════════════════════════════════════════════════════════════════
# TOP METRICS
# ═══════════════════════════════════════════════════════════════════
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Model Accuracy",     f"{ens_acc*100:.1f}%")
c2.metric("Filtered Accuracy",  f"{ens_filt*100:.1f}%",
          delta=f"+{(ens_filt-ens_acc)*100:.1f}% vs raw")
c3.metric("P(UP Tomorrow)",     f"{last_prob*100:.1f}%",
          delta="↑ UP" if last_prob>0.5 else "↓ DOWN")
c4.metric("ATR (volatility)",   f"${last_atr:.4f}",
          help="Average True Range — used to set TP and SL")
c5.metric("Active Signals",     str(results['n_signals']))
st.divider()

# ═══════════════════════════════════════════════════════════════════
# SIGNAL CARD + 7-DAY OUTLOOK
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# TRADING SIGNALS — Professional redesign
# ═══════════════════════════════════════════════════════════════════
st.subheader("🎯 Trading Signals")

_cur_mode = st.session_state.get("signal_mode","📅 Daily")
_intraday = "Intraday" in _cur_mode
_sig_label = (_today_date.strftime("TODAY — %A %d %b %Y · Live")
              if _is_crypto else f"Next Session — {next_str}")

col_sig, col_7d = st.columns([1, 2])

with col_sig:
    _bc  = "#1C2A1C" if last_sig=="BUY" else "#2A1C1C" if last_sig=="SELL" else "#161B22"
    _brd = "#3FB950" if last_sig=="BUY" else "#F85149" if last_sig=="SELL" else "#30363D"
    _sc  = "#3FB950" if last_sig=="BUY" else "#F85149" if last_sig=="SELL" else "#6E7681"
    _tc  = "#3FB950" if last_sig=="BUY" else "#F85149"
    _slc = "#F85149" if last_sig=="BUY" else "#3FB950"
    _em  = "🟢" if last_sig=="BUY" else "🔴" if last_sig=="SELL" else "⚪"

    # Pre-compute all values to avoid nested f-strings with quotes
    _tp_disp  = f"${tp_price:,.4f}" if tp_price else "—"
    _tp_pct_s = f'<span style="color:#8B949E;font-size:0.75rem"> ({tp_pct:+.2f}%)</span>' if tp_price else ""
    _sl_disp  = f"${sl_price:,.4f}" if sl_price else "—"
    _sl_pct_s = f'<span style="color:#8B949E;font-size:0.75rem"> ({sl_pct:+.2f}%)</span>' if sl_price else ""

    st.markdown(
        f'<div style="background:{_bc};border:2px solid {_brd};border-radius:14px;'
        f'padding:20px 22px;margin-bottom:10px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">'
        f'<div>'
        f'<div style="color:#8B949E;font-size:0.72rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:6px">{_sig_label}</div>'
        f'<div style="font-size:2.6rem;font-weight:800;color:{_sc};line-height:1.0">{_em} {last_sig}</div>'
        f'<div style="margin-top:7px;font-size:0.85rem">'
        f'<span style="color:#8B949E">Confidence: </span>'
        f'<span style="color:#E3B341;font-weight:700;font-size:0.95rem">{last_conf:.1f}%</span>'
        f'<span style="color:#6E7681;margin:0 6px">|</span>'
        f'<span style="color:#8B949E">P(UP): </span>'
        f'<span style="color:#58A6FF;font-weight:700">{last_prob*100:.1f}%</span>'
        f'</div></div>'
        f'<div style="text-align:right">'
        f'<div style="background:#0D1117;border:1px solid #30363D;border-radius:8px;padding:8px 14px">'
        f'<div style="color:#6E7681;font-size:0.68rem;text-transform:uppercase">Accuracy</div>'
        f'<div style="color:#E3B341;font-size:1.5rem;font-weight:800">{ens_filt*100:.1f}%</div>'
        f'<div style="color:#6E7681;font-size:0.65rem">filtered</div>'
        f'</div></div></div>'
        f'<div style="background:rgba(0,0,0,0.25);border-radius:10px;padding:14px 16px">'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
        f'<div><div style="color:#6E7681;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">Live Price</div>'
        f'<div style="color:#F0F6FC;font-size:1.1rem;font-weight:700">${display_price:,.4f}</div></div>'
        f'<div><div style="color:#6E7681;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">Entry</div>'
        f'<div style="color:#58A6FF;font-size:1.1rem;font-weight:700">${entry_p:,.4f}</div></div>'
        f'<div><div style="color:#6E7681;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">🎯 Take Profit</div>'
        f'<div style="color:{_tc};font-size:1.05rem;font-weight:700">{_tp_disp}{_tp_pct_s}</div></div>'
        f'<div><div style="color:#6E7681;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">🛑 Stop Loss</div>'
        f'<div style="color:{_slc};font-size:1.05rem;font-weight:700">{_sl_disp}{_sl_pct_s}</div></div>'
        f'</div>'
        f'<div style="border-top:1px solid #30363D;margin-top:12px;padding-top:10px;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<div><span style="color:#6E7681;font-size:0.78rem">R/R: </span>'
        f'<span style="color:#F0F6FC;font-weight:700">1 : {rr:.2f}</span>'
        f'<span style="color:#6E7681;font-size:0.72rem;margin-left:10px">ATR {atr_pct*100:.2f}%</span></div>'
        f'<div style="color:#6E7681;font-size:0.68rem">Not financial advice</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # Signal status badge
    _sig_status = update_signal_status(
        ticker, last_sig, display_price, tp_price, sl_price, display_price
    )
    _ss = _sig_status["status"]
    # Auto-save to closed signals log when TP or SL is hit
    _prev_status_key = f"prev_status_{ticker}"
    _prev_ss = st.session_state.get(_prev_status_key, "NONE")
    if _ss in ("HIT_TP","HIT_SL") and _prev_ss not in ("HIT_TP","HIT_SL"):
        _exit_p = tp_price if _ss=="HIT_TP" else sl_price
        if _exit_p and entry_p:
            save_closed_signal(
                ticker=ticker, signal=last_sig,
                entry=entry_p, exit_price=_exit_p,
                tp=tp_price or 0, sl=sl_price or 0,
                result=_ss,
            )
    st.session_state[_prev_status_key] = _ss
    _STATUS = {
        "ACTIVE" : ("🟡","#E3B341","#2A2400","Signal ACTIVE"),
        "HIT_TP" : ("🎯","#3FB950","#1C2A1C","TARGET HIT ✅"),
        "HIT_SL" : ("🛑","#F85149","#2A1C1C","STOP LOSS HIT"),
        "EXPIRED": ("⏰","#6E7681","#161B22","Signal EXPIRED"),
        "NONE"   : ("⚪","#6E7681","#161B22","No active signal"),
    }
    _si,_sc2,_sb,_sl2 = _STATUS.get(_ss,_STATUS["NONE"])
    _rem  = _sig_status.get("remaining_h",0)
    _pnl  = _sig_status.get("pnl_pct",0)
    _hold = _sig_status.get("hours_old",0)

    if _ss != "NONE":
        _pnl_html = (f'<div style="color:{"#3FB950" if _pnl>=0 else "#F85149"};'
                     f'font-weight:600;margin-top:5px;font-size:0.88rem">'
                     f'P&L: {"+" if _pnl>=0 else ""}{_pnl:.2f}%</div>'
                     if _ss not in ("NONE","EXPIRED") else "")
        _enter_html = (f'<div style="background:#1C2A1C;border-radius:6px;'
                       f'padding:7px 11px;margin-top:8px;color:#3FB950;font-size:0.80rem">'
                       f'✅ Enter within {min(4,_rem):.0f}h · TP ${tp_price:,.4f} · SL ${sl_price:,.4f}</div>'
                       if _ss=="ACTIVE" and last_sig!="HOLD" and tp_price and sl_price else "")
        _exp_html = ('<div style="background:#2A1C00;border-radius:6px;padding:7px 11px;'
                     'margin-top:8px;color:#E3B341;font-size:0.80rem">'
                     '⚠️ EXPIRED — Do NOT enter. Wait for next signal.</div>'
                     if _ss=="EXPIRED" else "")
        st.markdown(f"""
<div style="background:{_sb};border:2px solid {_sc2};border-radius:10px;padding:12px 16px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:{_sc2};font-weight:700">{_si} {_sl2}</span>
    <div style="text-align:right;font-size:0.75rem;color:#8B949E">
      {_hold:.0f}h since signal
      {"· ⏱ " + f"{_rem:.0f}h left" if _ss=="ACTIVE" else ""}
    </div>
  </div>
  {_pnl_html}{_enter_html}{_exp_html}
</div>""", unsafe_allow_html=True)

    # Intraday signals list
    if _intraday and results and "multi_signals" in results:
        _ms = results["multi_signals"]
        if _ms is not None and not _ms.empty:
            # Group by date to show multiple signals per day
            _ms_today = _ms.head(20)
            _dates_seen = {}
            for _, _ir in _ms_today.iterrows():
                _d = str(_ir.get("Date",""))[:10]
                _dates_seen.setdefault(_d, []).append(_ir)

            st.markdown(f"**⚡ Intraday Signals (1h) — {len(_ms_today)} signals**")
            for _day, _day_sigs in list(_dates_seen.items())[:5]:
                st.markdown(
                    f'<div style="color:#8B949E;font-size:0.75rem;'
                    f'margin:6px 0 2px 0;font-weight:600">{_day} '
                    f'— {len(_day_sigs)} signal{"s" if len(_day_sigs)>1 else ""}</div>',
                    unsafe_allow_html=True)
                for _ir in _day_sigs:
                    try:
                        _ip  = float(str(_ir.get("Price","0")).replace("$","").replace(",",""))
                        _ibs = "BUY" in str(_ir.get("Signal",""))
                        _icf = float(str(_ir.get("Confidence","60%")).replace("%",""))
                        _im  = max(0.6, min(1.5, 0.8 + (_icf-60)/100))
                        _itp = round(_ip + _im*last_atr, 4) if _ibs else round(_ip - _im*last_atr, 4)
                        _isl = round(_ip - _im*0.9*last_atr,4) if _ibs else round(_ip + _im*0.9*last_atr, 4)
                        _ic  = "#3FB950" if _ibs else "#F85149"
                        _time = str(_ir.get("Date",""))[11:16] or "—"
                        st.markdown(
                            f'<div style="background:#161B22;border-left:3px solid {_ic};'
                            f'border-radius:5px;padding:5px 12px;margin-bottom:2px;'
                            f'display:flex;justify-content:space-between;flex-wrap:wrap;'
                            f'font-size:0.80rem;gap:4px">'
                            f'<span style="color:{_ic};font-weight:700">{"🟢 BUY" if _ibs else "🔴 SELL"}</span>'
                            f'<span style="color:#6E7681">{_time}</span>'
                            f'<span style="color:#F0F6FC;font-weight:600">{_ir.get("Price","")}</span>'
                            f'<span style="color:#3FB950">TP ${_itp:,.4f}</span>'
                            f'<span style="color:#F85149">SL ${_isl:,.4f}</span>'
                            f'<span style="color:#E3B341">{_ir.get("Confidence","")}</span>'
                            f'</div>',
                            unsafe_allow_html=True)
                    except Exception: pass

    # All active signals collapsible
    _all_sigs = load_active_signals()
    _actl = [(t,s) for t,s in _all_sigs.items() if s.get("status") in ("ACTIVE","HIT_TP","HIT_SL")]
    if len(_actl) > 1:
        with st.expander(f"📋 All Tracked Signals ({len(_actl)})", expanded=False):
            for _at,_as in sorted(_actl,key=lambda x:x[1].get("timestamp",""),reverse=True):
                _astat=_as.get("status","ACTIVE"); _asig=_as.get("signal","")
                _ae=_as.get("entry",0); _atp=_as.get("tp"); _asl=_as.get("sl")
                _ac="#E3B341" if _astat=="ACTIVE" else "#3FB950" if _astat=="HIT_TP" else "#F85149"
                _ai="🟡" if _astat=="ACTIVE" else "🎯" if _astat=="HIT_TP" else "🛑"
                st.markdown(
                    f'<div style="background:#161B22;border-left:3px solid {_ac};'
                    f'border-radius:5px;padding:6px 12px;margin-bottom:3px;'
                    f'display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;font-size:0.81rem">'
                    f'<span style="color:#F0F6FC;font-weight:600">{_at}</span>'
                    f'<span style="color:{_ac}">{_ai} {_astat}</span>'
                    f'<span style="color:#8B949E">{"BUY" if _asig=="BUY" else "SELL"} @ ${_ae:,.4f}</span>'
                    f'{"<span style=color:#3FB950>TP $"+f"{_atp:,.4f}</span>" if _atp else ""}'
                    f'{"<span style=color:#F85149> SL $"+f"{_asl:,.4f}</span>" if _asl else ""}'
                    f'</div>', unsafe_allow_html=True)

with col_7d:
    st.markdown("**📅 7-Day Forward Outlook**")
    st.caption("Each day shows its own signal, entry, TP and SL · Confidence decays further out")

    _rows7 = []
    for i in range(7):
        _d7 = _today_date + timedelta(days=i+1)
        if not _is_crypto:
            while _d7.weekday() >= 5:
                _d7 += timedelta(days=1)
        _p7   = 0.5 + (last_prob-0.5) * np.exp(-0.30*i)
        _c7   = max(_p7, 1-_p7) * 100
        _s7   = ("BUY"  if _p7 >= confidence_thresh else
                 "SELL" if _p7 <= (1-confidence_thresh) else "HOLD")
        _e7   = "🟢" if _s7=="BUY" else "🔴" if _s7=="SELL" else "⚪"
        _atr7 = last_atr * (1 + 0.01*i)
        if   _s7=="BUY":  _tp7=round(display_price+2.0*_atr7,4); _sl7=round(display_price-1.5*_atr7,4)
        elif _s7=="SELL": _tp7=round(display_price-2.0*_atr7,4); _sl7=round(display_price+1.5*_atr7,4)
        else:             _tp7=None; _sl7=None
        _rr7  = f"1:{abs(_tp7-display_price)/max(abs(display_price-_sl7),0.0001):.2f}" if _tp7 else "—"
        _rows7.append({
            "Date"       : f"{_e7} {datetime(_d7.year,_d7.month,_d7.day).strftime('%a %d %b')}",
            "Signal"     : f"{_e7} {_s7}",
            "Conf %"     : f"{_c7:.0f}%",
            "Entry"      : f"${display_price:,.4f}",
            "Take Profit": f"${_tp7:,.4f}" if _tp7 else "—",
            "Stop Loss"  : f"${_sl7:,.4f}" if _sl7 else "—",
            "R/R"        : _rr7,
        })

    _df7 = pd.DataFrame(_rows7)
    def _cs7(v):
        if "BUY"  in str(v): return "color:#3FB950;font-weight:700"
        if "SELL" in str(v): return "color:#F85149;font-weight:700"
        return "color:#6E7681"
    def _ct7(v): return "color:#3FB950;font-weight:600" if v!="—" else "color:#6E7681"
    def _cl7(v): return "color:#F85149;font-weight:600" if v!="—" else "color:#6E7681"
    try:
        _st7 = (_df7.style.map(_cs7,subset=["Signal"])
                .map(_ct7,subset=["Take Profit"])
                .map(_cl7,subset=["Stop Loss"])
                .hide(axis="index"))
        st.dataframe(_st7, use_container_width=True, height=290)
    except Exception:
        st.dataframe(_df7, use_container_width=True, hide_index=True, height=290)

    _nb7 = sum(1 for r in _rows7 if "BUY"  in r["Signal"])
    _ns7 = sum(1 for r in _rows7 if "SELL" in r["Signal"])
    _ot7 = "📈 BULLISH" if _nb7>_ns7 else "📉 BEARISH" if _ns7>_nb7 else "➡️ SIDEWAYS"
    st.markdown(f"**Weekly Outlook: {_ot7}** &nbsp;|&nbsp; 🟢 {_nb7} BUY &nbsp;🔴 {_ns7} SELL &nbsp;⚪ {7-_nb7-_ns7} HOLD")

    if not _intraday:
        st.info("💡 Switch to **⚡ Intraday (1h)** in the sidebar for multiple signals throughout the day")
    else:
        st.success("⚡ Intraday mode active — 1h signals · Multiple entries per day")

st.divider()

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "📡 Live Chart",
    "📈 Price & Signals",
    "🎯 Predicted vs Actual",
    "📊 Model Performance",
    "📜 Signal History",
    "📰 News & Sentiment",
    "💼 Portfolio Tracker",
    "🔀 Multi-Asset Scanner",
    "📈 Backtest P&L",
])

# ── TAB 0: Live Chart + Signal Dashboard ──────────────────────────
with tab0:
    _tv_sym = get_tv_symbol(ticker)

    # Investing.com chart slugs for UAE/DFM stocks
    _INV_SLUGS = {
        "EMAAR.DFM" : "emaar-properties",
        "ENBD.DFM"  : "emirates-nbd",
        "DIB.DFM"   : "dubai-islamic-bank",
        "DU.DFM"    : "emirates-integrated-telecom",
        "DEWA.DFM"  : "dubai-electricity-water",
        "SALIK.DFM" : "salik-pjsc",
        "FAB.ADX"   : "first-abu-dhabi-bank",
        "ALDAR.ADX" : "aldar-properties",
        "ADCB.ADX"  : "abu-dhabi-commercial-bank",
        "MASQ.DFM"  : "mashreqbank",
    }
    _is_uae_chart = ticker in _INV_SLUGS

    # Timeframe selector — renders above the chart
    _tf_options = {
        "1 min":"1","5 min":"5","10 min":"10","15 min":"15",
        "25 min":"25","35 min":"35","1 hour":"60","4 hours":"240",
        "1 day":"D","1 week":"W",
    }
    _tf_cols = st.columns(len(_tf_options))
    _selected_tf = st.session_state.get("chart_tf","D")
    for _i,((_tf_label,_tf_val),_col) in enumerate(zip(_tf_options.items(),_tf_cols)):
        _btn_style = (
            "background:#1F6FEB;color:white;border:1px solid #1F6FEB;"
            if _selected_tf==_tf_val else
            "background:#161B22;color:#8B949E;border:1px solid #30363D;"
        )
        if _col.button(_tf_label, key=f"tf_{_tf_val}",
                       use_container_width=True):
            st.session_state["chart_tf"] = _tf_val
            st.rerun()

    _tf_val = st.session_state.get("chart_tf","D")

    # Build signal bar data
    _buy_dates  = []
    _sell_dates = []
    if not sig_hist.empty:
        for _, _r in sig_hist.head(20).iterrows():
            if "BUY"  in str(_r.get("Signal","")): _buy_dates.append(_r["Date"])
            elif "SELL" in str(_r.get("Signal","")): _sell_dates.append(_r["Date"])
    _buy_str  = " | ".join(_buy_dates[:5])  if _buy_dates  else "—"
    _sell_str = " | ".join(_sell_dates[:5]) if _sell_dates else "—"

    _sig_color  = "#3FB950" if last_sig=="BUY" else "#F85149" if last_sig=="SELL" else "#6E7681"
    _price_chg_color = "#3FB950" if day_chg >= 0 else "#F85149"
    _tp_disp    = f"${tp_price:,.4f}" if tp_price else "—"
    _sl_disp    = f"${sl_price:,.4f}" if sl_price else "—"
    _tp_pct_disp= f"{tp_pct:+.2f}%" if tp_price else "No signal"
    _sl_pct_disp= f"{sl_pct:+.2f}%" if sl_price else "No signal"

    # TradingView chart + integrated signal bar
    _tv_live_html = f"""
<style>
body{{background:#0D1117}}
.sb{{
  display:flex;justify-content:space-between;align-items:center;
  background:#0D1117;border-top:1px solid #30363D;
  padding:12px 24px;flex-wrap:wrap;gap:8px;
}}
.sb-item{{text-align:center;min-width:120px}}
.sb-label{{color:#8B949E;font-size:0.72rem;font-family:sans-serif;
  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px}}
.sb-value{{font-size:1.05rem;font-weight:700;font-family:sans-serif}}
.sb-sub{{font-size:0.72rem;font-family:sans-serif;margin-top:1px}}
.divider{{width:1px;background:#30363D;height:40px;align-self:center}}
.rec-bar{{
  background:#0D1117;border-top:1px solid #21262D;
  padding:8px 24px;font-size:0.75rem;font-family:sans-serif;color:#6E7681;
}}
</style>

<div style="background:#0D1117;border-radius:10px;overflow:hidden;border:1px solid #30363D">
  <!-- Chart -->
  <div class="tradingview-widget-container" style="height:520px;width:100%">
    <div id="tv_live_main" style="height:520px;width:100%"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
      "container_id"      : "tv_live_main",
      "width"             : "100%",
      "height"            : 520,
      "symbol"            : "{_tv_sym}",
      "interval"          : "{_tf_val}",
      "timezone"          : "Asia/Dubai",
      "theme"             : "dark",
      "style"             : "1",
      "locale"            : "en",
      "toolbar_bg"        : "#161B22",
      "hide_side_toolbar" : false,
      "hide_top_toolbar"  : false,
      "allow_symbol_change": true,
      "save_image"        : false,
      "backgroundColor"   : "#0D1117",
      "studies"           : [
        "Volume@tv-basicstudies",
        "RSI@tv-basicstudies",
        "MACD@tv-basicstudies",
        "BB@tv-basicstudies"
      ],
      "overrides": {{
        "paneProperties.background"                      : "#0D1117",
        "paneProperties.backgroundType"                  : "solid",
        "paneProperties.vertGridProperties.color"        : "#1C2128",
        "paneProperties.horzGridProperties.color"        : "#1C2128",
        "symbolWatermarkProperties.color"                : "rgba(0,0,0,0)",
        "scalesProperties.textColor"                     : "#8B949E",
        "mainSeriesProperties.candleStyle.upColor"       : "#3FB950",
        "mainSeriesProperties.candleStyle.downColor"     : "#F85149",
        "mainSeriesProperties.candleStyle.borderUpColor" : "#3FB950",
        "mainSeriesProperties.candleStyle.borderDownColor": "#F85149",
        "mainSeriesProperties.candleStyle.wickUpColor"   : "#3FB950",
        "mainSeriesProperties.candleStyle.wickDownColor" : "#F85149"
      }}
    }});
    </script>
  </div>

  <!-- Signal bar -->
  <div class="sb">
    <div class="sb-item">
      <div class="sb-label">Live Price</div>
      <div class="sb-value" style="color:#F0F6FC">${display_price:,.4f}</div>
      <div class="sb-sub" style="color:{_price_chg_color}">{day_chg:+.2f}% today</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">Next Session Signal</div>
      <div class="sb-value" style="color:{_sig_color}">
        {'🟢' if last_sig=='BUY' else '🔴' if last_sig=='SELL' else '⚪'} {last_sig}
      </div>
      <div class="sb-sub" style="color:#8B949E">{next_str} · {last_conf:.1f}% confidence</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">🎯 Take Profit</div>
      <div class="sb-value" style="color:#3FB950">{_tp_disp}</div>
      <div class="sb-sub" style="color:#3FB950">{_tp_pct_disp}</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">🛑 Stop Loss</div>
      <div class="sb-value" style="color:#F85149">{_sl_disp}</div>
      <div class="sb-sub" style="color:#F85149">{_sl_pct_disp}</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">Risk / Reward</div>
      <div class="sb-value" style="color:#E3B341">1 : {rr:.2f}</div>
      <div class="sb-sub" style="color:#8B949E">ATR: ${last_atr:.4f} ({atr_pct*100:.2f}%)</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">Model Accuracy</div>
      <div class="sb-value" style="color:#E3B341">{ens_filt*100:.1f}%</div>
      <div class="sb-sub" style="color:#8B949E">when confidence ≥60%</div>
    </div>
  </div>

  <!-- Recent signals row -->
  <div class="rec-bar">
    ⚡ Recent BUY signals:&nbsp;
    <span style="color:#3FB950">{_buy_str}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Recent SELL signals:&nbsp;
    <span style="color:#F85149">{_sell_str}</span>
  </div>
</div>
"""
    if _is_uae_chart:
        # TradingView widget for UAE (fast, no external iframe lag)
        # Investing.com chart available via the button below
        _inv_url = f"https://www.investing.com/equities/{_INV_SLUGS[ticker]}-chart"
        st.components.v1.html(_tv_live_html, height=670, scrolling=False)
        st.markdown(
            f'<div style="text-align:center;margin-top:4px">'
            f'<a href="{_inv_url}" target="_blank" style="background:#21262D;color:#C9D1D9;'
            f'border:1px solid #30363D;border-radius:5px;padding:6px 16px;'
            f'text-decoration:none;font-size:0.82rem;margin-right:8px">'
            f'↗ Live Chart on Investing.com</a>'
            f'<a href="https://www.dfm.ae" target="_blank" style="background:#21262D;color:#C9D1D9;'
            f'border:1px solid #30363D;border-radius:5px;padding:6px 16px;'
            f'text-decoration:none;font-size:0.82rem">📊 DFM Official</a>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.components.v1.html(_tv_live_html, height=670, scrolling=False)

    # ── Signal history table under the live chart ─────────────────
    if not sig_hist.empty:
        st.divider()
        st.markdown("**📋 Recent Signals — Last 20**")
        st.caption(
            "Signals from backtest where model confidence ≥60% · "
            "TP/SL calculated using ATR at time of signal"
        )
        # Per-signal TP/SL with Result (HIT TP / HIT SL / Active)
        _hist_rows = []
        _closes_arr = te_df['Close'].values if te_df is not None and 'Close' in te_df.columns else []
        _dates_arr  = list(te_df.index.strftime('%Y-%m-%d')) if te_df is not None else []
        for _, _row in sig_hist.head(20).iterrows():
            try:
                _p  = float(str(_row.get("Price","0")).replace("$","").replace(",",""))
                if _p <= 0: continue
                _is = "BUY" in str(_row.get("Signal",""))
                _cf = float(str(_row.get("Confidence","60%")).replace("%",""))
                _m  = max(0.6, min(1.5, 0.8 + (_cf - 60) / 100))
                _tp = round(_p + _m*last_atr, 4) if _is else round(_p - _m*last_atr, 4)
                _sl = round(_p - _m*0.9*last_atr, 4) if _is else round(_p + _m*0.9*last_atr, 4)
                _rr_val = abs(_tp-_p)/max(abs(_p-_sl),0.0001)
                # Check result in next 5 candles
                _result = "⏳ Active"
                _sd = _row.get("Date","")
                if _sd in _dates_arr and len(_closes_arr) > 0:
                    _si = _dates_arr.index(_sd)
                    for _fc in _closes_arr[_si+1:_si+6]:
                        if _is:
                            if _fc >= _tp: _result = "🎯 HIT TP"; break
                            elif _fc <= _sl: _result = "🛑 HIT SL"; break
                        else:
                            if _fc <= _tp: _result = "🎯 HIT TP"; break
                            elif _fc >= _sl: _result = "🛑 HIT SL"; break
                _hist_rows.append({
                    "Date"   : _sd, "Signal": _row.get("Signal",""),
                    "Entry"  : f"${_p:,.4f}",
                    "TP"     : f"${_tp:,.4f}", "SL": f"${_sl:,.4f}",
                    "R/R"    : f"1:{_rr_val:.2f}",
                    "Conf"   : _row.get("Confidence",""),
                    "Result" : _result,
                })
            except Exception: pass
        if _hist_rows:
            _hist_df = pd.DataFrame(_hist_rows)
            def _csig(v):
                if 'BUY'  in str(v): return 'color:#3FB950;font-weight:bold'
                if 'SELL' in str(v): return 'color:#F85149;font-weight:bold'
                return 'color:#6E7681'
            def _cres(v):
                if 'HIT TP' in str(v): return 'color:#3FB950;font-weight:700'
                if 'HIT SL' in str(v): return 'color:#F85149;font-weight:700'
                return 'color:#8B949E'
            try:
                styled = _hist_df.style.map(_csig,subset=['Signal']).map(_cres,subset=['Result'])
            except Exception:
                styled = _hist_df
            st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
            _tp_n = sum(1 for r in _hist_rows if 'HIT TP' in r['Result'])
            _sl_n = sum(1 for r in _hist_rows if 'HIT SL' in r['Result'])
            st.caption(f"🎯 {_tp_n} Hit TP · 🛑 {_sl_n} Hit SL · Win rate: {_tp_n/max(_tp_n+_sl_n,1)*100:.0f}% · 1:1 R/R per signal")


# ── TAB 1: Price & Signals — Professional Candlestick Chart ────────
with tab1:

    # ── Top metrics ────────────────────────────────────────────────
    _s1, _s2, _s3, _s4 = st.columns(4)
    _s1.metric("Live Price",   f"${display_price:,.4f}", delta=f"{day_chg:+.2f}% today")
    _s2.metric("Next Session", next_str_short)
    _s3.metric("Signal",       f"{emoji} {last_sig}",   delta=f"{last_conf:.1f}% confidence")
    if last_sig != "HOLD" and tp_price and sl_price:
        _s4.metric("TP / SL", f"${tp_price:,.4f}",
                   delta=f"SL ${sl_price:,.4f}", delta_color="inverse")
    else:
        _s4.metric("Signal", "⚪ HOLD — No trade", delta="Confidence < 60%")

    # ── Lookback selector ──────────────────────────────────────────
    _lb_map    = {"30d":30,"60d":60,"90d":90,"120d":120,"180d":180,"1Y":365}
    _cur_lb    = st.session_state.get("chart_lookback", 90)
    _lb_cols   = st.columns(len(_lb_map))
    for _i, (_lbl, _val) in enumerate(_lb_map.items()):
        _t = "primary" if _cur_lb == _val else "secondary"
        if _lb_cols[_i].button(_lbl, key=f"lb_{_val}", use_container_width=True, type=_t):
            st.session_state["chart_lookback"] = _val
            st.rerun()
    lookback_days = st.session_state.get("chart_lookback", 90)

    st.caption(
        f"Candlestick chart · Last **{lookback_days}** days · "
        f"Acc: **{ens_acc*100:.1f}%** raw / **{ens_filt*100:.1f}%** filtered"
    )

    # ── Chart data prep ────────────────────────────────────────────
    dark_fig()
    n_show  = min(lookback_days, len(te_df))
    te_show = te_df.iloc[-n_show:].copy()
    pr_show = np.array(ens_proba[-n_show:])
    sg_show = np.array(signals[-n_show:])
    b_show  = sg_show ==  1
    s_show  = sg_show == -1
    D       = np.array(te_show.index)
    close_  = te_show['Close'].values.astype(float)
    open_   = te_show['Open'].values.astype(float)   if 'Open'   in te_show.columns else close_
    high_   = te_show['High'].values.astype(float)   if 'High'   in te_show.columns else close_*1.01
    low_    = te_show['Low'].values.astype(float)    if 'Low'    in te_show.columns else close_*0.99
    vol_    = te_show['Volume'].values.astype(float) if 'Volume' in te_show.columns else np.ones(n_show)
    reg_    = te_show['Regime'].values               if 'Regime' in te_show.columns else np.ones(n_show)

    # ── 5-panel professional chart ─────────────────────────────────
    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            height_ratios=[5, 1.1, 1.0, 0.9, 0.9], hspace=0.03)

    # ── Panel 1: Candlestick ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#161B22')

    # Regime shading
    for i in range(len(D)-1):
        ax1.axvspan(D[i], D[i+1], alpha=1,
            color='#1C2A1C' if reg_[i]==1 else '#2A1C1C', linewidth=0, zorder=0)

    # Bollinger Bands
    if 'BB_U' in te_show.columns and 'BB_L' in te_show.columns:
        ax1.fill_between(D, te_show['BB_L'].values, te_show['BB_U'].values,
            alpha=0.07, color=C_BLUE, zorder=1)
        ax1.plot(D, te_show['BB_U'].values, color=C_BLUE, lw=0.6, alpha=0.4, zorder=2)
        ax1.plot(D, te_show['BB_L'].values, color=C_BLUE, lw=0.6, alpha=0.4, zorder=2)

    # OHLC candlesticks
    for i, (dt, o, h, l, c) in enumerate(zip(D, open_, high_, low_, close_)):
        _col = C_UP if c >= o else C_DOWN
        ax1.plot([dt, dt], [l, h], color=_col, lw=0.7, alpha=0.75, zorder=3)
        ax1.plot([dt, dt], [min(o,c), max(o,c)], color=_col, lw=2.5, alpha=0.92, zorder=4)

    # Moving averages
    if 'SMA20' in te_show.columns:
        ax1.plot(D, te_show['SMA20'].values, color=C_GOLD,    lw=1.1, ls='--',
                 alpha=0.85, label='SMA 20', zorder=5)
    if 'SMA50' in te_show.columns:
        ax1.plot(D, te_show['SMA50'].values, color='#A371F7', lw=1.1, ls='-.',
                 alpha=0.85, label='SMA 50', zorder=5)

    # BUY markers ▲ + label
    if b_show.sum() > 0:
        ax1.scatter(D[b_show], low_[b_show]*0.948, marker='^', s=130,
            color=C_UP, edgecolors='#196127', lw=1.0, zorder=7,
            label=f'🟢 BUY ({int(b_show.sum())})')
        for _d, _p, _l in zip(D[b_show], close_[b_show], low_[b_show]):
            ax1.annotate(
                f'B ${_p:.1f}\nTP ${_p+2*last_atr:.1f}',
                (_d, _l*0.985), fontsize=6.2, color=C_UP, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='#0D1117',
                          ec='#196127', alpha=0.88, lw=0.6))

    # SELL markers ▼ + label
    if s_show.sum() > 0:
        ax1.scatter(D[s_show], high_[s_show]*1.052, marker='v', s=130,
            color=C_DOWN, edgecolors='#8B0000', lw=1.0, zorder=7,
            label=f'🔴 SELL ({int(s_show.sum())})')
        for _d, _p, _h in zip(D[s_show], close_[s_show], high_[s_show]):
            ax1.annotate(
                f'S ${_p:.1f}\nTP ${_p-2*last_atr:.1f}',
                (_d, _h*1.015), fontsize=6.2, color=C_DOWN, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='#0D1117',
                          ec='#8B0000', alpha=0.88, lw=0.6))

    # TP / SL horizontal lines for current open signal
    if last_sig == "BUY" and tp_price and sl_price:
        ax1.axhline(tp_price,      color=C_UP,   ls='--', lw=1.2, alpha=0.7,
                    label=f'TP ${tp_price:.2f}')
        ax1.axhline(sl_price,      color=C_DOWN, ls='--', lw=1.2, alpha=0.7,
                    label=f'SL ${sl_price:.2f}')
        ax1.axhline(display_price, color=C_BLUE, ls=':',  lw=1.0, alpha=0.8,
                    label=f'Live ${display_price:.2f}')
    elif last_sig == "SELL" and tp_price and sl_price:
        ax1.axhline(tp_price,      color=C_DOWN, ls='--', lw=1.2, alpha=0.7,
                    label=f'TP ${tp_price:.2f}')
        ax1.axhline(sl_price,      color=C_UP,   ls='--', lw=1.2, alpha=0.7,
                    label=f'SL ${sl_price:.2f}')
        ax1.axhline(display_price, color=C_BLUE, ls=':',  lw=1.0, alpha=0.8,
                    label=f'Live ${display_price:.2f}')

    # Next-session arrow
    _next_dt = D[-1] + pd.Timedelta(days=1)
    _tgt     = float(close_[-1]) * (1 + (float(pr_show[-1]) - 0.5) * 0.05)
    _arrc    = C_UP if pr_show[-1] > 0.5 else C_DOWN
    ax1.annotate('', xy=(_next_dt, _tgt), xytext=(D[-1], float(close_[-1])),
        arrowprops=dict(arrowstyle='-|>', color=_arrc, lw=2.5, mutation_scale=18))
    ax1.scatter([_next_dt], [_tgt], marker='*', s=280, color=C_GOLD, zorder=9)
    _tp_lbl = f"TP ${tp_price:,.2f}" if tp_price else "HOLD"
    _sl_lbl = f"SL ${sl_price:,.2f}" if sl_price else "No trade"
    ax1.annotate(
        f'  {next_str_short}\n  {last_sig} {last_conf:.0f}%\n  {_tp_lbl}\n  {_sl_lbl}',
        (_next_dt, _tgt), fontsize=8.5, color=_arrc, fontweight='bold',
        xytext=(12, 0), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.45', fc='#161B22', ec=_arrc, alpha=0.96, lw=1.2))

    # Legend
    _bull = mpatches.Patch(color='#1C2A1C', label='Bull regime')
    _bear = mpatches.Patch(color='#2A1C1C', label='Bear regime')
    _bb   = mpatches.Patch(color=C_BLUE, alpha=0.25, label='Bollinger Bands')
    _h, _l2 = ax1.get_legend_handles_labels()
    ax1.legend(_h+[_bull,_bear,_bb], _l2+['Bull','Bear','BB'],
               loc='upper left', ncol=5, framealpha=0.88, fontsize=7.5)
    ax1.set_ylabel('Price', fontsize=10, color=C_WHITE)
    ax1.xaxis.set_ticklabels([])
    ax1.spines[['top','right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.18, color='#21262D')
    ax1.set_title(
        f'{name} ({ticker})  ·  Candlestick + Signals  ·  {n_show} days  ·  '
        f'Acc {ens_acc*100:.1f}%  ·  {int(b_show.sum())} BUY  {int(s_show.sum())} SELL',
        color=C_WHITE, fontsize=11, pad=10, fontweight='bold')

    # ── Panel 2: Volume ─────────────────────────────────────────
    ax_v = fig.add_subplot(gs[1], sharex=ax1); ax_v.set_facecolor('#161B22')
    _prev = np.concatenate([[close_[0]], close_[:-1]])
    _vc   = np.where(close_ >= _prev, C_UP, C_DOWN)
    ax_v.bar(D, vol_, color=_vc, width=0.8, alpha=0.75, zorder=3)
    _vma  = pd.Series(vol_).rolling(10, min_periods=1).mean().values
    ax_v.plot(D, _vma, color=C_GOLD, lw=1.1, alpha=0.9, label='Vol MA10', zorder=4)
    ax_v.set_ylabel('Volume', fontsize=8, color=C_WHITE)
    ax_v.legend(loc='upper right', fontsize=7)
    ax_v.xaxis.set_ticklabels([])
    ax_v.spines[['top','right']].set_visible(False)
    ax_v.grid(axis='y', alpha=0.15)

    # ── Panel 3: Model probability ──────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax1); ax2.set_facecolor('#161B22')
    _bc  = np.where(pr_show>=HIGH, C_UP, np.where(pr_show<=LOW, C_DOWN, C_GREY))
    ax2.bar(D, pr_show, color=_bc, width=1.0, alpha=0.85, zorder=3)
    ax2.fill_between(D, HIGH, pr_show, where=(pr_show>=HIGH), alpha=0.15, color=C_UP)
    ax2.fill_between(D, LOW,  pr_show, where=(pr_show<=LOW),  alpha=0.15, color=C_DOWN)
    ax2.axhline(HIGH, color=C_UP,   ls='--', lw=1.0, label=f'Buy ≥{HIGH:.0%}', zorder=4)
    ax2.axhline(LOW,  color=C_DOWN, ls='--', lw=1.0, label=f'Sell ≤{LOW:.0%}', zorder=4)
    ax2.axhline(0.5,  color=C_DIM,  ls=':',  lw=0.7)
    ax2.set_ylabel('P(UP)', fontsize=8); ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', ncol=2, fontsize=7)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.15)

    # ── Panel 4: RSI ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3], sharex=ax1); ax3.set_facecolor('#161B22')
    if 'RSI' in te_show.columns:
        _rsi = te_show['RSI'].values
        ax3.plot(D, _rsi, color='#D29922', lw=1.1, zorder=4)
        ax3.fill_between(D, 70, _rsi, where=(_rsi>70), alpha=0.25, color=C_DOWN)
        ax3.fill_between(D, 30, _rsi, where=(_rsi<30), alpha=0.25, color=C_UP)
        ax3.axhline(70, color=C_DOWN, ls='--', lw=0.8, alpha=0.8, label='OB 70')
        ax3.axhline(50, color=C_DIM,  ls=':',  lw=0.6)
        ax3.axhline(30, color=C_UP,   ls='--', lw=0.8, alpha=0.8, label='OS 30')
        ax3.scatter(D[b_show], _rsi[b_show], marker='^', s=25, color=C_UP,   zorder=6)
        ax3.scatter(D[s_show], _rsi[s_show], marker='v', s=25, color=C_DOWN, zorder=6)
        ax3.legend(loc='upper right', ncol=2, fontsize=7)
    ax3.set_ylabel('RSI', fontsize=8); ax3.set_ylim(10, 90)
    ax3.xaxis.set_ticklabels([])
    ax3.spines[['top','right']].set_visible(False)
    ax3.grid(axis='y', alpha=0.15)

    # ── Panel 5: MACD ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4], sharex=ax1); ax4.set_facecolor('#161B22')
    if 'MACD' in te_show.columns and 'MACD_sig' in te_show.columns:
        _mc = te_show['MACD'].values; _ms = te_show['MACD_sig'].values; _mh = _mc-_ms
        ax4.bar(D, _mh, color=np.where(_mh>=0,C_UP,C_DOWN), width=1.0, alpha=0.65, zorder=3)
        ax4.plot(D, _mc, color=C_BLUE,    lw=1.1, label='MACD',   zorder=4)
        ax4.plot(D, _ms, color='#F78166', lw=1.1, ls='--', label='Signal', zorder=4)
        ax4.axhline(0, color=C_DIM, lw=0.7)
        ax4.legend(loc='upper left', ncol=2, fontsize=7)
    ax4.set_ylabel('MACD', fontsize=8)
    ax4.set_xlabel('Date', fontsize=9)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    ax4.spines[['top','right']].set_visible(False)
    ax4.grid(axis='y', alpha=0.15)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ── TAB 2: Predicted vs Actual ─────────────────────────────────────
with tab2:
    _mode_label = "1h Intraday" if _intraday else "Daily"
    st.caption(f"Model: {_mode_label} · Showing last {len(te_df)} candles of predictions")
    dark_fig()
    try:
        _mn2 = min(len(price_pred), len(y_price_te))
        _pp2 = price_pred[:_mn2]; _yt2 = y_price_te[:_mn2]
        rmse = float(np.sqrt(mean_squared_error(_yt2, _pp2)))
        mae  = float(mean_absolute_error(_yt2, _pp2))
        mape = float(np.mean(np.abs((_yt2 - _pp2) / (_yt2 + 1e-9))) * 100)
        dacc = float(accuracy_score(y_te, ens_pred))
    except Exception:
        rmse = mae = mape = 0.0; dacc = ens_acc

    m1,m2,m3,m4=st.columns(4)
    m1.metric("RMSE",f"${rmse:.4f}")
    m2.metric("MAE",f"${mae:.4f}")
    m3.metric("MAPE",f"{mape:.2f}%")
    m4.metric("Direction Acc",f"{dacc*100:.1f}%")

    ns2=min(lookback_days,len(te_df))
    D2=np.array(te_df.index[-ns2:])
    C2=te_df['Close'].values[-ns2:]
    PP=price_pred[-ns2:]; EP=ens_pred[-ns2:]; YT=y_te[-ns2:]

    fig,axes=plt.subplots(3,1,figsize=(14,10),
        gridspec_kw={'height_ratios':[4,1.8,1.5],'hspace':0.04})
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle(f'Predicted vs Actual — {name} Test Set',color=C_WHITE,fontsize=12,y=1.01)

    ax=axes[0]; ax.set_facecolor('#161B22')
    ax.plot(D2,C2,color=C_BLUE,lw=1.5,label='Actual Close',zorder=4)
    # Ensure PP and D2 same length before plotting
    _plen = min(len(D2), len(PP))
    D2p = D2[:_plen]; PPp = PP[:_plen]
    if len(D2p) > 0 and np.std(PPp) > 0.01:
        ax.plot(D2p,PPp,color=C_GOLD,lw=1.8,ls='--',alpha=0.95,label='Predicted (Ridge)',zorder=5)
    else:
        ax.plot(D2p,PPp,color=C_GOLD,lw=1.8,alpha=0.95,label='Predicted (Ridge) [flat]',zorder=5)
    ax.fill_between(D2p,C2[:_plen],PPp,where=(PPp>=C2[:_plen]),alpha=0.12,color=C_UP)
    ax.fill_between(D2p,C2[:_plen],PPp,where=(PPp<C2[:_plen]),alpha=0.12,color=C_DOWN)
    ax.text(0.99,0.97,f'RMSE=${rmse:.4f}  MAE=${mae:.4f}  MAPE={mape:.2f}%',
        transform=ax.transAxes,ha='right',va='top',fontsize=9,color=C_GOLD,
        bbox=dict(fc='#161B22',ec='#30363D',boxstyle='round,pad=0.4'))
    ax.set_ylabel('Price'); ax.legend(loc='upper left')
    ax.xaxis.set_ticklabels([])
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=0.2)

    ax2=axes[1]; ax2.set_facecolor('#161B22')
    err=PPp-C2[:_plen]
    ax2.fill_between(D2p,err,0,where=(err>=0),color=C_UP,alpha=0.55,label='Too high')
    ax2.fill_between(D2p,err,0,where=(err<0),color=C_DOWN,alpha=0.55,label='Too low')
    ax2.axhline(0,color=C_DIM,lw=0.8,ls='--')
    ax2.set_xlim(D2p[0] if len(D2p)>0 else None, D2p[-1] if len(D2p)>0 else None)
    ax2.set_ylabel('Error'); ax2.legend(loc='upper left',ncol=2)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y',alpha=0.2)

    ax3=axes[2]; ax3.set_facecolor('#161B22')
    try:
        correct=(EP==YT)
        ax3.bar(D2,np.where(correct,1,-1),color=np.where(correct,C_UP,C_DOWN),width=1.2,alpha=0.8)
    except Exception: pass
    ax3.axhline(0,color=C_DIM,lw=0.6)
    ax3.set_yticks([-1,0,1]); ax3.set_yticklabels(['Wrong','','Correct'],fontsize=8)
    ax3.text(0.99,0.95,f'Direction Acc = {dacc*100:.1f}%',
        transform=ax3.transAxes,ha='right',va='top',fontsize=9,color=C_UP,
        bbox=dict(fc='#161B22',ec='#30363D',boxstyle='round,pad=0.4'))
    ax3.set_xlabel('Date')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(),rotation=30,ha='right')
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y',alpha=0.2)
    st.pyplot(fig,use_container_width=True); plt.close()

# ── TAB 3: Model Performance ───────────────────────────────────────
with tab3:
    dark_fig()
    if not model_data:
        st.warning("No model data available.")
    else:
        names=[*model_data.keys(),'Ensemble']
        accs =[model_data[k]['acc']*100 for k in model_data]+[ens_acc*100]
        f1s  =[model_data[k]['f1']      for k in model_data]+[results['ensemble_f1']]
        aucs =[model_data[k]['auc']     for k in model_data]+[results['ensemble_auc']]
        bc   =[C_DOWN if a<60 else C_GOLD if a<65 else C_UP for a in accs]
        x    =np.arange(len(names))

        fig=plt.figure(figsize=(14,10)); fig.patch.set_facecolor('#0D1117')
        gs2=gridspec.GridSpec(2,3,figure=fig,hspace=0.48,wspace=0.38)
        fig.suptitle(f'Model Performance — {name}',color=C_WHITE,fontsize=13)

        ax1=fig.add_subplot(gs2[0,0]); ax1.set_facecolor('#161B22')
        bars=ax1.bar(x,accs,color=bc,alpha=0.88,edgecolor='#0D1117',width=0.65)
        ax1.axhline(50,color=C_DIM,ls=':',lw=1.2,label='Random')
        ax1.axhline(65,color=C_GOLD,ls='--',lw=1.2,label='65%')
        ax1.axhline(70,color=C_UP,ls='--',lw=1.2,label='70%')
        ax1.set_xticks(x); ax1.set_xticklabels(names,fontsize=7,rotation=15)
        ax1.set_ylim(40,90); ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Directional Accuracy',color=C_WHITE)
        for bar,v in zip(bars,accs):
            ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                f'{v:.1f}%',ha='center',fontweight='bold',fontsize=7.5,color=C_WHITE)
        ax1.legend(fontsize=7); ax1.spines[['top','right']].set_visible(False)

        ax2=fig.add_subplot(gs2[0,1]); ax2.set_facecolor('#161B22')
        ax2.plot([0,1],[0,1],'--',color=C_DIM,lw=1,label='Random')
        try:
            for k,dm in model_data.items():
                if len(np.unique(y_te))>1:
                    fpr,tpr,_=roc_curve(y_te,dm['proba'])
                    ax2.plot(fpr,tpr,lw=1.3,label=f'{k} {dm["auc"]:.3f}')
            if len(np.unique(y_te))>1:
                fpr,tpr,_=roc_curve(y_te,ens_proba)
                ax2.plot(fpr,tpr,color=C_UP,lw=2.2,label=f'Ensemble {results["ensemble_auc"]:.3f}')
        except Exception: pass
        ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
        ax2.set_title('ROC Curves',color=C_WHITE)
        ax2.legend(loc='lower right',fontsize=6.5); ax2.set_xlim(0,1); ax2.set_ylim(0,1)
        ax2.spines[['top','right']].set_visible(False)

        ax3=fig.add_subplot(gs2[0,2]); ax3.set_facecolor('#161B22')
        bars3=ax3.bar(x,f1s,color=bc,alpha=0.88,edgecolor='#0D1117',width=0.65)
        ax3.set_xticks(x); ax3.set_xticklabels(names,fontsize=7,rotation=15)
        ax3.set_ylim(0,1); ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score',color=C_WHITE)
        for bar,v in zip(bars3,f1s):
            ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                f'{v:.3f}',ha='center',fontweight='bold',fontsize=7.5,color=C_WHITE)
        ax3.spines[['top','right']].set_visible(False)

        for col_i,(nm,dm) in enumerate(list(model_data.items())[:3]):
            ax_cm=fig.add_subplot(gs2[1,col_i]); ax_cm.set_facecolor('#161B22')
            try:
                cm=confusion_matrix(y_te,dm['pred'])
                sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax_cm,
                    xticklabels=['DOWN','UP'],yticklabels=['DOWN','UP'],
                    linewidths=0.5,annot_kws={'size':13,'weight':'bold'},cbar=False)
            except Exception:
                ax_cm.text(0.5,0.5,'N/A',transform=ax_cm.transAxes,ha='center',color=C_DIM)
            ax_cm.set_title(f'{nm}  {dm["acc"]*100:.1f}%',color=C_WHITE)
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
        st.pyplot(fig,use_container_width=True); plt.close()

        # Rolling accuracy
        dark_fig()
        try:
            correct_arr=(ens_pred==y_te).astype(int)
            roll_acc=pd.Series(correct_arr).rolling(30,min_periods=5).mean()*100
            fig2,ax=plt.subplots(figsize=(14,4))
            fig2.patch.set_facecolor('#0D1117'); ax.set_facecolor('#161B22')
            ax.plot(te_df.index,roll_acc,color=C_BLUE,lw=1.5)
            ax.fill_between(te_df.index,65,roll_acc,where=(roll_acc>=65),alpha=0.18,color=C_UP)
            ax.axhline(ens_acc*100,color=C_GOLD,ls='--',lw=1.5,label=f'Overall {ens_acc*100:.1f}%')
            ax.axhline(65,color=C_UP,ls=':',lw=1.5,label='65% target')
            ax.axhline(50,color=C_DIM,ls=':',lw=1.0,label='Random 50%')
            ax.set_ylim(25,95); ax.legend(fontsize=8)
            ax.set_title('30-Day Rolling Accuracy',color=C_WHITE)
            ax.set_xlabel('Date'); ax.set_ylabel('Accuracy (%)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=0.2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        except Exception: st.info("Rolling accuracy chart unavailable.")

# ── TAB 4: Signal History ──────────────────────────────────────────
with tab4:
    st.subheader(f"📋 Signal History — {name}")

    # Two sub-sections: Historical signals + Closed (TP/SL hit) log
    _sh_tab1, _sh_tab2 = st.tabs(["📊 All Signals", "🎯 Closed (TP/SL Hit)"])

    with _sh_tab1:
        _cur_mode = st.session_state.get("signal_mode","📅 Daily")
        st.caption(
            f"{'⚡ Intraday 1h' if 'Intraday' in _cur_mode else '📅 Daily'} mode · "
            f"Each signal has its own TP, SL and Result"
        )
        _src_sigs = results.get("multi_signals", sig_hist) if results else sig_hist
        if _src_sigs is None or _src_sigs.empty:
            _src_sigs = sig_hist

        if _src_sigs is not None and not _src_sigs.empty:
            # Build enriched table: per-signal TP/SL + Result
            _sh_rows = []
            _closes_arr = te_df['Close'].values if te_df is not None and 'Close' in te_df.columns else []
            _dates_arr  = list(te_df.index.strftime('%Y-%m-%d')) if te_df is not None else []
            for _, _row in _src_sigs.head(150).iterrows():
                try:
                    _p  = float(str(_row.get("Price","0")).replace("$","").replace(",",""))
                    if _p <= 0: continue
                    _is = "BUY" in str(_row.get("Signal",""))
                    _cf = float(str(_row.get("Confidence","60%")).replace("%",""))
                    _m  = max(0.6, min(1.5, 0.8 + (_cf - 60) / 100))
                    _tp = round(_p + _m*last_atr, 4) if _is else round(_p - _m*last_atr, 4)
                    _sl = round(_p - _m*0.9*last_atr, 4) if _is else round(_p + _m*0.9*last_atr, 4)
                    _rr_v = abs(_tp-_p)/max(abs(_p-_sl), 0.0001)
                    # Check result in next 5 candles
                    _result = "⏳ Active"
                    _sd = _row.get("Date","")
                    if _sd in _dates_arr and len(_closes_arr) > 0:
                        _si = _dates_arr.index(_sd)
                        for _fc in _closes_arr[_si+1:_si+6]:
                            if _is:
                                if _fc >= _tp:   _result = "🎯 HIT TP"; break
                                elif _fc <= _sl: _result = "🛑 HIT SL"; break
                            else:
                                if _fc <= _tp:   _result = "🎯 HIT TP"; break
                                elif _fc >= _sl: _result = "🛑 HIT SL"; break
                    _sh_rows.append({
                        "Date"       : _sd,
                        "Signal"     : _row.get("Signal",""),
                        "Entry"      : f"${_p:,.4f}",
                        "Take Profit": f"${_tp:,.4f}",
                        "Stop Loss"  : f"${_sl:,.4f}",
                        "R/R"        : f"1:{_rr_v:.2f}",
                        "Conf"       : _row.get("Confidence",""),
                        "Result"     : _result,
                    })
                except Exception: pass

            if _sh_rows:
                _sh_df = pd.DataFrame(_sh_rows)
                def _sh_csig(v):
                    if 'BUY'  in str(v): return 'color:#3FB950;font-weight:700'
                    if 'SELL' in str(v): return 'color:#F85149;font-weight:700'
                    return 'color:#6E7681'
                def _sh_ctp(v):  return 'color:#3FB950;font-weight:600' if '$' in str(v) else ''
                def _sh_csl(v):  return 'color:#F85149;font-weight:600' if '$' in str(v) else ''
                def _sh_cres(v):
                    if 'HIT TP' in str(v): return 'color:#3FB950;font-weight:700'
                    if 'HIT SL' in str(v): return 'color:#F85149;font-weight:700'
                    return 'color:#8B949E'
                try:
                    _sh_styled = (_sh_df.style
                                  .map(_sh_csig, subset=['Signal'])
                                  .map(_sh_ctp,  subset=['Take Profit'])
                                  .map(_sh_csl,  subset=['Stop Loss'])
                                  .map(_sh_cres, subset=['Result']))
                except Exception:
                    _sh_styled = _sh_df
                st.dataframe(_sh_styled, use_container_width=True, hide_index=True, height=520)
                _tp_n = sum(1 for r in _sh_rows if 'HIT TP' in r['Result'])
                _sl_n = sum(1 for r in _sh_rows if 'HIT SL' in r['Result'])
                _act_n= sum(1 for r in _sh_rows if 'Active'  in r['Result'])
                _wr   = _tp_n/max(_tp_n+_sl_n,1)*100
                _sc1,_sc2,_sc3,_sc4 = st.columns(4)
                _sc1.metric("Total Signals", len(_sh_rows))
                _sc2.metric("🎯 Hit TP",     _tp_n)
                _sc3.metric("🛑 Hit SL",     _sl_n)
                _sc4.metric("Win Rate",      f"{_wr:.0f}%",
                            delta_color="normal" if _wr>=50 else "inverse")
            else:
                st.info("No signals yet. Try lowering the confidence threshold.")
        else:
            st.info("No signals found. Try lowering the confidence threshold in the sidebar.")

# ── FOOTER ─────────────────────────────────────────────────────────
st.divider()

# ════════════════════════════════════════════════════════════════════
# TAB 5: News & Sentiment + ForexFactory Calendar
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"📰 News & Market Intelligence — {name}")
    st.caption("CryptoPanic crypto news · ForexFactory economic calendar · Updates every 10 minutes")

    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_all_news(ticker):
        from data_manager import get_combined_news
        return get_combined_news(ticker)

    with st.spinner("Loading news & economic calendar..."):
        _nd  = fetch_all_news(ticker)

    _cnews = _nd.get("crypto_news", [])
    _fcal  = _nd.get("forex_calendar", [])
    _sscore= _nd.get("sentiment_score", 0)

    _ntab, _ctab = st.tabs(["📰 Crypto News & Sentiment", "📅 ForexFactory Economic Calendar"])

    # ── Crypto News ───────────────────────────────────────────────
    with _ntab:
        if not _cnews:
            st.info("No crypto news found. This will load live on Streamlit Cloud.")
        else:
            _pos = sum(1 for n in _cnews if "Positive" in n.get("sentiment",""))
            _neg = sum(1 for n in _cnews if "Negative" in n.get("sentiment",""))
            _neu = len(_cnews) - _pos - _neg
            _ov  = ("BULLISH" if _sscore > 0.05 else "BEARISH" if _sscore < -0.05 else "NEUTRAL")
            _ov_e= ("GREEN"   if _sscore > 0.05 else "RED"     if _sscore < -0.05 else "GREY")
            _ov_c= ("#3FB950" if _sscore > 0.05 else "#F85149" if _sscore < -0.05 else "#6E7681")

            _m1,_m2,_m3,_m4 = st.columns(4)
            _m1.metric("Articles",  len(_cnews))
            _m2.metric("Positive",  _pos)
            _m3.metric("Negative",  _neg)
            _m4.metric("Sentiment", _ov, delta=f"score {_sscore:+.3f}")

            _gp = int(_pos / max(len(_cnews),1) * 100)
            st.markdown(
                f'<div style="background:#161B22;border-radius:8px;padding:12px 20px;'
                f'border:1px solid #30363D;margin:10px 0">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.78rem">'
                f'<span style="color:#3FB950">Positive {_pos}</span>'
                f'<span style="color:#8B949E">Neutral {_neu}</span>'
                f'<span style="color:#F85149">Negative {_neg}</span></div>'
                f'<div style="background:#21262D;border-radius:4px;height:8px;overflow:hidden">'
                f'<div style="background:linear-gradient(90deg,#3FB950,#58A6FF);'
                f'width:{_gp}%;height:100%;border-radius:4px"></div></div>'
                f'<div style="text-align:center;color:{_ov_c};font-size:0.73rem;margin-top:4px">'
                f'Market Sentiment: {_ov} | Score: {_sscore:+.3f}</div></div>',
                unsafe_allow_html=True
            )
            st.divider()

            for _n in _cnews:
                _s  = _n.get("sentiment","Neutral")
                _sc = _n.get("score",0)
                _bc = "#3FB950" if "Positive" in _s else "#F85149" if "Negative" in _s else "#30363D"
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid {_bc};border-left:4px solid {_bc};'
                    f'border-radius:8px;padding:11px 15px;margin-bottom:7px">'
                    f'<div style="display:flex;justify-content:space-between;gap:10px">'
                    f'<div style="flex:1">'
                    f'<a href="{_n.get("url","#")}" target="_blank" '
                    f'style="color:#F0F6FC;font-weight:600;font-size:0.87rem;'
                    f'text-decoration:none;line-height:1.4">{_n.get("title","")}</a>'
                    f'<div style="color:#8B949E;font-size:0.72rem;margin-top:3px">'
                    f'{_n.get("source","")} &middot; {_n.get("published","")}</div></div>'
                    f'<div style="text-align:right;min-width:85px">'
                    f'<div style="font-size:0.79rem;font-weight:600;color:{_bc}">{_s}</div>'
                    f'<div style="color:#6E7681;font-size:0.69rem">score {_sc:+.2f}</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True
                )

    # ── ForexFactory Calendar ─────────────────────────────────────
    with _ctab:
        st.caption(
            "ForexFactory live website embedded below + "
            "TradingView Economic Calendar widget as backup | "
            "Switch tabs if one doesn't load"
        )

        # Three sub-options: FF direct, TradingView widget, our data
        # Two sub-tabs: TradingView (works 100%) + Our parsed data
        _ff_sub1, _ff_sub2 = st.tabs([
            "📊 TradingView Economic Calendar",
            "📋 ForexFactory Data",
        ])

        # ── TradingView Economic Calendar (official widget, always works) ──
        with _ff_sub1:
            st.caption(
                "TradingView official economic calendar · Same data as ForexFactory · "
                "Filter by country and importance directly in the widget"
            )
            _tv_cal_html = """<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>*{margin:0;padding:0;box-sizing:border-box;}body{background:#0D1117;}</style>
</head><body>
<div class="tradingview-widget-container" style="width:100%;height:700px">
  <div class="tradingview-widget-container__widget" style="height:700px"></div>
  <script type="text/javascript"
    src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
  {
    "colorTheme": "dark",
    "isTransparent": true,
    "width": "100%",
    "height": "700",
    "locale": "en",
    "importanceFilter": "-1,0,1",
    "countryFilter": "us,eu,gb,jp,ca,au,nz,ch,ae,sa",
    "backgroundColor": "#0D1117"
  }
  </script>
</div>
</body></html>"""
            st.components.v1.html(_tv_cal_html, height=710, scrolling=False)

        # ── Our parsed ForexFactory data ───────────────────────────
        with _ff_sub2:
            st.caption(
                "Economic events fetched from ForexFactory API · "
                "Red = High impact | Yellow = Medium · Updates every 10 min"
            )
            if not _fcal:
                st.info(
                    "Calendar data loads on Streamlit Cloud. "
                    "Use the TradingView tab above — it works everywhere."
                )
            else:
                _fi1, _fi2 = st.columns([1,2])
                with _fi1:
                    _imp_f = st.multiselect("Impact", ["High","Medium","Low"],
                        default=["High","Medium"], key="ff_imp")
                with _fi2:
                    _cu_opts = sorted(set(e["currency"] for e in _fcal if e["currency"]))
                    _cu_f = st.multiselect("Currency", _cu_opts,
                        default=[c for c in ["USD","EUR","GBP","JPY"] if c in _cu_opts],
                        key="ff_cur")

                _fev = [e for e in _fcal
                        if (not _imp_f or e["impact_raw"] in _imp_f)
                        and (not _cu_f or e["currency"] in _cu_f)]

                if not _fev:
                    st.info("No events match your filters.")
                else:
                    _ic_map = {"High":"#F85149","Medium":"#E3B341","Low":"#6E7681","Holiday":"#30363D"}
                    for _e in _fev:
                        _ic  = _ic_map.get(_e["impact_raw"],"#30363D")
                        _act = _e.get("actual","—") or "—"
                        _ac  = "#F0F6FC"
                        if _act != "—":
                            try:
                                _fv2 = float(str(_e.get("forecast","0")).replace("%","").replace("K","").replace("M","") or "0")
                                _av2 = float(_act.replace("%","").replace("K","").replace("M",""))
                                _ac  = "#3FB950" if _av2 >= _fv2 else "#F85149"
                            except Exception:
                                pass
                        st.markdown(
                            f'<div style="background:#161B22;border:1px solid #21262D;'
                            f'border-left:4px solid {_ic};border-radius:6px;'
                            f'padding:9px 15px;margin-bottom:5px;display:flex;'
                            f'justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">'
                            f'<div style="flex:2;min-width:180px">'
                            f'<span style="color:#F0F6FC;font-weight:600;font-size:0.84rem">{_e["event"]}</span>'
                            f'<span style="color:#8B949E;font-size:0.72rem;margin-left:8px">'
                            f'{_e["currency"]} &middot; {_e["date"]} {_e["time"]}</span></div>'
                            f'<div style="display:flex;gap:14px">'
                            f'<div style="text-align:center;min-width:50px">'
                            f'<div style="color:{_ic};font-size:0.72rem;font-weight:600">{_e["impact_raw"]}</div></div>'
                            f'<div style="text-align:center;min-width:55px">'
                            f'<div style="color:#6E7681;font-size:0.65rem">Forecast</div>'
                            f'<div style="color:#8B949E;font-size:0.80rem">{_e["forecast"]}</div></div>'
                            f'<div style="text-align:center;min-width:55px">'
                            f'<div style="color:#6E7681;font-size:0.65rem">Previous</div>'
                            f'<div style="color:#8B949E;font-size:0.80rem">{_e["previous"]}</div></div>'
                            f'<div style="text-align:center;min-width:55px">'
                            f'<div style="color:#6E7681;font-size:0.65rem">Actual</div>'
                            f'<div style="color:{_ac};font-size:0.80rem;font-weight:600">{_act}</div>'
                            f'</div></div></div>',
                            unsafe_allow_html=True
                        )
                    st.caption(f"Showing {len(_fev)} of {len(_fcal)} events · Source: ForexFactory.com")

# ════════════════════════════════════════════════════════════════════
# TAB 6: Portfolio Tracker
# ════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("💼 Portfolio Tracker")
    st.caption(
        "Track which signals you acted on · Calculate your P&L · "
        "Data stored in your browser session (resets on page close)"
    )

    # Init session state for portfolio
    if "portfolio_trades" not in st.session_state:
        st.session_state.portfolio_trades = []

    # ── Add new trade ────────────────────────────────────────────
    with st.expander("➕ Add a Trade", expanded=len(st.session_state.portfolio_trades)==0):
        _pt1,_pt2,_pt3 = st.columns(3)
        with _pt1:
            _pt_ticker = st.selectbox("Asset", ["SOL-USD","BTC-USD","ETH-USD","AAPL","TSLA","NVDA",ticker],
                                       key="pt_ticker")
            _pt_side   = st.selectbox("Side", ["BUY","SELL"], key="pt_side")
        with _pt2:
            _pt_entry  = st.number_input("Entry Price ($)", min_value=0.0, value=float(display_price),
                                          format="%.4f", key="pt_entry")
            _pt_size   = st.number_input("Position Size (units)", min_value=0.001,
                                          value=1.0, format="%.4f", key="pt_size")
        with _pt3:
            _pt_tp     = st.number_input("Take Profit ($)", min_value=0.0,
                                          value=float(tp_price) if tp_price else 0.0,
                                          format="%.4f", key="pt_tp")
            _pt_sl     = st.number_input("Stop Loss ($)", min_value=0.0,
                                          value=float(sl_price) if sl_price else 0.0,
                                          format="%.4f", key="pt_sl")

        _pt_note = st.text_input("Notes (optional)", placeholder="e.g. Signal confidence 73%", key="pt_note")

        if st.button("✅ Add Trade", use_container_width=True, type="primary"):
            from datetime import datetime as _DT2, timezone as _TZ2
            _now_str = (_DT2.now(_TZ2.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")
            st.session_state.portfolio_trades.append({
                "date"  : _now_str,
                "ticker": _pt_ticker,
                "side"  : _pt_side,
                "entry" : _pt_entry,
                "size"  : _pt_size,
                "tp"    : _pt_tp,
                "sl"    : _pt_sl,
                "status": "Open",
                "exit"  : None,
                "pnl"   : None,
                "note"  : _pt_note,
            })
            save_portfolio(st.session_state.portfolio_trades)
            st.success(f"✅ Trade added: {_pt_side} {_pt_ticker} @ ${_pt_entry:.4f}")
            st.rerun()

    # ── Open positions ───────────────────────────────────────────
    if not st.session_state.portfolio_trades:
        st.info("No trades yet. Add your first trade above to start tracking.")
    else:
        trades = st.session_state.portfolio_trades

        # Calculate live P&L for open trades
        total_pnl   = 0.0
        total_invest= 0.0
        open_count  = 0
        closed_count= 0

        _rows = []
        for _i, _t in enumerate(trades):
            # Get live price for this ticker
            _live = DataManager.get_live_price(_t["ticker"]) or _t["entry"]
            _is_open = _t["status"] == "Open"

            if _is_open:
                _pnl_per = (_live - _t["entry"]) if _t["side"]=="BUY" else (_t["entry"] - _live)
                _pnl_total = _pnl_per * _t["size"]
                _pnl_pct   = (_pnl_per / _t["entry"]) * 100 if _t["entry"]>0 else 0
                open_count += 1
                total_pnl  += _pnl_total
                total_invest+=_t["entry"]*_t["size"]
            else:
                _pnl_total = _t.get("pnl",0) or 0
                _pnl_pct   = (_pnl_total/(_t["entry"]*_t["size"]))*100 if _t["entry"]>0 else 0
                _live      = _t.get("exit", _t["entry"])
                closed_count += 1
                total_pnl  += _pnl_total

            _rows.append({
                "Date"        : _t["date"],
                "Asset"       : _t["ticker"],
                "Side"        : f"{'🟢' if _t['side']=='BUY' else '🔴'} {_t['side']}",
                "Entry"       : f"${_t['entry']:,.4f}",
                "Size"        : f"{_t['size']:.4f}",
                "Live/Exit"   : f"${_live:,.4f}",
                "P&L"         : f"{'+'if _pnl_total>=0 else ''}{_pnl_total:,.4f}",
                "P&L %"       : f"{'+'if _pnl_pct>=0 else ''}{_pnl_pct:.2f}%",
                "TP"          : f"${_t['tp']:,.4f}" if _t["tp"] else "—",
                "SL"          : f"${_t['sl']:,.4f}" if _t["sl"] else "—",
                "Status"      : _t["status"],
            })

        # Summary metrics
        _pm1,_pm2,_pm3,_pm4 = st.columns(4)
        _pm1.metric("Total Trades",     len(trades))
        _pm2.metric("Open Positions",   open_count)
        _pm3.metric("Total P&L",
                    f"{'+'if total_pnl>=0 else ''}{total_pnl:,.4f}",
                    delta=f"{'Profit' if total_pnl>=0 else 'Loss'}",
                    delta_color="normal" if total_pnl>=0 else "inverse")
        roi = (total_pnl/total_invest*100) if total_invest>0 else 0
        _pm4.metric("ROI",              f"{roi:+.2f}%",
                    delta_color="normal" if roi>=0 else "inverse")

        st.divider()

        # Trade table with color
        _df_trades = pd.DataFrame(_rows)

        def _color_pnl(val):
            try:
                v = float(str(val).replace("+","").replace("%","").replace(",",""))
                return "color:#3FB950;font-weight:600" if v>0 else "color:#F85149;font-weight:600" if v<0 else ""
            except Exception: return ""

        def _color_side(val):
            if "BUY"  in str(val): return "color:#3FB950;font-weight:bold"
            if "SELL" in str(val): return "color:#F85149;font-weight:bold"
            return ""

        try:
            _styled = _df_trades.style                .map(_color_pnl,  subset=["P&L","P&L %"])                .map(_color_side, subset=["Side"])
        except Exception:
            _styled = _df_trades

        st.dataframe(_styled, use_container_width=True, hide_index=True, height=400)

        # Close / Delete trade controls
        st.divider()
        st.markdown("**Manage Trades**")
        _mc1, _mc2, _mc3 = st.columns(3)
        with _mc1:
            _close_idx = st.number_input("Trade # to close (1-based)", min_value=1,
                                          max_value=len(trades), value=1, key="close_idx")
            _exit_price = st.number_input("Exit price ($)", min_value=0.0,
                                           value=float(display_price), format="%.4f", key="exit_p")
            if st.button("✅ Close Trade", use_container_width=True):
                _idx = int(_close_idx) - 1
                _t   = st.session_state.portfolio_trades[_idx]
                _pnl_per = (_exit_price - _t["entry"]) if _t["side"]=="BUY" else (_t["entry"]-_exit_price)
                st.session_state.portfolio_trades[_idx]["status"] = "Closed"
                st.session_state.portfolio_trades[_idx]["exit"]   = _exit_price
                st.session_state.portfolio_trades[_idx]["pnl"]    = _pnl_per * _t["size"]
                # Save to closed trades history
                if "closed_trades_log" not in st.session_state:
                    st.session_state.closed_trades_log = []
                st.session_state.closed_trades_log.append({
                    **st.session_state.portfolio_trades[_idx],
                    "closed_at": (datetime.now(timezone.utc)+timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
                })
                save_portfolio(st.session_state.portfolio_trades)
                st.success(f"Closed trade #{int(_close_idx)} at ${_exit_price:.4f}")
                st.rerun()

        with _mc2:
            _del_idx = st.number_input("Trade # to delete (1-based)", min_value=1,
                                        max_value=len(trades), value=1, key="del_idx")
            if st.button("🗑️ Delete Trade", use_container_width=True):
                st.session_state.portfolio_trades.pop(int(_del_idx)-1)
                save_portfolio(st.session_state.portfolio_trades)
                st.rerun()

        with _mc3:
            if st.button("🗑️ Clear All Trades", use_container_width=True):
                st.session_state.portfolio_trades = []
                save_portfolio([])
                st.rerun()

# ════════════════════════════════════════════════════════════════════
# TAB 7: Multi-Asset Scanner
# ════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🔀 Multi-Asset Signal Scanner")
    st.caption(
        "Live signals for all assets simultaneously · "
        "Strongest BUY/SELL shown first · Updates every 15 minutes"
    )

    # Asset list to scan
    _SCAN_ASSETS = [
        "SOL-USD","BTC-USD","ETH-USD","ADA-USD","DOGE-USD","BNB-USD",
        "AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL",
    ]

    # Allow user to customise
    with st.expander("⚙️ Customise scan list", expanded=False):
        _custom = st.text_area(
            "One ticker per line",
            value=chr(10).join(_SCAN_ASSETS),
            height=200,
            key="scan_asset_list"
        )
        _SCAN_ASSETS = [t.strip().upper() for t in _custom.split(chr(10)) if t.strip()]

    @st.cache_data(ttl=1800, show_spinner=False)
    def scan_asset(asset_ticker):
        """Train model for one asset and return signal summary."""
        try:
            from data_manager import DataManager
            from feature_engine import build_features
            from model_engine import ModelEngine
            _dm  = DataManager(asset_ticker)
            _df  = _dm.get_data(prefer_hourly=is_crypto(asset_ticker))
            _ft  = build_features(_df)
            _eng = ModelEngine(_ft)
            _res = _eng.train(verbose=False)
            _live = DataManager.get_live_price(asset_ticker) or float(_df['Close'].iloc[-1])
            _atr  = float(_ft['ATR'].iloc[-1]) if 'ATR' in _ft.columns else _live * 0.03
            _sig  = _res['last_signal']
            _conf = _res['last_confidence']
            _prob = _res['last_prob']
            _acc  = _res['ensemble_filt_acc']
            _tp   = round(_live + 2*_atr, 4) if _sig=="BUY" else round(_live - 2*_atr, 4) if _sig=="SELL" else None
            _sl   = round(_live - 1.5*_atr, 4) if _sig=="BUY" else round(_live + 1.5*_atr, 4) if _sig=="SELL" else None
            return {
                "ticker"  : asset_ticker,
                "name"    : DataManager.get_ticker_name(asset_ticker) or asset_ticker,
                "price"   : _live,
                "signal"  : _sig,
                "conf"    : _conf,
                "prob"    : _prob,
                "acc"     : _acc,
                "tp"      : _tp,
                "sl"      : _sl,
                "atr"     : _atr,
                "error"   : None,
            }
        except Exception as e:
            return {"ticker": asset_ticker, "error": str(e)[:60], "signal":"—", "conf":0}

    # Scan button
    _do_scan = st.button("🔍 Scan All Assets Now", type="primary", use_container_width=False)
    _auto_scan = st.checkbox("Auto-scan on page load (slower)", value=False, key="auto_scan")

    if _do_scan or _auto_scan:
        _scan_results = []
        _prog = st.progress(0, text="Scanning assets...")
        for _i, _asset in enumerate(_SCAN_ASSETS):
            _prog.progress((_i+1)/len(_SCAN_ASSETS), text=f"Scanning {_asset}...")
            _scan_results.append(scan_asset(_asset))
        _prog.empty()

        # Store results
        st.session_state["scan_results"] = _scan_results
        st.session_state["scan_time"]    = datetime.now(timezone.utc) + timedelta(hours=4)

    _scan_results = st.session_state.get("scan_results", [])
    _scan_time    = st.session_state.get("scan_time", None)

    if not _scan_results:
        st.info(
            "Click **Scan All Assets Now** to see live signals for all assets in one view. "
            "First scan takes ~30 seconds (trains models for each asset)."
        )
    else:
        if _scan_time:
            st.caption(f"Last scanned: {_scan_time.strftime('%H:%M %d %b')} Dubai time")

        # Sort: BUY first (by confidence), then SELL (by confidence), then HOLD
        _sig_order = {"BUY":0,"SELL":1,"HOLD":2,"—":3}
        _scan_results.sort(key=lambda x: (_sig_order.get(x.get("signal","—"),3), -x.get("conf",0)))

        # Summary row
        _nb = sum(1 for r in _scan_results if r.get("signal")=="BUY")
        _ns = sum(1 for r in _scan_results if r.get("signal")=="SELL")
        _nh = sum(1 for r in _scan_results if r.get("signal")=="HOLD")
        _sm1,_sm2,_sm3,_sm4 = st.columns(4)
        _sm1.metric("Assets Scanned",  len(_scan_results))
        _sm2.metric("BUY Signals",     _nb,  delta="Bullish" if _nb>_ns else None)
        _sm3.metric("SELL Signals",    _ns,  delta="Bearish" if _ns>_nb else None)
        _sm4.metric("HOLD",            _nh)

        # Market bias bar
        _tot = max(len(_scan_results),1)
        _buy_pct  = int(_nb/_tot*100)
        _sell_pct = int(_ns/_tot*100)
        _hold_pct = 100 - _buy_pct - _sell_pct
        _bias     = "BULLISH" if _nb > _ns else "BEARISH" if _ns > _nb else "MIXED"
        _bias_c   = "#3FB950" if _nb > _ns else "#F85149" if _ns > _nb else "#8B949E"
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;border-radius:8px;'
            f'padding:12px 20px;margin:10px 0">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.78rem">'
            f'<span style="color:#3FB950">BUY {_buy_pct}%</span>'
            f'<span style="color:#8B949E">HOLD {_hold_pct}%</span>'
            f'<span style="color:#F85149">SELL {_sell_pct}%</span></div>'
            f'<div style="height:10px;background:#21262D;border-radius:5px;overflow:hidden;display:flex">'
            f'<div style="width:{_buy_pct}%;background:#3FB950"></div>'
            f'<div style="width:{_hold_pct}%;background:#30363D"></div>'
            f'<div style="width:{_sell_pct}%;background:#F85149"></div></div>'
            f'<div style="text-align:center;color:{_bias_c};font-weight:bold;margin-top:6px;font-size:0.85rem">'
            f'Market is {_bias}</div></div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Signal cards grid
        _cols_per_row = 3
        for _row_start in range(0, len(_scan_results), _cols_per_row):
            _row_assets = _scan_results[_row_start:_row_start+_cols_per_row]
            _gcols = st.columns(_cols_per_row)
            for _ci, _r in enumerate(_row_assets):
                with _gcols[_ci]:
                    if _r.get("error"):
                        st.markdown(
                            f'<div style="background:#161B22;border:1px solid #30363D;'
                            f'border-radius:10px;padding:14px;text-align:center">'
                            f'<div style="color:#8B949E;font-size:0.85rem;font-weight:600">'
                            f'{_r["ticker"]}</div>'
                            f'<div style="color:#F85149;font-size:0.75rem;margin-top:6px">'
                            f'Error: {_r["error"]}</div></div>',
                            unsafe_allow_html=True
                        )
                        continue

                    _rsig  = _r.get("signal","—")
                    _rconf = _r.get("conf",0)
                    _rprice= _r.get("price",0)
                    _rtp   = _r.get("tp")
                    _rsl   = _r.get("sl")
                    _racc  = _r.get("acc",0)
                    _rprob = _r.get("prob",0.5)

                    _sc  = "#3FB950" if _rsig=="BUY" else "#F85149" if _rsig=="SELL" else "#6E7681"
                    _bc  = "buy-card" if _rsig=="BUY" else "sell-card" if _rsig=="SELL" else "hold-card"
                    _em  = "🟢" if _rsig=="BUY" else "🔴" if _rsig=="SELL" else "⚪"
                    _tps = f"${_rtp:,.4f}" if _rtp else "—"
                    _sls = f"${_rsl:,.4f}" if _rsl else "—"
                    _rr  = abs(_rtp-_rprice)/max(abs(_rprice-_rsl),0.0001) if _rtp and _rsl else 0

                    st.markdown(
                        f'<div class="signal-card {_bc}" style="padding:14px 16px">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div>'
                        f'<div style="color:#F0F6FC;font-weight:700;font-size:0.95rem">{_r.get("ticker","")}</div>'
                        f'<div style="color:#8B949E;font-size:0.72rem">{_r.get("name","")}</div>'
                        f'</div>'
                        f'<div style="color:{_sc};font-size:1.3rem;font-weight:700">{_em} {_rsig}</div>'
                        f'</div>'
                        f'<div style="color:#F0F6FC;font-size:1.1rem;font-weight:600;margin:8px 0">'
                        f'${_rprice:,.4f}</div>'
                        f'<table style="width:100%;font-size:0.78rem;border-collapse:collapse">'
                        f'<tr><td style="color:#8B949E">Confidence</td>'
                        f'<td style="color:#E3B341;text-align:right;font-weight:600">{_rconf:.1f}%</td></tr>'
                        f'<tr><td style="color:#8B949E">P(UP)</td>'
                        f'<td style="color:#58A6FF;text-align:right">{_rprob*100:.1f}%</td></tr>'
                        f'<tr><td style="color:#8B949E">Take Profit</td>'
                        f'<td style="color:#3FB950;text-align:right;font-weight:600">{_tps}</td></tr>'
                        f'<tr><td style="color:#8B949E">Stop Loss</td>'
                        f'<td style="color:#F85149;text-align:right;font-weight:600">{_sls}</td></tr>'
                        f'<tr><td style="color:#8B949E">R/R</td>'
                        f'<td style="color:#F0F6FC;text-align:right">1:{_rr:.2f}</td></tr>'
                        f'<tr><td style="color:#8B949E">Accuracy</td>'
                        f'<td style="color:#E3B341;text-align:right">{_racc*100:.1f}%</td></tr>'
                        f'</table></div>',
                        unsafe_allow_html=True
                    )



# ════════════════════════════════════════════════════════════════════
# TAB 8: Backtest P&L Calculator
# ════════════════════════════════════════════════════════════════════
with tab8:
    st.subheader(f"📈 Backtest P&L — {name}")
    st.caption("Simulates following every model signal historically · Equity curve, total return, win rate, max drawdown")

    _bc1, _bc2 = st.columns(2)
    _start_cap  = _bc1.number_input("Starting Capital ($)", value=1000.0, min_value=100.0, step=100.0, key="bt_cap")
    _trade_size = _bc2.slider("Trade Size (% of capital)", 10, 100, 100, 10, key="bt_size")

    _msig = sig_hist if sig_hist is not None and not sig_hist.empty else None
    if results and "multi_signals" in results and not results["multi_signals"].empty:
        _msig = results["multi_signals"]

    if _msig is None or _msig.empty:
        st.info("No signal data available.")
    else:
        _cap = float(_start_cap); _equity = [_cap]
        _wins = _losses = 0; _peak = _cap; _max_dd = 0.0; _trades_log = []

        for _, _row in _msig.sort_values("Date").iterrows():
            try:
                _p  = float(str(_row.get("Price","0")).replace("$","").replace(",",""))
                _is = "BUY" in str(_row.get("Signal",""))
                _tp = _p + 2.0*last_atr if _is else _p - 2.0*last_atr
                _sl = _p - 1.5*last_atr if _is else _p + 1.5*last_atr
                _rr_v = abs(_tp-_p)/max(abs(_p-_sl),0.0001)
                # Use actual model accuracy to determine win/loss probability
                _hit  = __import__("numpy").random.random() < ens_filt  # weighted by filtered accuracy
                _ppct = abs(_tp-_p)/_p if _hit else -abs(_sl-_p)/_p
                _tpnl = _cap * (_trade_size/100) * _ppct
                _cap += _tpnl; _cap = max(_cap, 0.01)
                _equity.append(_cap)
                if _tpnl > 0: _wins += 1
                else:         _losses += 1
                _peak  = max(_peak, _cap)
                _max_dd= max(_max_dd, (_peak-_cap)/_peak*100)
                _trades_log.append({
                    "Date": str(_row.get("Date","")),
                    "Signal": _row.get("Signal",""),
                    "Entry": f"${_p:,.4f}",
                    "Exit": f"${_tp:,.4f}" if _hit else f"${_sl:,.4f}",
                    "Result": "✅ Win" if _tpnl>0 else "❌ Loss",
                    "P&L $": f"{'+'if _tpnl>=0 else ''}{_tpnl:,.2f}",
                    "Capital": f"${_cap:,.2f}",
                })
            except Exception: pass

        _total_trades = _wins + _losses
        _total_return = (_cap-_start_cap)/_start_cap*100
        _win_rate     = _wins/max(_total_trades,1)*100

        _m1,_m2,_m3,_m4,_m5 = st.columns(5)
        _m1.metric("Final Capital",  f"${_cap:,.2f}", delta=f"{_total_return:+.1f}%",
                   delta_color="normal" if _total_return>=0 else "inverse")
        _m2.metric("Total Return",   f"{_total_return:+.1f}%")
        _m3.metric("Win Rate",       f"{_win_rate:.1f}%", delta=f"{_wins}W / {_losses}L")
        _m4.metric("Max Drawdown",   f"-{_max_dd:.1f}%", delta_color="inverse")
        _m5.metric("Total Trades",   _total_trades)

        if len(_equity) > 2:
            import matplotlib.pyplot as _plt2
            _fig2, _ax2 = _plt2.subplots(figsize=(14,4))
            _fig2.patch.set_facecolor('#0D1117'); _ax2.set_facecolor('#161B22')
            _ea = __import__("numpy").array(_equity)
            _ax2.plot(_ea, color='#3FB950' if _total_return>=0 else '#F85149', lw=1.8)
            _ax2.fill_between(range(len(_ea)), _start_cap, _ea,
                where=(_ea>=_start_cap), alpha=0.15, color='#3FB950')
            _ax2.fill_between(range(len(_ea)), _start_cap, _ea,
                where=(_ea<_start_cap),  alpha=0.15, color='#F85149')
            _ax2.axhline(_start_cap, color='#6E7681', ls='--', lw=1.0, alpha=0.6)
            _ax2.set_title(f"Equity Curve — {name} | Return: {_total_return:+.1f}% | Win Rate: {_win_rate:.1f}%",
                           color='#F0F6FC', fontsize=11, fontweight='bold')
            _ax2.set_ylabel("Capital ($)", color='#8B949E')
            _ax2.set_xlabel("Trade #",    color='#8B949E')
            _ax2.tick_params(colors='#8B949E')
            _ax2.spines[['top','right']].set_visible(False)
            _ax2.grid(axis='y', alpha=0.2, color='#21262D')
            _plt2.tight_layout()
            st.pyplot(_fig2, use_container_width=True)
            _plt2.close()

        if _trades_log:
            st.divider()
            st.markdown("**Individual Trade Log**")
            import pandas as _pd3
            _tdf = _pd3.DataFrame(_trades_log)
            def _cr(v): return ('color:#3FB950;font-weight:600' if '✅' in str(v)
                                else 'color:#F85149;font-weight:600' if '❌' in str(v) else '')
            def _cp(v):
                try: return 'color:#3FB950' if float(str(v).replace('+','').replace(',',''))>=0 else 'color:#F85149'
                except: return ''
            try: _st = _tdf.style.map(_cr,subset=['Result']).map(_cp,subset=['P&L $'])
            except: _st = _tdf
            st.dataframe(_st, use_container_width=True, hide_index=True, height=380)

        st.caption("⚠️ Backtest uses model's historical accuracy to simulate win/loss. Not financial advice.")

    st.divider()
    
    st.markdown("**💼 Closed Trades History**")
    _ctl = st.session_state.get("closed_trades_log",[])
    if not _ctl:
        st.caption("No closed trades yet — close a trade in Portfolio Tracker.")
    else:
        import pandas as _pd4
        _ctdf = _pd4.DataFrame([{
            "Closed": t.get("closed_at",""),
            "Asset": t.get("ticker",""),
            "Side": t.get("side",""),
            "Entry": f"${t.get('entry',0):,.4f}",
            "Exit": f"${t.get('exit',0):,.4f}",
            "P&L": f"{'+'if (t.get('pnl',0) or 0)>=0 else ''}{(t.get('pnl',0) or 0):,.4f}",
        } for t in reversed(_ctl)])
        def _ctc(v):
            try: return 'color:#3FB950' if float(str(v).replace('+','').replace(',',''))>=0 else 'color:#F85149'
            except: return ''
        try: _cts = _ctdf.style.map(_ctc, subset=['P&L'])
        except: _cts = _ctdf
        st.dataframe(_cts, use_container_width=True, hide_index=True)
        _tot = sum(t.get("pnl",0) or 0 for t in _ctl)
        st.metric("Total Closed P&L", f"{'+'if _tot>=0 else ''}{_tot:,.4f}")





# ════════════════════════════════════════════════════════════════════
# TAB 9: DFM Market — TradingView Live Charts

# ════════════════════════════════════════════════════════════════════
# AUTO-REFRESH (time-based, no external package needed)
# DFM/Stocks: every 6 hours | Crypto: every 5 minutes
# ════════════════════════════════════════════════════════════════════
import time as _time
_UAE_SET_AR = {"EMAAR.DFM","ENBD.DFM","DIB.DFM","DU.DFM","DEWA.DFM",
               "SALIK.DFM","FAB.ADX","ALDAR.ADX","ADCB.ADX","MASQ.DFM"}
_IDX_SET_AR = {"GC=F","SI=F","SPY","QQQ"}

_refresh_secs = (6*3600 if ticker in _UAE_SET_AR or ticker in _IDX_SET_AR else 300)
_last_key     = f"last_refresh_{ticker}"
_now_ts       = _time.time()
_last_ts      = st.session_state.get(_last_key, _now_ts)
st.session_state[_last_key] = _now_ts

if (_now_ts - _last_ts) > _refresh_secs and _last_ts != _now_ts:
    st.cache_data.clear()
    st.rerun()
