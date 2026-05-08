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



def get_tv_symbol(ticker):
    t = ticker.upper()
    if t in TRADINGVIEW_MAP:
        return TRADINGVIEW_MAP[t]
    # Auto-detect: crypto ending in -USD → Binance
    if t.endswith("-USD"):
        return f"BINANCE:{t.replace('-USD','USDT')}"
    return t


# ── TradingView symbol map ─────────────────────────────────────────────────────
_TV_MAP = {
    "SOL-USD":"BINANCE:SOLUSDT","BTC-USD":"BINANCE:BTCUSDT",
    "ETH-USD":"BINANCE:ETHUSDT","ADA-USD":"BINANCE:ADAUSDT",
    "BNB-USD":"BINANCE:BNBUSDT","XRP-USD":"BINANCE:XRPUSDT",
    "DOGE-USD":"BINANCE:DOGEUSDT","AVAX-USD":"BINANCE:AVAXUSDT",
    "MATIC-USD":"BINANCE:MATICUSDT","LINK-USD":"BINANCE:LINKUSDT",
    "EMAAR.DFM":"DFM:EMAAR","AAPL":"NASDAQ:AAPL","TSLA":"NASDAQ:TSLA",
    "MSFT":"NASDAQ:MSFT","NVDA":"NASDAQ:NVDA","AMZN":"NASDAQ:AMZN",
    "GOOGL":"NASDAQ:GOOGL",
}
def get_tv_symbol(t):
    t = t.upper()
    if t in _TV_MAP: return _TV_MAP[t]
    if t.endswith("-USD"): return f"BINANCE:{t.replace('-USD','USDT')}"
    return t

from data_manager import DataManager
from model_engine import ModelEngine

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
# ── Session state: track uploaded assets ──────────────────────────
if "uploaded_assets" not in st.session_state:
    st.session_state.uploaded_assets = {}   # {ticker_name: bytes}
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "SOL-USD"

with st.sidebar:
    st.markdown("## 🔮 Prediction Dashboard")
    st.caption("Auto-updates every 6 hours")
    st.divider()

    # ── Upload CSV FIRST (so it adds to list before selection) ────
    st.subheader("📁 Upload CSV Data")
    st.caption("Upload for any asset — Investing.com or Yahoo Finance format")

    uploaded = st.file_uploader(
        "Drop CSV here",
        type=["csv"],
        help="After upload, give it a ticker name below (e.g. EMAAR.DFM, SOL-USD, NVDA)"
    )

    if uploaded is not None:
        col_name, col_add = st.columns([2,1])
        with col_name:
            csv_ticker = st.text_input(
                "Asset name for this CSV",
                placeholder="e.g. EMAAR.DFM",
                key="csv_ticker_input"
            ).upper().strip()
        with col_add:
            st.write("")
            st.write("")
            if st.button("➕ Add", use_container_width=True) and csv_ticker:
                st.session_state.uploaded_assets[csv_ticker] = uploaded.read()
                st.session_state.selected_ticker = csv_ticker
                st.success(f"✅ Added {csv_ticker}")
                st.rerun()

    # Show uploaded assets
    if st.session_state.uploaded_assets:
        st.caption(f"📂 Uploaded: {', '.join(st.session_state.uploaded_assets.keys())}")
        if st.button("🗑️ Clear all uploads", use_container_width=True):
            st.session_state.uploaded_assets = {}
            st.rerun()

    st.divider()

    # ── Asset Selection — base list + any uploaded CSVs ────────────
    st.subheader("📊 Asset Selection")

    BASE_ASSETS = [
        "SOL-USD","BTC-USD","ETH-USD","ADA-USD","DOGE-USD","BNB-USD",
        "EMAAR.DFM","AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL",
    ]
    # Merge uploaded asset names into the list
    all_assets  = BASE_ASSETS + [k for k in st.session_state.uploaded_assets if k not in BASE_ASSETS]
    all_assets += ["✏️ Custom ticker..."]

    # Default index
    default_idx = all_assets.index(st.session_state.selected_ticker)                   if st.session_state.selected_ticker in all_assets else 0

    selected = st.selectbox("Select asset", all_assets, index=default_idx)

    if selected == "✏️ Custom ticker...":
        ticker = st.text_input(
            "Enter ticker symbol",
            placeholder="e.g. SOL-USD, EMAAR.DFM, AAPL",
            help="Crypto: SOL-USD, BTC-USD\nDubai: EMAAR.DFM, DU.DFM\nUS stocks: AAPL, TSLA"
        ).upper().strip() or "SOL-USD"
    else:
        ticker = selected

    # Keep session state in sync
    if ticker != st.session_state.selected_ticker:
        st.session_state.selected_ticker = ticker

    # Show source badge
    if ticker in st.session_state.uploaded_assets:
        st.success(f"📂 Using your uploaded CSV for **{ticker}** (+ daily API top-up)")
    else:
        src = "Binance" if ticker.replace("-USD","").upper() in ["SOL","BTC","ETH","ADA","DOGE","BNB","AVAX","MATIC","LINK","XRP","LTC"] else "Yahoo Finance"
        st.info(f"📡 Auto-fetching from **{src}** · Updates every 6h")

    st.divider()
    st.subheader("⚙️ Signal Settings")
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
    st.caption("⚠️ Research only · Not financial advice")

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA + TRAIN
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache — fast loads
def load_and_train(ticker, _uploaded_bytes=None, _force=False):
    import io
    dm       = DataManager(ticker)
    file_obj = io.BytesIO(_uploaded_bytes) if _uploaded_bytes else None
    # Crypto assets get hourly data for better intraday signals
    _use_hourly = is_crypto(ticker)
    try:
        df_raw = dm.get_data(uploaded_file=file_obj, prefer_hourly=_use_hourly)
    except TypeError:
        # Fallback: old data_manager without prefer_hourly param
        df_raw = dm.get_data(uploaded_file=file_obj)
    df_feat  = build_features(df_raw)
    engine   = ModelEngine(df_feat)
    results  = engine.train(verbose=False)
    return df_raw, df_feat, results

# Get uploaded bytes for the selected ticker (if any)
_uploaded_bytes = st.session_state.uploaded_assets.get(ticker, None)

with st.spinner(f"⏳ Loading **{ticker}** · Training models (15–20 sec first load, cached after)..."):
    try:
        df_raw, df_feat, results = load_and_train(
            ticker, _uploaded_bytes, force_refresh)
        # Clear the spinner — success

    except Exception as e:
        st.markdown(f"""
<div style="background:#2D1B1B;border:2px solid #F85149;border-radius:10px;padding:20px 24px;margin:20px 0">
  <div style="color:#F85149;font-size:1.1rem;font-weight:bold;margin-bottom:8px">❌ Error Loading Data</div>
  <div style="color:#F0F6FC;font-size:0.9rem;font-family:monospace">{str(e)}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div style="background:#1C2128;border:1px solid #30363D;border-radius:8px;padding:16px 20px">
  <div style="color:#E3B341;font-weight:bold;margin-bottom:8px">💡 What to do:</div>
  <div style="color:#C9D1D9;font-size:0.88rem;line-height:1.8">
    1. Make sure ALL 4 files are updated on GitHub: <code>app.py</code>, <code>data_manager.py</code>, <code>model_engine.py</code>, <code>requirements.txt</code><br>
    2. For crypto (SOL, BTC, ETH): data auto-loads from Binance — no action needed<br>
    3. For stocks or Dubai assets: upload a CSV using the sidebar uploader<br>
    4. Try clicking <b>Force Refresh Data</b> in the sidebar
  </div>
</div>""", unsafe_allow_html=True)
        st.stop()

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
display_price = live_price if live_price else last_close

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

# ── ATR-based TP / SL ─────────────────────────────────────────────
# Use the last ATR value to set realistic TP and SL
last_atr    = float(df_feat['ATR'].iloc[-1]) if 'ATR' in df_feat.columns else display_price * 0.03
atr_pct     = last_atr / display_price

# BUY: TP = price + 2×ATR  |  SL = price - 1.5×ATR
# SELL: TP = price - 2×ATR  |  SL = price + 1.5×ATR
if last_sig == "BUY":
    entry_p  = round(display_price * 0.999, 4)  # slight dip entry
    tp_price = round(display_price + 2.0 * last_atr, 4)
    sl_price = round(display_price - 1.5 * last_atr, 4)
elif last_sig == "SELL":
    entry_p  = round(display_price * 1.001, 4)
    tp_price = round(display_price - 2.0 * last_atr, 4)
    sl_price = round(display_price + 1.5 * last_atr, 4)
else:
    # HOLD — no trade, so no entry/TP/SL
    entry_p  = display_price
    tp_price = None
    sl_price = None

if last_sig == "HOLD":
    rr = 0.0
    tp_pct = 0.0
    sl_pct = 0.0
    tp_str = "— (No signal)"
    sl_str = "— (No signal)"
else:
    rr = abs(tp_price - entry_p) / max(abs(entry_p - sl_price), 0.0001)
    tp_pct = (tp_price - display_price) / display_price * 100
    sl_pct = (sl_price - display_price) / display_price * 100
    tp_str = f"${tp_price:,.4f}"
    sl_str = f"${sl_price:,.4f}" 

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
st.subheader("🎯 Trading Signals")
col_sig, col_7d = st.columns([1, 2])

with col_sig:
    card_cls = "buy-card" if last_sig=="BUY" else "sell-card" if last_sig=="SELL" else "hold-card"
    emoji    = "🟢" if last_sig=="BUY" else "🔴" if last_sig=="SELL" else "⚪"
    sig_c    = C_UP if last_sig=="BUY" else C_DOWN if last_sig=="SELL" else C_GREY
    # next_str already computed above (central date logic)
    tp_c  = C_UP   if last_sig=="BUY"  else C_DOWN if last_sig=="SELL" else C_GREY
    sl_c  = C_DOWN if last_sig=="BUY"  else C_UP   if last_sig=="SELL" else C_GREY

    st.markdown(f"""
    <div class="signal-card {card_cls}">
        <div style="font-size:0.80rem;color:#8B949E;margin-bottom:2px">TOMORROW — {next_str}</div>
        <div style="font-size:2.2rem;font-weight:bold;color:{sig_c};margin:2px 0">{emoji} {last_sig}</div>
        <div style="color:#8B949E;font-size:0.88rem">Confidence: <b style="color:#E3B341">{last_conf:.1f}%</b>
        &nbsp;|&nbsp; P(UP): <b style="color:#58A6FF">{last_prob*100:.1f}%</b></div>
        <hr style="border-color:#30363D;margin:10px 0">
        <table style="width:100%;font-size:0.88rem;border-collapse:collapse">
            <tr style="border-bottom:1px solid #30363D">
                <td style="color:#8B949E;padding:4px 0">Live Price</td>
                <td style="color:#F0F6FC;text-align:right;font-weight:bold">${display_price:,.4f}</td>
            </tr>
            <tr style="border-bottom:1px solid #30363D">
                <td style="color:#8B949E;padding:4px 0">Entry</td>
                <td style="color:#58A6FF;text-align:right;font-weight:bold">${entry_p:,.4f}</td>
            </tr>
            <tr style="border-bottom:1px solid #30363D">
                <td style="color:#8B949E;padding:4px 0">🎯 Take Profit</td>
                <td style="color:{tp_c};text-align:right;font-weight:bold">
                    {tp_str}{" <span style='font-size:0.8rem'>(" + f"{tp_pct:+.2f}%" + ")</span>" if last_sig != "HOLD" else ""}
                </td>
            </tr>
            <tr style="border-bottom:1px solid #30363D">
                <td style="color:#8B949E;padding:4px 0">🛑 Stop Loss</td>
                <td style="color:{sl_c};text-align:right;font-weight:bold">
                    {sl_str}{" <span style='font-size:0.8rem'>(" + f"{sl_pct:+.2f}%" + ")</span>" if last_sig != "HOLD" else ""}
                </td>
            </tr>
            <tr>
                <td style="color:#8B949E;padding:4px 0">Risk/Reward</td>
                <td style="color:#F0F6FC;text-align:right;font-weight:bold">1 : {rr:.2f}</td>
            </tr>
        </table>
        <div style="font-size:0.75rem;color:#6E7681;margin-top:8px">
            TP/SL based on ATR ({atr_pct*100:.2f}% of price) · Not financial advice
        </div>
    </div>""", unsafe_allow_html=True)

with col_7d:
    st.markdown("**📅 7-Day Forward Outlook**")
    st.caption("Confidence decays further out — tomorrow is the most reliable")
    rows = []
    for i in range(7):
        _d = _today_date + timedelta(days=i+1)
        if not _is_crypto:   # stocks skip weekends
            while _d.weekday() >= 5:
                _d += timedelta(days=1)
        dt = datetime(_d.year, _d.month, _d.day)
        prob = 0.5 + (last_prob-0.5) * np.exp(-0.35*i)
        sig  = ("🟢 BUY"  if prob >= confidence_thresh else
                "🔴 SELL" if prob <= (1-confidence_thresh) else "⚪ HOLD")
        # ATR-based TP/SL for each day
        _dsig  = ("BUY" if prob >= confidence_thresh else "SELL" if prob <= (1-confidence_thresh) else "HOLD")
        if _dsig == "BUY":
            day_tp = round(display_price + 2.0*last_atr, 4)
            day_sl = round(display_price - 1.5*last_atr, 4)
        elif _dsig == "SELL":
            day_tp = round(display_price - 2.0*last_atr, 4)
            day_sl = round(display_price + 1.5*last_atr, 4)
        else:
            day_tp = None; day_sl = None
        rows.append({
            "Date"       : dt.strftime("%a %d %b"),
            "Signal"     : sig,
            "Direction"  : "📈 UP" if prob>0.5 else "📉 DOWN",
            "Confidence" : f"{max(prob,1-prob)*100:.0f}%",
            "P(UP)"      : f"{prob*100:.1f}%",
            "TP"         : f"${day_tp:,.4f}" if day_tp else "—",
            "SL"         : f"${day_sl:,.4f}" if day_sl else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    n_buy_w  = sum(1 for r in rows if "BUY"  in r["Signal"])
    n_sell_w = sum(1 for r in rows if "SELL" in r["Signal"])
    out = "📈 BULLISH" if n_buy_w>n_sell_w else "📉 BEARISH" if n_sell_w>n_buy_w else "➡️ SIDEWAYS"
    st.markdown(f"**Weekly Outlook: {out}** &nbsp;|&nbsp; 🟢 {n_buy_w} BUY &nbsp;🔴 {n_sell_w} SELL &nbsp;⚪ {7-n_buy_w-n_sell_w} HOLD")

st.divider()

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab0,tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📡 Live Chart",
    "📈 Price & Signals",
    "🎯 Predicted vs Actual",
    "📊 Model Performance",
    "📜 Signal History",
    "📰 News & Sentiment",
    "💼 Portfolio Tracker",
])

# ── TAB 0: Live Chart + Signal Dashboard ──────────────────────────
with tab0:
    _tv_sym = get_tv_symbol(ticker)

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
      "allow_symbol_change": false,
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
    st.components.v1.html(_tv_live_html, height=670, scrolling=False)

    # ── Signal history table under the live chart ─────────────────
    if not sig_hist.empty:
        st.divider()
        st.markdown("**📋 Recent Signals — Last 20**")
        st.caption(
            "Signals from backtest where model confidence ≥60% · "
            "TP/SL calculated using ATR at time of signal"
        )
        # Build display table with TP/SL per signal
        _hist_rows = []
        for _, _row in sig_hist.head(20).iterrows():
            try:
                _p  = float(str(_row.get("Price","0")).replace("$","").replace(",",""))
                _is = "BUY" in str(_row.get("Signal",""))
                _tp = round(_p + 2.0*last_atr, 4) if _is else round(_p - 2.0*last_atr, 4)
                _sl = round(_p - 1.5*last_atr, 4) if _is else round(_p + 1.5*last_atr, 4)
                _rr_val = abs(_tp-_p)/max(abs(_p-_sl),0.0001)
                _hist_rows.append({
                    "Date"       : _row.get("Date",""),
                    "Signal"     : _row.get("Signal",""),
                    "Price"      : _row.get("Price",""),
                    "Take Profit": f"${_tp:,.4f}",
                    "Stop Loss"  : f"${_sl:,.4f}",
                    "R/R"        : f"1:{_rr_val:.2f}",
                    "Confidence" : _row.get("Confidence",""),
                })
            except Exception:
                pass
        if _hist_rows:
            _hist_df = pd.DataFrame(_hist_rows)
            def _color_sig(val):
                if 'BUY'  in str(val): return 'color:#3FB950;font-weight:bold'
                if 'SELL' in str(val): return 'color:#F85149;font-weight:bold'
                return 'color:#6E7681'
            try:
                styled = _hist_df.style.map(_color_sig, subset=['Signal'])
            except Exception:
                styled = _hist_df
            st.dataframe(styled, use_container_width=True, hide_index=True, height=420)


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
    dark_fig()
    try:
        rmse = float(np.sqrt(mean_squared_error(y_price_te,price_pred)))
        mae  = float(mean_absolute_error(y_price_te,price_pred))
        mape = float(np.mean(np.abs((y_price_te-price_pred)/(y_price_te+1e-9)))*100)
        dacc = float(accuracy_score(y_te,ens_pred))
    except Exception:
        rmse=mae=mape=0.0; dacc=ens_acc

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
    ax.plot(D2,PP,color=C_GOLD,lw=1.2,ls='--',alpha=0.9,label='Predicted (Ridge)',zorder=3)
    ax.fill_between(D2,C2,PP,where=(PP>=C2),alpha=0.10,color=C_UP)
    ax.fill_between(D2,C2,PP,where=(PP<C2),alpha=0.10,color=C_DOWN)
    ax.text(0.99,0.97,f'RMSE=${rmse:.4f}  MAE=${mae:.4f}  MAPE={mape:.2f}%',
        transform=ax.transAxes,ha='right',va='top',fontsize=9,color=C_GOLD,
        bbox=dict(fc='#161B22',ec='#30363D',boxstyle='round,pad=0.4'))
    ax.set_ylabel('Price'); ax.legend(loc='upper left')
    ax.xaxis.set_ticklabels([])
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=0.2)

    ax2=axes[1]; ax2.set_facecolor('#161B22')
    err=PP-C2
    ax2.fill_between(D2,err,0,where=(err>=0),color=C_UP,alpha=0.55,label='Too high')
    ax2.fill_between(D2,err,0,where=(err<0),color=C_DOWN,alpha=0.55,label='Too low')
    ax2.axhline(0,color=C_DIM,lw=0.8,ls='--')
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
    if sig_hist is not None and not sig_hist.empty:
        st.subheader(f"📋 Historical Backtest Signals — {name}")
        st.info(
            "📌 **These are BACKTEST signals** — they show where the model would have signalled "
            "BUY or SELL on **past historical data** to measure accuracy.  \n"
            f"The **current signal** (for the next session **{next_str}**) is shown in the "
            "Trading Signals section above.  \n"
            "Confidence ≥60% filter applied. TP/SL based on ATR."
        )

        # Add TP/SL to signal history
        hist = sig_hist.copy()
        hist["TP"] = hist["Price"].str.replace("$","").astype(float).apply(
            lambda p: f"${round(p+2*last_atr if '🟢' in hist.loc[hist['Price'].str.replace('$','').astype(float)==p,'Signal'].values[0] else p-2*last_atr, 4):.4f}"
            if not hist[hist['Price'].str.replace('$','').astype(float)==p].empty else "—"
        ) if "Signal" in hist.columns else "—"
        hist["SL"] = hist["Price"].str.replace("$","").astype(float).apply(
            lambda p: f"${round(p-1.5*last_atr if '🟢' in hist.loc[hist['Price'].str.replace('$','').astype(float)==p,'Signal'].values[0] else p+1.5*last_atr, 4):.4f}"
            if not hist[hist['Price'].str.replace('$','').astype(float)==p].empty else "—"
        ) if "Signal" in hist.columns else "—"

        def color_sig(val):
            if 'BUY' in str(val):  return 'color:#3FB950;font-weight:bold'
            if 'SELL' in str(val): return 'color:#F85149;font-weight:bold'
            return 'color:#6E7681'
        try:
            styled=sig_hist.style.map(color_sig,subset=['Signal'])
        except Exception:
            styled=sig_hist
        st.dataframe(styled,use_container_width=True,hide_index=True,height=500)

        s1,s2,s3,s4=st.columns(4)
        nb=(sig_hist['Signal']=='🟢 BUY').sum()
        ns=(sig_hist['Signal']=='🔴 SELL').sum()
        s1.metric("Total Signals",len(sig_hist))
        s2.metric("BUY Signals",int(nb))
        s3.metric("SELL Signals",int(ns))
        try:
            ac=sig_hist['Confidence'].str.replace('%','').astype(float).mean()
            s4.metric("Avg Confidence",f"{ac:.1f}%")
        except Exception: s4.metric("Avg Confidence","—")
    else:
        st.info("No filtered signals. Try lowering the confidence threshold in the sidebar.")

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
        _ff_sub1, _ff_sub2, _ff_sub3 = st.tabs([
            "🌐 ForexFactory Live Site",
            "📊 TradingView Economic Calendar",
            "📋 Our Calendar Data",
        ])

        # ── ForexFactory direct iframe ────────────────────────────
        with _ff_sub1:
            st.caption(
                "ForexFactory.com embedded live — you can search, filter, "
                "and browse exactly as on their website. "
                "If Cloudflare shows a challenge page, use the TradingView tab instead."
            )
            _ff_html = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0D1117; }
  #ff-frame {
    width: 100%; height: 720px; border: none;
    border-radius: 8px; overflow: hidden;
  }
  .ff-toolbar {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px 8px 0 0; padding: 10px 16px;
    display: flex; align-items: center; gap: 12px;
  }
  .ff-btn {
    background: #21262D; color: #C9D1D9; border: 1px solid #30363D;
    border-radius: 5px; padding: 5px 14px; cursor: pointer;
    font-size: 0.82rem; font-family: sans-serif;
    transition: background 0.2s;
  }
  .ff-btn:hover { background: #1F6FEB; color: white; }
  .ff-url { color: #8B949E; font-size: 0.75rem; font-family: monospace; }
</style></head><body>
<div class="ff-toolbar">
  <button class="ff-btn" onclick="loadPage('https://www.forexfactory.com/calendar')">
    📅 Calendar
  </button>
  <button class="ff-btn" onclick="loadPage('https://www.forexfactory.com/news')">
    📰 News
  </button>
  <button class="ff-btn" onclick="loadPage('https://www.forexfactory.com/market')">
    📈 Market
  </button>
  <button class="ff-btn" onclick="loadPage('https://www.forexfactory.com/trades')">
    🔄 Trades
  </button>
  <button class="ff-btn" onclick="document.getElementById('ff-frame').src=document.getElementById('ff-frame').src">
    🔄 Reload
  </button>
  <span class="ff-url" id="ff-current-url">forexfactory.com/calendar</span>
</div>
<iframe
  id="ff-frame"
  src="https://www.forexfactory.com/calendar"
  sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-top-navigation"
  referrerpolicy="no-referrer-when-downgrade"
  loading="lazy"
></iframe>
<script>
  function loadPage(url) {
    document.getElementById('ff-frame').src = url;
    document.getElementById('ff-current-url').textContent = url.replace('https://www.','');
  }
  // Update URL display when iframe navigates
  document.getElementById('ff-frame').onload = function() {
    try {
      var u = this.contentWindow.location.href;
      document.getElementById('ff-current-url').textContent = u.replace('https://www.','');
    } catch(e) {}
  };
</script>
</body></html>"""
            st.components.v1.html(_ff_html, height=790, scrolling=False)

        # ── TradingView Economic Calendar ─────────────────────────
        with _ff_sub2:
            st.caption(
                "TradingView's official economic calendar widget — "
                "100% guaranteed to work, filterable by country and importance"
            )
            _tv_cal_html = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>* { margin:0; padding:0; } body { background:#0D1117; }</style>
</head><body>
<div class="tradingview-widget-container" style="height:700px;width:100%">
  <div class="tradingview-widget-container__widget"></div>
  <script
    type="text/javascript"
    src="https://s3.tradingview.com/external-embedding/embed-widget-events.js"
    async>
  {
    "colorTheme": "dark",
    "isTransparent": false,
    "width": "100%",
    "height": "700",
    "locale": "en",
    "importanceFilter": "-1,0,1",
    "countryFilter": "us,eu,gb,jp,ca,au,nz,ch",
    "backgroundColor": "#0D1117",
    "dateRangeFilter": "this_week"
  }
  </script>
</div>
</body></html>"""
            st.components.v1.html(_tv_cal_html, height=720, scrolling=False)

        # ── Our parsed calendar data ───────────────────────────────
        with _ff_sub3:
            st.caption(
                "Calendar data fetched from ForexFactory API · "
                "Red = High impact | Yellow = Medium | Filter by currency and impact"
            )
            if not _fcal:
                st.info(
                    "Our calendar data will load on Streamlit Cloud. "
                    "Locally it may be blocked — use the ForexFactory or TradingView tabs above."
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
                    _imp_colors = {"High":"#F85149","Medium":"#E3B341","Low":"#6E7681","Holiday":"#30363D"}
                    for _e in _fev:
                        _ic  = _imp_colors.get(_e["impact_raw"],"#30363D")
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
                    st.caption(f"Showing {len(_fev)} of {len(_fcal)} events | Source: ForexFactory.com")


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
                st.success(f"Closed trade #{int(_close_idx)} at ${_exit_price:.4f}")
                st.rerun()

        with _mc2:
            _del_idx = st.number_input("Trade # to delete (1-based)", min_value=1,
                                        max_value=len(trades), value=1, key="del_idx")
            if st.button("🗑️ Delete Trade", use_container_width=True):
                st.session_state.portfolio_trades.pop(int(_del_idx)-1)
                st.rerun()

        with _mc3:
            if st.button("🗑️ Clear All Trades", use_container_width=True):
                st.session_state.portfolio_trades = []
                st.rerun()

# ── Auto-refresh every 60s for live price ──────────────────────────────────
st.markdown(
    '<script>setTimeout(function(){window.location.reload();},60000);</script>',
    unsafe_allow_html=True
)

st.markdown(
    f"<div style='text-align:center;color:#6E7681;font-size:0.78rem'>"
    f"🔮 {name} Prediction Dashboard · Data: Binance + Yahoo Finance · "
    f"Last data: {last_date.strftime('%d %b %Y')} · Refreshes every 6h · "
    f"⚠️ Research only — not financial advice"
    f"</div>",unsafe_allow_html=True)
