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
    st.caption("Data auto-refreshes every 6h")
    st.divider()
    st.caption("⚠️ Research only · Not financial advice")

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA + TRAIN
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=21600, show_spinner=False)   # cache 6 hours
def load_and_train(ticker, _uploaded_bytes=None, _force=False):
    """
    Load data and train models.
    _uploaded_bytes: raw CSV bytes if user uploaded a file for this ticker.
    If bytes provided, they are used as the historical base and the API
    appends any newer rows on top (so data stays current even after upload).
    """
    import io
    dm       = DataManager(ticker)
    file_obj = io.BytesIO(_uploaded_bytes) if _uploaded_bytes else None
    df_raw   = dm.get_data(uploaded_file=file_obj)
    df_feat  = build_features(df_raw)
    engine   = ModelEngine(df_feat)
    results  = engine.train(verbose=False)
    return df_raw, df_feat, results

# Get uploaded bytes for the selected ticker (if any)
_uploaded_bytes = st.session_state.uploaded_assets.get(ticker, None)

with st.spinner(f"🔄 Loading **{ticker}** data and training models..."):
    try:
        df_raw, df_feat, results = load_and_train(
            ticker, _uploaded_bytes, force_refresh)
    except Exception as e:
        st.error(str(e))
        st.info(
            "💡 **Tips:**\n"
            "- For Emaar: upload a CSV from Investing.com, name it `EMAAR.DFM`\n"
            "- Check the ticker is correct (e.g. `SOL-USD`, `AAPL`, `EMAAR.DFM`)\n"
            "- Make sure you have internet connection"
        )
        st.stop()

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
    # Smart date: find next weekday after last candle
    from datetime import datetime as _dt, timezone as _tz
    _now_dubai   = _dt.now(_tz.utc) + timedelta(hours=4)   # Dubai = UTC+4
    _today       = _now_dubai.date()
    _last        = last_date.date() if hasattr(last_date, 'date') else last_date
    # If the last candle is today (data already updated), signal applies to tomorrow
    # If the last candle is yesterday or older, next trading day is what matters
    _next = _last + timedelta(days=1)
    # Skip weekends (markets closed Sat/Sun for most assets)
    while _next.weekday() >= 5:   # 5=Saturday, 6=Sunday
        _next += timedelta(days=1)
    next_str = _next.strftime('%A %d %b %Y')

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
        dt   = last_date + timedelta(days=i+1)
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
tab0,tab1,tab2,tab3,tab4 = st.tabs([
    "📡 Live Chart","📈 Price & Signals","🎯 Predicted vs Actual",
    "📊 Model Performance","📜 Signal History"
])

# ── TAB 0: Live TradingView Chart ─────────────────────────────────
with tab0:
    tv_symbol = get_tv_symbol(ticker)
    st.markdown(f"**📡 Live Chart — {name} (`{tv_symbol}`)**")
    st.caption("Real-time data from TradingView · Updates live · Switch timeframes inside the chart")

    tv_html = f"""<div class="tradingview-widget-container" style="height:580px;width:100%">
      <div id="tv_chart_main"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
        new TradingView.widget({{
          "container_id"     : "tv_chart_main",
          "width"            : "100%",
          "height"           : 560,
          "symbol"           : "{tv_symbol}",
          "interval"         : "D",
          "timezone"         : "Asia/Dubai",
          "theme"            : "dark",
          "style"            : "1",
          "locale"           : "en",
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"]
        }});
      </script>
    </div>"""
    st.components.v1.html(tv_html, height=580, scrolling=False)

    st.caption("💡 Inside the chart: click timeframe buttons (1m 5m 1h 4h 1D 1W) in the toolbar")
    _sc1, _sc2, _sc3 = st.columns(3)
    _sc1.metric("Live Price",      f"${display_price:,.4f}", delta=f"{day_chg:+.2f}%")
    _sc2.metric("Tomorrow Signal", f"{emoji} {last_sig}",    delta=f"Confidence: {last_conf:.1f}%")
    if last_sig != "HOLD" and tp_price and sl_price:
        _sc3.metric("TP / SL",
            f"TP: ${tp_price:,.4f}",
            delta=f"SL: ${sl_price:,.4f}",
            delta_color="inverse")
    else:
        _sc3.metric("Signal", "⚪ HOLD", delta="Wait for clearer signal")

# ── TAB 1: Price & Signals ─────────────────────────────────────────
with tab1:
    dark_fig()
    n_show  = min(lookback_days, len(te_df))
    te_show = te_df.iloc[-n_show:].copy()
    pr_show = np.array(ens_proba[-n_show:])
    sg_show = np.array(signals[-n_show:])
    b_show  = sg_show ==  1
    s_show  = sg_show == -1
    D       = np.array(te_show.index)
    cl      = te_show['Close'].values.astype(float)
    hi      = te_show['High'].values.astype(float)
    lo      = te_show['Low'].values.astype(float)
    reg     = te_show['Regime'].values if 'Regime' in te_show.columns else np.ones(n_show)

    fig = plt.figure(figsize=(14,12))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(4,1,figure=fig,height_ratios=[4.5,1.2,1.0,1.0],hspace=0.04)

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#161B22')
    for i in range(len(D)-1):
        ax1.axvspan(D[i],D[i+1],alpha=1,
            color='#1C2A1C' if reg[i]==1 else '#2A1C1C',linewidth=0,zorder=0)
    ax1.plot(D,cl,color=C_BLUE,lw=1.4,label='Close',zorder=4)
    if 'SMA20' in te_show.columns: ax1.plot(D,te_show['SMA20'].values,color=C_GOLD,lw=0.9,ls='--',alpha=0.8,label='SMA20')
    if 'SMA50' in te_show.columns: ax1.plot(D,te_show['SMA50'].values,color='#A371F7',lw=0.9,ls='-.',alpha=0.8,label='SMA50')
    if b_show.sum()>0:
        ax1.scatter(D[b_show],lo[b_show]*0.955,marker='^',s=110,color=C_UP,
            edgecolors='#196127',lw=0.6,zorder=6,label=f'BUY ({int(b_show.sum())} in window)')
        for d,p in zip(D[b_show],cl[b_show]):
            ax1.annotate(f'TP:${round(p+2*last_atr,2)}',
                (d,p*0.935),fontsize=6.5,color=C_UP,ha='center',
                bbox=dict(boxstyle='round,pad=0.15',fc='#0D1117',ec='none',alpha=0.8))
    if s_show.sum()>0:
        ax1.scatter(D[s_show],hi[s_show]*1.045,marker='v',s=110,color=C_DOWN,
            edgecolors='#8B0000',lw=0.6,zorder=6,label=f'SELL ({int(s_show.sum())} in window)')
        for d,p in zip(D[s_show],cl[s_show]):
            ax1.annotate(f'TP:${round(p-2*last_atr,2)}',
                (d,p*1.065),fontsize=6.5,color=C_DOWN,ha='center',
                bbox=dict(boxstyle='round,pad=0.15',fc='#0D1117',ec='none',alpha=0.8))
    # Tomorrow arrow
    next_dt = D[-1]+pd.Timedelta(days=1)
    tgt     = float(cl[-1])*(1+(float(pr_show[-1])-0.5)*0.06)
    arrc    = C_UP if pr_show[-1]>0.5 else C_DOWN
    ax1.annotate('',xy=(next_dt,tgt),xytext=(D[-1],float(cl[-1])),
        arrowprops=dict(arrowstyle='-|>',color=arrc,lw=2.5,mutation_scale=16))
    ax1.scatter([next_dt],[tgt],marker='*',s=220,color=C_GOLD,zorder=9)
    _tp_txt = f"TP:${tp_price:,.2f}" if tp_price is not None else "HOLD"
    _sl_txt = f"SL:${sl_price:,.2f}" if sl_price is not None else "No trade"
    ax1.annotate(f'  TOMORROW\n  {last_sig}\n  {_tp_txt}\n  {_sl_txt}',
        (next_dt,tgt),fontsize=8,color=arrc,fontweight='bold',
        xytext=(8,0),textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.35',fc='#161B22',ec='#30363D',alpha=0.95))
    bull_p=mpatches.Patch(color='#1C2A1C',label='Bull Regime')
    bear_p=mpatches.Patch(color='#2A1C1C',label='Bear Regime')
    h,l2=ax1.get_legend_handles_labels()
    ax1.legend(h+[bull_p,bear_p],l2+['Bull','Bear'],loc='upper left',ncol=4,framealpha=0.85)
    ax1.set_ylabel('Price'); ax1.xaxis.set_ticklabels([])
    ax1.spines[['top','right']].set_visible(False); ax1.grid(axis='y',alpha=0.2)
    ax1.set_title(f'{name} ({ticker}) — Signals  |  {n_show}d shown  |  '
                  f'Acc:{ens_acc*100:.1f}%  |  {int(b_show.sum())}BUY {int(s_show.sum())}SELL',
        color=C_WHITE,fontsize=11,pad=6)

    ax2=fig.add_subplot(gs[1],sharex=ax1); ax2.set_facecolor('#161B22')
    bc=np.where(pr_show>=HIGH,C_UP,np.where(pr_show<=LOW,C_DOWN,C_GREY))
    ax2.bar(D,pr_show,color=bc,width=1.0,alpha=0.85)
    ax2.fill_between(D,HIGH,pr_show,where=(pr_show>=HIGH),alpha=0.2,color=C_UP)
    ax2.fill_between(D,LOW,pr_show,where=(pr_show<=LOW),alpha=0.2,color=C_DOWN)
    ax2.axhline(HIGH,color=C_UP,ls='--',lw=1.0,label=f'Buy≥{HIGH:.0%}')
    ax2.axhline(LOW,color=C_DOWN,ls='--',lw=1.0,label=f'Sell≤{LOW:.0%}')
    ax2.axhline(0.5,color=C_DIM,ls=':',lw=0.7)
    ax2.set_ylabel('P(UP)'); ax2.set_ylim(0,1); ax2.legend(loc='upper right',ncol=2)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y',alpha=0.2)

    ax3=fig.add_subplot(gs[2],sharex=ax1); ax3.set_facecolor('#161B22')
    if 'RSI' in te_show.columns:
        rsi=te_show['RSI'].values
        ax3.plot(D,rsi,color='#D29922',lw=0.9)
        ax3.fill_between(D,70,rsi,where=(rsi>70),alpha=0.25,color=C_DOWN)
        ax3.fill_between(D,30,rsi,where=(rsi<30),alpha=0.25,color=C_UP)
        ax3.axhline(70,color=C_DOWN,ls='--',lw=0.8,alpha=0.7)
        ax3.axhline(30,color=C_UP,ls='--',lw=0.8,alpha=0.7)
        if b_show.sum()>0: ax3.scatter(D[b_show],rsi[b_show],marker='^',s=35,color=C_UP,zorder=5)
        if s_show.sum()>0: ax3.scatter(D[s_show],rsi[s_show],marker='v',s=35,color=C_DOWN,zorder=5)
    ax3.set_ylabel('RSI'); ax3.set_ylim(10,90)
    ax3.xaxis.set_ticklabels([])
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y',alpha=0.2)

    ax4=fig.add_subplot(gs[3],sharex=ax1); ax4.set_facecolor('#161B22')
    if 'MACD' in te_show.columns and 'MACD_sig' in te_show.columns:
        mc=te_show['MACD'].values; ms=te_show['MACD_sig'].values; mh=mc-ms
        ax4.plot(D,mc,color=C_BLUE,lw=1.0,label='MACD')
        ax4.plot(D,ms,color='#F78166',lw=1.0,ls='--',label='Signal')
        ax4.bar(D,mh,color=np.where(mh>=0,C_UP,C_DOWN),width=1.0,alpha=0.55)
        ax4.axhline(0,color=C_DIM,lw=0.7); ax4.legend(loc='upper left',ncol=2)
    ax4.set_ylabel('MACD'); ax4.set_xlabel('Date')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax4.xaxis.get_majorticklabels(),rotation=30,ha='right')
    ax4.spines[['top','right']].set_visible(False); ax4.grid(axis='y',alpha=0.2)
    st.pyplot(fig,use_container_width=True); plt.close()

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
        st.subheader(f"📋 All Buy/Sell Signals — {name} Test Period")
        st.caption("Only signals where model confidence ≥60% shown | Includes ATR-based TP/SL")

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
st.markdown(
    f"<div style='text-align:center;color:#6E7681;font-size:0.78rem'>"
    f"🔮 {name} Prediction Dashboard · Data: Binance + Yahoo Finance · "
    f"Last data: {last_date.strftime('%d %b %Y')} · Refreshes every 6h · "
    f"⚠️ Research only — not financial advice"
    f"</div>",unsafe_allow_html=True)
