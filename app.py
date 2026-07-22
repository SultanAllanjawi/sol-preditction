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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap');

@keyframes fadeInUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes popIn{0%{opacity:0;transform:scale(0.85)}70%{transform:scale(1.04)}100%{opacity:1;transform:scale(1)}}
@keyframes pulseDot{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(45,212,191,0.55)}50%{opacity:0.75;box-shadow:0 0 0 6px rgba(45,212,191,0)}}
@keyframes pulseDotRed{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(251,113,133,0.55)}50%{opacity:0.75;box-shadow:0 0 0 6px rgba(251,113,133,0)}}
@keyframes pulseDotGold{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(245,158,11,0.5)}50%{opacity:0.7;box-shadow:0 0 0 6px rgba(245,158,11,0)}}
@keyframes gradientShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes fillBar{from{width:0}to{width:var(--tw)}}
@keyframes glowPulse{0%,100%{box-shadow:0 10px 26px rgba(245,158,11,0.35)}50%{box-shadow:0 10px 40px rgba(245,158,11,0.65)}}

* { font-family: 'Inter', -apple-system, sans-serif; }
code, .stCode, [data-testid="stMetricDelta"] svg { font-family: 'JetBrains Mono', monospace !important; }

.stApp{
  color:#E2E8F0;
  background:
    radial-gradient(circle at 12% 8%, rgba(45,212,191,0.12), transparent 42%),
    radial-gradient(circle at 88% 15%, rgba(45,212,191,0.08), transparent 40%),
    radial-gradient(circle at 50% 100%, rgba(245,158,11,0.07), transparent 45%),
    #070812;
}
.stat-pop{display:inline-block;animation:popIn .5s cubic-bezier(.26,1.4,.4,1) both}
.status-dot{display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:8px;vertical-align:middle}
.status-dot.buy{background:#2DD4BF;animation:pulseDot 1.6s infinite}
.status-dot.sell{background:#FB7185;animation:pulseDotRed 1.6s infinite}
.status-dot.hold{background:#F59E0B;animation:pulseDotGold 1.6s infinite}
.outlook-row{animation:fadeInUp .4s ease both}
.block-container{padding-top:2.6rem}
.block-container > div{animation:fadeInUp .45s ease both}
/* orchestrated stagger: each top-level section fades in slightly after the previous */
.block-container > div:nth-child(1){animation-delay:0s}
.block-container > div:nth-child(2){animation-delay:.04s}
.block-container > div:nth-child(3){animation-delay:.08s}
.block-container > div:nth-child(4){animation-delay:.12s}
.block-container > div:nth-child(5){animation-delay:.16s}
.block-container > div:nth-child(6){animation-delay:.20s}
.block-container > div:nth-child(7){animation-delay:.24s}
.block-container > div:nth-child(n+8){animation-delay:.28s}
/* charts/images fade in like everything else instead of popping in instantly */
[data-testid="stImage"] img{animation:fadeInUp .5s ease both}
/* shimmer on the loading spinner text */
[data-testid="stSpinner"] p{
  background:linear-gradient(90deg,#94A3B8 0%,#2DD4BF 50%,#94A3B8 100%);
  background-size:200% auto;-webkit-background-clip:text;background-clip:text;
  color:transparent;animation:gradientShift 1.8s linear infinite;
}

/* header toolbar */
header[data-testid="stHeader"]{background:#070812;z-index:999}
#MainMenu, footer{visibility:hidden}
/* hide Streamlit's auto heading-anchor icon (stray floating link/+ glyph on hover) */
[data-testid="stHeaderActionElements"], .stMarkdown a.anchor-link, [data-testid="stHeadingWithActionElements"] svg{
  display:none!important;
}

/* metric tiles */
[data-testid="metric-container"]{
  background:linear-gradient(160deg,#161B2C 0%,#10121C 100%);
  border:1px solid #1E2333;border-radius:14px;padding:16px 20px;
  box-shadow:0 8px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  transition:transform .25s cubic-bezier(.2,.8,.3,1), border-color .2s ease, box-shadow .25s ease;
}
[data-testid="metric-container"]:hover{
  border-color:#2DD4BF; transform:translateY(-4px);
  box-shadow:0 16px 32px rgba(45,212,191,0.22), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stMetricValue"]{color:#FFFFFF;font-size:1.7rem!important;font-weight:800}
[data-testid="stMetricLabel"]{color:#94A3B8;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:600}
[data-testid="stMetricDelta"]{font-weight:700;border-radius:20px;padding:2px 9px;display:inline-flex}
[data-testid="stMetricDelta"] svg{display:none}

/* live indicator dot on the price header */
.live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2DD4BF;
  margin-right:6px;animation:pulseDot 1.8s infinite}
.live-dot.down{background:#FB7185;animation-name:pulseDotRed}

.signal-card{
  background:linear-gradient(160deg,#161B2C 0%,#10121C 100%);
  border:1px solid #1E2333;border-radius:16px;padding:18px 22px;margin:6px 0;
  box-shadow:0 10px 30px rgba(0,0,0,0.4);
  animation:fadeInUp .5s ease both;
  transition:transform .25s cubic-bezier(.2,.8,.3,1), border-color .2s ease, box-shadow .25s ease;
}
.signal-card:hover{
  transform:translateY(-4px);
  border-color:#2DD4BF; box-shadow:0 16px 32px rgba(45,212,191,0.18);
}
.buy-card{border-left:5px solid #2DD4BF}.sell-card{border-left:5px solid #FB7185}.hold-card{border-left:5px solid #64748B}
hr{border-color:#1A1E2B}

/* generic tilt-card — apply class="tilt-card" to any custom HTML card div */
.tilt-card{transition:transform .25s cubic-bezier(.2,.8,.3,1), border-color .2s ease, box-shadow .25s ease}
.tilt-card:hover{
  transform:translateY(-4px)!important;
  border-color:#2DD4BF!important; box-shadow:0 16px 32px rgba(45,212,191,0.22)!important;
}

/* smooth tab panel transitions (client-side, no rerun) */
[data-baseweb="tab-panel"]{animation:fadeInUp .35s ease both}

/* smooth expander open/close */
[data-testid="stExpander"] details{transition:background .2s ease}
[data-testid="stExpander"] summary svg{transition:transform .25s ease}

/* sidebar */
[data-testid="stSidebar"]{background:#08090F;border-right:1px solid #1A1E2B}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0.5rem}
[data-testid="stSidebar"] .block-container{animation:fadeInUp .45s ease both}

/* alerts / info boxes — override Streamlit's default blue */
.stAlert, [data-testid="stNotification"], [data-testid^="stAlertContent"]{
  background:#10121C!important;border:1px solid #1E2333!important;
  border-radius:10px!important;color:#E2E8F0!important;
}
.stAlert p, [data-testid^="stAlertContent"] p{color:#E2E8F0!important}
[data-testid="stAlertContentInfo"]{border-left:3px solid #2DD4BF!important}
[data-testid="stAlertContentSuccess"]{border-left:3px solid #2DD4BF!important}
[data-testid="stAlertContentWarning"]{border-left:3px solid #F59E0B!important}
[data-testid="stAlertContentError"]{border-left:3px solid #FB7185!important}
[data-testid="stAlertContentInfo"] svg{color:#2DD4BF!important;fill:#2DD4BF!important}
[data-testid="stAlertContentSuccess"] svg{color:#2DD4BF!important;fill:#2DD4BF!important}
[data-testid="stAlertContentWarning"] svg{color:#F59E0B!important;fill:#F59E0B!important}
[data-testid="stAlertContentError"] svg{color:#FB7185!important;fill:#FB7185!important}

/* radio groups — pill-style hover + gold accent */
[data-testid="stRadio"] label{
  border-radius:8px;padding:4px 8px;margin:1px 0;transition:background .15s ease;
  accent-color:#2DD4BF;
}
[data-testid="stRadio"] label:hover{background:rgba(45,212,191,0.08)}
[data-testid="stRadio"] [data-baseweb="radio"] div:first-child{border-color:#1E2333!important}
[data-testid="stCheckbox"]{accent-color:#2DD4BF}

/* tabs */
.stTabs [data-baseweb="tab-list"]{background:#10121C;border-radius:10px;padding:4px;gap:2px}
.stTabs [data-baseweb="tab"]{color:#94A3B8;border-radius:8px;transition:color .15s ease}
.stTabs [data-baseweb="tab"]:hover{color:#E2E8F0}
.stTabs [aria-selected="true"]{
  color:#FFFFFF!important;
  background:linear-gradient(90deg,#2DD4BF,#F59E0B,#2DD4BF);
  background-size:200% auto; animation:gradientShift 6s ease infinite;
  border-bottom:2px solid transparent!important;
}

/* buttons */
.stButton>button{
  border-radius:9px;border:1px solid #1E2333;transition:all .15s ease;font-weight:600;
}
.stButton>button:hover{border-color:#2DD4BF;box-shadow:0 0 16px rgba(45,212,191,0.35);transform:translateY(-1px)}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#2DD4BF,#F59E0B);border:none;
  box-shadow:0 4px 16px rgba(45,212,191,0.35);
}
.stButton>button[kind="primary"]:hover{box-shadow:0 6px 22px rgba(45,212,191,0.5);transform:translateY(-2px)}

/* inline code / badges (ticker chips etc) */
code{
  background:linear-gradient(135deg,#2DD4BF22,#F59E0B22)!important;
  color:#2DD4BF!important;border:1px solid #2DD4BF55;border-radius:6px;
  padding:2px 7px!important;font-weight:700;
}

/* inputs, selects, expanders */
.stTextInput input,.stNumberInput input,.stSelectbox [data-baseweb="select"]>div{
  background:#10121C!important;border:1px solid #1E2333!important;border-radius:8px!important;
  transition:border-color .15s ease;
}
.stTextInput input:focus,.stSelectbox [data-baseweb="select"]>div:focus-within{
  border-color:#2DD4BF!important;box-shadow:0 0 0 2px rgba(45,212,191,0.25)!important;
}
[data-testid="stExpander"]{
  background:#0F0B1C;border:1px solid #1A1E2B;border-radius:12px;overflow:hidden;
}
.stSlider [role="slider"]{background:#2DD4BF!important}
[data-testid="stFileUploaderDropzone"]{
  background:#0F0B1C!important;border:1.5px dashed #1E2333!important;border-radius:12px!important;
  transition:border-color .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#2DD4BF!important}

/* toast / alerts */
[data-testid="stToast"]{background:#10121C;border:1px solid #1E2333;border-radius:12px}
.stAlert{border-radius:10px;animation:fadeInUp .3s ease both}

/* chat bubbles */
[data-testid="stChatMessage"]{
  background:#10121C;border:1px solid #1A1E2B;border-radius:12px;animation:fadeInUp .3s ease both;
}

/* dataframe */
[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;border:1px solid #1A1E2B}

::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#070812}
::-webkit-scrollbar-thumb{background:#1E2333;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#2DD4BF}
</style>""", unsafe_allow_html=True)

C_UP='#2DD4BF';C_DOWN='#FB7185';C_BLUE='#2DD4BF'
C_GOLD='#F59E0B';C_GREY='#64748B';C_WHITE='#FFFFFF';C_DIM='#94A3B8'



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

@st.cache_data(ttl=30, show_spinner=False)
def get_live_price_cached(ticker):
    return DataManager.get_live_price(ticker)

@st.cache_data(ttl=20, show_spinner=False)
def get_order_book_cached(ticker):
    return DataManager.get_order_book(ticker, limit=10)

@st.cache_data(ttl=60, show_spinner=False)
def get_24h_stats_cached(ticker):
    return DataManager.get_24h_stats(ticker)

@st.cache_data(ttl=300, show_spinner=False)
def get_market_cap_cached(ticker):
    return DataManager.get_market_cap(ticker)

@st.cache_data(ttl=1800, show_spinner=False)
def get_fear_greed_cached():
    return DataManager.get_fear_greed_index()

@st.cache_data(ttl=600, show_spinner=False)
def get_active_wallets_cached(ticker):
    return DataManager.get_active_wallets(ticker)

@st.cache_data(ttl=1800, show_spinner=False)
def get_fear_greed_history_cached(days=14):
    return DataManager.get_fear_greed_history(days)

@st.cache_data(ttl=1800, show_spinner=False)
def get_market_chart_cached(ticker, days=14):
    return DataManager.get_market_chart(ticker, days)

@st.cache_data(ttl=300, show_spinner=False)
def get_price_history_lite_cached(ticker, days=14):
    return DataManager.get_price_history_lite(ticker, days)

def sparkline_svg(values, color="#2DD4BF", width=100, height=32):
    """Tiny inline SVG area-sparkline from a real numeric series. No JS, no external libs."""
    if not values or len(values) < 2:
        return f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:{height}px"></svg>'
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    pts = [(i/(n-1)*width, height - ((v-lo)/rng)*height*0.85 - height*0.05) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{height} " + line + f" {width},{height}"
    gid = f"sg{abs(hash(tuple(values)))%100000}"
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="width:100%;height:{height}px">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.45"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )

def empty_state(icon, title, subtitle, cta=None):
    _cta_html = (f'<div style="color:#2DD4BF;font-size:0.78rem;font-weight:700;margin-top:12px">{cta}</div>'
                 if cta else "")
    st.markdown(
        f'<div style="background:linear-gradient(160deg,#161B2C 0%,#10121C 100%);'
        f'border:1px dashed #1E2333;border-radius:14px;padding:32px 24px;text-align:center;margin:8px 0">'
        f'<div style="width:44px;height:44px;border-radius:12px;margin:0 auto 12px;'
        f'background:linear-gradient(135deg,#2DD4BF30,#F59E0B20);border:1px solid #2DD4BF50;'
        f'display:flex;align-items:center;justify-content:center;font-size:1.2rem">{icon}</div>'
        f'<div style="color:#FFFFFF;font-weight:700;font-size:0.95rem">{title}</div>'
        f'<div style="color:#94A3B8;font-size:0.82rem;margin-top:4px;line-height:1.5">{subtitle}</div>'
        f'{_cta_html}</div>', unsafe_allow_html=True
    )

def dark_fig():
    plt.rcParams.update({
        'figure.facecolor':'#070812','axes.facecolor':'#10121C',
        'axes.edgecolor':'#1E2333','axes.labelcolor':'#E2E8F0',
        'xtick.color':'#94A3B8','ytick.color':'#94A3B8','text.color':'#E2E8F0',
        'grid.color':'#1A1E2B','grid.linewidth':0.5,'legend.facecolor':'#10121C',
        'legend.edgecolor':'#1E2333','legend.fontsize':8,'font.size':9,
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

    # ── Page navigation ────────────────────────────────────────────
    if "app_page" not in st.session_state:
        st.session_state.app_page = "Dashboard"
    _nav1, _nav2 = st.columns(2)
    if _nav1.button("🔮 Dashboard", use_container_width=True,
                     type="primary" if st.session_state.app_page=="Dashboard" else "secondary"):
        st.session_state.app_page = "Dashboard"; st.rerun()
    if _nav2.button("💼 Portfolio", use_container_width=True,
                     type="primary" if st.session_state.app_page=="Portfolio" else "secondary"):
        st.session_state.app_page = "Portfolio"; st.rerun()
    st.divider()

    # ── Upload CSV ──────────────────────────────────────────────────
    with st.expander("📁 Upload CSV Data", expanded=False):
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
    
    # ── Asset Selection — base list + any uploaded CSVs ────────────
    with st.expander("📊 Asset Selection", expanded=True):
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
    
    # ── Signal Settings ────────────────────────────────────────────
    with st.expander("⚙️ Signal Settings", expanded=False):
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
# STANDALONE PORTFOLIO PAGE — independent of the selected asset/model,
# loads instantly instead of waiting on ML training
# ═══════════════════════════════════════════════════════════════════
def render_portfolio_page():
    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:flex-end;'
        'flex-wrap:wrap;gap:8px;margin-bottom:14px">'
        '<div><div style="font-size:1.4rem;font-weight:800;color:#FFFFFF">💼 Portfolio</div>'
        '<div style="color:#94A3B8;font-size:0.8rem;margin-top:2px">'
        'Track which signals you acted on · P&amp;L updates live on refresh</div></div>'
        '<div style="display:flex;align-items:center;gap:6px;color:#475569;font-size:0.72rem">'
        '<span class="live-dot"></span>Session data · resets on page close</div>'
        '</div>', unsafe_allow_html=True
    )

    if "portfolio_trades" not in st.session_state:
        st.session_state.portfolio_trades = load_portfolio()

    with st.expander("➕ Add a Trade", expanded=len(st.session_state.portfolio_trades)==0):
        _pt1,_pt2,_pt3 = st.columns(3)
        with _pt1:
            _pt_ticker = st.selectbox("Asset", all_assets_full, key="pf_pt_ticker")
            _pt_side   = st.selectbox("Side", ["BUY","SELL"], key="pf_pt_side")
        with _pt2:
            _pt_live_default = get_live_price_cached(_pt_ticker) or 0.0
            _pt_entry  = st.number_input("Entry Price ($)", min_value=0.0, value=float(_pt_live_default),
                                          format="%.4f", key="pf_pt_entry")
            _pt_size   = st.number_input("Position Size (units)", min_value=0.001,
                                          value=1.0, format="%.4f", key="pf_pt_size")
        with _pt3:
            _pt_tp = st.number_input("Take Profit ($)", min_value=0.0, value=0.0, format="%.4f", key="pf_pt_tp")
            _pt_sl = st.number_input("Stop Loss ($)",   min_value=0.0, value=0.0, format="%.4f", key="pf_pt_sl")

        _pt_note = st.text_input("Notes (optional)", placeholder="e.g. Signal confidence 73%", key="pf_pt_note")

        if st.button("✅ Add Trade", use_container_width=True, type="primary", key="pf_add_trade"):
            _now_str = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")
            st.session_state.portfolio_trades.append({
                "date": _now_str, "ticker": _pt_ticker, "side": _pt_side,
                "entry": _pt_entry, "size": _pt_size, "tp": _pt_tp, "sl": _pt_sl,
                "status": "Open", "exit": None, "pnl": None, "note": _pt_note,
            })
            save_portfolio(st.session_state.portfolio_trades)
            st.success(f"✅ Trade added: {_pt_side} {_pt_ticker} @ ${_pt_entry:.4f}")
            st.rerun()

    if not st.session_state.portfolio_trades:
        empty_state("💼", "No trades logged yet",
                     "Trades you add here track live P&amp;L against your entry, TP and SL automatically.")
        return

    trades = st.session_state.portfolio_trades
    total_pnl = 0.0; total_invest = 0.0; open_count = 0
    _rows = []
    for _t in trades:
        _live = get_live_price_cached(_t["ticker"]) or _t["entry"]
        _is_open = _t["status"] == "Open"
        if _is_open:
            _pnl_per = (_live - _t["entry"]) if _t["side"]=="BUY" else (_t["entry"] - _live)
            _pnl_total = _pnl_per * _t["size"]
            _pnl_pct = (_pnl_per / _t["entry"]) * 100 if _t["entry"]>0 else 0
            open_count += 1
            total_pnl += _pnl_total
            total_invest += _t["entry"]*_t["size"]
        else:
            _pnl_total = _t.get("pnl",0) or 0
            _pnl_pct = (_pnl_total/(_t["entry"]*_t["size"]))*100 if _t["entry"]>0 else 0
            _live = _t.get("exit", _t["entry"])
            total_pnl += _pnl_total
        _rows.append({
            "date": _t["date"], "ticker": _t["ticker"], "side": _t["side"],
            "entry": _t["entry"], "size": _t["size"], "live": _live,
            "pnl": _pnl_total, "pnl_pct": _pnl_pct,
            "tp": _t["tp"], "sl": _t["sl"], "status": _t["status"],
        })

    roi = (total_pnl/total_invest*100) if total_invest>0 else 0
    _pnl_c = "#2DD4BF" if total_pnl>=0 else "#FB7185"
    _roi_c = "#2DD4BF" if roi>=0 else "#FB7185"
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px">'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">Total Trades</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.35rem;font-weight:800;margin-top:3px">{len(trades)}</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">Open Positions</div>'
        f'<div class="stat-pop" style="color:#2DD4BF;font-size:1.35rem;font-weight:800;margin-top:3px">{open_count}</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">Total P&amp;L</div>'
        f'<div class="stat-pop" style="color:{_pnl_c};font-size:1.35rem;font-weight:800;margin-top:3px">'
        f'{"+" if total_pnl>=0 else ""}{total_pnl:,.2f}</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">ROI</div>'
        f'<div class="stat-pop" style="color:{_roi_c};font-size:1.35rem;font-weight:800;margin-top:3px">{roi:+.2f}%</div></div>'
        '</div>', unsafe_allow_html=True
    )

    _pf_left, _pf_right = st.columns([2, 1])
    with _pf_left:
        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">📈 Portfolio Performance</div>',
            unsafe_allow_html=True
        )
        try:
            _sorted_rows = sorted(_rows, key=lambda r: r["date"])
            _cum = np.cumsum([r["pnl"] for r in _sorted_rows])
            dark_fig()
            _figp, _axp = plt.subplots(figsize=(9, 3.2))
            _figp.patch.set_facecolor('#070812'); _axp.set_facecolor('#10121C')
            _xs = range(len(_cum))
            _lc = '#2DD4BF' if (len(_cum)==0 or _cum[-1]>=0) else '#FB7185'
            _axp.plot(_xs, _cum, color=_lc, lw=1.8, marker='o', ms=3.5)
            _axp.fill_between(_xs, 0, _cum, where=(_cum>=0), alpha=0.14, color='#2DD4BF')
            _axp.fill_between(_xs, 0, _cum, where=(_cum<0),  alpha=0.14, color='#FB7185')
            _axp.axhline(0, color='#6B7290', lw=0.8, ls='--', alpha=0.6)
            _axp.set_xticks(_xs); _axp.set_xticklabels([r["date"][5:10] for r in _sorted_rows],
                                                         rotation=30, ha='right', fontsize=7.5)
            _axp.set_ylabel('Cumulative P&L ($)', fontsize=8.5)
            _axp.spines[['top','right']].set_visible(False)
            _axp.grid(axis='y', alpha=0.15)
            plt.tight_layout()
            st.pyplot(_figp, use_container_width=True); plt.close()
        except Exception:
            st.caption("Add at least one trade to see the performance curve.")

    with _pf_right:
        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">🏆 Top Assets</div>',
            unsafe_allow_html=True
        )
        _by_asset = {}
        for r in _rows:
            _by_asset.setdefault(r["ticker"], 0.0)
            _by_asset[r["ticker"]] += r["pnl"]
        _asset_ranked = sorted(_by_asset.items(), key=lambda x: -abs(x[1]))[:5]
        _ASSET_GRAD = {
            "SOL-USD": "linear-gradient(135deg,#9945FF,#14F195)",
            "BTC-USD": "linear-gradient(135deg,#F7931A,#2DD4BF)",
            "ETH-USD": "linear-gradient(135deg,#3C3C3D,#8C8C8C)",
        }
        _asset_html = ""
        for _at, _apnl in _asset_ranked:
            _ac = "#2DD4BF" if _apnl>=0 else "#FB7185"
            _ag = _ASSET_GRAD.get(_at, "linear-gradient(135deg,#2DD4BF,#F59E0B)")
            _asset_html += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:8px 0;border-bottom:1px solid #141726">'
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="width:26px;height:26px;border-radius:7px;background:{_ag};'
                f'display:flex;align-items:center;justify-content:center;font-size:0.7rem;'
                f'font-weight:800;color:#0B0F1A">{_at[0]}</div>'
                f'<span style="color:#E2E8F0;font-size:0.82rem;font-weight:600">{_at}</span></div>'
                f'<span style="color:{_ac};font-size:0.82rem;font-weight:700">'
                f'{"+" if _apnl>=0 else ""}{_apnl:,.2f}</span></div>'
            )
        if not _asset_html:
            _asset_html = '<div style="color:#475569;font-size:0.8rem">No assets yet</div>'
        st.markdown(
            f'<div style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
            f'border-radius:12px;padding:6px 14px">{_asset_html}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
        'letter-spacing:.04em;margin:20px 0 8px">📋 Transactions</div>',
        unsafe_allow_html=True
    )
    _tx_hdr = "".join(
        f'<th style="text-align:left;padding:8px 10px;color:#94A3B8;font-size:0.7rem;'
        f'text-transform:uppercase;letter-spacing:.03em;border-bottom:1px solid #1A1E2B">{h}</th>'
        for h in ["Date","Asset","Side","Entry","Size","Live / Exit","P&L","Status"]
    )
    _tx_body = ""
    for _i, r in enumerate(_rows):
        _sc = "#2DD4BF" if r["side"]=="BUY" else "#FB7185"
        _pc = "#2DD4BF" if r["pnl"]>=0 else "#FB7185"
        _stc, _stb = ("#2DD4BF","rgba(45,212,191,0.12)") if r["status"]=="Open" else ("#64748B","rgba(100,116,139,0.14)")
        _tx_body += (
            f'<tr class="outlook-row" style="animation-delay:{_i*0.05:.2f}s">'
            f'<td style="padding:9px 10px;color:#94A3B8;font-size:0.8rem;border-bottom:1px solid #141726">{r["date"]}</td>'
            f'<td style="padding:9px 10px;color:#FFFFFF;font-weight:600;font-size:0.82rem;border-bottom:1px solid #141726">{r["ticker"]}</td>'
            f'<td style="padding:9px 10px;border-bottom:1px solid #141726">'
            f'<span class="status-dot {"buy" if r["side"]=="BUY" else "sell"}" style="width:8px;height:8px"></span>'
            f'<span style="color:{_sc};font-weight:700;font-size:0.8rem">{r["side"]}</span></td>'
            f'<td style="padding:9px 10px;color:#E2E8F0;font-size:0.82rem;border-bottom:1px solid #141726">${r["entry"]:,.4f}</td>'
            f'<td style="padding:9px 10px;color:#E2E8F0;font-size:0.82rem;border-bottom:1px solid #141726">{r["size"]:.4f}</td>'
            f'<td style="padding:9px 10px;color:#E2E8F0;font-size:0.82rem;border-bottom:1px solid #141726">${r["live"]:,.4f}</td>'
            f'<td style="padding:9px 10px;color:{_pc};font-weight:700;font-size:0.82rem;border-bottom:1px solid #141726">'
            f'{"+" if r["pnl"]>=0 else ""}{r["pnl"]:,.2f} ({"+" if r["pnl_pct"]>=0 else ""}{r["pnl_pct"]:.2f}%)</td>'
            f'<td style="padding:9px 10px;border-bottom:1px solid #141726">'
            f'<span class="chip" style="color:{_stc};background:{_stb};font-size:0.68rem;font-weight:700;'
            f'padding:3px 9px;border-radius:999px">{r["status"].upper()}</span></td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:14px;padding:6px 4px;overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse"><thead><tr>{_tx_hdr}</tr></thead>'
        f'<tbody>{_tx_body}</tbody></table></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
        'letter-spacing:.04em;margin:20px 0 8px">⚙️ Manage Trades</div>',
        unsafe_allow_html=True
    )
    _mc1, _mc2, _mc3 = st.columns(3)
    with _mc1:
        _close_idx = st.number_input("Trade # to close (1-based)", min_value=1,
                                      max_value=len(trades), value=1, key="pf_close_idx")
        _exit_price = st.number_input("Exit price ($)", min_value=0.0,
                                       value=float(get_live_price_cached(trades[int(_close_idx)-1]["ticker"]) or 0.0),
                                       format="%.4f", key="pf_exit_p")
        if st.button("✅ Close Trade", use_container_width=True, key="pf_close_btn"):
            _idx = int(_close_idx) - 1
            _t = st.session_state.portfolio_trades[_idx]
            _pnl_per = (_exit_price - _t["entry"]) if _t["side"]=="BUY" else (_t["entry"]-_exit_price)
            st.session_state.portfolio_trades[_idx]["status"] = "Closed"
            st.session_state.portfolio_trades[_idx]["exit"] = _exit_price
            st.session_state.portfolio_trades[_idx]["pnl"] = _pnl_per * _t["size"]
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
                                    max_value=len(trades), value=1, key="pf_del_idx")
        if st.button("🗑️ Delete Trade", use_container_width=True, key="pf_del_btn"):
            st.session_state.portfolio_trades.pop(int(_del_idx)-1)
            save_portfolio(st.session_state.portfolio_trades)
            st.rerun()

    with _mc3:
        if st.button("🗑️ Clear All Trades", use_container_width=True, key="pf_clear_btn"):
            st.session_state.portfolio_trades = []
            save_portfolio([])
            st.rerun()

    _ctl = st.session_state.get("closed_trades_log", [])
    if _ctl:
        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin:20px 0 8px">📜 Closed Trades History</div>',
            unsafe_allow_html=True
        )
        _ctdf = pd.DataFrame([{
            "Closed": t.get("closed_at",""), "Asset": t.get("ticker",""), "Side": t.get("side",""),
            "Entry": f"${t.get('entry',0):,.4f}", "Exit": f"${t.get('exit',0):,.4f}",
            "P&L": f"{'+' if (t.get('pnl',0) or 0)>=0 else ''}{(t.get('pnl',0) or 0):,.4f}",
        } for t in reversed(_ctl)])
        st.dataframe(_ctdf, use_container_width=True, hide_index=True)

if st.session_state.get("app_page","Dashboard") == "Portfolio":
    render_portfolio_page()
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA + TRAIN
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=21600, show_spinner=False)  # 6-hour cache
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
<div style="background:#241521;border:2px solid #FB7185;border-radius:10px;padding:20px 24px;margin:20px 0">
  <div style="color:#FB7185;font-size:1.1rem;font-weight:bold;margin-bottom:8px">
    ❌ Error Loading Data for {ticker}
  </div>
  <div style="color:#FFFFFF;font-size:0.88rem">{str(e).split(chr(10))[0]}</div>
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
<div style="background:#1A1826;border:1px solid #F59E0B;border-radius:8px;padding:16px 20px">
  <div style="color:#F59E0B;font-weight:bold;margin-bottom:10px">
    🇦🇪 {ticker} — Auto-fetch failed
  </div>
  <div style="color:#E2E8F0;font-size:0.88rem;line-height:1.9">
    Auto-fetch via yfinance/Stooq is not returning data for this stock.<br>
    <b>Quick fix — download CSV manually (30 seconds):</b><br><br>
    <b>Option 1 — Yahoo Finance:</b><br>
    &nbsp;&nbsp;→ <a href="{_yf_csv}" target="_blank" 
      style="color:#2DD4BF">Click to download {_yf_sym} CSV directly</a><br><br>
    <b>Option 2 — Investing.com:</b><br>
    &nbsp;&nbsp;→ <a href="{_inv_url}" target="_blank"
      style="color:#2DD4BF">Open {ticker} historical data on Investing.com</a>
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
                f'background:#2DD4BF;color:white;border-radius:5px;padding:8px 16px;'
                f'text-decoration:none;font-size:0.85rem;margin-top:4px">'
                f'⬇️ Download CSV for {_yf_sym}</a>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
<div style="background:#1A1826;border:1px solid #1E2333;border-radius:8px;padding:16px 20px">
  <div style="color:#F59E0B;font-weight:bold;margin-bottom:8px">💡 What to do:</div>
  <div style="color:#E2E8F0;font-size:0.88rem;line-height:1.8">
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
live_price = get_live_price_cached(ticker)
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

# ── Smart TP/SL: per-asset dollar cap so TP is reachable same day ──
_conf_mult = max(0.6, min(1.5, 0.8 + (last_conf - 60) / 100))
# Per-asset max TP/SL distance in dollars
_ASSET_MAX_DIST = {
    "GC=F"   : 30,    # Gold: max $30
    "ETH-USD": 30,    # ETH: max $30
    "BTC-USD": 300,   # BTC: max $300
}
_max_dist = _ASSET_MAX_DIST[ticker] if ticker in _ASSET_MAX_DIST else _conf_mult * last_atr
_tp_dist  = min(_conf_mult * last_atr, _max_dist)
_sl_dist  = _tp_dist * 0.9  # 1:1 R/R (SL slightly tighter)

if last_sig == "BUY":
    entry_p  = round(display_price, 4)
    tp_price = round(display_price + _tp_dist, 4)
    sl_price = round(display_price - _sl_dist, 4)
elif last_sig == "SELL":
    entry_p  = round(display_price, 4)
    tp_price = round(display_price - _tp_dist, 4)
    sl_price = round(display_price + _sl_dist, 4)
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
# HERO HEADER — big price + range selector (Nexus-style)
# ═══════════════════════════════════════════════════════════════════
_hero_name  = DataManager.get_ticker_name(ticker) or ticker
_hero_pair  = f'{ticker.replace("-USD","")} <span style="color:#475569">/</span> USD' if ticker.upper().endswith("-USD") else ticker
_hero_chg_c = "#2DD4BF" if day_chg >= 0 else "#FB7185"
_hero_arrow = "&#8599;" if day_chg >= 0 else "&#8600;"
_hero_mc    = get_market_cap_cached(ticker) if is_crypto(ticker) else None
_hero_s24   = get_24h_stats_cached(ticker) if is_crypto(ticker) else None
_hero_vol   = (f"${_hero_s24['volume_quote']/1e6:.1f}M" if _hero_s24 else
               (f"${_hero_mc['volume_24h']/1e6:.1f}M" if _hero_mc else None))
_hero_cap   = f"${_hero_mc['market_cap']/1e9:.2f}B" if _hero_mc and _hero_mc.get('market_cap') else None
_hero_sub   = " &middot; ".join(filter(None, [
    f"Vol {_hero_vol}" if _hero_vol else None,
    f"Cap {_hero_cap}" if _hero_cap else None,
])) or f"{len(df_feat):,} trading days"

st.markdown(
    f'<div style="color:#94A3B8;font-size:0.78rem;display:flex;align-items:center;gap:6px;margin-bottom:6px">'
    f'<span class="live-dot"></span>Live feed &middot; auto-refreshing &middot; {_hero_name}</div>'
    f'<div style="font-size:2.2rem;font-weight:800;line-height:1.1">'
    f'{_hero_pair} <span style="color:#475569">&middot;</span> '
    f'<span style="font-family:ui-monospace,monospace">${display_price:,.4f}</span></div>'
    f'<div style="display:flex;align-items:center;gap:10px;margin-top:6px;font-size:0.85rem">'
    f'<span style="background:{_hero_chg_c}18;color:{_hero_chg_c};border-radius:6px;padding:2px 8px;'
    f'font-weight:700">{_hero_arrow} {day_chg:+.2f}%</span>'
    f'<span style="color:#94A3B8">{_hero_sub}</span></div>',
    unsafe_allow_html=True
)
_hero_ranges = ["1H","24H","7D","1M","1Y","ALL"]
if "hero_range" not in st.session_state:
    st.session_state.hero_range = "24H"
_hero_cols = st.columns(len(_hero_ranges) + 6)  # push buttons right, leave room
for _hi, _hr in enumerate(_hero_ranges):
    if _hero_cols[_hi+6].button(_hr, key=f"hero_rng_{_hr}", use_container_width=True,
                                 type="primary" if st.session_state.hero_range==_hr else "secondary"):
        st.session_state.hero_range = _hr
        st.rerun()
st.caption("This selector doesn't drive a chart yet — the reference's own version doesn't either. "
           "Real range control lives in the Price & Signals tab's lookback buttons.")
st.divider()

# ═══════════════════════════════════════════════════════════════════
# HEADER + PROTOCOL/TOKEN METRICS — card-grid dashboard layout
# ═══════════════════════════════════════════════════════════════════
name = DataManager.get_ticker_name(ticker) or ticker
data_src = "Binance" if any(ticker.replace("-USD","").upper() == k.replace("-USD","") for k in ["SOL","BTC","ETH"]) else "Yahoo Finance"
st.markdown(
    f'<div style="margin-bottom:4px">'
    f'<span style="font-size:1.5rem">🔮</span> '
    f'<span style="font-size:1.4rem;font-weight:800;color:#FFFFFF">{name}</span> '
    f'<code style="font-size:0.95rem">{ticker}</code>'
    f'<div style="color:#94A3B8;font-size:0.8rem;margin-top:2px">'
    f'Source: {data_src} · Last candle {last_date.strftime("%d %b %Y")} · {len(df_feat):,} trading days</div>'
    f'</div>', unsafe_allow_html=True
)
st.divider()

_dot_cls    = "live-dot" if day_chg >= 0 else "live-dot down"
_chg_color  = "#2DD4BF" if day_chg >= 0 else "#FB7185"
_price_label = "Live Price" if live_price else "Last Close"

def _metric_bar(pct, grad="linear-gradient(90deg,#2DD4BF,#F59E0B)", delay=0):
    pct = max(2, min(100, pct))
    return (f'<div style="background:#1A1E2B;border-radius:6px;height:6px;margin-top:9px;overflow:hidden">'
            f'<div style="--tw:{pct}%;width:0;height:100%;background:{grad};border-radius:6px;'
            f'animation:fillBar 1s ease-out {delay}s forwards"></div></div>')

_prob_grad = "linear-gradient(90deg,#2DD4BF,#14B8A6)" if last_prob > 0.5 else "linear-gradient(90deg,#FB7185,#E11D48)"
_acc_bar  = _metric_bar(ens_acc*100, delay=0.0)
_filt_bar = _metric_bar(ens_filt*100, delay=0.1)
_prob_bar = _metric_bar(last_prob*100, _prob_grad, delay=0.2)
_atr_bar  = _metric_bar(min(100, atr_pct*100*10), delay=0.3)

col_left, col_right = st.columns([2.1, 1])

with col_left:
    st.markdown(
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C 0%,#10121C 100%);'
        f'border:1px solid #1E2333;border-radius:16px;padding:20px 22px;height:100%;'
        f'box-shadow:0 10px 30px rgba(0,0,0,0.4)">'
        f'<div style="color:#2DD4BF;font-weight:700;font-size:0.9rem;margin-bottom:16px;'
        f'text-transform:uppercase;letter-spacing:0.04em">📊 Model Metrics</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px 28px">'
        f'<div><div style="color:#94A3B8;font-size:0.74rem;text-transform:uppercase">Model Accuracy</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.55rem;font-weight:800;animation-delay:.05s">{ens_acc*100:.1f}%</div>{_acc_bar}</div>'
        f'<div><div style="color:#94A3B8;font-size:0.74rem;text-transform:uppercase">Filtered Accuracy</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.55rem;font-weight:800;animation-delay:.15s">{ens_filt*100:.1f}%'
        f'<span style="color:#2DD4BF;font-size:0.78rem;font-weight:600"> +{(ens_filt-ens_acc)*100:.1f}%</span></div>{_filt_bar}</div>'
        f'<div><div style="color:#94A3B8;font-size:0.74rem;text-transform:uppercase">P(UP Tomorrow)</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.55rem;font-weight:800;animation-delay:.25s">{last_prob*100:.1f}%</div>{_prob_bar}</div>'
        f'<div><div style="color:#94A3B8;font-size:0.74rem;text-transform:uppercase">ATR Volatility</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.55rem;font-weight:800;animation-delay:.35s">${last_atr:.4f}</div>{_atr_bar}</div>'
        f'</div></div>', unsafe_allow_html=True
    )

with col_right:
    st.markdown(
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C 0%,#10121C 100%);'
        f'border:1px solid #1E2333;border-radius:16px;padding:16px 20px;'
        f'box-shadow:0 10px 30px rgba(0,0,0,0.4);margin-bottom:12px">'
        f'<div style="color:#94A3B8;font-size:0.72rem;text-transform:uppercase;font-weight:600">'
        f'<span class="{_dot_cls}"></span>{_price_label}</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.75rem;font-weight:800;margin-top:2px">${display_price:,.4f}</div>'
        f'<div style="color:{_chg_color};font-weight:700;font-size:0.88rem">{day_chg:+.2f}%</div>'
        f'</div>'
        f'<div class="tilt-card" style="background:linear-gradient(135deg,#2DD4BF,#F59E0B);border-radius:16px;'
        f'padding:16px 20px;box-shadow:0 10px 26px rgba(245,158,11,0.35);animation:glowPulse 2.4s ease-in-out infinite">'
        f'<div style="color:#0B1220;font-size:0.72rem;text-transform:uppercase;font-weight:700">Active Signals</div>'
        f'<div class="stat-pop" style="color:#0B1220;font-size:1.6rem;font-weight:900;animation-delay:.2s">{results["n_signals"]}</div>'
        f'<div style="color:#0B1220;font-size:0.7rem;font-weight:700;opacity:0.75">'
        f'🟢 {int((signals==1).sum())} buy &middot; 🔴 {int((signals==-1).sum())} sell</div>'
        f'</div>',
        unsafe_allow_html=True
    )
st.divider()

# ═══════════════════════════════════════════════════════════════════
# MARKET PULSE — order book, sentiment, fear/greed, market cap
# (crypto only — hidden entirely for stocks/UAE/commodities)
# ═══════════════════════════════════════════════════════════════════
if is_crypto(ticker):
    _mc   = get_market_cap_cached(ticker)
    _s24  = get_24h_stats_cached(ticker)
    _fng  = get_fear_greed_cached()
    _aw   = get_active_wallets_cached(ticker)
    _mchart = get_market_chart_cached(ticker)
    _fng_hist = get_fear_greed_history_cached()

    st.markdown(
        '<div style="color:#2DD4BF;font-weight:700;font-size:0.9rem;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:14px">🌐 Market Pulse — Live</div>',
        unsafe_allow_html=True
    )

    # ── Top stat row ──────────────────────────────────────────
    _mcap_str = f"${_mc['market_cap']/1e9:.2f}B" if _mc and _mc['market_cap'] else "—"
    _vol_str  = f"${_s24['volume_quote']/1e6:.1f}M" if _s24 else (f"${_mc['volume_24h']/1e6:.1f}M" if _mc else "—")
    _chg24    = _s24['price_change_pct'] if _s24 else (_mc['change_24h'] if _mc else 0)
    _chg_c    = "#2DD4BF" if _chg24 >= 0 else "#FB7185"
    _fng_val  = _fng['value'] if _fng else 50
    _fng_cls  = _fng['classification'] if _fng else "Neutral"
    _fng_c    = ("#FB7185" if _fng_val<=25 else "#F59E0B" if _fng_val<=45 else
                 "#A78BFA" if _fng_val<=55 else "#2DD4BF" if _fng_val<=75 else "#5EEAD4")
    _trades_str = f"{_s24['trade_count']:,}" if _s24 else "—"
    _aw_str = f"{_aw['active_addresses']:,}" if _aw else "N/A"
    _aw_sub = "unique BTC addresses, 24h" if _aw else "no free source for this asset"

    _spk_mcap  = sparkline_svg(_mchart['market_caps'], "#2DD4BF") if _mchart and _mchart.get('market_caps') else ""
    _spk_vol   = sparkline_svg(_mchart['volumes'], "#F59E0B") if _mchart and _mchart.get('volumes') else ""
    _spk_fng   = sparkline_svg(_fng_hist, "#FB7185") if _fng_hist else ""
    _spk_aw    = sparkline_svg(_aw['history'], "#38BDF8") if _aw and _aw.get('history') else ""

    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px">'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px;overflow:hidden">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">Market Cap</div>'
        f'<div class="stat-pop" style="color:#FFFFFF;font-size:1.3rem;font-weight:800;margin-top:3px">{_mcap_str}</div>'
        f'<div style="margin-top:6px">{_spk_mcap}</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px;overflow:hidden">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">24h Volume</div>'
        f'<div class="stat-pop" style="color:{_chg_c};font-size:1.3rem;font-weight:800;margin-top:3px">{_vol_str}</div>'
        f'<div style="color:{_chg_c};font-size:0.72rem;font-weight:600">{_chg24:+.2f}% (24h)</div>'
        f'<div style="margin-top:2px">{_spk_vol}</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px;overflow:hidden">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">Fear &amp; Greed</div>'
        f'<div class="stat-pop" style="color:{_fng_c};font-size:1.3rem;font-weight:800;margin-top:3px">{_fng_val} &middot; {_fng_cls}</div>'
        f'<div style="margin-top:6px">{_spk_fng}</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">24h Trades</div>'
        f'<div class="stat-pop" style="color:#A78BFA;font-size:1.3rem;font-weight:800;margin-top:3px">{_trades_str}</div>'
        f'<div style="color:#475569;font-size:0.68rem">trade count &mdash; no free history source</div></div>'
        f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
        f'border-radius:12px;padding:14px 16px;overflow:hidden">'
        f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:.05em">Active Wallets</div>'
        f'<div class="stat-pop" style="color:{"#FFFFFF" if _aw else "#475569"};font-size:1.3rem;font-weight:800;margin-top:3px">{_aw_str}</div>'
        f'<div style="color:#475569;font-size:0.68rem">{_aw_sub}</div>'
        f'<div style="margin-top:2px">{_spk_aw}</div></div>'
        '</div>', unsafe_allow_html=True
    )

    _pulse_left, _pulse_center, _pulse_right = st.columns([1, 1.6, 1])

    # ── Sentiment Rate + Social Sentiment + Order Book ────────
    with _pulse_left:
        _sent_lbl  = "Bullish" if _sentiment>0.1 else "Bearish" if _sentiment<-0.1 else "Neutral"
        _sent_c    = "#2DD4BF" if _sentiment>0.1 else "#FB7185" if _sentiment<-0.1 else "#64748B"
        _sent_pct  = int((_sentiment + 1) / 2 * 100)  # -1..1 -> 0..100

        # Real "equalizer" bars: last 14 days' actual |% change|, not decorative noise
        _eq_n = min(14, len(df_feat))
        _eq_vals = (df_feat['Close'].pct_change().abs().fillna(0).values[-_eq_n:] * 100)
        _eq_max = max(_eq_vals.max(), 0.01)
        _eq_bars = "".join(
            f'<div style="width:5px;border-radius:2px;height:{max(15,v/_eq_max*100):.0f}%;'
            f'background:linear-gradient(180deg,#2DD4BF,#0F766E)"></div>'
            for v in _eq_vals
        )
        st.markdown(
            f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);'
            f'border:1px solid #1E2333;border-radius:12px;padding:16px;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
            f'<span style="font-size:0.85rem;font-weight:700">Sentiment Rate</span>'
            f'<span style="border:1px solid #F59E0B55;background:#F59E0B18;color:#F59E0B;'
            f'padding:2px 8px;border-radius:6px;font-size:0.65rem;text-transform:uppercase;'
            f'font-weight:700">{_sent_lbl}</span></div>'
            f'<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:10px">'
            f'<div><div class="stat-pop" style="font-size:1.7rem;font-weight:800">{_sent_pct}%</div>'
            f'<div style="color:{_sent_c};font-size:0.72rem;margin-top:2px">news score {_sentiment:+.3f}</div></div>'
            f'<div style="display:flex;align-items:flex-end;gap:2px;height:36px">{_eq_bars}</div>'
            f'</div></div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">💬 Social Sentiment</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);'
            f'border:1px solid #1E2333;border-radius:12px;padding:16px;margin-bottom:12px">'
            f'<div style="color:{_sent_c};font-size:1.4rem;font-weight:800">{_sent_lbl}</div>'
            f'<div style="color:#94A3B8;font-size:0.78rem;margin:2px 0 10px">CryptoPanic score: {_sentiment:+.3f}</div>'
            f'<div style="background:#1A1E2B;border-radius:6px;height:8px;overflow:hidden">'
            f'<div style="width:{_sent_pct}%;height:100%;background:linear-gradient(90deg,#FB7185,#64748B,#2DD4BF);'
            f'border-radius:6px"></div></div>'
            f'<div style="display:flex;justify-content:space-between;color:#475569;font-size:0.66rem;margin-top:3px">'
            f'<span>Bearish</span><span>Neutral</span><span>Bullish</span></div>'
            f'</div>', unsafe_allow_html=True
        )

        @st.fragment(run_every=8)
        def _order_book_fragment(_ticker):
            _ob_f = get_order_book_cached(_ticker)
            st.markdown(
                '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
                'letter-spacing:.04em;margin-bottom:8px;display:flex;align-items:center;gap:8px">'
                '📖 Order Book — Top 8 <span class="live-dot" style="margin:0"></span>'
                '<span style="color:#475569;font-weight:400;text-transform:none;font-size:0.68rem">'
                'updates every 8s</span></div>',
                unsafe_allow_html=True
            )
            if not _ob_f or not _ob_f.get("bids") or not _ob_f.get("asks"):
                empty_state("📖", "Order book unavailable", "Binance depth data didn't load — try refreshing.")
            else:
                _max_qty = max(
                    max((q for _, q in _ob_f["bids"]), default=1),
                    max((q for _, q in _ob_f["asks"]), default=1),
                ) or 1
                _bid_rows = ""
                for _p, _q in _ob_f["bids"][:8]:
                    _w = min(100, _q / _max_qty * 100)
                    _bid_rows += (
                        f'<div style="position:relative;padding:3px 8px;font-size:0.76rem;'
                        f'font-family:ui-monospace,monospace">'
                        f'<div style="position:absolute;inset:0;background:rgba(45,212,191,0.12);'
                        f'width:{_w:.0f}%;border-radius:3px"></div>'
                        f'<div style="position:relative;display:flex;justify-content:space-between">'
                        f'<span style="color:#2DD4BF;font-weight:700">${_p:,.4f}</span>'
                        f'<span style="color:#94A3B8">{_q:.3f}</span></div></div>'
                    )
                _ask_rows = ""
                for _p, _q in _ob_f["asks"][:8]:
                    _w = min(100, _q / _max_qty * 100)
                    _ask_rows += (
                        f'<div style="position:relative;padding:3px 8px;font-size:0.76rem;'
                        f'font-family:ui-monospace,monospace">'
                        f'<div style="position:absolute;inset:0;background:rgba(251,113,133,0.12);'
                        f'width:{_w:.0f}%;border-radius:3px"></div>'
                        f'<div style="position:relative;display:flex;justify-content:space-between">'
                        f'<span style="color:#FB7185;font-weight:700">${_p:,.4f}</span>'
                        f'<span style="color:#94A3B8">{_q:.3f}</span></div></div>'
                    )
                st.markdown(
                    f'<div style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
                    f'border-radius:12px;padding:10px;display:grid;grid-template-columns:1fr 1fr;gap:12px">'
                    f'<div><div style="color:#2DD4BF;font-size:0.68rem;text-transform:uppercase;'
                    f'font-weight:700;margin-bottom:4px">Bids</div>{_bid_rows}</div>'
                    f'<div><div style="color:#FB7185;font-size:0.68rem;text-transform:uppercase;'
                    f'font-weight:700;margin-bottom:4px">Asks</div>{_ask_rows}</div>'
                    f'</div>', unsafe_allow_html=True
                )

        _order_book_fragment(ticker)

    # ── Center: hero momentum chart + 24h range + comparison mini-charts ──
    with _pulse_center:
        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">📈 Momentum Stream — last 48 candles</div>',
            unsafe_allow_html=True
        )
        try:
            dark_fig()
            _nshow = min(48, len(df_feat))
            _dshow = df_feat.iloc[-_nshow:]
            _pchg  = _dshow['Close'].pct_change().fillna(0).values * 100
            _vol_n = (_dshow['Volume'].values / max(_dshow['Volume'].max(), 1e-9)) if 'Volume' in _dshow.columns else np.zeros(_nshow)
            _atrn  = (_dshow['ATR'].values / max(_dshow['ATR'].max(), 1e-9)) if 'ATR' in _dshow.columns else np.zeros(_nshow)
            _figm, _axm = plt.subplots(figsize=(9, 3.6))
            _figm.patch.set_facecolor('#070812'); _axm.set_facecolor('#10121C')
            _xm = np.arange(_nshow)
            _axm.bar(_xm, _pchg, color='#2DD4BF', alpha=0.6, width=0.7, label='Price momentum %')
            _axm.bar(_xm, _vol_n*_pchg.std()*2, color='#F59E0B', alpha=0.4, width=0.4, label='Volume (norm.)')
            _axm.bar(_xm, -_atrn*_pchg.std()*1.5, color='#A78BFA', alpha=0.5, width=0.4, label='Volatility (norm.)')
            _axm.axhline(0, color='#475569', lw=0.8, ls='--', alpha=0.6)
            # Real-data callout box (last candle's actual values, not a fake tooltip)
            _cx, _cy = _nshow-1, _pchg[-1]
            _axm.annotate(
                f"{_dshow.index[-1].strftime('%d %b')}\nmomentum {_pchg[-1]:+.2f}%\nP(UP) {last_prob*100:.0f}%",
                xy=(_cx, _cy), xytext=(-90, 18), textcoords='offset points', fontsize=7.5,
                color='#E2E8F0', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', fc='#161B2C', ec='#2DD4BF', alpha=0.95),
                arrowprops=dict(arrowstyle='-', color='#2DD4BF', lw=0.8),
            )
            _axm.set_xticks([]); _axm.legend(loc='upper left', ncol=3, fontsize=7, framealpha=0.15)
            _axm.spines[['top','right','left']].set_visible(False)
            _axm.set_yticks([])
            plt.tight_layout()
            st.pyplot(_figm, use_container_width=True); plt.close()
        except Exception:
            st.caption("Momentum chart unavailable for this asset right now.")

        if _s24:
            _rng_pct = ((display_price - _s24['low']) / max(_s24['high']-_s24['low'], 0.0001)) * 100
            st.markdown(
                f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);'
                f'border:1px solid #1E2333;border-radius:12px;padding:14px 16px;margin-bottom:12px">'
                f'<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;margin-bottom:6px">24h Range</div>'
                f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:6px">'
                f'<span style="color:#FB7185">${_s24["low"]:,.4f}</span>'
                f'<span style="color:#2DD4BF;font-weight:700">${display_price:,.4f}</span>'
                f'<span style="color:#2DD4BF">${_s24["high"]:,.4f}</span></div>'
                f'<div style="background:#1A1E2B;border-radius:6px;height:6px;position:relative">'
                f'<div style="position:absolute;left:{max(0,min(96,_rng_pct)):.0f}%;top:-3px;width:12px;height:12px;'
                f'border-radius:50%;background:#2DD4BF;box-shadow:0 0 8px rgba(45,212,191,0.6)"></div></div>'
                f'</div>', unsafe_allow_html=True
            )

        # Two comparison mini-cards: the current asset + one other default crypto
        _cmp_other = "BTC-USD" if ticker.upper() != "BTC-USD" else "ETH-USD"
        _cmp_pairs  = [(ticker, name), (_cmp_other, DataManager.get_ticker_name(_cmp_other) or _cmp_other)]
        _mini_cols = st.columns(2)
        for _mi, (_mt, _mn) in enumerate(_cmp_pairs):
            with _mini_cols[_mi]:
                _mprice = get_live_price_cached(_mt)
                _m24 = get_24h_stats_cached(_mt)
                _mchg = _m24['price_change_pct'] if _m24 else 0.0
                _mc_c = "#2DD4BF" if _mchg >= 0 else "#FB7185"
                st.markdown(
                    f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);'
                    f'border:1px solid #1E2333;border-radius:12px;padding:14px">'
                    f'<div style="font-size:0.82rem;font-weight:700">{_mn}</div>'
                    f'<div style="color:#475569;font-size:0.7rem;margin-bottom:6px">{_mt} &middot; 24h</div>'
                    f'<div style="font-family:ui-monospace,monospace;font-size:0.95rem">'
                    f'{"$"+format(_mprice,",.4f") if _mprice else "—"}</div>'
                    f'<div style="color:{_mc_c};font-size:0.75rem">{_mchg:+.2f}%</div>'
                    f'</div>', unsafe_allow_html=True
                )

    # ── Right: Execution Rate, Top Movers, Portfolio snippet ──
    with _pulse_right:
        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">⚡ Execution Rate</div>',
            unsafe_allow_html=True
        )
        _exec_delta = (ens_filt - ens_acc) * 100
        st.markdown(
            f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);'
            f'border:1px solid #1E2333;border-radius:12px;padding:16px;margin-bottom:12px">'
            f'<div style="display:flex;align-items:flex-end;justify-content:space-between">'
            f'<div class="stat-pop" style="font-size:1.9rem;font-weight:800">{ens_filt*100:.0f}%</div>'
            f'<div style="color:#2DD4BF;font-size:0.72rem">filtered vs raw <b>{_exec_delta:+.1f}%</b></div></div>'
            f'<div style="margin-top:8px">{_metric_bar(ens_filt*100)}</div>'
            f'<div style="color:#475569;font-size:0.68rem;margin-top:4px">accuracy on high-confidence signals only</div>'
            f'</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">🔀 Top Movers</div>',
            unsafe_allow_html=True
        )
        _mover_pool = [t for t in ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","DOGE-USD","AVAX-USD"] if t != ticker.upper()][:4]
        _mover_rows = ""
        for _mt in _mover_pool:
            _m24b = get_24h_stats_cached(_mt)
            _mpx  = get_live_price_cached(_mt)
            if not _m24b or not _mpx:
                continue
            _mch = _m24b['price_change_pct']
            _mc2 = "#2DD4BF" if _mch >= 0 else "#FB7185"
            _mhist = get_price_history_lite_cached(_mt)
            _mspk = sparkline_svg(_mhist, _mc2, width=64, height=28) if _mhist else ""
            _mover_rows += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 0;border-top:1px solid #1A1E2B">'
                f'<div><div style="font-size:0.82rem;font-weight:700">{_mt.replace("-USD","")}</div>'
                f'<div style="color:#475569;font-size:0.68rem">{DataManager.get_ticker_name(_mt) or _mt}</div></div>'
                f'<div style="width:64px;margin:0 8px">{_mspk}</div>'
                f'<div style="text-align:right"><div style="font-family:ui-monospace,monospace;font-size:0.78rem">'
                f'${_mpx:,.4f}</div><div style="color:{_mc2};font-size:0.7rem">{_mch:+.2f}%</div></div></div>'
            )
        st.markdown(
            f'<div style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px solid #1E2333;'
            f'border-radius:12px;padding:6px 14px;margin-bottom:12px">'
            f'{_mover_rows or "<div style=\'color:#475569;font-size:0.78rem;padding:8px 0\'>No mover data right now</div>"}'
            f'</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div style="color:#2DD4BF;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:8px">💼 Portfolio</div>',
            unsafe_allow_html=True
        )
        _pf_trades = st.session_state.get("portfolio_trades", [])
        if not _pf_trades:
            st.markdown(
                '<div style="background:linear-gradient(160deg,#161B2C,#10121C);border:1px dashed #1E2333;'
                'border-radius:12px;padding:16px;text-align:center;color:#475569;font-size:0.8rem">'
                'No trades yet &mdash; add one on the Portfolio page</div>', unsafe_allow_html=True
            )
        else:
            _pf_pnl = 0.0
            for _pt in _pf_trades:
                if _pt["status"] == "Open":
                    _pfl = get_live_price_cached(_pt["ticker"]) or _pt["entry"]
                    _pf_pnl += ((_pfl-_pt["entry"]) if _pt["side"]=="BUY" else (_pt["entry"]-_pfl)) * _pt["size"]
                else:
                    _pf_pnl += _pt.get("pnl",0) or 0
            _pf_c = "#2DD4BF" if _pf_pnl>=0 else "#FB7185"
            st.markdown(
                f'<div class="tilt-card" style="background:linear-gradient(160deg,#161B2C,#10121C);'
                f'border:1px solid #1E2333;border-radius:12px;padding:16px;margin-bottom:12px">'
                f'<div style="color:{_pf_c};font-size:1.5rem;font-weight:800">'
                f'{"+" if _pf_pnl>=0 else ""}{_pf_pnl:,.2f}</div>'
                f'<div style="color:#475569;font-size:0.72rem">{len(_pf_trades)} trade(s) &middot; full breakdown on Portfolio page</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown(
            f'<a href="#" onclick="return false" style="display:block;text-decoration:none;'
            f'background:linear-gradient(90deg,#2DD4BF18,#A78BFA18);border:1px solid #1E2333;'
            f'border-radius:12px;padding:14px 16px;color:inherit">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><div style="font-size:0.85rem;font-weight:700">🤖 Ask the AI Assistant</div>'
            f'<div style="color:#475569;font-size:0.7rem;margin-top:2px">scroll down for live Q&amp;A on {name}</div></div>'
            f'<span style="color:#A78BFA">&rsaquo;</span></div></a>',
            unsafe_allow_html=True
        )
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
    _bc  = "#0F2420" if last_sig=="BUY" else "#241521" if last_sig=="SELL" else "#10121C"
    _brd = "#2DD4BF" if last_sig=="BUY" else "#FB7185" if last_sig=="SELL" else "#1E2333"
    _sc  = "#2DD4BF" if last_sig=="BUY" else "#FB7185" if last_sig=="SELL" else "#64748B"
    _tc  = "#2DD4BF" if last_sig=="BUY" else "#FB7185"
    _slc = "#FB7185" if last_sig=="BUY" else "#2DD4BF"
    _dotcls2 = "buy" if last_sig=="BUY" else "sell" if last_sig=="SELL" else "hold"
    _em  = f'<span class="status-dot {_dotcls2}"></span>'

    # Pre-compute all values to avoid nested f-strings with quotes
    _tp_disp  = f"${tp_price:,.4f}" if tp_price else "—"
    _tp_pct_s = f'<span style="color:#94A3B8;font-size:0.75rem"> ({tp_pct:+.2f}%)</span>' if tp_price else ""
    _sl_disp  = f"${sl_price:,.4f}" if sl_price else "—"
    _sl_pct_s = f'<span style="color:#94A3B8;font-size:0.75rem"> ({sl_pct:+.2f}%)</span>' if sl_price else ""

    st.markdown(
        f'<div style="background:{_bc};border:2px solid {_brd};border-radius:14px;'
        f'padding:20px 22px;margin-bottom:10px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">'
        f'<div>'
        f'<div style="color:#94A3B8;font-size:0.72rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:6px">{_sig_label}</div>'
        f'<div class="stat-pop" style="font-size:2.6rem;font-weight:800;color:{_sc};line-height:1.0">{_em} {last_sig}</div>'
        f'<div style="margin-top:7px;font-size:0.85rem">'
        f'<span style="color:#94A3B8">Confidence: </span>'
        f'<span style="color:#F59E0B;font-weight:700;font-size:0.95rem">{last_conf:.1f}%</span>'
        f'<span style="color:#64748B;margin:0 6px">|</span>'
        f'<span style="color:#94A3B8">P(UP): </span>'
        f'<span style="color:#2DD4BF;font-weight:700">{last_prob*100:.1f}%</span>'
        f'</div></div>'
        f'<div style="text-align:right">'
        f'<div style="background:#070812;border:1px solid #1E2333;border-radius:8px;padding:8px 14px">'
        f'<div style="color:#64748B;font-size:0.68rem;text-transform:uppercase">Accuracy</div>'
        f'<div style="color:#F59E0B;font-size:1.5rem;font-weight:800">{ens_filt*100:.1f}%</div>'
        f'<div style="color:#64748B;font-size:0.65rem">filtered</div>'
        f'</div></div></div>'
        f'<div style="background:rgba(0,0,0,0.25);border-radius:10px;padding:14px 16px">'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
        f'<div><div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">Live Price</div>'
        f'<div style="color:#FFFFFF;font-size:1.1rem;font-weight:700">${display_price:,.4f}</div></div>'
        f'<div><div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">Entry</div>'
        f'<div style="color:#2DD4BF;font-size:1.1rem;font-weight:700">${entry_p:,.4f}</div></div>'
        f'<div><div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">🎯 Take Profit</div>'
        f'<div style="color:{_tc};font-size:1.05rem;font-weight:700">{_tp_disp}{_tp_pct_s}</div></div>'
        f'<div><div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;margin-bottom:2px">🛑 Stop Loss</div>'
        f'<div style="color:{_slc};font-size:1.05rem;font-weight:700">{_sl_disp}{_sl_pct_s}</div></div>'
        f'</div>'
        f'<div style="border-top:1px solid #1E2333;margin-top:12px;padding-top:10px;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<div><span style="color:#64748B;font-size:0.78rem">R/R: </span>'
        f'<span style="color:#FFFFFF;font-weight:700">1 : {rr:.2f}</span>'
        f'<span style="color:#64748B;font-size:0.72rem;margin-left:10px">ATR {atr_pct*100:.2f}%</span></div>'
        f'<div style="color:#64748B;font-size:0.68rem">Not financial advice</div>'
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
        "ACTIVE" : ("🟡","#F59E0B","#2E2410","Signal ACTIVE"),
        "HIT_TP" : ("🎯","#2DD4BF","#0F2420","TARGET HIT ✅"),
        "HIT_SL" : ("🛑","#FB7185","#241521","STOP LOSS HIT"),
        "EXPIRED": ("⏰","#64748B","#10121C","Signal EXPIRED"),
        "NONE"   : ("⚪","#64748B","#10121C","No active signal"),
    }
    _si,_sc2,_sb,_sl2 = _STATUS.get(_ss,_STATUS["NONE"])
    _rem  = _sig_status.get("remaining_h",0)
    _pnl  = _sig_status.get("pnl_pct",0)
    _hold = _sig_status.get("hours_old",0)

    if _ss != "NONE":
        _pnl_html = (f'<div style="color:{"#2DD4BF" if _pnl>=0 else "#FB7185"};'
                     f'font-weight:600;margin-top:5px;font-size:0.88rem">'
                     f'P&L: {"+" if _pnl>=0 else ""}{_pnl:.2f}%</div>'
                     if _ss not in ("NONE","EXPIRED") else "")
        _enter_html = (f'<div style="background:#0F2420;border-radius:6px;'
                       f'padding:7px 11px;margin-top:8px;color:#2DD4BF;font-size:0.80rem">'
                       f'✅ Enter within {min(4,_rem):.0f}h · TP ${tp_price:,.4f} · SL ${sl_price:,.4f}</div>'
                       if _ss=="ACTIVE" and last_sig!="HOLD" and tp_price and sl_price else "")
        _exp_html = ('<div style="background:#2A1C00;border-radius:6px;padding:7px 11px;'
                     'margin-top:8px;color:#F59E0B;font-size:0.80rem">'
                     '⚠️ EXPIRED — Do NOT enter. Wait for next signal.</div>'
                     if _ss=="EXPIRED" else "")
        _hold_str = f"{_hold:.0f}h since signal"
        _rem_str  = f"· ⏱ {_rem:.0f}h left" if _ss == "ACTIVE" else ""
        st.markdown(
            f'<div style="background:{_sb};border:2px solid {_sc2};border-radius:10px;padding:12px 16px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="color:{_sc2};font-weight:700">{_si} {_sl2}</span>'
            f'<div style="text-align:right;font-size:0.75rem;color:#94A3B8">'
            f'{_hold_str} {_rem_str}</div>'
            f'</div>'
            f'{_pnl_html}{_enter_html}{_exp_html}'
            f'</div>',
            unsafe_allow_html=True
        )

    # Intraday signals list
    if _intraday and results and "multi_signals" in results:
        _ms = results["multi_signals"]
        if _ms is not None and not _ms.empty:
            from datetime import datetime as _dt_cls, timezone as _tz_cls, timedelta as _td_cls
            _now_sig = _dt_cls.now(_tz_cls.utc) + _td_cls(hours=4)  # Dubai time

            # Group all signals by date
            _ms_all = _ms.head(30)
            _days_map = {}
            for _, _ir in _ms_all.iterrows():
                _full_date = str(_ir.get("Date",""))
                _day_key   = _full_date[:10]
                _days_map.setdefault(_day_key, []).append(_ir)

            total_sigs = sum(len(v) for v in _days_map.values())
            st.markdown(f"**⚡ Intraday Signals (1h) — {total_sigs} signals across {len(_days_map)} days**")

            for _day, _day_sigs in list(_days_map.items())[:7]:
                n = len(_day_sigs)
                st.markdown(
                    f'<div style="color:#94A3B8;font-size:0.74rem;margin:8px 0 3px 0;'
                    f'font-weight:600;border-bottom:1px solid #1A1E2B;padding-bottom:3px">'
                    f'📅 {_day} — <span style="color:#2DD4BF">{n} signal{"s" if n>1 else ""}</span></div>',
                    unsafe_allow_html=True)
                for _ir in _day_sigs:
                    try:
                        _ip   = float(str(_ir.get("Price","0")).replace("$","").replace(",",""))
                        _ibs  = "BUY" in str(_ir.get("Signal",""))
                        _icf  = float(str(_ir.get("Confidence","60%")).replace("%",""))
                        _im   = max(0.6, min(1.5, 0.8 + (_icf-60)/100))
                        _itp  = round(_ip + _im*last_atr, 4) if _ibs else round(_ip - _im*last_atr, 4)
                        _isl  = round(_ip - _im*0.9*last_atr, 4) if _ibs else round(_ip + _im*0.9*last_atr, 4)
                        _ic   = "#2DD4BF" if _ibs else "#FB7185"
                        _full = str(_ir.get("Date",""))
                        _time = _full[11:16] if len(_full) > 10 else "00:00"

                        # Expiry: signal valid for 2 hours after signal time
                        _expired = False
                        try:
                            _sig_dt  = _dt_cls.strptime(_full[:16], "%Y-%m-%d %H:%M")
                            _exp_dt  = _sig_dt + _td_cls(hours=2)
                            _now_naive = _now_sig.replace(tzinfo=None)
                            _expired = _now_naive > _exp_dt
                        except Exception:
                            pass

                        _status_badge = (
                            '<span style="color:#64748B;font-size:0.72rem">⏰ EXPIRED</span>'
                            if _expired else
                            '<span style="color:#2DD4BF;font-size:0.72rem">✅ LIVE</span>'
                            if _day == str(_now_sig.date()) else
                            '<span style="color:#94A3B8;font-size:0.72rem">📋 PAST</span>'
                        )

                        st.markdown(
                            f'<div style="background:{"#14141C" if _expired else "#10121C"};'
                            f'border-left:3px solid {"#1E2333" if _expired else _ic};'
                            f'border-radius:5px;padding:6px 12px;margin-bottom:2px;'
                            f'display:flex;justify-content:space-between;flex-wrap:wrap;'
                            f'font-size:0.80rem;gap:4px;opacity:{"0.5" if _expired else "1"}">'
                            f'<span style="color:{"#555" if _expired else _ic};font-weight:700">'
                            f'{"🟢 BUY" if _ibs else "🔴 SELL"}</span>'
                            f'<span style="color:#94A3B8">{_time}</span>'
                            f'<span style="color:{"#555" if _expired else "#FFFFFF"};font-weight:600">{_ir.get("Price","")}</span>'
                            f'<span style="color:{"#555" if _expired else "#2DD4BF"}">TP ${_itp:,.4f}</span>'
                            f'<span style="color:{"#555" if _expired else "#FB7185"}">SL ${_isl:,.4f}</span>'
                            f'<span style="color:#F59E0B">{_ir.get("Confidence","")}</span>'
                            f'{_status_badge}'
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
                _ac="#F59E0B" if _astat=="ACTIVE" else "#2DD4BF" if _astat=="HIT_TP" else "#FB7185"
                _ai="🟡" if _astat=="ACTIVE" else "🎯" if _astat=="HIT_TP" else "🛑"
                st.markdown(
                    f'<div style="background:#10121C;border-left:3px solid {_ac};'
                    f'border-radius:5px;padding:6px 12px;margin-bottom:3px;'
                    f'display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;font-size:0.81rem">'
                    f'<span style="color:#FFFFFF;font-weight:600">{_at}</span>'
                    f'<span style="color:{_ac}">{_ai} {_astat}</span>'
                    f'<span style="color:#94A3B8">{"BUY" if _asig=="BUY" else "SELL"} @ ${_ae:,.4f}</span>'
                    f'{"<span style=color:#2DD4BF>TP $"+f"{_atp:,.4f}</span>" if _atp else ""}'
                    f'{"<span style=color:#FB7185> SL $"+f"{_asl:,.4f}</span>" if _asl else ""}'
                    f'</div>', unsafe_allow_html=True)

# ── AI Assistant (inline, below signal card) ─────────────────────
_ai_chat_key = f"ai_msgs_{ticker}"
if _ai_chat_key not in st.session_state:
    st.session_state[_ai_chat_key] = []

# Build full app context for AI
_ai_pf = st.session_state.get("portfolio_trades", [])
_ai_open_trades = [t for t in _ai_pf if t.get("status","") == "Open"]
_ai_port_lines = [
    f"  {t['ticker']} {t['side']} @ ${t['entry']:.4f} | TP:${t.get('tp',0):.4f} SL:${t.get('sl',0):.4f}"
    for t in _ai_open_trades[:5]
]
_ai_port_str = chr(10).join(_ai_port_lines) if _ai_port_lines else "  None"

_ai_sig_lines = []
try:
    for _, _sr in sig_hist.head(8).iterrows():
        _ai_sig_lines.append(
            f"  {str(_sr.get('Date',''))[:10]} | {_sr.get('Signal','')} | "
            f"Price:{_sr.get('Price','')} | Conf:{_sr.get('Confidence','')}"
        )
except Exception:
    pass
_ai_sig_str = chr(10).join(_ai_sig_lines) if _ai_sig_lines else "  None"

_ai_model_info = ""
try:
    for _mn, _md in results.get("model_data", {}).items():
        _ai_model_info += f"  {_mn}: {_md['acc']*100:.1f}% acc, F1={_md['f1']:.3f}" + chr(10)
except Exception:
    pass

_ai_tp_str   = f"${tp_price:,.4f}" if tp_price else "N/A"
_ai_sl_str   = f"${sl_price:,.4f}" if sl_price else "N/A"
_ai_pr_str   = f"${display_price:,.4f}"
_ai_atr_str  = f"${last_atr:.4f}"
_ai_mode_str = st.session_state.get("signal_mode", "Daily")
_ai_cat_str  = st.session_state.get("asset_category", "")

# Fetch live news for AI context (keep short to avoid 413)
_ai_news_str = "Not available"
try:
    from data_manager import get_combined_news as _gcn
    _ai_nd   = _gcn(ticker)
    _ai_arts = _ai_nd.get("crypto_news", [])
    _ai_sent = _ai_nd.get("sentiment_score", 0)
    _sent_lbl = "Bullish" if _ai_sent>0.1 else "Bearish" if _ai_sent<-0.1 else "Neutral"
    _nl = [f"{str(_an.get('published_at',''))[:10]}: {_an.get('title','')[:60]}"
           for _an in _ai_arts[:3]]
    _ai_news_str = _sent_lbl + " | " + " || ".join(_nl) if _nl else _sent_lbl
except Exception:
    pass

_ai_system = (
    f"You are a trading AI assistant. {name} ({ticker}) | Price:{_ai_pr_str} | {last_sig} {last_conf:.0f}% conf | TP:{_ai_tp_str} SL:{_ai_sl_str} | Acc:{ens_filt*100:.0f}% | News:{_ai_news_str[:100]}. Answer any question concisely."
)

st.markdown(
    '<div style="color:#A78BFA;font-weight:700;font-size:0.82rem;text-transform:uppercase;'
    'letter-spacing:.04em;margin:4px 0 8px;display:flex;align-items:center;gap:6px">'
    '<span style="width:6px;height:6px;border-radius:50%;background:#A78BFA;display:inline-block"></span>'
    '🤖 AI Assistant</div>', unsafe_allow_html=True
)

# Quick question buttons + clear
_aq1, _aq2, _aq3, _aq4 = st.columns(4)
_ai_quick_map = {
    _aq1: "Is there a signal right now?",
    _aq2: "What are the TP and SL?",
    _aq3: "Should I enter this trade?",
    _aq4: "How accurate is the model?",
}
for _aqcol, _aqq in _ai_quick_map.items():
    if _aqcol.button(_aqq, key=f"aiq_{abs(hash(_aqq+ticker))}", use_container_width=True):
        st.session_state[_ai_chat_key].append({"role": "user", "content": _aqq})
        st.rerun()

# Clear chat button — always visible
if st.session_state[_ai_chat_key]:
    if st.button("🗑️ Clear Chat", key=f"ai_clear_{ticker}", type="secondary"):
        st.session_state[_ai_chat_key] = []
        st.rerun()

# Inline text input (stays in place, not stuck to bottom)
_ai_col_inp, _ai_col_btn = st.columns([5,1])
with _ai_col_inp:
    _ai_q = st.text_input(
        "Ask", label_visibility="collapsed",
        placeholder=f"Ask anything about {name} or trading...",
        key=f"ai_text_{ticker}"
    )
with _ai_col_btn:
    _ai_send = st.button("Send ➤", key=f"ai_send_{ticker}", use_container_width=True, type="primary")

# Show conversation history
for _am in st.session_state[_ai_chat_key][-8:]:
    with st.chat_message(_am["role"]):
        st.markdown(_am["content"])

if (_ai_send or _ai_q) and _ai_q.strip():
    _ai_msg = _ai_q.strip()
    st.session_state[_ai_chat_key].append({"role": "user", "content": _ai_msg})
    with st.chat_message("user"):
        st.markdown(_ai_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                import requests as _rqa
                _groq_key = st.secrets.get("GROQ_API_KEY", "")
                # Last 3 messages, 300 chars each to stay under token limit
                _msgs_groq = [
                    {"role": m["role"], "content": str(m["content"])[:300]}
                    for m in st.session_state[_ai_chat_key][-3:]
                ]
                # Use compound-beta for web search capability
                _groq_payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": _ai_system}] + _msgs_groq,
                    "max_tokens": 1024,
                    "temperature": 0.4,
                }
                _groq_resp = _rqa.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {_groq_key}",
                    },
                    json=_groq_payload,
                    timeout=45,
                )
                if _groq_resp.status_code == 200:
                    _resp_json = _groq_resp.json()
                    _ans = _resp_json["choices"][0]["message"]["content"]
                    # Show web search sources if used
                    _executed = _resp_json.get("choices",[{}])[0].get("message",{}).get("executed_tools",[])
                    if _executed:
                        _ans += chr(10) + chr(10) + "---" + chr(10) + "*🔍 Searched the web for real-time data*"
                    st.markdown(_ans)
                    st.session_state[_ai_chat_key].append({"role": "assistant", "content": _ans})
                elif _groq_resp.status_code == 404:
                    # compound-beta not available, fallback to llama
                    _groq_resp2 = _rqa.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Content-Type":"application/json","Authorization":f"Bearer {_groq_key}"},
                        json={"model":"llama-3.3-70b-versatile","messages":[{"role":"system","content":_ai_system}]+_msgs_groq,"max_tokens":1024,"temperature":0.4},
                        timeout=30,
                    )
                    if _groq_resp2.status_code == 200:
                        _ans = _groq_resp2.json()["choices"][0]["message"]["content"]
                        st.markdown(_ans)
                        st.session_state[_ai_chat_key].append({"role":"assistant","content":_ans})
                    else:
                        _err = f"Error {_groq_resp2.status_code}: {_groq_resp2.text[:200]}"
                        st.error(_err)
                        st.session_state[_ai_chat_key].append({"role":"assistant","content":_err})
                else:
                    _err = f"Error {_groq_resp.status_code}: {_groq_resp.text[:200]}"
                    st.error(_err)
                    st.session_state[_ai_chat_key].append({"role": "assistant", "content": _err})
            except Exception as _ex:
                _err = f"Connection error: {str(_ex)[:150]}"
                st.error(_err)
                st.session_state[_ai_chat_key].append({"role": "assistant", "content": _err})

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
        _7cap = display_price * 0.009
        _7dist = min(_conf_mult * _atr7, _7cap)
        if   _s7=="BUY":  _tp7=round(display_price+_7dist,4); _sl7=round(display_price-_7dist*0.9,4)
        elif _s7=="SELL": _tp7=round(display_price-_7dist,4); _sl7=round(display_price+_7dist*0.9,4)
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

    def _sig_color7(v):
        if "BUY"  in str(v): return "#2DD4BF"
        if "SELL" in str(v): return "#FB7185"
        return "#64748B"

    _hdr7 = "".join(
        f'<th style="text-align:left;padding:8px 10px;color:#94A3B8;font-size:0.72rem;'
        f'text-transform:uppercase;letter-spacing:0.03em;border-bottom:1px solid #1A1E2B">{h}</th>'
        for h in ["Date","Signal","Conf %","Entry","Take Profit","Stop Loss","R/R"]
    )
    _body7 = ""
    for _i7, _r7 in enumerate(_rows7):
        _sc7 = _sig_color7(_r7["Signal"])
        _tp_c7 = "#2DD4BF" if _r7["Take Profit"] != "—" else "#64748B"
        _sl_c7 = "#FB7185" if _r7["Stop Loss"]   != "—" else "#64748B"
        _body7 += (
            f'<tr class="outlook-row" style="animation-delay:{_i7*0.07:.2f}s">'
            f'<td style="padding:8px 10px;color:#E2E8F0;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["Date"]}</td>'
            f'<td style="padding:8px 10px;color:{_sc7};font-weight:700;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["Signal"]}</td>'
            f'<td style="padding:8px 10px;color:#E2E8F0;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["Conf %"]}</td>'
            f'<td style="padding:8px 10px;color:#E2E8F0;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["Entry"]}</td>'
            f'<td style="padding:8px 10px;color:{_tp_c7};font-weight:600;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["Take Profit"]}</td>'
            f'<td style="padding:8px 10px;color:{_sl_c7};font-weight:600;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["Stop Loss"]}</td>'
            f'<td style="padding:8px 10px;color:#E2E8F0;font-size:0.85rem;border-bottom:1px solid #141726">{_r7["R/R"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="background:linear-gradient(160deg,#161B2C 0%,#10121C 100%);'
        f'border:1px solid #1E2333;border-radius:14px;padding:6px 4px;overflow-x:auto;'
        f'box-shadow:0 8px 24px rgba(0,0,0,0.35)">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>{_hdr7}</tr></thead><tbody>{_body7}</tbody></table></div>',
        unsafe_allow_html=True
    )

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
tab0,tab1,tab2,tab3,tab4,tab5,tab7,tab8 = st.tabs([
    "📡 Live Chart",
    "📈 Price & Signals",
    "🎯 Predicted vs Actual",
    "📊 Model Performance",
    "📜 Signal History",
    "📰 News & Sentiment",
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
        _tf_type = "primary" if _selected_tf==_tf_val else "secondary"
        if _col.button(_tf_label, key=f"tf_{_tf_val}",
                       use_container_width=True, type=_tf_type):
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

    _sig_color  = "#2DD4BF" if last_sig=="BUY" else "#FB7185" if last_sig=="SELL" else "#64748B"
    _price_chg_color = "#2DD4BF" if day_chg >= 0 else "#FB7185"
    _tp_disp    = f"${tp_price:,.4f}" if tp_price else "—"
    _sl_disp    = f"${sl_price:,.4f}" if sl_price else "—"
    _tp_pct_disp= f"{tp_pct:+.2f}%" if tp_price else "No signal"
    _sl_pct_disp= f"{sl_pct:+.2f}%" if sl_price else "No signal"

    # TradingView chart + integrated signal bar
    _tv_live_html = f"""
<style>
body{{background:#070812}}
.sb{{
  display:flex;justify-content:space-between;align-items:center;
  background:#070812;border-top:1px solid #1E2333;
  padding:12px 24px;flex-wrap:wrap;gap:8px;
}}
.sb-item{{text-align:center;min-width:120px}}
.sb-label{{color:#94A3B8;font-size:0.72rem;font-family:sans-serif;
  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px}}
.sb-value{{font-size:1.05rem;font-weight:700;font-family:sans-serif}}
.sb-sub{{font-size:0.72rem;font-family:sans-serif;margin-top:1px}}
.divider{{width:1px;background:#1E2333;height:40px;align-self:center}}
.rec-bar{{
  background:#070812;border-top:1px solid #1A1E2B;
  padding:8px 24px;font-size:0.75rem;font-family:sans-serif;color:#64748B;
}}
</style>

<div style="background:#070812;border-radius:10px;overflow:hidden;border:1px solid #1E2333">
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
      "toolbar_bg"        : "#10121C",
      "hide_side_toolbar" : false,
      "hide_top_toolbar"  : false,
      "allow_symbol_change": true,
      "save_image"        : false,
      "backgroundColor"   : "#070812",
      "studies"           : [
        "Volume@tv-basicstudies",
        "RSI@tv-basicstudies",
        "MACD@tv-basicstudies",
        "BB@tv-basicstudies"
      ],
      "overrides": {{
        "paneProperties.background"                      : "#070812",
        "paneProperties.backgroundType"                  : "solid",
        "paneProperties.vertGridProperties.color"        : "#1A1826",
        "paneProperties.horzGridProperties.color"        : "#1A1826",
        "symbolWatermarkProperties.color"                : "rgba(0,0,0,0)",
        "scalesProperties.textColor"                     : "#94A3B8",
        "mainSeriesProperties.candleStyle.upColor"       : "#2DD4BF",
        "mainSeriesProperties.candleStyle.downColor"     : "#FB7185",
        "mainSeriesProperties.candleStyle.borderUpColor" : "#2DD4BF",
        "mainSeriesProperties.candleStyle.borderDownColor": "#FB7185",
        "mainSeriesProperties.candleStyle.wickUpColor"   : "#2DD4BF",
        "mainSeriesProperties.candleStyle.wickDownColor" : "#FB7185"
      }}
    }});
    </script>
  </div>

  <!-- Signal bar -->
  <div class="sb">
    <div class="sb-item">
      <div class="sb-label">Live Price</div>
      <div class="sb-value" style="color:#FFFFFF">${display_price:,.4f}</div>
      <div class="sb-sub" style="color:{_price_chg_color}">{day_chg:+.2f}% today</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">Next Session Signal</div>
      <div class="sb-value" style="color:{_sig_color}">
        {'🟢' if last_sig=='BUY' else '🔴' if last_sig=='SELL' else '⚪'} {last_sig}
      </div>
      <div class="sb-sub" style="color:#94A3B8">{next_str} · {last_conf:.1f}% confidence</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">🎯 Take Profit</div>
      <div class="sb-value" style="color:#2DD4BF">{_tp_disp}</div>
      <div class="sb-sub" style="color:#2DD4BF">{_tp_pct_disp}</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">🛑 Stop Loss</div>
      <div class="sb-value" style="color:#FB7185">{_sl_disp}</div>
      <div class="sb-sub" style="color:#FB7185">{_sl_pct_disp}</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">Risk / Reward</div>
      <div class="sb-value" style="color:#F59E0B">1 : {rr:.2f}</div>
      <div class="sb-sub" style="color:#94A3B8">ATR: ${last_atr:.4f} ({atr_pct*100:.2f}%)</div>
    </div>
    <div class="divider"></div>
    <div class="sb-item">
      <div class="sb-label">Model Accuracy</div>
      <div class="sb-value" style="color:#F59E0B">{ens_filt*100:.1f}%</div>
      <div class="sb-sub" style="color:#94A3B8">when confidence ≥60%</div>
    </div>
  </div>

  <!-- Recent signals row -->
  <div class="rec-bar">
    ⚡ Recent BUY signals:&nbsp;
    <span style="color:#2DD4BF">{_buy_str}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Recent SELL signals:&nbsp;
    <span style="color:#FB7185">{_sell_str}</span>
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
            f'<a href="{_inv_url}" target="_blank" style="background:#1A1E2B;color:#E2E8F0;'
            f'border:1px solid #1E2333;border-radius:5px;padding:6px 16px;'
            f'text-decoration:none;font-size:0.82rem;margin-right:8px">'
            f'↗ Live Chart on Investing.com</a>'
            f'<a href="https://www.dfm.ae" target="_blank" style="background:#1A1E2B;color:#E2E8F0;'
            f'border:1px solid #1E2333;border-radius:5px;padding:6px 16px;'
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
        # Per-signal TP/SL with Result — only signals from 2026-05-06 onward
        _hist_rows = []
        _closes_arr = te_df['Close'].values if te_df is not None and 'Close' in te_df.columns else []
        _dates_arr  = list(te_df.index.strftime('%Y-%m-%d')) if te_df is not None else []
        _sig_filter_date = "2026-05-06"
        _filtered_hist = sig_hist[sig_hist['Date'].astype(str).str[:10] >= _sig_filter_date] if 'Date' in sig_hist.columns else sig_hist
        for _, _row in _filtered_hist.head(20).iterrows():
            try:
                _p  = float(str(_row.get("Price","0")).replace("$","").replace(",",""))
                if _p <= 0: continue
                _is = "BUY" in str(_row.get("Signal",""))
                _cf = float(str(_row.get("Confidence","60%")).replace("%",""))
                _m    = max(0.6, min(1.5, 0.8 + (_cf - 60) / 100))
                _mcap = {"GC=F":30,"ETH-USD":30,"BTC-USD":300}.get(ticker, _m*last_atr)
                _dist = min(_m * last_atr, _mcap)
                _tp   = round(_p + _dist, 4) if _is else round(_p - _dist, 4)
                _sl   = round(_p - _dist*0.9, 4) if _is else round(_p + _dist*0.9, 4)
                _rr_val = abs(_tp-_p)/max(abs(_p-_sl),0.0001)
                # Check result in next 5 candles
                _result = "⏳ Active"
                _sd = str(_row.get("Date",""))
                _sd10 = _sd[:10]  # date only for lookup
                if _sd10 in _dates_arr and len(_closes_arr) > 0:
                    _si = _dates_arr.index(_sd10)
                    for _fc in _closes_arr[_si+1:_si+6]:
                        if not (_fc==_fc): continue  # skip NaN
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
                if 'BUY'  in str(v): return 'color:#2DD4BF;font-weight:bold'
                if 'SELL' in str(v): return 'color:#FB7185;font-weight:bold'
                return 'color:#64748B'
            def _cres(v):
                if 'HIT TP' in str(v): return 'color:#2DD4BF;font-weight:700'
                if 'HIT SL' in str(v): return 'color:#FB7185;font-weight:700'
                return 'color:#94A3B8'
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
    fig.patch.set_facecolor('#070812')
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            height_ratios=[5, 1.1, 1.0, 0.9, 0.9], hspace=0.03)

    # ── Panel 1: Candlestick ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#10121C')

    # Regime shading
    for i in range(len(D)-1):
        ax1.axvspan(D[i], D[i+1], alpha=1,
            color='#0F2420' if reg_[i]==1 else '#241521', linewidth=0, zorder=0)

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
            color=C_UP, edgecolors='#0F766E', lw=1.0, zorder=7,
            label=f'🟢 BUY ({int(b_show.sum())})')
        for _d, _p, _l in zip(D[b_show], close_[b_show], low_[b_show]):
            ax1.annotate(
                f'B ${_p:.1f}\nTP ${_p+2*last_atr:.1f}',
                (_d, _l*0.985), fontsize=6.2, color=C_UP, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='#070812',
                          ec='#0F766E', alpha=0.88, lw=0.6))

    # SELL markers ▼ + label
    if s_show.sum() > 0:
        ax1.scatter(D[s_show], high_[s_show]*1.052, marker='v', s=130,
            color=C_DOWN, edgecolors='#9F1239', lw=1.0, zorder=7,
            label=f'🔴 SELL ({int(s_show.sum())})')
        for _d, _p, _h in zip(D[s_show], close_[s_show], high_[s_show]):
            ax1.annotate(
                f'S ${_p:.1f}\nTP ${_p-2*last_atr:.1f}',
                (_d, _h*1.015), fontsize=6.2, color=C_DOWN, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='#070812',
                          ec='#9F1239', alpha=0.88, lw=0.6))

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
        bbox=dict(boxstyle='round,pad=0.45', fc='#10121C', ec=_arrc, alpha=0.96, lw=1.2))

    # Legend
    _bull = mpatches.Patch(color='#0F2420', label='Bull regime')
    _bear = mpatches.Patch(color='#241521', label='Bear regime')
    _bb   = mpatches.Patch(color=C_BLUE, alpha=0.25, label='Bollinger Bands')
    _h, _l2 = ax1.get_legend_handles_labels()
    ax1.legend(_h+[_bull,_bear,_bb], _l2+['Bull','Bear','BB'],
               loc='upper left', ncol=5, framealpha=0.88, fontsize=7.5)
    ax1.set_ylabel('Price', fontsize=10, color=C_WHITE)
    ax1.xaxis.set_ticklabels([])
    ax1.spines[['top','right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.18, color='#1A1E2B')
    ax1.set_title(
        f'{name} ({ticker})  ·  Candlestick + Signals  ·  {n_show} days  ·  '
        f'Acc {ens_acc*100:.1f}%  ·  {int(b_show.sum())} BUY  {int(s_show.sum())} SELL',
        color=C_WHITE, fontsize=11, pad=10, fontweight='bold')

    # ── Panel 2: Volume ─────────────────────────────────────────
    ax_v = fig.add_subplot(gs[1], sharex=ax1); ax_v.set_facecolor('#10121C')
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
    ax2 = fig.add_subplot(gs[2], sharex=ax1); ax2.set_facecolor('#10121C')
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
    ax3 = fig.add_subplot(gs[3], sharex=ax1); ax3.set_facecolor('#10121C')
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
    ax4 = fig.add_subplot(gs[4], sharex=ax1); ax4.set_facecolor('#10121C')
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
    fig.patch.set_facecolor('#070812')
    fig.suptitle(f'Predicted vs Actual — {name} Test Set',color=C_WHITE,fontsize=12,y=1.01)

    ax=axes[0]; ax.set_facecolor('#10121C')
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
        bbox=dict(fc='#10121C',ec='#1E2333',boxstyle='round,pad=0.4'))
    ax.set_ylabel('Price'); ax.legend(loc='upper left')
    ax.xaxis.set_ticklabels([])
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=0.2)

    ax2=axes[1]; ax2.set_facecolor('#10121C')
    err=PPp-C2[:_plen]
    ax2.fill_between(D2p,err,0,where=(err>=0),color=C_UP,alpha=0.55,label='Too high')
    ax2.fill_between(D2p,err,0,where=(err<0),color=C_DOWN,alpha=0.55,label='Too low')
    ax2.axhline(0,color=C_DIM,lw=0.8,ls='--')
    ax2.set_xlim(D2p[0] if len(D2p)>0 else None, D2p[-1] if len(D2p)>0 else None)
    ax2.set_ylabel('Error'); ax2.legend(loc='upper left',ncol=2)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y',alpha=0.2)

    ax3=axes[2]; ax3.set_facecolor('#10121C')
    try:
        correct=(EP==YT)
        ax3.bar(D2,np.where(correct,1,-1),color=np.where(correct,C_UP,C_DOWN),width=1.2,alpha=0.8)
    except Exception: pass
    ax3.axhline(0,color=C_DIM,lw=0.6)
    ax3.set_yticks([-1,0,1]); ax3.set_yticklabels(['Wrong','','Correct'],fontsize=8)
    ax3.text(0.99,0.95,f'Direction Acc = {dacc*100:.1f}%',
        transform=ax3.transAxes,ha='right',va='top',fontsize=9,color=C_UP,
        bbox=dict(fc='#10121C',ec='#1E2333',boxstyle='round,pad=0.4'))
    ax3.set_xlabel('Date')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(),rotation=30,ha='right')
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y',alpha=0.2)
    st.pyplot(fig,use_container_width=True); plt.close()

# ── TAB 3: Model Performance ───────────────────────────────────────
with tab3:
    dark_fig()
    if not model_data:
        empty_state("📊", "No model data available", "Try switching assets or forcing a refresh from the sidebar.")
    else:
        names=[*model_data.keys(),'Ensemble']
        accs =[model_data[k]['acc']*100 for k in model_data]+[ens_acc*100]
        f1s  =[model_data[k]['f1']      for k in model_data]+[results['ensemble_f1']]
        aucs =[model_data[k]['auc']     for k in model_data]+[results['ensemble_auc']]
        bc   =[C_DOWN if a<60 else C_GOLD if a<65 else C_UP for a in accs]
        x    =np.arange(len(names))

        fig=plt.figure(figsize=(14,10)); fig.patch.set_facecolor('#070812')
        gs2=gridspec.GridSpec(2,3,figure=fig,hspace=0.48,wspace=0.38)
        fig.suptitle(f'Model Performance — {name}',color=C_WHITE,fontsize=13)

        ax1=fig.add_subplot(gs2[0,0]); ax1.set_facecolor('#10121C')
        bars=ax1.bar(x,accs,color=bc,alpha=0.88,edgecolor='#070812',width=0.65)
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

        ax2=fig.add_subplot(gs2[0,1]); ax2.set_facecolor('#10121C')
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

        ax3=fig.add_subplot(gs2[0,2]); ax3.set_facecolor('#10121C')
        bars3=ax3.bar(x,f1s,color=bc,alpha=0.88,edgecolor='#070812',width=0.65)
        ax3.set_xticks(x); ax3.set_xticklabels(names,fontsize=7,rotation=15)
        ax3.set_ylim(0,1); ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score',color=C_WHITE)
        for bar,v in zip(bars3,f1s):
            ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                f'{v:.3f}',ha='center',fontweight='bold',fontsize=7.5,color=C_WHITE)
        ax3.spines[['top','right']].set_visible(False)

        for col_i,(nm,dm) in enumerate(list(model_data.items())[:3]):
            ax_cm=fig.add_subplot(gs2[1,col_i]); ax_cm.set_facecolor('#10121C')
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
            fig2.patch.set_facecolor('#070812'); ax.set_facecolor('#10121C')
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
            _sig_filter = "2026-05-06"
            _src_filtered = _src_sigs[_src_sigs['Date'].astype(str).str[:10] >= _sig_filter] if 'Date' in _src_sigs.columns else _src_sigs
            for _, _row in _src_filtered.head(150).iterrows():
                try:
                    _p  = float(str(_row.get("Price","0")).replace("$","").replace(",",""))
                    if _p <= 0: continue
                    _is = "BUY" in str(_row.get("Signal",""))
                    _cf = float(str(_row.get("Confidence","60%")).replace("%",""))
                    _m    = max(0.6, min(1.5, 0.8 + (_cf - 60) / 100))
                    _mcap4= {"GC=F":30,"ETH-USD":30,"BTC-USD":300}.get(ticker, _m*last_atr)
                    _dist4= min(_m * last_atr, _mcap4)
                    _tp   = round(_p + _dist4, 4) if _is else round(_p - _dist4, 4)
                    _sl   = round(_p - _dist4*0.9, 4) if _is else round(_p + _dist4*0.9, 4)
                    _rr_v = abs(_tp-_p)/max(abs(_p-_sl), 0.0001)
                    # Check result in next 5 candles (date-only match)
                    _result = "⏳ Active"
                    _sd = str(_row.get("Date",""))
                    _sd10 = _sd[:10]
                    if _sd10 in _dates_arr and len(_closes_arr) > 0:
                        _si = _dates_arr.index(_sd10)
                        for _fc in _closes_arr[_si+1:_si+6]:
                            if not (_fc == _fc): continue  # skip NaN
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
                    if 'BUY'  in str(v): return 'color:#2DD4BF;font-weight:700'
                    if 'SELL' in str(v): return 'color:#FB7185;font-weight:700'
                    return 'color:#64748B'
                def _sh_ctp(v):  return 'color:#2DD4BF;font-weight:600' if '$' in str(v) else ''
                def _sh_csl(v):  return 'color:#FB7185;font-weight:600' if '$' in str(v) else ''
                def _sh_cres(v):
                    if 'HIT TP' in str(v): return 'color:#2DD4BF;font-weight:700'
                    if 'HIT SL' in str(v): return 'color:#FB7185;font-weight:700'
                    return 'color:#94A3B8'
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
                empty_state("📋", "No signals yet", "Try lowering the confidence threshold in the sidebar.")
        else:
            empty_state("📋", "No signals found", "Try lowering the confidence threshold in the sidebar.")

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
            empty_state("📰", "No crypto news found", "This loads live once deployed on Streamlit Cloud.")
        else:
            _pos = sum(1 for n in _cnews if "Positive" in n.get("sentiment",""))
            _neg = sum(1 for n in _cnews if "Negative" in n.get("sentiment",""))
            _neu = len(_cnews) - _pos - _neg
            _ov  = ("BULLISH" if _sscore > 0.05 else "BEARISH" if _sscore < -0.05 else "NEUTRAL")
            _ov_e= ("GREEN"   if _sscore > 0.05 else "RED"     if _sscore < -0.05 else "GREY")
            _ov_c= ("#2DD4BF" if _sscore > 0.05 else "#FB7185" if _sscore < -0.05 else "#64748B")

            _m1,_m2,_m3,_m4 = st.columns(4)
            _m1.metric("Articles",  len(_cnews))
            _m2.metric("Positive",  _pos)
            _m3.metric("Negative",  _neg)
            _m4.metric("Sentiment", _ov, delta=f"score {_sscore:+.3f}")

            _gp = int(_pos / max(len(_cnews),1) * 100)
            st.markdown(
                f'<div style="background:#10121C;border-radius:8px;padding:12px 20px;'
                f'border:1px solid #1E2333;margin:10px 0">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.78rem">'
                f'<span style="color:#2DD4BF">Positive {_pos}</span>'
                f'<span style="color:#94A3B8">Neutral {_neu}</span>'
                f'<span style="color:#FB7185">Negative {_neg}</span></div>'
                f'<div style="background:#1A1E2B;border-radius:4px;height:8px;overflow:hidden">'
                f'<div style="background:linear-gradient(90deg,#2DD4BF,#F59E0B);'
                f'width:{_gp}%;height:100%;border-radius:4px"></div></div>'
                f'<div style="text-align:center;color:{_ov_c};font-size:0.73rem;margin-top:4px">'
                f'Market Sentiment: {_ov} | Score: {_sscore:+.3f}</div></div>',
                unsafe_allow_html=True
            )
            st.divider()

            for _n in _cnews:
                _s  = _n.get("sentiment","Neutral")
                _sc = _n.get("score",0)
                _bc = "#2DD4BF" if "Positive" in _s else "#FB7185" if "Negative" in _s else "#1E2333"
                st.markdown(
                    f'<div style="background:#10121C;border:1px solid {_bc};border-left:4px solid {_bc};'
                    f'border-radius:8px;padding:11px 15px;margin-bottom:7px">'
                    f'<div style="display:flex;justify-content:space-between;gap:10px">'
                    f'<div style="flex:1">'
                    f'<a href="{_n.get("url","#")}" target="_blank" '
                    f'style="color:#FFFFFF;font-weight:600;font-size:0.87rem;'
                    f'text-decoration:none;line-height:1.4">{_n.get("title","")}</a>'
                    f'<div style="color:#94A3B8;font-size:0.72rem;margin-top:3px">'
                    f'{_n.get("source","")} &middot; {_n.get("published","")}</div></div>'
                    f'<div style="text-align:right;min-width:85px">'
                    f'<div style="font-size:0.79rem;font-weight:600;color:{_bc}">{_s}</div>'
                    f'<div style="color:#64748B;font-size:0.69rem">score {_sc:+.2f}</div>'
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
<style>*{margin:0;padding:0;box-sizing:border-box;}body{background:#070812;}</style>
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
    "backgroundColor": "#070812"
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
                    _ic_map = {"High":"#FB7185","Medium":"#F59E0B","Low":"#64748B","Holiday":"#1E2333"}
                    for _e in _fev:
                        _ic  = _ic_map.get(_e["impact_raw"],"#1E2333")
                        _act = _e.get("actual","—") or "—"
                        _ac  = "#FFFFFF"
                        if _act != "—":
                            try:
                                _fv2 = float(str(_e.get("forecast","0")).replace("%","").replace("K","").replace("M","") or "0")
                                _av2 = float(_act.replace("%","").replace("K","").replace("M",""))
                                _ac  = "#2DD4BF" if _av2 >= _fv2 else "#FB7185"
                            except Exception:
                                pass
                        st.markdown(
                            f'<div style="background:#10121C;border:1px solid #1A1E2B;'
                            f'border-left:4px solid {_ic};border-radius:6px;'
                            f'padding:9px 15px;margin-bottom:5px;display:flex;'
                            f'justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">'
                            f'<div style="flex:2;min-width:180px">'
                            f'<span style="color:#FFFFFF;font-weight:600;font-size:0.84rem">{_e["event"]}</span>'
                            f'<span style="color:#94A3B8;font-size:0.72rem;margin-left:8px">'
                            f'{_e["currency"]} &middot; {_e["date"]} {_e["time"]}</span></div>'
                            f'<div style="display:flex;gap:14px">'
                            f'<div style="text-align:center;min-width:50px">'
                            f'<div style="color:{_ic};font-size:0.72rem;font-weight:600">{_e["impact_raw"]}</div></div>'
                            f'<div style="text-align:center;min-width:55px">'
                            f'<div style="color:#64748B;font-size:0.65rem">Forecast</div>'
                            f'<div style="color:#94A3B8;font-size:0.80rem">{_e["forecast"]}</div></div>'
                            f'<div style="text-align:center;min-width:55px">'
                            f'<div style="color:#64748B;font-size:0.65rem">Previous</div>'
                            f'<div style="color:#94A3B8;font-size:0.80rem">{_e["previous"]}</div></div>'
                            f'<div style="text-align:center;min-width:55px">'
                            f'<div style="color:#64748B;font-size:0.65rem">Actual</div>'
                            f'<div style="color:{_ac};font-size:0.80rem;font-weight:600">{_act}</div>'
                            f'</div></div></div>',
                            unsafe_allow_html=True
                        )
                    st.caption(f"Showing {len(_fev)} of {len(_fcal)} events · Source: ForexFactory.com")

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
        empty_state("🔀", "No scan run yet",
                     "Click Scan All Assets Now to see live signals for every asset in one view "
                     "&mdash; first scan takes ~30 seconds.")
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
        _bias_c   = "#2DD4BF" if _nb > _ns else "#FB7185" if _ns > _nb else "#94A3B8"
        st.markdown(
            f'<div style="background:#10121C;border:1px solid #1E2333;border-radius:8px;'
            f'padding:12px 20px;margin:10px 0">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.78rem">'
            f'<span style="color:#2DD4BF">BUY {_buy_pct}%</span>'
            f'<span style="color:#94A3B8">HOLD {_hold_pct}%</span>'
            f'<span style="color:#FB7185">SELL {_sell_pct}%</span></div>'
            f'<div style="height:10px;background:#1A1E2B;border-radius:5px;overflow:hidden;display:flex">'
            f'<div style="width:{_buy_pct}%;background:#2DD4BF"></div>'
            f'<div style="width:{_hold_pct}%;background:#1E2333"></div>'
            f'<div style="width:{_sell_pct}%;background:#FB7185"></div></div>'
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
                            f'<div style="background:#10121C;border:1px solid #1E2333;'
                            f'border-radius:10px;padding:14px;text-align:center">'
                            f'<div style="color:#94A3B8;font-size:0.85rem;font-weight:600">'
                            f'{_r["ticker"]}</div>'
                            f'<div style="color:#FB7185;font-size:0.75rem;margin-top:6px">'
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

                    _sc  = "#2DD4BF" if _rsig=="BUY" else "#FB7185" if _rsig=="SELL" else "#64748B"
                    _bc  = "buy-card" if _rsig=="BUY" else "sell-card" if _rsig=="SELL" else "hold-card"
                    _em  = "🟢" if _rsig=="BUY" else "🔴" if _rsig=="SELL" else "⚪"
                    _tps = f"${_rtp:,.4f}" if _rtp else "—"
                    _sls = f"${_rsl:,.4f}" if _rsl else "—"
                    _rr  = abs(_rtp-_rprice)/max(abs(_rprice-_rsl),0.0001) if _rtp and _rsl else 0

                    st.markdown(
                        f'<div class="signal-card {_bc}" style="padding:14px 16px">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div>'
                        f'<div style="color:#FFFFFF;font-weight:700;font-size:0.95rem">{_r.get("ticker","")}</div>'
                        f'<div style="color:#94A3B8;font-size:0.72rem">{_r.get("name","")}</div>'
                        f'</div>'
                        f'<div style="color:{_sc};font-size:1.3rem;font-weight:700">{_em} {_rsig}</div>'
                        f'</div>'
                        f'<div style="color:#FFFFFF;font-size:1.1rem;font-weight:600;margin:8px 0">'
                        f'${_rprice:,.4f}</div>'
                        f'<table style="width:100%;font-size:0.78rem;border-collapse:collapse">'
                        f'<tr><td style="color:#94A3B8">Confidence</td>'
                        f'<td style="color:#F59E0B;text-align:right;font-weight:600">{_rconf:.1f}%</td></tr>'
                        f'<tr><td style="color:#94A3B8">P(UP)</td>'
                        f'<td style="color:#2DD4BF;text-align:right">{_rprob*100:.1f}%</td></tr>'
                        f'<tr><td style="color:#94A3B8">Take Profit</td>'
                        f'<td style="color:#2DD4BF;text-align:right;font-weight:600">{_tps}</td></tr>'
                        f'<tr><td style="color:#94A3B8">Stop Loss</td>'
                        f'<td style="color:#FB7185;text-align:right;font-weight:600">{_sls}</td></tr>'
                        f'<tr><td style="color:#94A3B8">R/R</td>'
                        f'<td style="color:#FFFFFF;text-align:right">1:{_rr:.2f}</td></tr>'
                        f'<tr><td style="color:#94A3B8">Accuracy</td>'
                        f'<td style="color:#F59E0B;text-align:right">{_racc*100:.1f}%</td></tr>'
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
        empty_state("📈", "No signal data available", "Try switching assets or forcing a refresh from the sidebar.")
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
            _fig2.patch.set_facecolor('#070812'); _ax2.set_facecolor('#10121C')
            _ea = __import__("numpy").array(_equity)
            _ax2.plot(_ea, color='#2DD4BF' if _total_return>=0 else '#FB7185', lw=1.8)
            _ax2.fill_between(range(len(_ea)), _start_cap, _ea,
                where=(_ea>=_start_cap), alpha=0.15, color='#2DD4BF')
            _ax2.fill_between(range(len(_ea)), _start_cap, _ea,
                where=(_ea<_start_cap),  alpha=0.15, color='#FB7185')
            _ax2.axhline(_start_cap, color='#64748B', ls='--', lw=1.0, alpha=0.6)
            _ax2.set_title(f"Equity Curve — {name} | Return: {_total_return:+.1f}% | Win Rate: {_win_rate:.1f}%",
                           color='#FFFFFF', fontsize=11, fontweight='bold')
            _ax2.set_ylabel("Capital ($)", color='#94A3B8')
            _ax2.set_xlabel("Trade #",    color='#94A3B8')
            _ax2.tick_params(colors='#94A3B8')
            _ax2.spines[['top','right']].set_visible(False)
            _ax2.grid(axis='y', alpha=0.2, color='#1A1E2B')
            _plt2.tight_layout()
            st.pyplot(_fig2, use_container_width=True)
            _plt2.close()

        if _trades_log:
            st.divider()
            st.markdown("**Individual Trade Log**")
            import pandas as _pd3
            _tdf = _pd3.DataFrame(_trades_log)
            def _cr(v): return ('color:#2DD4BF;font-weight:600' if '✅' in str(v)
                                else 'color:#FB7185;font-weight:600' if '❌' in str(v) else '')
            def _cp(v):
                try: return 'color:#2DD4BF' if float(str(v).replace('+','').replace(',',''))>=0 else 'color:#FB7185'
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
            try: return 'color:#2DD4BF' if float(str(v).replace('+','').replace(',',''))>=0 else 'color:#FB7185'
            except: return ''
        try: _cts = _ctdf.style.map(_ctc, subset=['P&L'])
        except: _cts = _ctdf
        st.dataframe(_cts, use_container_width=True, hide_index=True)
        _tot = sum(t.get("pnl",0) or 0 for t in _ctl)
        st.metric("Total Closed P&L", f"{'+'if _tot>=0 else ''}{_tot:,.4f}")

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;padding-top:16px;border-top:1px solid #1A1E2B;'
    'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">'
    '<div style="display:flex;align-items:center;gap:7px;color:#475569;font-size:0.75rem">'
    '<span style="width:6px;height:6px;border-radius:50%;background:#2DD4BF;display:inline-block"></span>'
    'Prediction Dashboard &middot; research tool, not financial advice</div>'
    '<div style="color:#475569;font-size:0.72rem">Data refreshes every 6h</div>'
    '</div>', unsafe_allow_html=True
)





# ════════════════════════════════════════════════════════════════════
