"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SOL/USD Auto-Updating Prediction Dashboard                                 ║
║  Fetches live price daily · Retrains models · Predicts tomorrow             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOL/USD Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0D1117; color: #C9D1D9; }
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { color: #F0F6FC; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #8B949E; font-size: 0.85rem; }
    [data-testid="stMetricDelta"] { font-size: 1rem !important; }

    /* Cards */
    .signal-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .buy-card  { border-left: 5px solid #3FB950; }
    .sell-card { border-left: 5px solid #F85149; }
    .hold-card { border-left: 5px solid #6E7681; }

    /* Table */
    .signal-table th {
        background: #1F6FEB;
        color: white;
        padding: 8px 12px;
        text-align: center;
    }
    .signal-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #30363D;
        text-align: center;
    }
    .signal-table tr:nth-child(even) { background: #1C2128; }

    /* Divider */
    hr { border-color: #30363D; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #161B22; border-right: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────────
from data_manager import DataManager
from model_engine import ModelEngine
from feature_engine import build_features

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cryptologos.cc/logos/solana-sol-logo.png", width=60)
    st.title("SOL/USD Predictor")
    st.caption("Auto-updating · Retrains daily")
    st.divider()

    st.subheader("⚙️ Settings")
    confidence_thresh = st.slider(
        "Signal Confidence Threshold",
        min_value=0.50, max_value=0.80, value=0.60, step=0.01,
        help="Only show BUY/SELL when model is this confident. Higher = fewer but better signals."
    )
    lookback_days = st.selectbox(
        "Chart lookback",
        [30, 60, 90, 120, 180, 365],
        index=2,
        help="How many days to show in the price chart"
    )
    st.divider()

    st.subheader("🔄 Data")
    force_refresh = st.button("🔄 Refresh Data Now", use_container_width=True)
    st.caption("Data auto-refreshes every 24h")
    st.divider()

    st.subheader("📋 About")
    st.caption(
        "This dashboard fetches live SOL/USD price data from CoinGecko, "
        "retrains ML models daily, and generates next-day and weekly predictions. "
        "**Not financial advice.**"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (cached, refreshes every 24 hours)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)  # 86400s = 24 hours
def load_and_train(force=False):
    """Fetch live data, build features, train models. Cached 24h."""
    dm      = DataManager()
    df_raw  = dm.get_data()
    df_feat = build_features(df_raw)
    engine  = ModelEngine(df_feat)
    results = engine.train()
    return df_raw, df_feat, engine, results

with st.spinner("🔄 Loading data and training models..."):
    cache_key = datetime.now(timezone.utc).strftime("%Y-%m-%d") + str(force_refresh)
    try:
        df_raw, df_feat, engine, results = load_and_train(force_refresh)
        data_ok = True
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        st.info("Make sure you have an internet connection. Retrying...")
        data_ok = False
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
last_date    = df_feat.index[-1]
last_close   = float(df_feat['Close'].iloc[-1])
prev_close   = float(df_feat['Close'].iloc[-2])
day_change   = (last_close - prev_close) / prev_close * 100

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown(f"## 🔮 SOL/USD Prediction Dashboard")
    st.caption(f"Last updated: **{last_date.strftime('%A, %d %B %Y')}**  ·  "
               f"Data: {df_feat.index[0].strftime('%b %Y')} → {last_date.strftime('%b %Y')}  ·  "
               f"{len(df_feat):,} trading days")
with col_refresh:
    st.metric("Last Price", f"${last_close:.2f}",
              delta=f"{day_change:+.2f}%",
              delta_color="normal")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ═══════════════════════════════════════════════════════════════════════════════
ens_acc      = results['ensemble_acc']
ens_filt_acc = results['ensemble_filt_acc']
n_signals    = results['n_signals']
last_signal  = results['last_signal']
last_conf    = results['last_confidence']
last_prob    = results['last_prob']

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Model Accuracy",     f"{ens_acc*100:.1f}%",    help="Overall ensemble direction accuracy on test set")
col2.metric("Filtered Accuracy",  f"{ens_filt_acc*100:.1f}%", delta=f"+{(ens_filt_acc-ens_acc)*100:.1f}% vs raw", help="Accuracy when confidence ≥60%")
col3.metric("P(Price UP Tomorrow)",f"{last_prob*100:.1f}%", delta=("UP ↑" if last_prob > 0.5 else "DOWN ↓"))
col4.metric("Signal Confidence",   f"{last_conf:.1f}%",     help="How confident the model is in tomorrow's signal")
col5.metric("Active Signals (test)",str(n_signals),         help="Total BUY+SELL signals in test period")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TOMORROW SIGNAL + 7-DAY OUTLOOK
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Trading Signals")

col_sig, col_7d = st.columns([1, 2])

with col_sig:
    # Signal card
    entry_p  = round(last_close * 0.995,  2)
    target_p = round(last_close * (1 + abs(last_prob - 0.5) * 0.07), 2)
    stop_p   = round(last_close * 0.97,   2)
    rr       = abs(target_p - entry_p) / max(abs(entry_p - stop_p), 0.01)

    if last_signal == 'BUY':
        card_class = "buy-card"
        emoji      = "🟢"
        sig_color  = "#3FB950"
    elif last_signal == 'SELL':
        card_class = "sell-card"
        emoji      = "🔴"
        sig_color  = "#F85149"
    else:
        card_class = "hold-card"
        emoji      = "⚪"
        sig_color  = "#6E7681"

    next_day_str = (last_date + timedelta(days=1)).strftime('%A %d %b %Y')

    st.markdown(f"""
    <div class="signal-card {card_class}">
        <div style="font-size:0.85rem;color:#8B949E;margin-bottom:4px">TOMORROW — {next_day_str}</div>
        <div style="font-size:2.2rem;font-weight:bold;color:{sig_color}">{emoji} {last_signal}</div>
        <div style="font-size:0.9rem;color:#8B949E;margin-top:8px">Confidence: <b style="color:#E3B341">{last_conf:.1f}%</b></div>
        <div style="font-size:0.9rem;color:#8B949E">P(UP): <b style="color:#58A6FF">{last_prob*100:.1f}%</b></div>
        <hr style="border-color:#30363D;margin:12px 0">
        <table style="width:100%;font-size:0.9rem">
            <tr><td style="color:#8B949E">Current</td><td style="color:#F0F6FC;text-align:right"><b>${last_close:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Entry</td><td style="color:#3FB950;text-align:right"><b>${entry_p:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Target</td><td style="color:#E3B341;text-align:right"><b>${target_p:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Stop Loss</td><td style="color:#F85149;text-align:right"><b>${stop_p:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Risk/Reward</td><td style="color:#F0F6FC;text-align:right"><b>1 : {rr:.2f}</b></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_7d:
    st.markdown("**📅 7-Day Forward Outlook**")
    st.caption("Confidence decays further out — closer days are more reliable")

    # Build 7-day table
    week_rows = []
    for i in range(7):
        day_dt   = last_date + timedelta(days=i+1)
        day_prob = 0.5 + (last_prob - 0.5) * np.exp(-0.35 * i)
        day_sig  = "🟢 BUY" if day_prob >= confidence_thresh else \
                   "🔴 SELL" if day_prob <= (1 - confidence_thresh) else "⚪ HOLD"
        day_dir  = "📈 UP" if day_prob > 0.5 else "📉 DOWN"
        day_conf = max(day_prob, 1 - day_prob) * 100
        week_rows.append({
            "Day"           : day_dt.strftime("%a %d %b"),
            "Signal"        : day_sig,
            "Direction"     : day_dir,
            "Confidence"    : f"{day_conf:.0f}%",
            "P(UP)"         : f"{day_prob*100:.1f}%",
        })

    week_df = pd.DataFrame(week_rows)
    st.dataframe(
        week_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Day"       : st.column_config.TextColumn("Date"),
            "Signal"    : st.column_config.TextColumn("Signal"),
            "Direction" : st.column_config.TextColumn("Direction"),
            "Confidence": st.column_config.TextColumn("Confidence"),
            "P(UP)"     : st.column_config.TextColumn("P(Price UP)"),
        }
    )

    # Summary
    n_buy  = sum(1 for r in week_rows if 'BUY'  in r['Signal'])
    n_sell = sum(1 for r in week_rows if 'SELL' in r['Signal'])
    n_hold = 7 - n_buy - n_sell
    weekly_outlook = "📈 BULLISH" if n_buy > n_sell else "📉 BEARISH" if n_sell > n_buy else "➡️ SIDEWAYS"
    st.markdown(
        f"**Weekly Outlook: {weekly_outlook}**  ·  "
        f"🟢 {n_buy} BUY  ·  🔴 {n_sell} SELL  ·  ⚪ {n_hold} HOLD"
    )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Signals",
    "🎯 Predicted vs Actual",
    "📊 Model Performance",
    "📜 Signal History"
])

# Shared plot style
def setup_dark_fig():
    plt.rcParams.update({
        'figure.facecolor' : '#0D1117',
        'axes.facecolor'   : '#161B22',
        'axes.edgecolor'   : '#30363D',
        'axes.labelcolor'  : '#C9D1D9',
        'xtick.color'      : '#8B949E',
        'ytick.color'      : '#8B949E',
        'text.color'       : '#C9D1D9',
        'grid.color'       : '#21262D',
        'grid.linewidth'   : 0.5,
        'legend.facecolor' : '#161B22',
        'legend.edgecolor' : '#30363D',
        'legend.fontsize'  : 8,
        'font.size'        : 9,
    })

C_UP   = '#3FB950'; C_DOWN = '#F85149'; C_BLUE = '#58A6FF'
C_GOLD = '#E3B341'; C_GREY = '#6E7681'; C_WHITE= '#F0F6FC'; C_DIM  = '#8B949E'

# ── TAB 1: Price & Signals ─────────────────────────────────────────────────────
with tab1:
    setup_dark_fig()

    te_df      = results['te_df']
    ens_proba  = results['ens_proba']
    signals    = results['signals']
    HIGH       = results['HIGH']
    LOW        = results['LOW']

    # Slice to lookback
    n_show  = min(lookback_days, len(te_df))
    te_show = te_df.iloc[-n_show:]
    pr_show = ens_proba[-n_show:]
    sg_show = signals[-n_show:]
    buy_s   = sg_show ==  1
    sell_s  = sg_show == -1

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(4, 1, figure=fig,
        height_ratios=[4.5, 1.2, 1.0, 1.0], hspace=0.04)

    # Panel 1: Price
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#161B22')
    dates = te_show.index
    close = te_show['Close'].values
    high  = te_show['High'].values
    low   = te_show['Low'].values
    regime= te_show['Regime'].values

    for i in range(len(dates)-1):
        ax1.axvspan(dates[i], dates[i+1], alpha=1,
            color='#1C2A1C' if regime[i]==1 else '#2A1C1C', linewidth=0, zorder=0)

    ax1.plot(dates, close, color=C_BLUE, lw=1.4, label='SOL Close', zorder=4)
    ax1.plot(dates, te_show['SMA20'].values, color=C_GOLD, lw=0.9, ls='--', alpha=0.8, label='SMA 20')
    ax1.plot(dates, te_show['SMA50'].values, color='#A371F7', lw=0.9, ls='-.', alpha=0.8, label='SMA 50')

    ax1.scatter(dates[buy_s],  low[buy_s]*0.955,
        marker='^', s=100, color=C_UP, edgecolors='#196127', lw=0.6, zorder=6,
        label=f'BUY ({buy_s.sum()})')
    ax1.scatter(dates[sell_s], high[sell_s]*1.045,
        marker='v', s=100, color=C_DOWN, edgecolors='#8B0000', lw=0.6, zorder=6,
        label=f'SELL ({sell_s.sum()})')

    for dt in dates[buy_s]:  ax1.axvline(dt, color=C_UP,  alpha=0.08, lw=0.7, zorder=1)
    for dt in dates[sell_s]: ax1.axvline(dt, color=C_DOWN,alpha=0.08, lw=0.7, zorder=1)

    # Tomorrow arrow
    next_dt  = dates[-1] + pd.Timedelta(days=1)
    tgt      = close[-1] * (1 + (float(ens_proba[-1]) - 0.5) * 0.08)
    arr_c    = C_UP if ens_proba[-1] > 0.5 else C_DOWN
    ax1.annotate('', xy=(next_dt, tgt), xytext=(dates[-1], close[-1]),
        arrowprops=dict(arrowstyle='-|>', color=arr_c, lw=2.5, mutation_scale=16))
    ax1.scatter([next_dt], [tgt], marker='*', s=220, color=C_GOLD, zorder=9)
    ax1.annotate(f'  TOMORROW\n  ${tgt:.2f}\n  {last_signal}',
        (next_dt, tgt), fontsize=8.5, color=arr_c, fontweight='bold',
        xytext=(8, 0), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.35', fc='#161B22', ec='#30363D', alpha=0.95))

    bull_p = mpatches.Patch(color='#1C2A1C', label='Bull Regime')
    bear_p = mpatches.Patch(color='#2A1C1C', label='Bear Regime')
    h, l2  = ax1.get_legend_handles_labels()
    ax1.legend(h+[bull_p,bear_p], l2+['Bull','Bear'],
        loc='upper left', ncol=4, framealpha=0.85)
    ax1.set_ylabel('Price (USD)'); ax1.xaxis.set_ticklabels([])
    ax1.spines[['top','right']].set_visible(False); ax1.grid(axis='y', alpha=0.2)

    # Panel 2: Probability
    ax2 = fig.add_subplot(gs[1], sharex=ax1); ax2.set_facecolor('#161B22')
    bc  = np.where(pr_show>=HIGH, C_UP, np.where(pr_show<=LOW, C_DOWN, C_GREY))
    ax2.bar(dates, pr_show, color=bc, width=1.0, alpha=0.85)
    ax2.fill_between(dates, HIGH, pr_show, where=(pr_show>=HIGH), alpha=0.2, color=C_UP)
    ax2.fill_between(dates, LOW,  pr_show, where=(pr_show<=LOW),  alpha=0.2, color=C_DOWN)
    ax2.axhline(HIGH, color=C_UP,  ls='--', lw=1.0)
    ax2.axhline(LOW,  color=C_DOWN,ls='--', lw=1.0)
    ax2.axhline(0.5,  color=C_DIM, ls=':',  lw=0.7)
    ax2.set_ylabel('P(UP)'); ax2.set_ylim(0,1)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y', alpha=0.2)

    # Panel 3: RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1); ax3.set_facecolor('#161B22')
    rsi = te_show['RSI'].values
    ax3.plot(dates, rsi, color='#D29922', lw=0.9)
    ax3.fill_between(dates, 70, rsi, where=(rsi>70), alpha=0.25, color=C_DOWN)
    ax3.fill_between(dates, 30, rsi, where=(rsi<30), alpha=0.25, color=C_UP)
    ax3.axhline(70, color=C_DOWN, ls='--', lw=0.8, alpha=0.7)
    ax3.axhline(30, color=C_UP,   ls='--', lw=0.8, alpha=0.7)
    ax3.set_ylabel('RSI'); ax3.set_ylim(10,90)
    ax3.xaxis.set_ticklabels([])
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y', alpha=0.2)

    # Panel 4: MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1); ax4.set_facecolor('#161B22')
    macd = te_show['MACD'].values; msig = te_show['MACD_sig'].values
    mh   = macd - msig
    ax4.plot(dates, macd, color=C_BLUE, lw=1.0, label='MACD')
    ax4.plot(dates, msig, color='#F78166', lw=1.0, ls='--', label='Signal')
    ax4.bar(dates, mh, color=np.where(mh>=0, C_UP, C_DOWN), width=1.0, alpha=0.55)
    ax4.axhline(0, color=C_DIM, lw=0.7)
    ax4.set_ylabel('MACD'); ax4.set_xlabel('Date')
    ax4.legend(loc='upper left', ncol=2)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax4.spines[['top','right']].set_visible(False); ax4.grid(axis='y', alpha=0.2)

    fig.suptitle(f'SOL/USD — Buy/Sell Signals  |  Last {lookback_days} days  |  '
                 f'Ensemble Acc: {ens_acc*100:.1f}%',
        color=C_WHITE, fontsize=12, y=1.01)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── TAB 2: Predicted vs Actual ─────────────────────────────────────────────────
with tab2:
    setup_dark_fig()
    price_pred = results['price_pred']
    y_price_te = results['y_price_te']
    ens_pred   = results['ens_pred']
    y_te       = results['y_te']

    n_show2  = min(lookback_days, len(te_df))
    d2       = te_df.index[-n_show2:]
    c2       = te_df['Close'].values[-n_show2:]
    pp2      = price_pred[-n_show2:]
    ep2      = ens_pred[-n_show2:]
    yt2      = y_te[-n_show2:]

    rmse = float(np.sqrt(np.mean((pp2 - c2)**2)))
    mae  = float(np.mean(np.abs(pp2 - c2)))
    mape = float(np.mean(np.abs((c2 - pp2) / (c2 + 1e-9))) * 100)
    dir_acc = accuracy_score(yt2, ep2)

    c1, c2m, c3, c4 = st.columns(4)
    c1.metric("RMSE",  f"${rmse:.2f}")
    c2m.metric("MAE",  f"${mae:.2f}")
    c3.metric("MAPE",  f"{mape:.2f}%")
    c4.metric("Direction Acc", f"{dir_acc*100:.1f}%")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios':[4, 1.8, 1.5], 'hspace':0.04})
    fig.patch.set_facecolor('#0D1117')

    ax = axes[0]; ax.set_facecolor('#161B22')
    ax.plot(d2, c2,   color=C_BLUE, lw=1.5, label='Actual Close', zorder=4)
    ax.plot(d2, pp2, color=C_GOLD, lw=1.2, ls='--', alpha=0.9, label='Predicted (Ridge)', zorder=3)
    ax.fill_between(d2, c2, pp2, where=(pp2>=c2), alpha=0.10, color=C_UP)
    ax.fill_between(d2, c2, pp2, where=(pp2< c2), alpha=0.10, color=C_DOWN)
    ax.text(0.99, 0.97, f'RMSE=${rmse:.2f}  MAE=${mae:.2f}  MAPE={mape:.2f}%',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=C_GOLD,
        bbox=dict(fc='#161B22', ec='#30363D', boxstyle='round,pad=0.4'))
    ax.set_ylabel('Price (USD)'); ax.legend(loc='upper left')
    ax.xaxis.set_ticklabels([])
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.2)

    ax2 = axes[1]; ax2.set_facecolor('#161B22')
    err = pp2 - c2
    ax2.fill_between(d2, err, 0, where=(err>=0), color=C_UP,   alpha=0.55, label='Too high')
    ax2.fill_between(d2, err, 0, where=(err< 0), color=C_DOWN, alpha=0.55, label='Too low')
    ax2.axhline(0, color=C_DIM, lw=0.8, ls='--')
    ax2.set_ylabel('Error (USD)'); ax2.legend(loc='upper left', ncol=2)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y', alpha=0.2)

    ax3 = axes[2]; ax3.set_facecolor('#161B22')
    correct = (ep2 == yt2)
    ax3.bar(d2, np.where(correct, 1, -1),
        color=np.where(correct, C_UP, C_DOWN), width=1.2, alpha=0.8)
    ax3.axhline(0, color=C_DIM, lw=0.6)
    ax3.set_yticks([-1,0,1]); ax3.set_yticklabels(['Wrong','','Correct'], fontsize=8)
    ax3.text(0.99, 0.95, f'Direction Acc = {dir_acc*100:.1f}%',
        transform=ax3.transAxes, ha='right', va='top', fontsize=9, color=C_UP,
        bbox=dict(fc='#161B22', ec='#30363D', boxstyle='round,pad=0.4'))
    ax3.set_xlabel('Date')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y', alpha=0.2)

    fig.suptitle('Predicted vs Actual Price — Test Set', color=C_WHITE, fontsize=12, y=1.01)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── TAB 3: Model Performance ───────────────────────────────────────────────────
with tab3:
    setup_dark_fig()
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score, roc_auc_score

    model_data = results['model_data']

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle('Model Performance Summary', color=C_WHITE, fontsize=13)

    names  = list(model_data.keys()) + ['Ensemble']
    accs   = [model_data[k]['acc']*100 for k in model_data] + [results['ensemble_acc']*100]
    f1s    = [model_data[k]['f1']      for k in model_data] + [results['ensemble_f1']]
    aucs   = [model_data[k]['auc']     for k in model_data] + [results['ensemble_auc']]
    bc     = [C_DOWN if a<60 else C_GOLD if a<65 else C_UP for a in accs]
    x      = np.arange(len(names))

    ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor('#161B22')
    bars = ax1.bar(x, accs, color=bc, alpha=0.88, edgecolor='#0D1117', width=0.65)
    ax1.axhline(50, color=C_DIM, ls=':', lw=1.2, label='Random')
    ax1.axhline(65, color=C_GOLD,ls='--',lw=1.2, label='65%')
    ax1.axhline(70, color=C_UP,  ls='--',lw=1.2, label='70%')
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylim(40, 90); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Directional Accuracy', color=C_WHITE)
    for bar, v in zip(bars, accs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.1f}%', ha='center', fontweight='bold', fontsize=8, color=C_WHITE)
    ax1.legend(fontsize=7); ax1.spines[['top','right']].set_visible(False)

    ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor('#161B22')
    ax2.plot([0,1],[0,1],'--',color=C_DIM,lw=1,label='Random')
    y_te_plot = results['y_te']
    for k, d_m in model_data.items():
        fpr, tpr, _ = roc_curve(y_te_plot, d_m['proba'])
        ax2.plot(fpr, tpr, lw=1.3, label=f'{k}  {d_m["auc"]:.3f}')
    fpr, tpr, _ = roc_curve(y_te_plot, results['ens_proba'])
    ax2.plot(fpr, tpr, color=C_UP, lw=2.2, label=f'Ensemble  {results["ensemble_auc"]:.3f}')
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curves', color=C_WHITE)
    ax2.legend(loc='lower right', fontsize=7)
    ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    ax2.spines[['top','right']].set_visible(False)

    ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor('#161B22')
    bars3 = ax3.bar(x, f1s, color=bc, alpha=0.88, edgecolor='#0D1117', width=0.65)
    ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=8)
    ax3.set_ylim(0,1); ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score', color=C_WHITE)
    for bar,v in zip(bars3,f1s):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{v:.3f}', ha='center', fontweight='bold', fontsize=8, color=C_WHITE)
    ax3.spines[['top','right']].set_visible(False)

    # Confusion matrix
    best_key = max(model_data.keys(), key=lambda k: model_data[k]['acc'])
    for col, (nm, dat) in enumerate(list(model_data.items())[:3]):
        ax_cm = fig.add_subplot(gs[1,col]); ax_cm.set_facecolor('#161B22')
        cm    = confusion_matrix(y_te_plot, dat['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
            xticklabels=['DOWN','UP'], yticklabels=['DOWN','UP'],
            linewidths=0.5, annot_kws={'size':13,'weight':'bold'}, cbar=False)
        ax_cm.set_title(f'{nm}  Acc={dat["acc"]*100:.1f}%', color=C_WHITE)
        ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Rolling accuracy
    setup_dark_fig()
    correct_arr = (results['ens_pred'] == results['y_te']).astype(int)
    roll_acc    = pd.Series(correct_arr).rolling(30, min_periods=5).mean() * 100

    fig2, ax = plt.subplots(figsize=(14, 4))
    fig2.patch.set_facecolor('#0D1117'); ax.set_facecolor('#161B22')
    ax.plot(te_df.index, roll_acc, color=C_BLUE, lw=1.5)
    ax.fill_between(te_df.index, 65, roll_acc, where=(roll_acc>=65), alpha=0.18, color=C_UP)
    ax.axhline(results['ensemble_acc']*100, color=C_GOLD, ls='--', lw=1.5,
        label=f'Overall {results["ensemble_acc"]*100:.1f}%')
    ax.axhline(65, color=C_UP,  ls=':', lw=1.5, label='65% target')
    ax.axhline(50, color=C_DIM, ls=':', lw=1.0, label='Random 50%')
    ax.set_ylim(25,95); ax.legend(fontsize=8)
    ax.set_title('30-Day Rolling Accuracy', color=C_WHITE)
    ax.set_xlabel('Date'); ax.set_ylabel('Accuracy (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.2)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ── TAB 4: Signal History ──────────────────────────────────────────────────────
with tab4:
    sig_hist = results['signal_history']
    if not sig_hist.empty:
        st.subheader("📋 All Buy/Sell Signals — Test Period")
        st.caption("Only signals where model confidence ≥60% are shown")

        def color_signal(val):
            if 'BUY'  in str(val): return 'color: #3FB950; font-weight: bold'
            if 'SELL' in str(val): return 'color: #F85149; font-weight: bold'
            return 'color: #6E7681'

        styled = sig_hist.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        n_buy_h  = (sig_hist['Signal']=='🟢 BUY').sum()
        n_sell_h = (sig_hist['Signal']=='🔴 SELL').sum()
        col1.metric("Total Signals", len(sig_hist))
        col2.metric("BUY Signals",   n_buy_h)
        col3.metric("SELL Signals",  n_sell_h)
        avg_conf = sig_hist['Confidence'].str.replace('%','').astype(float).mean()
        col4.metric("Avg Confidence", f"{avg_conf:.1f}%")
    else:
        st.info("No filtered signals generated yet.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#6E7681;font-size:0.8rem'>"
    "⚠️ For research and educational purposes only · Not financial advice · "
    "Data: CoinGecko API · Models retrain daily"
    "</div>",
    unsafe_allow_html=True
)

# Local import for metrics used inside tabs
from sklearn.metrics import accuracy_score
