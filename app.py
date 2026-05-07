"""
SOL/USD Auto-Updating Prediction Dashboard
Fetches live price daily · Retrains models · Predicts tomorrow
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

# ── ALL sklearn imports at the top (fixes NameError) ──────────────────────────
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
    mean_squared_error, mean_absolute_error,
)

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
    .stApp { background-color: #0D1117; color: #C9D1D9; }
    .block-container { padding-top: 1.5rem; }
    [data-testid="metric-container"] {
        background: #161B22; border: 1px solid #30363D;
        border-radius: 10px; padding: 16px 20px;
    }
    [data-testid="stMetricValue"]  { color: #F0F6FC; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"]  { color: #8B949E; font-size: 0.85rem; }
    [data-testid="stMetricDelta"]  { font-size: 1rem !important; }
    .signal-card {
        background: #161B22; border: 1px solid #30363D;
        border-radius: 12px; padding: 20px 24px; margin: 8px 0;
    }
    .buy-card  { border-left: 5px solid #3FB950; }
    .sell-card { border-left: 5px solid #F85149; }
    .hold-card { border-left: 5px solid #6E7681; }
    hr { border-color: #30363D; }
    [data-testid="stSidebar"] { background: #161B22; border-right: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_UP   = '#3FB950'; C_DOWN = '#F85149'; C_BLUE = '#58A6FF'
C_GOLD = '#E3B341'; C_GREY = '#6E7681'; C_WHITE= '#F0F6FC'; C_DIM  = '#8B949E'

# ── App imports ────────────────────────────────────────────────────────────────
from data_manager import DataManager
from model_engine import ModelEngine
from feature_engine import build_features

# ── Dark plot style helper ─────────────────────────────────────────────────────
def setup_dark_fig():
    plt.rcParams.update({
        'figure.facecolor' : '#0D1117', 'axes.facecolor'   : '#161B22',
        'axes.edgecolor'   : '#30363D', 'axes.labelcolor'  : '#C9D1D9',
        'xtick.color'      : '#8B949E', 'ytick.color'      : '#8B949E',
        'text.color'       : '#C9D1D9', 'grid.color'       : '#21262D',
        'grid.linewidth'   : 0.5,       'legend.facecolor' : '#161B22',
        'legend.edgecolor' : '#30363D', 'legend.fontsize'  : 8,
        'font.size'        : 9,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔮 SOL/USD Predictor")
    st.caption("Auto-updating · Retrains daily")
    st.divider()
    st.subheader("⚙️ Settings")
    confidence_thresh = st.slider(
        "Signal Confidence Threshold",
        min_value=0.50, max_value=0.80, value=0.60, step=0.01,
        help="Only show BUY/SELL when model is this confident.",
    )
    lookback_days = st.selectbox(
        "Chart lookback", [30, 60, 90, 120, 180, 365], index=2,
    )
    st.divider()
    st.subheader("🔄 Data")
    force_refresh = st.button("🔄 Refresh Data Now", use_container_width=True)
    st.caption("Data auto-refreshes every 24h")
    st.divider()
    st.caption("⚠️ For research only · Not financial advice")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA  (cached 24 hours)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def load_and_train(_force=False):
    dm      = DataManager()
    df_raw  = dm.get_data()
    df_feat = build_features(df_raw)
    engine  = ModelEngine(df_feat)
    results = engine.train()
    return df_raw, df_feat, results

with st.spinner("🔄 Loading data and training models..."):
    try:
        df_raw, df_feat, results = load_and_train(force_refresh)
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        st.info("Please check your internet connection or upload Solana_Historical_Data.csv to the repo.")
        st.stop()

# ── Unpack results ─────────────────────────────────────────────────────────────
last_date    = df_feat.index[-1]
last_close   = float(df_feat['Close'].iloc[-1])
prev_close   = float(df_feat['Close'].iloc[-2])
day_change   = (last_close - prev_close) / prev_close * 100

ens_acc      = results['ensemble_acc']
ens_filt_acc = results['ensemble_filt_acc']
last_signal  = results['last_signal']
last_conf    = results['last_confidence']
last_prob    = results['last_prob']
n_signals    = results['n_signals']
te_df        = results['te_df']
ens_proba    = results['ens_proba']
ens_pred     = results['ens_pred']
y_te         = results['y_te']
signals      = results['signals']
HIGH         = results['HIGH']
LOW          = results['LOW']
price_pred   = results['price_pred']
y_price_te   = results['y_price_te']
model_data   = results['model_data']
sig_history  = results['signal_history']

# ── Align lengths (safety check) ──────────────────────────────────────────────
# All arrays must be the same length as te_df
n = min(len(te_df), len(ens_proba), len(ens_pred), len(y_te),
        len(signals), len(price_pred), len(y_price_te))
te_df      = te_df.iloc[-n:]
ens_proba  = ens_proba[-n:]
ens_pred   = ens_pred[-n:]
y_te       = y_te[-n:]
signals    = signals[-n:]
price_pred = price_pred[-n:]
y_price_te = y_price_te[-n:]
buy_m      = signals ==  1
sell_m     = signals == -1

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
col_h, col_p = st.columns([5, 1])
with col_h:
    st.markdown("## 🔮 SOL/USD Prediction Dashboard")
    st.caption(
        f"Last updated: **{last_date.strftime('%A, %d %B %Y')}**  ·  "
        f"Data: {df_feat.index[0].strftime('%b %Y')} → {last_date.strftime('%b %Y')}  ·  "
        f"{len(df_feat):,} trading days"
    )
with col_p:
    st.metric("Last Price", f"${last_close:.2f}", delta=f"{day_change:+.2f}%")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TOP METRICS
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model Accuracy",       f"{ens_acc*100:.1f}%")
c2.metric("Filtered Accuracy",    f"{ens_filt_acc*100:.1f}%",
          delta=f"+{(ens_filt_acc-ens_acc)*100:.1f}% vs raw")
c3.metric("P(UP Tomorrow)",       f"{last_prob*100:.1f}%",
          delta="UP ↑" if last_prob > 0.5 else "DOWN ↓")
c4.metric("Signal Confidence",    f"{last_conf:.1f}%")
c5.metric("Active Signals",       str(n_signals))
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TOMORROW SIGNAL + 7-DAY OUTLOOK
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Trading Signals")
col_sig, col_7d = st.columns([1, 2])

with col_sig:
    entry_p  = round(last_close * 0.995, 2)
    target_p = round(last_close * (1 + abs(last_prob - 0.5) * 0.07), 2)
    stop_p   = round(last_close * 0.97,  2)
    rr       = abs(target_p - entry_p) / max(abs(entry_p - stop_p), 0.01)

    card_class = "buy-card" if last_signal=="BUY" else "sell-card" if last_signal=="SELL" else "hold-card"
    emoji      = "🟢" if last_signal=="BUY" else "🔴" if last_signal=="SELL" else "⚪"
    sig_color  = C_UP  if last_signal=="BUY" else C_DOWN if last_signal=="SELL" else C_GREY
    next_str   = (last_date + timedelta(days=1)).strftime('%A %d %b %Y')

    st.markdown(f"""
    <div class="signal-card {card_class}">
        <div style="font-size:0.82rem;color:#8B949E">TOMORROW — {next_str}</div>
        <div style="font-size:2.2rem;font-weight:bold;color:{sig_color};margin:4px 0">{emoji} {last_signal}</div>
        <div style="font-size:0.9rem;color:#8B949E">Confidence: <b style="color:#E3B341">{last_conf:.1f}%</b></div>
        <div style="font-size:0.9rem;color:#8B949E">P(UP): <b style="color:#58A6FF">{last_prob*100:.1f}%</b></div>
        <hr style="border-color:#30363D;margin:10px 0">
        <table style="width:100%;font-size:0.88rem">
            <tr><td style="color:#8B949E">Current</td>  <td style="color:#F0F6FC;text-align:right"><b>${last_close:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Entry</td>    <td style="color:#3FB950;text-align:right"><b>${entry_p:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Target</td>   <td style="color:#E3B341;text-align:right"><b>${target_p:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">Stop Loss</td><td style="color:#F85149;text-align:right"><b>${stop_p:.2f}</b></td></tr>
            <tr><td style="color:#8B949E">R/R</td>      <td style="color:#F0F6FC;text-align:right"><b>1 : {rr:.2f}</b></td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

with col_7d:
    st.markdown("**📅 7-Day Forward Outlook**")
    st.caption("Confidence decays further out — tomorrow is most reliable")
    week_rows = []
    for i in range(7):
        day_dt   = last_date + timedelta(days=i+1)
        day_prob = 0.5 + (last_prob - 0.5) * np.exp(-0.35 * i)
        day_sig  = ("🟢 BUY"  if day_prob >= confidence_thresh else
                    "🔴 SELL" if day_prob <= (1-confidence_thresh) else "⚪ HOLD")
        week_rows.append({
            "Date"      : day_dt.strftime("%a %d %b"),
            "Signal"    : day_sig,
            "Direction" : "📈 UP" if day_prob > 0.5 else "📉 DOWN",
            "Confidence": f"{max(day_prob,1-day_prob)*100:.0f}%",
            "P(UP)"     : f"{day_prob*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(week_rows), use_container_width=True, hide_index=True)

    n_buy_w  = sum(1 for r in week_rows if "BUY"  in r["Signal"])
    n_sell_w = sum(1 for r in week_rows if "SELL" in r["Signal"])
    outlook  = "📈 BULLISH" if n_buy_w>n_sell_w else "📉 BEARISH" if n_sell_w>n_buy_w else "➡️ SIDEWAYS"
    st.markdown(f"**Weekly Outlook: {outlook}**  ·  🟢 {n_buy_w} BUY  ·  🔴 {n_sell_w} SELL  ·  ⚪ {7-n_buy_w-n_sell_w} HOLD")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Signals",
    "🎯 Predicted vs Actual",
    "📊 Model Performance",
    "📜 Signal History",
])

# ── TAB 1: Price & Signals ─────────────────────────────────────────────────────
with tab1:
    setup_dark_fig()
    n_show  = min(lookback_days, len(te_df))
    te_show = te_df.iloc[-n_show:]
    pr_show = ens_proba[-n_show:]
    sg_show = signals[-n_show:]
    b_show  = sg_show ==  1
    s_show  = sg_show == -1
    dates   = te_show.index
    close   = te_show['Close'].values
    high    = te_show['High'].values
    low     = te_show['Low'].values
    regime  = te_show.get('Regime', pd.Series(np.ones(len(te_show)), index=te_show.index)).values

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[4.5,1.2,1.0,1.0], hspace=0.04)

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#161B22')
    for i in range(len(dates)-1):
        ax1.axvspan(dates[i], dates[i+1], alpha=1,
            color='#1C2A1C' if regime[i]==1 else '#2A1C1C', linewidth=0, zorder=0)
    ax1.plot(dates, close, color=C_BLUE, lw=1.4, label='SOL Close', zorder=4)
    if 'SMA20' in te_show.columns:
        ax1.plot(dates, te_show['SMA20'].values, color=C_GOLD, lw=0.9, ls='--', alpha=0.8, label='SMA 20')
    if 'SMA50' in te_show.columns:
        ax1.plot(dates, te_show['SMA50'].values, color='#A371F7', lw=0.9, ls='-.', alpha=0.8, label='SMA 50')
    if b_show.sum() > 0:
        ax1.scatter(dates[b_show], low[b_show]*0.955,
            marker='^', s=100, color=C_UP, edgecolors='#196127', lw=0.6, zorder=6,
            label=f'BUY ({b_show.sum()})')
    if s_show.sum() > 0:
        ax1.scatter(dates[s_show], high[s_show]*1.045,
            marker='v', s=100, color=C_DOWN, edgecolors='#8B0000', lw=0.6, zorder=6,
            label=f'SELL ({s_show.sum()})')
    # Tomorrow arrow
    next_dt = dates[-1] + pd.Timedelta(days=1)
    tgt     = float(close[-1]) * (1 + (float(ens_proba[-1]) - 0.5) * 0.08)
    arr_c   = C_UP if ens_proba[-1] > 0.5 else C_DOWN
    ax1.annotate('', xy=(next_dt, tgt), xytext=(dates[-1], float(close[-1])),
        arrowprops=dict(arrowstyle='-|>', color=arr_c, lw=2.5, mutation_scale=16))
    ax1.scatter([next_dt], [tgt], marker='*', s=220, color=C_GOLD, zorder=9)
    ax1.annotate(f'  TOMORROW\n  ${tgt:.2f}\n  {last_signal}',
        (next_dt, tgt), fontsize=8.5, color=arr_c, fontweight='bold',
        xytext=(8,0), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.35', fc='#161B22', ec='#30363D', alpha=0.95))
    bull_p = mpatches.Patch(color='#1C2A1C', label='Bull Regime')
    bear_p = mpatches.Patch(color='#2A1C1C', label='Bear Regime')
    h, l2  = ax1.get_legend_handles_labels()
    ax1.legend(h+[bull_p,bear_p], l2+['Bull','Bear'], loc='upper left', ncol=4, framealpha=0.85)
    ax1.set_ylabel('Price (USD)'); ax1.xaxis.set_ticklabels([])
    ax1.spines[['top','right']].set_visible(False); ax1.grid(axis='y', alpha=0.2)
    ax1.set_title(f'SOL/USD — Buy/Sell Signals  |  Ensemble  |  Acc: {ens_acc*100:.1f}%  |  Last {n_show} days',
        color=C_WHITE, fontsize=11, pad=6)

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

    ax3 = fig.add_subplot(gs[2], sharex=ax1); ax3.set_facecolor('#161B22')
    if 'RSI' in te_show.columns:
        rsi = te_show['RSI'].values
        ax3.plot(dates, rsi, color='#D29922', lw=0.9)
        ax3.fill_between(dates, 70, rsi, where=(rsi>70), alpha=0.25, color=C_DOWN)
        ax3.fill_between(dates, 30, rsi, where=(rsi<30), alpha=0.25, color=C_UP)
        ax3.axhline(70, color=C_DOWN, ls='--', lw=0.8, alpha=0.7)
        ax3.axhline(30, color=C_UP,   ls='--', lw=0.8, alpha=0.7)
    ax3.set_ylabel('RSI'); ax3.set_ylim(10,90)
    ax3.xaxis.set_ticklabels([])
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y', alpha=0.2)

    ax4 = fig.add_subplot(gs[3], sharex=ax1); ax4.set_facecolor('#161B22')
    if 'MACD' in te_show.columns and 'MACD_sig' in te_show.columns:
        macd = te_show['MACD'].values; msig = te_show['MACD_sig'].values
        mh   = macd - msig
        ax4.plot(dates, macd, color=C_BLUE, lw=1.0, label='MACD')
        ax4.plot(dates, msig, color='#F78166', lw=1.0, ls='--', label='Signal')
        ax4.bar(dates, mh, color=np.where(mh>=0, C_UP, C_DOWN), width=1.0, alpha=0.55)
        ax4.axhline(0, color=C_DIM, lw=0.7)
        ax4.legend(loc='upper left', ncol=2)
    ax4.set_ylabel('MACD'); ax4.set_xlabel('Date')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax4.spines[['top','right']].set_visible(False); ax4.grid(axis='y', alpha=0.2)

    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── TAB 2: Predicted vs Actual ─────────────────────────────────────────────────
with tab2:
    setup_dark_fig()

    # Compute metrics safely
    try:
        rmse     = float(np.sqrt(mean_squared_error(y_price_te, price_pred)))
        mae_val  = float(mean_absolute_error(y_price_te, price_pred))
        mape_val = float(np.mean(np.abs((y_price_te - price_pred) / (y_price_te + 1e-9))) * 100)
        dir_acc  = float(accuracy_score(y_te, ens_pred))
    except Exception:
        rmse = mae_val = mape_val = 0.0
        dir_acc = float(ens_acc)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",          f"${rmse:.2f}")
    m2.metric("MAE",           f"${mae_val:.2f}")
    m3.metric("MAPE",          f"{mape_val:.2f}%")
    m4.metric("Direction Acc", f"{dir_acc*100:.1f}%")

    n_show2 = min(lookback_days, len(te_df))
    d2  = te_df.index[-n_show2:]
    c2  = te_df['Close'].values[-n_show2:]
    pp2 = price_pred[-n_show2:]
    ep2 = ens_pred[-n_show2:]
    yt2 = y_te[-n_show2:]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios':[4, 1.8, 1.5], 'hspace':0.04})
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle('Predicted vs Actual Price — Test Set', color=C_WHITE, fontsize=12, y=1.01)

    ax = axes[0]; ax.set_facecolor('#161B22')
    ax.plot(d2, c2,  color=C_BLUE, lw=1.5, label='Actual Close', zorder=4)
    ax.plot(d2, pp2, color=C_GOLD, lw=1.2, ls='--', alpha=0.9, label='Predicted (Ridge)', zorder=3)
    ax.fill_between(d2, c2, pp2, where=(pp2>=c2), alpha=0.10, color=C_UP)
    ax.fill_between(d2, c2, pp2, where=(pp2< c2), alpha=0.10, color=C_DOWN)
    ax.text(0.99, 0.97, f'RMSE=${rmse:.2f}  MAE=${mae_val:.2f}  MAPE={mape_val:.2f}%',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=C_GOLD,
        bbox=dict(fc='#161B22', ec='#30363D', boxstyle='round,pad=0.4'))
    ax.set_ylabel('Price (USD)'); ax.legend(loc='upper left')
    ax.xaxis.set_ticklabels([])
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.2)

    ax2 = axes[1]; ax2.set_facecolor('#161B22')
    err = pp2 - c2
    ax2.fill_between(d2, err, 0, where=(err>=0), color=C_UP,   alpha=0.55, label='Predicted too high')
    ax2.fill_between(d2, err, 0, where=(err< 0), color=C_DOWN, alpha=0.55, label='Predicted too low')
    ax2.axhline(0, color=C_DIM, lw=0.8, ls='--')
    ax2.set_ylabel('Error (USD)'); ax2.legend(loc='upper left', ncol=2)
    ax2.xaxis.set_ticklabels([])
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y', alpha=0.2)

    ax3 = axes[2]; ax3.set_facecolor('#161B22')
    try:
        correct = (ep2 == yt2)
        ax3.bar(d2, np.where(correct, 1, -1),
            color=np.where(correct, C_UP, C_DOWN), width=1.2, alpha=0.8)
    except Exception:
        ax3.text(0.5, 0.5, 'Direction chart unavailable', transform=ax3.transAxes,
            ha='center', color=C_DIM)
    ax3.axhline(0, color=C_DIM, lw=0.6)
    ax3.set_yticks([-1,0,1]); ax3.set_yticklabels(['Wrong','','Correct'], fontsize=8)
    ax3.text(0.99, 0.95, f'Direction Acc = {dir_acc*100:.1f}%',
        transform=ax3.transAxes, ha='right', va='top', fontsize=9, color=C_UP,
        bbox=dict(fc='#161B22', ec='#30363D', boxstyle='round,pad=0.4'))
    ax3.set_xlabel('Date')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y', alpha=0.2)

    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── TAB 3: Model Performance ───────────────────────────────────────────────────
with tab3:
    setup_dark_fig()

    if not model_data:
        st.warning("No model data available yet.")
    else:
        names = list(model_data.keys()) + ['Ensemble']
        accs  = [model_data[k]['acc']*100 for k in model_data] + [ens_acc*100]
        f1s   = [model_data[k]['f1']      for k in model_data] + [results['ensemble_f1']]
        aucs  = [model_data[k]['auc']     for k in model_data] + [results['ensemble_auc']]
        bc    = [C_DOWN if a<60 else C_GOLD if a<65 else C_UP for a in accs]
        x     = np.arange(len(names))

        n_cols = min(len(model_data), 3)
        fig    = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#0D1117')
        gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
        fig.suptitle('Model Performance Summary', color=C_WHITE, fontsize=13)

        # Accuracy bars
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

        # ROC curves
        ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor('#161B22')
        ax2.plot([0,1],[0,1],'--',color=C_DIM,lw=1,label='Random')
        try:
            for k, dm in model_data.items():
                if len(np.unique(y_te)) > 1:
                    fpr, tpr, _ = roc_curve(y_te, dm['proba'])
                    ax2.plot(fpr, tpr, lw=1.3, label=f'{k}  {dm["auc"]:.3f}')
            if len(np.unique(y_te)) > 1:
                fpr, tpr, _ = roc_curve(y_te, ens_proba)
                ax2.plot(fpr, tpr, color=C_UP, lw=2.2, label=f'Ensemble  {results["ensemble_auc"]:.3f}')
        except Exception:
            pass
        ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
        ax2.set_title('ROC Curves', color=C_WHITE)
        ax2.legend(loc='lower right', fontsize=7)
        ax2.set_xlim(0,1); ax2.set_ylim(0,1)
        ax2.spines[['top','right']].set_visible(False)

        # F1 bars
        ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor('#161B22')
        bars3 = ax3.bar(x, f1s, color=bc, alpha=0.88, edgecolor='#0D1117', width=0.65)
        ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=8)
        ax3.set_ylim(0,1); ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score', color=C_WHITE)
        for bar, v in zip(bars3, f1s):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', fontweight='bold', fontsize=8, color=C_WHITE)
        ax3.spines[['top','right']].set_visible(False)

        # Confusion matrices
        for col_idx, (nm, dm) in enumerate(list(model_data.items())[:3]):
            ax_cm = fig.add_subplot(gs[1, col_idx]); ax_cm.set_facecolor('#161B22')
            try:
                cm = confusion_matrix(y_te, dm['pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['DOWN','UP'], yticklabels=['DOWN','UP'],
                    linewidths=0.5, annot_kws={'size':13,'weight':'bold'}, cbar=False)
            except Exception:
                ax_cm.text(0.5, 0.5, 'N/A', transform=ax_cm.transAxes, ha='center', color=C_DIM)
            ax_cm.set_title(f'{nm}  Acc={dm["acc"]*100:.1f}%', color=C_WHITE)
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')

        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Rolling accuracy chart
        setup_dark_fig()
        try:
            correct_arr = (ens_pred == y_te).astype(int)
            roll_acc    = pd.Series(correct_arr).rolling(30, min_periods=5).mean() * 100

            fig2, ax = plt.subplots(figsize=(14, 4))
            fig2.patch.set_facecolor('#0D1117'); ax.set_facecolor('#161B22')
            ax.plot(te_df.index, roll_acc, color=C_BLUE, lw=1.5)
            ax.fill_between(te_df.index, 65, roll_acc, where=(roll_acc>=65), alpha=0.18, color=C_UP)
            ax.axhline(ens_acc*100, color=C_GOLD, ls='--', lw=1.5, label=f'Overall {ens_acc*100:.1f}%')
            ax.axhline(65, color=C_UP,  ls=':', lw=1.5, label='65% target')
            ax.axhline(50, color=C_DIM, ls=':', lw=1.0, label='Random 50%')
            ax.set_ylim(25, 95); ax.legend(fontsize=8)
            ax.set_title('30-Day Rolling Accuracy', color=C_WHITE)
            ax.set_xlabel('Date'); ax.set_ylabel('Accuracy (%)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.2)
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        except Exception:
            st.info("Rolling accuracy chart unavailable.")

# ── TAB 4: Signal History ──────────────────────────────────────────────────────
with tab4:
    if sig_history is not None and not sig_history.empty:
        st.subheader("📋 All Buy/Sell Signals — Test Period")
        st.caption("Only signals where model confidence ≥60% are shown")

        # Style using map (not deprecated applymap)
        def color_signal(val):
            if 'BUY'  in str(val): return 'color: #3FB950; font-weight: bold'
            if 'SELL' in str(val): return 'color: #F85149; font-weight: bold'
            return 'color: #6E7681'

        try:
            styled = sig_history.style.map(color_signal, subset=['Signal'])
        except Exception:
            try:
                styled = sig_history.style.applymap(color_signal, subset=['Signal'])
            except Exception:
                styled = sig_history

        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        s1, s2, s3, s4 = st.columns(4)
        n_buy_h  = (sig_history['Signal'] == '🟢 BUY').sum()
        n_sell_h = (sig_history['Signal'] == '🔴 SELL').sum()
        s1.metric("Total Signals", len(sig_history))
        s2.metric("BUY Signals",   int(n_buy_h))
        s3.metric("SELL Signals",  int(n_sell_h))
        try:
            avg_conf = sig_history['Confidence'].str.replace('%','').astype(float).mean()
            s4.metric("Avg Confidence", f"{avg_conf:.1f}%")
        except Exception:
            s4.metric("Avg Confidence", "—")
    else:
        st.info("No filtered signals yet. Try lowering the confidence threshold in the sidebar.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#6E7681;font-size:0.8rem'>"
    "⚠️ For research and educational purposes only · Not financial advice · "
    "Data: CoinGecko API · Models retrain daily"
    "</div>",
    unsafe_allow_html=True,
)
