# 🔮 SOL/USD Auto-Updating Prediction Dashboard

A Streamlit web app that automatically fetches live Solana prices every day,
retrains ML models, and generates buy/sell signals with 7-day outlooks.

---

## 📁 File Structure

```
solana_app/
├── app.py              ← Main Streamlit dashboard
├── data_manager.py     ← Auto-fetches live prices from CoinGecko
├── feature_engine.py   ← 60+ technical indicator features
├── model_engine.py     ← Trains RNN, LSTM, BiLSTM, GRU, CNN-LSTM, ARIMAX, GB
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── data/               ← Auto-created, stores cached price data
    └── sol_prices.csv  ← Updated automatically every 24h
```

---

## 🚀 Deploy to Streamlit Cloud (Free — 3 steps)

### Step 1 — Push to GitHub
1. Create a free GitHub account at github.com
2. Create a new repository called `sol-prediction`
3. Upload all files from this folder to the repository root

### Step 2 — Connect to Streamlit Cloud
1. Go to **share.streamlit.io** (free)
2. Sign in with GitHub
3. Click **New app**
4. Select your `sol-prediction` repository
5. Main file path: `app.py`
6. Click **Deploy**

### Step 3 — Done ✅
Your app will be live at:
`https://[your-username]-sol-prediction-app.streamlit.app`

It will automatically:
- Refresh data every 24 hours from CoinGecko
- Retrain all models on the latest data
- Update tomorrow's signal and 7-day outlook

---

## 🔄 How Auto-Updating Works

```
Every 24 hours:
┌─────────────────────────────────────────────────────────────┐
│ 1. DataManager fetches latest SOL prices from CoinGecko API │
│ 2. New rows are appended to data/sol_prices.csv             │
│ 3. 60+ features are recalculated on the updated dataset     │
│ 4. All models retrain on the full history                   │
│ 5. New predictions are generated for tomorrow               │
│ 6. Dashboard updates automatically                          │
└─────────────────────────────────────────────────────────────┘
```

The `@st.cache_data(ttl=86400)` decorator in `app.py` controls the 24h refresh.
Change `ttl=3600` for hourly refresh if you want more frequent updates.

---

## 💻 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: place your existing CSV for faster startup
# (put Solana_Historical_Data.csv in the same folder)

# Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

---

## 📊 What the Dashboard Shows

| Section | Content |
|---|---|
| **Top metrics** | Model accuracy, P(UP tomorrow), confidence, signal count |
| **Tomorrow Signal** | BUY / SELL / HOLD with entry, target, stop loss, R/R ratio |
| **7-Day Outlook** | Day-by-day signals for the next week |
| **Price & Signals chart** | Full price chart with BUY/SELL arrows, RSI, MACD |
| **Predicted vs Actual** | Model price prediction accuracy on test set |
| **Model Performance** | Accuracy, ROC curves, F1, confusion matrices |
| **Signal History** | Full log of all historical BUY/SELL signals |

---

## ⚙️ Customisation

### Change confidence threshold
In the sidebar, drag the **Signal Confidence** slider.
Higher = fewer but more reliable signals.

### Change models
Edit `model_engine.py` → add/remove models from `dl_builders` dict.

### Change features
Edit `feature_engine.py` → add indicators, then update `FEATURE_COLS` list.

### Change refresh frequency
In `app.py`, line with `@st.cache_data(ttl=86400)`:
- `ttl=3600`  → refresh every hour
- `ttl=86400` → refresh every 24 hours (default)

---

## ⚠️ Disclaimer

This dashboard is for **educational and research purposes only**.
It does not constitute financial advice. Always manage your own risk.
Past prediction accuracy does not guarantee future results.
Cryptocurrency markets are highly volatile and unpredictable.
