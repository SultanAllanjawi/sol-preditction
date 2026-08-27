# Signal Engine

AI-driven trading signals for crypto, US equities, and Dubai Financial Market stocks — a
real 4-model ensemble (RNN + Random Forest + Gradient Boosting + XGBoost) trained on live
market data, served through a FastAPI backend to a Next.js dashboard.

This replaces the original Streamlit app (kept for reference in [`legacy-streamlit/`](legacy-streamlit))
with a two-service architecture built for a Vercel + Railway deploy:

```
/backend   FastAPI service — data fetch, feature engineering, the model ensemble, backtesting,
           portfolio/signal persistence. Deploys to Railway (or any host that runs a Dockerfile).
/web       Next.js 14 dashboard — deploys to Vercel. Talks to /backend over HTTPS/JSON.
```

Every number on the dashboard is real: live prices from Binance/CoinGecko/Yahoo, models trained
on that live data, and a backtest that walks actual subsequent price action rather than
simulating outcomes.

## Local development

**Backend**

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8010
```

Copy `.env.example` to `.env` if you need to change `CORS_ORIGINS` or `TRAIN_TTL_MINUTES`
(defaults work for local dev against `http://localhost:3000`).

**Frontend**

```bash
cd web
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL to your backend's URL
npm run dev
```

Open the printed localhost URL. The first request for any ticker takes ~10-20s (training the
ensemble live); a background scheduler in the backend keeps `tickers.py`'s `DEFAULT_TRACKED`
list warm so those tickers are instant.

## Deploying

**Backend → Railway**
1. New Railway project → Deploy from GitHub → set the root directory to `backend/`.
2. Railway auto-detects the `Dockerfile`. Add a persistent volume mounted at `/app/data` so
   cached OHLCV data and portfolio/signal history survive redeploys.
3. Set env var `CORS_ORIGINS` to your Vercel domain (comma-separated if you have a preview + prod URL).
4. Note the deployed URL (e.g. `https://your-app.up.railway.app`).

**Frontend → Vercel**
1. New Vercel project → import this repo → set **Root Directory** to `web/`.
2. Set env var `NEXT_PUBLIC_API_URL` to the Railway backend URL from above.
3. Deploy. Vercel auto-detects Next.js — no other config needed.

## What's real vs. what to know

- **Models**: identical to the original engine (`backend/model_engine.py`, unmodified) — a
  2-seed-averaged numpy RNN, RandomForest, GradientBoosting, and XGBoost, weighted into an
  ensemble that excludes any model scoring under 55% accuracy.
- **Backtest** (`backend/backtest.py`): rewritten from the original's random-coin-flip simulation
  (which never looked at real subsequent prices) to a deterministic walk-forward test against
  actual High/Low data for TP/SL touches.
- **Not financial advice.** Signals reflect historical model accuracy, not a guarantee of future
  performance — no model predicts markets perfectly.

## Repo layout

- `backend/` — FastAPI app, routers per concern (`predict`, `market`, `news`, `portfolio`, `signals`, `scanner`), `pipeline.py` (fetch→features→train), `cache.py` (TTL cache + retrain scheduler), `backtest.py`.
- `web/` — Next.js App Router, `src/app/dashboard/[ticker]/*` for the per-ticker tabs, `src/app/portfolio` for the trade tracker, `src/lib/api.ts` for the typed backend client, `src/hooks/*` for TanStack Query hooks.
- `legacy-streamlit/` — the original Streamlit app, kept for reference only.
