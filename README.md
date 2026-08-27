# Signal Engine

AI-driven trading signals for crypto, US equities, and Dubai Financial Market stocks — a
real 4-model ensemble (RNN + Random Forest + Gradient Boosting + XGBoost) trained on live
market data, served through a FastAPI backend to a Next.js dashboard.

This replaces the original Streamlit app (kept for reference in [`legacy-streamlit/`](legacy-streamlit))
with a two-service architecture built for a Vercel + Render deploy:

```
/backend   FastAPI service — data fetch, feature engineering, the model ensemble, backtesting,
           portfolio/signal persistence. Deploys to Render's free web-service tier (or any host
           that runs a Dockerfile).
/web       Next.js 14 dashboard — deploys to Vercel. Talks to /backend over HTTPS/JSON.
```

**Live**: frontend at https://web-seven-brown-10.vercel.app. The backend currently runs
**locally** (see below) rather than on Render — Render's free-tier CPU is throttled to a
fraction of a core, which made a cold ticker take minutes to train instead of seconds.
Running it on real hardware fixed that, at the cost of the backend only being reachable
while it's actually running.

**Backend is running on this machine, exposed via a Cloudflare quick tunnel** (no account
needed): `cloudflared tunnel --url http://127.0.0.1:8010`. Two things to know:
- The tunnel URL is random and changes every time `cloudflared` restarts — if it drops,
  regenerate it and update Vercel's `NEXT_PUBLIC_API_URL` (`vercel env rm/add` + `vercel deploy --prod`).
- The live site only works while both the local backend (`uvicorn`) and the tunnel are running.

`render.yaml` is still in the repo if you want to move the backend back to Render (or a paid
instance type on it) later — nothing about the app changes, only where it's hosted. See
[Deploying](#deploying).

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

**Backend → Render**

Deployed from [`render.yaml`](render.yaml) at the repo root, which points Render at `backend/`
with the Docker runtime and a free-tier plan. Since the repo is public, Render can build
straight from the GitHub URL without installing its GitHub App.

1. Render dashboard → New → Blueprint → paste this repo's URL → Apply. (Or, if you'd rather
   not use the Blueprint, New → Web Service → Public Git Repository, set root directory to
   `backend/`, runtime Docker, plan Free.)
2. Set env vars `CORS_ORIGINS` (your Vercel domain(s), comma-separated) and `TRAIN_TTL_MINUTES`
   (default `180`).
3. Note the deployed URL (e.g. `https://your-app.onrender.com`).

Free tier has no persistent disk (`backend/data/` resets on restart) and spins down after
~15 min idle (30-60s cold start on the next request). To remove that tradeoff, switch the
service to a paid instance type with a disk attached at `/app/data`, or move to a host like
Railway/Fly.io that includes a persistent volume — the Dockerfile doesn't change either way.

**Frontend → Vercel**
1. New Vercel project → import this repo → set **Root Directory** to `web/`.
2. Set env var `NEXT_PUBLIC_API_URL` to the Render backend URL from above (safe to expose —
   it's a public API endpoint, which is why it's a `NEXT_PUBLIC_` var).
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
