FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# On a host with a persistent volume mounted here, cached OHLCV/portfolio/signal
# JSON survives redeploys; DATA_DIR in data_manager.py/persistence.py is
# relative ("data"), so we just need the working dir's data/ to be the mount.
# On a free-tier host with no volume, this is plain container storage — it
# resets on every redeploy/restart, which just means a cold refetch, not a crash.
RUN mkdir -p /app/data

EXPOSE 8000
# Shell form (not exec form) so ${PORT} expands — Render/Railway/most PaaS hosts
# inject PORT and expect the app to bind to it; 8000 is the local-dev fallback.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
