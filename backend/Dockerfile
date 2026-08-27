FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway mounts a persistent volume here so cached OHLCV/portfolio/signal
# JSON survives redeploys; DATA_DIR in data_manager.py/persistence.py is
# relative ("data"), so we just need the working dir's data/ to be the mount.
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
