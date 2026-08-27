import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import cache as model_cache
from routers import market, news, portfolio, predict, scanner, signals

app = FastAPI(title="Signal Engine API", version="1.0.0")

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(market.router)
app.include_router(news.router)
app.include_router(portfolio.router)
app.include_router(signals.router)
app.include_router(scanner.router)


@app.on_event("startup")
def _startup():
    model_cache.start_scheduler()


@app.on_event("shutdown")
def _shutdown():
    model_cache.stop_scheduler()


@app.get("/health")
def health():
    return {"status": "ok"}
