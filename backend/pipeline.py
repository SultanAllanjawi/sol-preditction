"""
pipeline.py — data fetch -> feature engineering -> ensemble training, as one call.
Mirrors legacy-streamlit/app.py's `load_and_train()` so accuracy/behavior is unchanged.
"""
from __future__ import annotations

import pandas as pd

from data_manager import DataManager, is_crypto
from feature_engine import build_features
from model_engine import ModelEngine


def get_sentiment_score(ticker: str) -> float:
    try:
        news = DataManager.get_news_sentiment(ticker, limit=10)
        if news:
            return round(sum(n.get("score", 0) for n in news) / len(news), 3)
    except Exception:
        pass
    return 0.0


def load_and_train(ticker: str, prefer_hourly: bool = True) -> dict:
    """Returns {df_raw, df_feat, results, sentiment_score, trained_at}."""
    dm = DataManager(ticker)
    sentiment_score = get_sentiment_score(ticker)

    df_raw = dm.get_data(prefer_hourly=prefer_hourly and is_crypto(ticker))
    if df_raw is None or len(df_raw) < 100:
        raise RuntimeError(
            f"Could not load enough data for {ticker} (got "
            f"{0 if df_raw is None else len(df_raw)} rows, need >= 100)."
        )

    df_feat = build_features(df_raw, sentiment_score=sentiment_score)
    engine = ModelEngine(df_feat)
    results = engine.train(
        verbose=False,
        sentiment_score=sentiment_score,
        is_crypto=is_crypto(ticker),
    )
    return {
        "ticker": ticker,
        "df_raw": df_raw,
        "df_feat": df_feat,
        "results": results,
        "sentiment_score": sentiment_score,
        "rows": len(df_raw),
    }
