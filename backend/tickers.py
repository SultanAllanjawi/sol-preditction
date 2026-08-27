"""
tickers.py — canonical ticker universe shared by every router.
Mirrors the categories the old Streamlit app exposed (Crypto / US stocks / DFM).
"""

CRYPTO = {
    "SOL-USD":  "Solana",
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "ADA-USD":  "Cardano",
    "DOGE-USD": "Dogecoin",
    "BNB-USD":  "BNB",
    "AVAX-USD": "Avalanche",
    "XRP-USD":  "XRP",
}

US_STOCKS = {
    "AAPL":  "Apple",
    "TSLA":  "Tesla",
    "NVDA":  "NVIDIA",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon",
    "GOOGL": "Google",
}

DFM = {
    "EMAAR.DFM": "Emaar Properties",
    "ENBD.DFM":  "Emirates NBD",
    "DIB.DFM":   "Dubai Islamic Bank",
    "DU.DFM":    "du (EITC)",
    "DEWA.DFM":  "DEWA",
    "SALIK.DFM": "Salik",
    "MASQ.DFM":  "Mashreq",
}

ALL_TICKERS = {**CRYPTO, **US_STOCKS, **DFM}

CATEGORY_OF = (
    {t: "crypto" for t in CRYPTO}
    | {t: "us_stock" for t in US_STOCKS}
    | {t: "dfm" for t in DFM}
)

# Tickers the retrain scheduler keeps warm at all times (small, fast set).
# Everything else in ALL_TICKERS still works — first request just trains on demand.
DEFAULT_TRACKED = ["SOL-USD", "BTC-USD", "ETH-USD", "AAPL", "EMAAR.DFM", "ENBD.DFM"]


def ticker_list() -> list[dict]:
    return [
        {"ticker": t, "name": name, "category": CATEGORY_OF[t]}
        for t, name in ALL_TICKERS.items()
    ]


def normalize(ticker: str) -> str:
    return ticker.upper().strip()


def is_known(ticker: str) -> bool:
    return normalize(ticker) in ALL_TICKERS
