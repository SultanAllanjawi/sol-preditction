// Exact mapping from legacy-streamlit/app.py (TV_SYMBOLS dict) — ticker -> TradingView symbol.
const TV_SYMBOLS: Record<string, string> = {
  "SOL-USD": "BINANCE:SOLUSDT",
  "BTC-USD": "BINANCE:BTCUSDT",
  "ETH-USD": "BINANCE:ETHUSDT",
  "ADA-USD": "BINANCE:ADAUSDT",
  "BNB-USD": "BINANCE:BNBUSDT",
  "XRP-USD": "BINANCE:XRPUSDT",
  "DOGE-USD": "BINANCE:DOGEUSDT",
  "AVAX-USD": "BINANCE:AVAXUSDT",
  "AAPL": "NASDAQ:AAPL",
  "TSLA": "NASDAQ:TSLA",
  "MSFT": "NASDAQ:MSFT",
  "NVDA": "NASDAQ:NVDA",
  "AMZN": "NASDAQ:AMZN",
  "GOOGL": "NASDAQ:GOOGL",
  "EMAAR.DFM": "DFM:EMAAR",
  "ENBD.DFM": "DFM:EMIRATESNBD",
  "DIB.DFM": "DFM:DIB",
  "DU.DFM": "DFM:DU",
  "DEWA.DFM": "DFM:DEWA",
  "SALIK.DFM": "DFM:SALIK",
  "MASQ.DFM": "DFM:MASQ",
};

export function tradingViewSymbol(ticker: string): string {
  if (TV_SYMBOLS[ticker]) return TV_SYMBOLS[ticker];
  if (ticker.endsWith("-USD")) return `BINANCE:${ticker.replace("-USD", "USDT")}`;
  return ticker;
}
