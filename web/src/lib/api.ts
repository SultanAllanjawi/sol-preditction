const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      // ignore, use statusText
    }
    throw new Error(detail || `Request to ${path} failed with ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Types ──────────────────────────────────────────────────────────

export type Category = "crypto" | "us_stock" | "dfm";

export interface TickerInfo {
  ticker: string;
  name: string;
  category: Category;
}

export interface ModelMetrics {
  accuracy: number;
  f1: number;
  auc: number;
}

export interface SignalHistoryRow {
  date: string | null;
  price: number | null;
  signal: "BUY" | "SELL" | "HOLD";
  prob_up: number;
  confidence: number;
}

export interface PricePrediction {
  rmse: number;
  mae: number;
  dates: string[];
  predicted: number[];
  actual: number[];
}

export interface PredictResponse {
  ticker: string;
  trained_at: string | null;
  rows_used: number;
  sentiment_score: number;
  last_signal: "BUY" | "SELL" | "HOLD";
  last_confidence: number;
  last_prob_up: number;
  thresholds: { high: number; low: number };
  ensemble: {
    accuracy: number;
    filtered_accuracy: number;
    f1: number;
    auc: number;
    models_used: string[];
    models_excluded: string[];
    best_model: string;
  };
  models: Record<string, ModelMetrics>;
  n_signals: number;
  signal_history: SignalHistoryRow[];
  price_prediction: PricePrediction;
}

export interface OhlcvResponse {
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
  sma20: (number | null)[];
  sma50: (number | null)[];
  rsi: (number | null)[];
  macd: (number | null)[];
  macd_sig: (number | null)[];
  bb_u: (number | null)[];
  bb_l: (number | null)[];
  signal: number[];
}

export interface BacktestTrade {
  date: string;
  signal: "BUY" | "SELL";
  entry: number;
  exit: number;
  outcome: "TP" | "SL";
  pnl: number;
  capital: number;
}

export interface BacktestResponse {
  starting_capital: number;
  final_capital: number;
  total_return_pct: number;
  win_rate_pct: number;
  wins: number;
  losses: number;
  total_trades: number;
  max_drawdown_pct: number;
  equity_curve: number[];
  trades: BacktestTrade[];
}

export interface LivePrice {
  ticker: string;
  price: number;
  is_crypto: boolean;
  stats_24h?: {
    price_change_pct: number;
    volume_base: number;
    volume_quote: number;
    trade_count: number;
    high: number;
    low: number;
  };
  market_cap?: Record<string, unknown>;
}

export interface FearGreed {
  value: number;
  classification: string;
  history: number[];
}

export interface NewsItem {
  title?: string;
  score?: number;
  [key: string]: unknown;
}

export interface CalendarEvent {
  date: string;
  time: string;
  currency: string;
  event: string;
  impact: string;
  impact_raw: string;
  forecast: string;
  previous: string;
  actual: string;
}

export interface ScannerRow {
  ticker: string;
  name: string;
  category: Category;
  signal: "BUY" | "SELL" | "HOLD";
  confidence: number;
  ensemble_accuracy: number;
  trained_at: string | null;
}

export interface Trade {
  id: number;
  date: string;
  ticker: string;
  side: "BUY" | "SELL";
  entry: number;
  size: number;
  tp: number | null;
  sl: number | null;
  status: "Open" | "Closed";
  exit: number | null;
  pnl: number | null;
  note: string;
  live_price?: number;
  pnl_pct?: number;
}

export interface PortfolioResponse {
  trades: Trade[];
  summary: {
    total_trades: number;
    open_positions: number;
    total_pnl: number;
    roi_pct: number;
  };
}

export interface NewTradeInput {
  ticker: string;
  side: "BUY" | "SELL";
  entry: number;
  size: number;
  tp?: number | null;
  sl?: number | null;
  note?: string;
}

export interface ActiveSignal {
  ticker: string;
  live_price: number | null;
  status: string;
  message: string;
  pnl_pct: number;
  hours_old?: number;
  remaining_h?: number;
  valid_until?: boolean;
  signal?: string;
  entry?: number;
  tp?: number | null;
  sl?: number | null;
}

// ── API surface ────────────────────────────────────────────────────

export const api = {
  tickers: () => apiFetch<TickerInfo[]>("/api/tickers"),
  predict: (ticker: string, refresh = false) =>
    apiFetch<PredictResponse>(`/api/predict/${encodeURIComponent(ticker)}${refresh ? "?refresh=true" : ""}`),
  ohlcv: (ticker: string, limit = 500) =>
    apiFetch<OhlcvResponse>(`/api/ohlcv/${encodeURIComponent(ticker)}?limit=${limit}`),
  backtest: (ticker: string, startingCapital = 1000, tradeSizePct = 100) =>
    apiFetch<BacktestResponse>(
      `/api/backtest/${encodeURIComponent(ticker)}?starting_capital=${startingCapital}&trade_size_pct=${tradeSizePct}`
    ),
  live: (ticker: string) => apiFetch<LivePrice>(`/api/market/live/${encodeURIComponent(ticker)}`),
  fearGreed: () => apiFetch<FearGreed>("/api/market/fear-greed"),
  news: (ticker: string) =>
    apiFetch<{ crypto_news: NewsItem[]; forex_calendar: CalendarEvent[] }>(`/api/news/${encodeURIComponent(ticker)}`),
  calendar: () => apiFetch<{ events: CalendarEvent[] }>("/api/calendar"),
  scanner: () => apiFetch<{ tickers: ScannerRow[] }>("/api/scanner"),
  activeSignal: (ticker: string) => apiFetch<ActiveSignal>(`/api/signals/active/${encodeURIComponent(ticker)}`),
  closedSignals: () => apiFetch<unknown[]>("/api/signals/history"),

  portfolio: {
    list: () => apiFetch<PortfolioResponse>("/api/portfolio"),
    add: (trade: NewTradeInput) =>
      apiFetch<Trade>("/api/portfolio", { method: "POST", body: JSON.stringify(trade) }),
    close: (id: number, exitPrice: number) =>
      apiFetch<Trade>(`/api/portfolio/${id}/close`, {
        method: "POST",
        body: JSON.stringify({ exit_price: exitPrice }),
      }),
    remove: (id: number) => apiFetch<{ deleted: number }>(`/api/portfolio/${id}`, { method: "DELETE" }),
    closed: () => apiFetch<unknown[]>("/api/portfolio/closed"),
  },
};
