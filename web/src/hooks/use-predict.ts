"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function usePredict(ticker: string) {
  return useQuery({
    queryKey: ["predict", ticker],
    queryFn: () => api.predict(ticker),
    refetchInterval: 60_000,
    enabled: Boolean(ticker),
  });
}

export function useOhlcv(ticker: string, limit = 500) {
  return useQuery({
    queryKey: ["ohlcv", ticker, limit],
    queryFn: () => api.ohlcv(ticker, limit),
    refetchInterval: 60_000,
    enabled: Boolean(ticker),
  });
}

export function useBacktest(ticker: string, startingCapital: number, tradeSizePct: number) {
  return useQuery({
    queryKey: ["backtest", ticker, startingCapital, tradeSizePct],
    queryFn: () => api.backtest(ticker, startingCapital, tradeSizePct),
    enabled: Boolean(ticker),
  });
}
