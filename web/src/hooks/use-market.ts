"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useLivePrice(ticker: string) {
  return useQuery({
    queryKey: ["live", ticker],
    queryFn: () => api.live(ticker),
    refetchInterval: 15_000,
    enabled: Boolean(ticker),
  });
}

export function useFearGreed() {
  return useQuery({
    queryKey: ["fear-greed"],
    queryFn: api.fearGreed,
    refetchInterval: 5 * 60_000,
  });
}

export function useActiveSignal(ticker: string) {
  return useQuery({
    queryKey: ["active-signal", ticker],
    queryFn: () => api.activeSignal(ticker),
    refetchInterval: 30_000,
    enabled: Boolean(ticker),
  });
}
