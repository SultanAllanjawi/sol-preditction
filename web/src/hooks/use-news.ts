"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useNews(ticker: string) {
  return useQuery({
    queryKey: ["news", ticker],
    queryFn: () => api.news(ticker),
    refetchInterval: 5 * 60_000,
    enabled: Boolean(ticker),
  });
}

export function useCalendar() {
  return useQuery({
    queryKey: ["calendar"],
    queryFn: api.calendar,
    refetchInterval: 15 * 60_000,
  });
}
