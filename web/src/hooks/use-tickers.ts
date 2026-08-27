"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useTickers() {
  return useQuery({
    queryKey: ["tickers"],
    queryFn: api.tickers,
    staleTime: Infinity,
  });
}
