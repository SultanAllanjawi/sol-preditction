"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type NewTradeInput } from "@/lib/api";

export function usePortfolio() {
  return useQuery({
    queryKey: ["portfolio"],
    queryFn: api.portfolio.list,
    refetchInterval: 20_000,
  });
}

export function usePortfolioMutations() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ["portfolio"] });

  const add = useMutation({
    mutationFn: (trade: NewTradeInput) => api.portfolio.add(trade),
    onSuccess: invalidate,
  });
  const close = useMutation({
    mutationFn: ({ id, exitPrice }: { id: number; exitPrice: number }) => api.portfolio.close(id, exitPrice),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (id: number) => api.portfolio.remove(id),
    onSuccess: invalidate,
  });

  return { add, close, remove };
}
