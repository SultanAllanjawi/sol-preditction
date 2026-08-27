"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useScanner() {
  return useQuery({
    queryKey: ["scanner"],
    queryFn: api.scanner,
    refetchInterval: 60_000,
  });
}
