"use client";

import { useLivePrice } from "@/hooks/use-market";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

export function LivePriceBadge({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useLivePrice(ticker);

  if (isLoading) return <Skeleton className="h-7 w-28" />;
  if (isError || !data) return <span className="text-sm text-muted-foreground">Price unavailable</span>;

  const change = data.stats_24h?.price_change_pct;
  const up = (change ?? 0) >= 0;

  return (
    <div className="flex items-baseline gap-2">
      <span className="font-mono text-2xl font-bold tabular-nums">
        ${data.price < 10 ? data.price.toFixed(4) : data.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </span>
      {change !== undefined && (
        <span className={cn("text-sm font-medium tabular-nums", up ? "text-signal-buy" : "text-signal-sell")}>
          {up ? "+" : ""}
          {change.toFixed(2)}% (24h)
        </span>
      )}
    </div>
  );
}
