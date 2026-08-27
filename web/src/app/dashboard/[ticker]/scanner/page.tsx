"use client";

import Link from "next/link";
import { useScanner } from "@/hooks/use-scanner";
import { SignalBadge } from "@/components/signal-badge";
import { StatTile } from "@/components/stat-tile";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { cn } from "@/lib/utils";

const BORDER_BY_SIGNAL = {
  BUY: "border-l-signal-buy",
  SELL: "border-l-signal-sell",
  HOLD: "border-l-signal-hold",
} as const;

export default function ScannerPage() {
  const scanner = useScanner();

  if (scanner.isLoading) return <LoadingPanel label="Scanning tracked tickers…" />;
  if (scanner.isError) return <ErrorPanel message={(scanner.error as Error).message} />;
  const rows = scanner.data!.tickers;

  if (!rows.length)
    return (
      <EmptyPanel
        title="No tickers warmed up yet"
        hint="Visit a ticker's Overview tab once to train it — the scanner only shows tickers already cached."
      />
    );

  const buy = rows.filter((r) => r.signal === "BUY").length;
  const sell = rows.filter((r) => r.signal === "SELL").length;
  const hold = rows.filter((r) => r.signal === "HOLD").length;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Assets Scanned" value={rows.length} />
        <StatTile label="Buy" value={buy} tone="buy" />
        <StatTile label="Sell" value={sell} tone="sell" />
        <StatTile label="Hold" value={hold} />
      </div>

      <div className="surface-panel h-2 overflow-hidden rounded-full">
        <div className="flex h-full w-full">
          <div className="bg-signal-buy" style={{ width: `${(buy / rows.length) * 100}%` }} />
          <div className="bg-signal-hold" style={{ width: `${(hold / rows.length) * 100}%` }} />
          <div className="bg-signal-sell" style={{ width: `${(sell / rows.length) * 100}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {rows.map((r) => (
          <Link
            key={r.ticker}
            href={`/dashboard/${r.ticker}/overview`}
            className={cn("surface-panel flex flex-col gap-2 rounded-2xl border-l-4 p-4", BORDER_BY_SIGNAL[r.signal])}
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="font-bold">{r.ticker}</div>
                <div className="text-xs text-muted-foreground">{r.name}</div>
              </div>
              <SignalBadge signal={r.signal} size="md" />
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
              <span className="text-muted-foreground">Confidence</span>
              <span className="text-right font-mono tabular-nums">{r.confidence.toFixed(1)}%</span>
              <span className="text-muted-foreground">Accuracy</span>
              <span className="text-right font-mono tabular-nums">{r.ensemble_accuracy.toFixed(1)}%</span>
              <span className="text-muted-foreground">Category</span>
              <span className="text-right capitalize">{r.category.replace("_", " ")}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
