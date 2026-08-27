"use client";

import Link from "next/link";
import { useScanner } from "@/hooks/use-scanner";
import { SignalBadge } from "@/components/signal-badge";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

const CATEGORY_LABEL = { crypto: "Crypto", us_stock: "US Equity", dfm: "DFM" } as const;

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

  return (
    <div className="surface-panel rounded-xl p-2">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead>Ticker</TableHead>
            <TableHead>Category</TableHead>
            <TableHead>Signal</TableHead>
            <TableHead className="text-right">Confidence</TableHead>
            <TableHead className="text-right">Ensemble Accuracy</TableHead>
            <TableHead className="text-right">Last Trained</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.ticker}>
              <TableCell>
                <Link href={`/dashboard/${r.ticker}/overview`} className="font-medium hover:underline">
                  {r.ticker}
                </Link>
                <div className="text-xs text-muted-foreground">{r.name}</div>
              </TableCell>
              <TableCell>
                <Badge variant="outline">{CATEGORY_LABEL[r.category]}</Badge>
              </TableCell>
              <TableCell>
                <SignalBadge signal={r.signal} size="sm" />
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">{r.confidence.toFixed(1)}%</TableCell>
              <TableCell className="text-right font-mono tabular-nums">{r.ensemble_accuracy.toFixed(1)}%</TableCell>
              <TableCell className="text-right text-xs text-muted-foreground">
                {r.trained_at ? new Date(r.trained_at).toLocaleTimeString() : "—"}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
