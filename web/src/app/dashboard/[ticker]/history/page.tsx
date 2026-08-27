"use client";

import { use } from "react";
import { usePredict } from "@/hooks/use-predict";
import { SignalBadge } from "@/components/signal-badge";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function HistoryPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const predict = usePredict(ticker);

  if (predict.isLoading) return <LoadingPanel label="Loading signal history…" />;
  if (predict.isError) return <ErrorPanel message={(predict.error as Error).message} />;
  const rows = predict.data!.signal_history;

  if (!rows.length) return <EmptyPanel title="No signals fired yet" hint="The ensemble hasn't crossed a confidence threshold in this window." />;

  return (
    <div className="surface-panel rounded-xl p-2">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead>Date</TableHead>
            <TableHead>Signal</TableHead>
            <TableHead className="text-right">Price</TableHead>
            <TableHead className="text-right">P(up)</TableHead>
            <TableHead className="text-right">Confidence</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.slice(0, 200).map((r, i) => (
            <TableRow key={`${r.date}-${i}`}>
              <TableCell className="font-mono text-xs text-muted-foreground">
                {r.date ? new Date(r.date).toLocaleString() : "—"}
              </TableCell>
              <TableCell>
                <SignalBadge signal={r.signal} size="sm" />
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {r.price !== null ? `$${r.price.toFixed(4)}` : "—"}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                {(r.prob_up * 100).toFixed(1)}%
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">{r.confidence.toFixed(1)}%</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
