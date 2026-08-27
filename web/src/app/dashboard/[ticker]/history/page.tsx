"use client";

import { use } from "react";
import { useQuery } from "@tanstack/react-query";
import { usePredict } from "@/hooks/use-predict";
import { api } from "@/lib/api";
import { SignalBadge } from "@/components/signal-badge";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ClosedSignal {
  ticker: string;
  signal: string;
  entry: number;
  exit: number;
  result: string;
  pnl_pct: number;
  closed_at: string;
}

function useClosedSignals() {
  return useQuery({ queryKey: ["closed-signals"], queryFn: () => api.closedSignals() as Promise<ClosedSignal[]> });
}

export default function HistoryPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const predict = usePredict(ticker);
  const closed = useClosedSignals();

  if (predict.isLoading) return <LoadingPanel label="Loading signal history…" />;
  if (predict.isError) return <ErrorPanel message={(predict.error as Error).message} />;
  const rows = predict.data!.signal_history;

  return (
    <Tabs defaultValue="all">
      <TabsList>
        <TabsTrigger value="all">📊 All Signals</TabsTrigger>
        <TabsTrigger value="closed">🎯 Closed (TP/SL Hit)</TabsTrigger>
      </TabsList>

      <TabsContent value="all" className="mt-4">
        {!rows.length ? (
          <EmptyPanel title="No signals fired yet" hint="The ensemble hasn't crossed a confidence threshold in this window." />
        ) : (
          <div className="surface-panel rounded-2xl p-2">
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
        )}
      </TabsContent>

      <TabsContent value="closed" className="mt-4">
        {closed.isLoading && <LoadingPanel label="Loading closed signals…" rows={3} />}
        {closed.isError && <ErrorPanel message={(closed.error as Error).message} />}
        {closed.data && closed.data.length === 0 && (
          <EmptyPanel title="No closed signals yet" hint="Signals appear here once they hit take-profit or stop-loss." />
        )}
        {closed.data && closed.data.length > 0 && (
          <div className="surface-panel rounded-2xl p-2">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Closed</TableHead>
                  <TableHead>Asset</TableHead>
                  <TableHead>Signal</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Exit</TableHead>
                  <TableHead className="text-right">P&amp;L %</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {closed.data
                  .filter((c) => c.ticker === ticker)
                  .reverse()
                  .map((c, i) => (
                    <TableRow key={i}>
                      <TableCell className="text-xs text-muted-foreground">{c.closed_at}</TableCell>
                      <TableCell className="font-medium">{c.ticker}</TableCell>
                      <TableCell>
                        <SignalBadge signal={c.signal as "BUY" | "SELL"} size="sm" />
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums">${c.entry.toFixed(4)}</TableCell>
                      <TableCell className="text-right font-mono tabular-nums">${c.exit.toFixed(4)}</TableCell>
                      <TableCell
                        className={`text-right font-mono tabular-nums ${c.pnl_pct >= 0 ? "text-signal-buy" : "text-signal-sell"}`}
                      >
                        {c.pnl_pct >= 0 ? "+" : ""}
                        {c.pnl_pct.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </div>
        )}
      </TabsContent>
    </Tabs>
  );
}
