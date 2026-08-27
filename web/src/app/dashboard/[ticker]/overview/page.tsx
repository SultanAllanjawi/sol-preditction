"use client";

import { use } from "react";
import { usePredict } from "@/hooks/use-predict";
import { useLivePrice, useActiveSignal } from "@/hooks/use-market";
import { tradingViewSymbol } from "@/lib/tradingview";
import { SignalBadge } from "@/components/signal-badge";
import { LoadingPanel, ErrorPanel } from "@/components/state-views";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

function Divider() {
  return <div className="hidden h-10 w-px bg-border sm:block" />;
}

function SignalItem({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-1 flex-col items-center gap-1 px-2 py-3 text-center">
      <div className="text-[0.68rem] font-semibold uppercase tracking-[0.05em] text-muted-foreground">{label}</div>
      <div className="font-mono text-sm font-bold">{children}</div>
    </div>
  );
}

export default function LiveChartPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);

  const predict = usePredict(ticker);
  const live = useLivePrice(ticker);
  const active = useActiveSignal(ticker);
  const symbol = tradingViewSymbol(ticker);

  return (
    <div className="flex flex-col gap-4">
      <div className="surface-panel overflow-hidden rounded-2xl">
        <iframe
          key={symbol}
          title="TradingView chart"
          src={`https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=D&theme=dark&style=1&locale=en&hide_top_toolbar=0&withdateranges=1`}
          className="h-[520px] w-full border-0"
        />

        <div className="flex flex-col divide-y divide-border sm:flex-row sm:divide-x sm:divide-y-0">
          <SignalItem label="Live Price">
            {live.data ? `$${live.data.price.toLocaleString(undefined, { maximumFractionDigits: 4 })}` : "—"}
          </SignalItem>
          <Divider />
          <SignalItem label="Next Session Signal">
            {predict.data ? <SignalBadge signal={predict.data.last_signal} size="sm" /> : "—"}
          </SignalItem>
          <Divider />
          <SignalItem label="Take Profit">
            {active.data?.tp ? `$${active.data.tp.toFixed(4)}` : "—"}
          </SignalItem>
          <Divider />
          <SignalItem label="Stop Loss">
            {active.data?.sl ? `$${active.data.sl.toFixed(4)}` : "—"}
          </SignalItem>
          <Divider />
          <SignalItem label="Risk-Reward">
            {active.data?.tp && active.data?.sl && active.data?.entry
              ? (Math.abs(active.data.tp - active.data.entry) / Math.max(Math.abs(active.data.entry - active.data.sl), 0.0001)).toFixed(2)
              : "—"}
          </SignalItem>
          <Divider />
          <SignalItem label="Model Accuracy">
            {predict.data ? `${(predict.data.ensemble.accuracy * 100).toFixed(1)}%` : "—"}
          </SignalItem>
        </div>
      </div>

      <div className="surface-panel rounded-2xl p-4">
        <div className="mb-3 text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">
          Recent Signals — Last 20
        </div>
        {predict.isLoading && <LoadingPanel label="Loading signals…" rows={3} />}
        {predict.isError && <ErrorPanel message={(predict.error as Error).message} />}
        {predict.data && (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Date</TableHead>
                <TableHead>Signal</TableHead>
                <TableHead className="text-right">Price</TableHead>
                <TableHead className="text-right">Confidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predict.data.signal_history.slice(0, 20).map((r, i) => (
                <TableRow key={i}>
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {r.date ? new Date(r.date).toLocaleString() : "—"}
                  </TableCell>
                  <TableCell>
                    <SignalBadge signal={r.signal} size="sm" />
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {r.price !== null ? `$${r.price.toFixed(4)}` : "—"}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.confidence.toFixed(1)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
