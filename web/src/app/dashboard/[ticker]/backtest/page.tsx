"use client";

import { use, useState } from "react";
import { useBacktest } from "@/hooks/use-predict";
import { StatTile } from "@/components/stat-tile";
import { LineChart } from "@/components/line-chart";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function BacktestPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const [capital, setCapital] = useState(1000);
  const [sizePct, setSizePct] = useState(100);

  const backtest = useBacktest(ticker, capital, sizePct);

  return (
    <div className="flex flex-col gap-4">
      <div className="surface-panel flex flex-wrap items-end gap-4 rounded-xl p-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="capital" className="text-xs text-muted-foreground">
            Starting Capital ($)
          </Label>
          <Input
            id="capital"
            type="number"
            min={100}
            step={100}
            value={capital}
            onChange={(e) => setCapital(Number(e.target.value) || 100)}
            className="w-36"
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="size" className="text-xs text-muted-foreground">
            Trade Size (% of capital)
          </Label>
          <Input
            id="size"
            type="number"
            min={10}
            max={100}
            step={10}
            value={sizePct}
            onChange={(e) => setSizePct(Number(e.target.value) || 10)}
            className="w-36"
          />
        </div>
        <div className="text-xs text-muted-foreground">
          ⚠️ Backtest uses the model&apos;s historical accuracy to simulate win/loss. Not financial advice.
        </div>
      </div>

      {backtest.isLoading && <LoadingPanel label="Running backtest…" />}
      {backtest.isError && <ErrorPanel message={(backtest.error as Error).message} />}

      {backtest.data && backtest.data.total_trades === 0 && (
        <EmptyPanel title="No signals to backtest" hint="Try switching assets or waiting for the ensemble to fire a signal." />
      )}

      {backtest.data && backtest.data.total_trades > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            <StatTile
              label="Final Capital"
              value={`$${backtest.data.final_capital.toLocaleString()}`}
              tone={backtest.data.total_return_pct >= 0 ? "buy" : "sell"}
              hint={`${backtest.data.total_return_pct >= 0 ? "+" : ""}${backtest.data.total_return_pct.toFixed(1)}%`}
            />
            <StatTile label="Win Rate" value={`${backtest.data.win_rate_pct.toFixed(1)}%`} hint={`${backtest.data.wins}W / ${backtest.data.losses}L`} />
            <StatTile label="Max Drawdown" value={`-${backtest.data.max_drawdown_pct.toFixed(1)}%`} tone="sell" />
            <StatTile label="Total Trades" value={backtest.data.total_trades} />
            <StatTile label="Avg P&L / Trade" value={`$${(backtest.data.trades.reduce((s, t) => s + t.pnl, 0) / backtest.data.total_trades).toFixed(2)}`} />
          </div>

          <div className="surface-panel rounded-2xl p-4">
            <div className="mb-3 text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">
              Portfolio Performance
            </div>
            <LineChart
              categories={backtest.data.equity_curve.map((_, i) => i)}
              series={[{ label: "Capital", color: backtest.data.total_return_pct >= 0 ? "#5EEAD4" : "#FB7185", values: backtest.data.equity_curve }]}
              asDates={false}
            />
          </div>

          <div className="surface-panel rounded-2xl p-2">
            <div className="px-2 pt-2 text-sm font-semibold">Individual Trade Log</div>
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Date</TableHead>
                  <TableHead>Signal</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Exit</TableHead>
                  <TableHead>Result</TableHead>
                  <TableHead className="text-right">P&amp;L $</TableHead>
                  <TableHead className="text-right">Capital</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {backtest.data.trades.slice(0, 200).map((t, i) => (
                  <TableRow key={i}>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {new Date(t.date).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Badge variant={t.signal === "BUY" ? "default" : "destructive"}>{t.signal}</Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">${t.entry.toFixed(4)}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">${t.exit.toFixed(4)}</TableCell>
                    <TableCell>
                      <span className={cn("text-xs font-medium", t.pnl >= 0 ? "text-signal-buy" : "text-signal-sell")}>
                        {t.pnl >= 0 ? "✅ Win" : "❌ Loss"}
                      </span>
                    </TableCell>
                    <TableCell className={cn("text-right font-mono tabular-nums", t.pnl >= 0 ? "text-signal-buy" : "text-signal-sell")}>
                      {t.pnl >= 0 ? "+" : ""}
                      {t.pnl.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">${t.capital.toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </div>
  );
}
