"use client";

import { useState } from "react";
import { toast } from "sonner";
import { usePortfolio, usePortfolioMutations } from "@/hooks/use-portfolio";
import { useTickers } from "@/hooks/use-tickers";
import { StatTile } from "@/components/stat-tile";
import { LineChart } from "@/components/line-chart";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { Plus, X, Check } from "lucide-react";
import type { Trade } from "@/lib/api";

function StatusChip({ status }: { status: string }) {
  const isOpen = status === "Open";
  return (
    <span
      className="rounded-full px-2.5 py-0.5 text-[0.68rem] font-bold"
      style={{
        color: isOpen ? "#2dd4bf" : "#64748b",
        background: isOpen ? "rgba(45,212,191,0.12)" : "rgba(100,116,139,0.14)",
      }}
    >
      {status.toUpperCase()}
    </span>
  );
}

function assetInitial(ticker: string) {
  return ticker.replace(/[^A-Za-z]/g, "").slice(0, 1).toUpperCase() || "?";
}

export default function PortfolioPage() {
  const portfolio = usePortfolio();
  const { data: tickers } = useTickers();
  const { add, close, remove } = usePortfolioMutations();
  const [open, setOpen] = useState(false);

  const [form, setForm] = useState({
    ticker: "SOL-USD",
    side: "BUY" as "BUY" | "SELL",
    entry: "",
    size: "1",
    tp: "",
    sl: "",
    note: "",
  });

  async function submit() {
    if (!form.entry || Number(form.entry) <= 0) {
      toast.error("Enter a valid entry price");
      return;
    }
    try {
      await add.mutateAsync({
        ticker: form.ticker,
        side: form.side,
        entry: Number(form.entry),
        size: Number(form.size) || 1,
        tp: form.tp ? Number(form.tp) : null,
        sl: form.sl ? Number(form.sl) : null,
        note: form.note,
      });
      toast.success(`Trade added: ${form.side} ${form.ticker} @ $${form.entry}`);
      setOpen(false);
      setForm((f) => ({ ...f, entry: "", tp: "", sl: "", note: "" }));
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  if (portfolio.isLoading) return <LoadingPanel label="Loading portfolio…" />;
  if (portfolio.isError) return <ErrorPanel message={(portfolio.error as Error).message} />;
  const { trades, summary } = portfolio.data!;

  const byDate = [...trades].sort((a, b) => a.date.localeCompare(b.date));
  const cumulativePnl = byDate.reduce<number[]>((acc, t) => {
    acc.push((acc.at(-1) ?? 0) + (t.pnl ?? 0));
    return acc;
  }, []);
  const topAssets = [...trades].sort((a, b) => Math.abs(b.pnl ?? 0) - Math.abs(a.pnl ?? 0)).slice(0, 5);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-bold">💼 Portfolio</h1>
          <p className="text-xs text-muted-foreground">
            Track which signals you acted on · P&amp;L updates live on refresh
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[0.72rem] text-muted-foreground">
            <span className="status-dot buy" />
            Persisted · updates live
          </div>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger
              render={
                <Button>
                  <Plus className="size-4" />
                  Add Trade
                </Button>
              }
            />
            <DialogContent>
              <DialogHeader>
                <DialogTitle>➕ Add a Trade</DialogTitle>
              </DialogHeader>
              <div className="grid grid-cols-2 gap-3 py-2">
                <div className="flex flex-col gap-1.5">
                  <Label>Asset</Label>
                  <Select value={form.ticker} onValueChange={(v) => v && setForm((f) => ({ ...f, ticker: v }))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {tickers?.map((t) => (
                        <SelectItem key={t.ticker} value={t.ticker}>
                          {t.ticker}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>Side</Label>
                  <Select
                    value={form.side}
                    onValueChange={(v) => v && setForm((f) => ({ ...f, side: v as "BUY" | "SELL" }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BUY">BUY</SelectItem>
                      <SelectItem value="SELL">SELL</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>Entry Price ($)</Label>
                  <Input
                    type="number"
                    value={form.entry}
                    onChange={(e) => setForm((f) => ({ ...f, entry: e.target.value }))}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>Position Size (units)</Label>
                  <Input
                    type="number"
                    value={form.size}
                    onChange={(e) => setForm((f) => ({ ...f, size: e.target.value }))}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>Take Profit ($)</Label>
                  <Input type="number" value={form.tp} onChange={(e) => setForm((f) => ({ ...f, tp: e.target.value }))} />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>Stop Loss ($)</Label>
                  <Input type="number" value={form.sl} onChange={(e) => setForm((f) => ({ ...f, sl: e.target.value }))} />
                </div>
                <div className="col-span-2 flex flex-col gap-1.5">
                  <Label>Notes (optional)</Label>
                  <Input
                    placeholder="e.g. Signal confidence 73%"
                    value={form.note}
                    onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={submit} disabled={add.isPending} className="w-full">
                  {add.isPending ? "Adding…" : "✅ Add Trade"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Total Trades" value={summary.total_trades} />
        <StatTile label="Open Positions" value={summary.open_positions} tone="buy" />
        <StatTile
          label="Total P&L"
          value={`${summary.total_pnl >= 0 ? "+" : ""}${summary.total_pnl.toFixed(2)}`}
          tone={summary.total_pnl >= 0 ? "buy" : "sell"}
        />
        <StatTile
          label="ROI"
          value={`${summary.roi_pct >= 0 ? "+" : ""}${summary.roi_pct.toFixed(2)}%`}
          tone={summary.roi_pct >= 0 ? "buy" : "sell"}
        />
      </div>

      {trades.length === 0 ? (
        <EmptyPanel
          title="No trades logged yet"
          hint="Trades you add here track live P&L against your entry, TP, and SL automatically."
        />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[2fr_1fr]">
            <div className="surface-panel rounded-2xl p-4">
              <div className="mb-3 text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">
                📈 Portfolio Performance
              </div>
              <LineChart
                categories={cumulativePnl.map((_, i) => i)}
                series={[
                  {
                    label: "Cumulative P&L",
                    color: (cumulativePnl.at(-1) ?? 0) >= 0 ? "#2dd4bf" : "#fb7185",
                    values: cumulativePnl,
                  },
                ]}
                asDates={false}
                height={240}
              />
            </div>
            <div className="surface-panel rounded-2xl p-4">
              <div className="mb-3 text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">
                🏆 Top Assets
              </div>
              <div className="flex flex-col gap-2.5">
                {topAssets.map((t, i) => (
                  <div key={i} className="flex items-center gap-2.5">
                    <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-secondary text-xs font-bold">
                      {assetInitial(t.ticker)}
                    </div>
                    <span className="flex-1 truncate text-sm font-medium">{t.ticker}</span>
                    <span
                      className={cn(
                        "font-mono text-xs font-bold tabular-nums",
                        (t.pnl ?? 0) >= 0 ? "text-signal-buy" : "text-signal-sell"
                      )}
                    >
                      {(t.pnl ?? 0) >= 0 ? "+" : ""}
                      {(t.pnl ?? 0).toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="surface-panel rounded-2xl p-2">
            <div className="px-2 pt-2 text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">
              📋 Transactions
            </div>
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Date</TableHead>
                  <TableHead>Asset</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead className="text-right">Live / Exit</TableHead>
                  <TableHead className="text-right">P&amp;L</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.map((t: Trade) => (
                  <TableRow key={t.id}>
                    <TableCell className="text-xs text-muted-foreground">{t.date}</TableCell>
                    <TableCell className="font-medium">{t.ticker}</TableCell>
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5">
                        <span className={cn("status-dot", t.side === "BUY" ? "buy" : "sell")} />
                        <span className={cn("font-bold", t.side === "BUY" ? "text-signal-buy" : "text-signal-sell")}>
                          {t.side}
                        </span>
                      </span>
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">${t.entry.toFixed(4)}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{t.size}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      ${(t.status === "Open" ? t.live_price ?? t.entry : t.exit ?? t.entry).toFixed(4)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "text-right font-mono tabular-nums",
                        (t.pnl ?? 0) >= 0 ? "text-signal-buy" : "text-signal-sell"
                      )}
                    >
                      {(t.pnl ?? 0) >= 0 ? "+" : ""}
                      {(t.pnl ?? 0).toFixed(2)}
                      {t.pnl_pct !== undefined && (
                        <span className="ml-1 text-[10px] text-muted-foreground">
                          ({t.pnl_pct >= 0 ? "+" : ""}
                          {t.pnl_pct.toFixed(1)}%)
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      <StatusChip status={t.status} />
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        {t.status === "Open" && (
                          <Button
                            size="icon"
                            variant="ghost"
                            title="Close at live price"
                            onClick={() =>
                              close
                                .mutateAsync({ id: t.id, exitPrice: t.live_price ?? t.entry })
                                .then(() => toast.success(`Closed ${t.ticker}`))
                                .catch((e) => toast.error(e.message))
                            }
                          >
                            <Check className="size-4" />
                          </Button>
                        )}
                        <Button
                          size="icon"
                          variant="ghost"
                          title="Delete"
                          onClick={() =>
                            remove
                              .mutateAsync(t.id)
                              .then(() => toast.success("Trade removed"))
                              .catch((e) => toast.error(e.message))
                          }
                        >
                          <X className="size-4" />
                        </Button>
                      </div>
                    </TableCell>
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
