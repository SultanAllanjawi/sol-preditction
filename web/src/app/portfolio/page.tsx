"use client";

import { useState } from "react";
import { toast } from "sonner";
import { usePortfolio, usePortfolioMutations } from "@/hooks/use-portfolio";
import { useTickers } from "@/hooks/use-tickers";
import { StatTile } from "@/components/stat-tile";
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
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Plus, X, Check } from "lucide-react";

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

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Portfolio Tracker</h1>
          <p className="text-sm text-muted-foreground">
            Track which signals you acted on — P&amp;L updates live from real prices.
          </p>
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
              <DialogTitle>Add a Trade</DialogTitle>
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
                <Label>Position Size</Label>
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
                <Label>Notes</Label>
                <Input
                  placeholder="e.g. Signal confidence 73%"
                  value={form.note}
                  onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={submit} disabled={add.isPending} className="w-full">
                {add.isPending ? "Adding…" : "Add Trade"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
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
        <div className="surface-panel rounded-xl p-2">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Date</TableHead>
                <TableHead>Asset</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Live / Exit</TableHead>
                <TableHead className="text-right">P&amp;L</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades.map((t) => (
                <TableRow key={t.id}>
                  <TableCell className="text-xs text-muted-foreground">{t.date}</TableCell>
                  <TableCell className="font-medium">{t.ticker}</TableCell>
                  <TableCell>
                    <Badge variant={t.side === "BUY" ? "default" : "destructive"}>{t.side}</Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">${t.entry.toFixed(4)}</TableCell>
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
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{t.status}</Badge>
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
      )}
    </div>
  );
}
