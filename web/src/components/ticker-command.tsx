"use client";

import { Command } from "cmdk";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Search, Bitcoin, Landmark, Building2 } from "lucide-react";
import { useTickers } from "@/hooks/use-tickers";
import type { Category } from "@/lib/api";

const CATEGORY_META: Record<Category, { label: string; icon: typeof Bitcoin }> = {
  crypto: { label: "Crypto", icon: Bitcoin },
  us_stock: { label: "US Equities", icon: Landmark },
  dfm: { label: "Dubai Financial Market", icon: Building2 },
};

export function TickerCommand() {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const { data: tickers } = useTickers();

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  function select(ticker: string) {
    setOpen(false);
    router.push(`/dashboard/${encodeURIComponent(ticker)}/overview`);
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-lg border border-border bg-secondary/50 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
      >
        <Search className="size-3.5" />
        <span className="hidden sm:inline">Jump to ticker</span>
        <kbd className="ml-2 rounded border border-border bg-background px-1.5 py-0.5 font-mono text-[10px]">
          &#8984;K
        </kbd>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[15vh] backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <div
            className="w-full max-w-lg overflow-hidden rounded-xl border border-border bg-popover shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <Command loop shouldFilter>
              <div className="flex items-center gap-2 border-b border-border px-3.5">
                <Search className="size-4 text-muted-foreground" />
                <Command.Input
                  autoFocus
                  placeholder="Search SOL-USD, AAPL, EMAAR.DFM…"
                  className="h-12 w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                />
              </div>
              <Command.List className="max-h-80 overflow-y-auto p-2">
                <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
                  No matching ticker.
                </Command.Empty>
                {(["crypto", "us_stock", "dfm"] as Category[]).map((cat) => {
                  const items = tickers?.filter((t) => t.category === cat) ?? [];
                  if (items.length === 0) return null;
                  const meta = CATEGORY_META[cat];
                  return (
                    <Command.Group
                      key={cat}
                      heading={meta.label}
                      className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted-foreground"
                    >
                      {items.map((t) => (
                        <Command.Item
                          key={t.ticker}
                          value={`${t.ticker} ${t.name}`}
                          onSelect={() => select(t.ticker)}
                          className="flex cursor-pointer items-center justify-between rounded-md px-2.5 py-2 text-sm data-[selected=true]:bg-accent"
                        >
                          <span className="flex items-center gap-2">
                            <meta.icon className="size-3.5 text-muted-foreground" />
                            <span className="font-medium">{t.ticker}</span>
                          </span>
                          <span className="text-xs text-muted-foreground">{t.name}</span>
                        </Command.Item>
                      ))}
                    </Command.Group>
                  );
                })}
              </Command.List>
            </Command>
          </div>
        </div>
      )}
    </>
  );
}
