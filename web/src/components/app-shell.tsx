"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { LayoutDashboard, Wallet, Bitcoin, Landmark, Building2 } from "lucide-react";
import { TickerCommand } from "@/components/ticker-command";
import { useTickers } from "@/hooks/use-tickers";
import { cn } from "@/lib/utils";
import type { Category } from "@/lib/api";

const CATEGORY_ICON: Record<Category, typeof Bitcoin> = {
  crypto: Bitcoin,
  us_stock: Landmark,
  dfm: Building2,
};

function currentTicker(pathname: string): string | null {
  const m = pathname.match(/^\/dashboard\/([^/]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const activeTicker = currentTicker(pathname);
  const { data: tickers } = useTickers();
  const isPortfolio = pathname.startsWith("/portfolio");

  return (
    <div className="flex min-h-screen w-full">
      <aside className="sidebar-background hidden w-64 shrink-0 flex-col border-r border-sidebar-border md:flex">
        <div className="px-5 py-5">
          <div className="text-base font-bold tracking-tight">🔮 Prediction Dashboard</div>
          <div className="text-[11px] text-muted-foreground">Auto-updates every 30 minutes</div>
        </div>

        <nav className="flex flex-col gap-0.5 px-3 py-2">
          <Link
            href={`/dashboard/${activeTicker ?? "SOL-USD"}/overview`}
            className={cn(
              "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              !isPortfolio ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-foreground"
            )}
          >
            <LayoutDashboard className="size-4" />
            Dashboard
          </Link>
          <Link
            href="/portfolio"
            className={cn(
              "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              isPortfolio ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-foreground"
            )}
          >
            <Wallet className="size-4" />
            Portfolio
          </Link>
        </nav>

        <div className="mt-2 flex-1 overflow-y-auto px-3 pb-4">
          <div className="px-2 pb-1.5 pt-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Watchlist
          </div>
          <div className="flex flex-col gap-0.5">
            {tickers?.map((t) => {
              const Icon = CATEGORY_ICON[t.category];
              const active = t.ticker === activeTicker && !isPortfolio;
              return (
                <Link
                  key={t.ticker}
                  href={`/dashboard/${t.ticker}/overview`}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-2.5 py-1.5 text-[13px] transition-colors",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-foreground"
                  )}
                >
                  <Icon className="size-3.5 shrink-0 opacity-70" />
                  <span className="truncate font-medium">{t.ticker}</span>
                </Link>
              );
            })}
          </div>
        </div>

      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between border-b border-border bg-background/80 px-4 backdrop-blur md:px-6">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="relative flex size-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal-buy opacity-60" />
              <span className="relative inline-flex size-2 rounded-full bg-signal-buy" />
            </span>
            Live
          </div>
          <TickerCommand />
        </header>
        <motion.main
          key={pathname}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="flex-1 px-4 py-6 md:px-6"
        >
          {children}
          <div className="mt-8 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4">
            <div className="flex items-center gap-1.5 text-[0.75rem] text-muted-foreground">
              <span className="inline-block size-1.5 rounded-full bg-signal-buy" />
              Prediction Dashboard &middot; research tool, not financial advice
            </div>
            <div className="text-[0.72rem] text-muted-foreground">Data refreshes every 6h</div>
          </div>
        </motion.main>
      </div>
    </div>
  );
}
