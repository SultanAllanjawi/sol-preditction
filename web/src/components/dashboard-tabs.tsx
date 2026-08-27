"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const TABS = [
  { slug: "overview", label: "Overview" },
  { slug: "predictions", label: "Predicted vs Actual" },
  { slug: "performance", label: "Model Performance" },
  { slug: "history", label: "Signal History" },
  { slug: "backtest", label: "Backtest P&L" },
  { slug: "news", label: "News & Sentiment" },
  { slug: "scanner", label: "Scanner" },
] as const;

export function DashboardTabs({ ticker }: { ticker: string }) {
  const pathname = usePathname();

  return (
    <div className="scrollbar-none -mx-1 flex gap-1 overflow-x-auto px-1">
      {TABS.map((tab) => {
        const href = `/dashboard/${ticker}/${tab.slug}`;
        const active = pathname === href;
        return (
          <Link
            key={tab.slug}
            href={href}
            className={cn(
              "relative shrink-0 rounded-md px-3 py-1.5 text-sm font-medium whitespace-nowrap transition-colors",
              active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            {active && <span className="absolute inset-0 -z-10 rounded-md bg-accent" />}
            {tab.label}
          </Link>
        );
      })}
    </div>
  );
}
