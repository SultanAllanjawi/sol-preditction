"use client";

import { use, useState } from "react";
import { useNews, useCalendar } from "@/hooks/use-news";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

function Tabs({ value, onChange }: { value: "news" | "calendar"; onChange: (v: "news" | "calendar") => void }) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-secondary/40 p-1">
      {(["news", "calendar"] as const).map((t) => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className={cn(
            "rounded-md px-3 py-1.5 text-sm font-medium capitalize transition-colors",
            value === t ? "bg-accent text-foreground" : "text-muted-foreground"
          )}
        >
          {t === "news" ? "Crypto News" : "Economic Calendar"}
        </button>
      ))}
    </div>
  );
}

export default function NewsPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const [tab, setTab] = useState<"news" | "calendar">("news");
  const news = useNews(ticker);
  const calendar = useCalendar();

  return (
    <div className="flex flex-col gap-4">
      <Tabs value={tab} onChange={setTab} />

      {tab === "news" && (
        <>
          {news.isLoading && <LoadingPanel label="Loading sentiment feed…" />}
          {news.isError && <ErrorPanel message={(news.error as Error).message} />}
          {news.data && news.data.crypto_news.length === 0 && (
            <EmptyPanel title="No fresh news for this ticker" hint="Try the economic calendar tab, or check back later." />
          )}
          <div className="flex flex-col gap-2">
            {news.data?.crypto_news.map((item, i) => (
              <div key={i} className="surface-panel rounded-xl p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="text-sm font-medium">{String(item.title ?? "Untitled")}</div>
                  {typeof item.score === "number" && (
                    <Badge variant={item.score > 0.1 ? "default" : item.score < -0.1 ? "destructive" : "secondary"}>
                      {item.score > 0.1 ? "Bullish" : item.score < -0.1 ? "Bearish" : "Neutral"}
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {tab === "calendar" && (
        <>
          {calendar.isLoading && <LoadingPanel label="Loading economic calendar…" />}
          {calendar.isError && <ErrorPanel message={(calendar.error as Error).message} />}
          {calendar.data && calendar.data.events.length === 0 && <EmptyPanel title="No upcoming events" />}
          <div className="surface-panel overflow-hidden rounded-xl">
            <table className="w-full text-sm">
              <thead className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-4 py-2.5">Date</th>
                  <th className="px-4 py-2.5">Time</th>
                  <th className="px-4 py-2.5">Currency</th>
                  <th className="px-4 py-2.5">Event</th>
                  <th className="px-4 py-2.5">Impact</th>
                  <th className="px-4 py-2.5 text-right">Forecast</th>
                  <th className="px-4 py-2.5 text-right">Previous</th>
                </tr>
              </thead>
              <tbody>
                {calendar.data?.events.slice(0, 100).map((e, i) => (
                  <tr key={i} className="border-b border-border/50 last:border-0">
                    <td className="px-4 py-2 text-muted-foreground">{e.date}</td>
                    <td className="px-4 py-2 text-muted-foreground">{e.time}</td>
                    <td className="px-4 py-2 font-medium">{e.currency}</td>
                    <td className="px-4 py-2">{e.event}</td>
                    <td className="px-4 py-2">{e.impact}</td>
                    <td className="px-4 py-2 text-right font-mono">{e.forecast}</td>
                    <td className="px-4 py-2 text-right font-mono text-muted-foreground">{e.previous}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
