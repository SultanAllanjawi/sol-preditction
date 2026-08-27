"use client";

import { use } from "react";
import { useNews, useCalendar } from "@/hooks/use-news";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";
import { Badge } from "@/components/ui/badge";
import { StatTile } from "@/components/stat-tile";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function NewsPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const news = useNews(ticker);
  const calendar = useCalendar();

  const positive = news.data?.crypto_news.filter((n) => typeof n.score === "number" && n.score > 0.1).length ?? 0;
  const negative = news.data?.crypto_news.filter((n) => typeof n.score === "number" && n.score < -0.1).length ?? 0;
  const total = news.data?.crypto_news.length ?? 0;
  const avgScore = total
    ? (news.data!.crypto_news.reduce((s, n) => s + (typeof n.score === "number" ? n.score : 0), 0) / total).toFixed(2)
    : "0.00";

  return (
    <Tabs defaultValue="news">
      <TabsList>
        <TabsTrigger value="news">📰 Crypto News &amp; Sentiment</TabsTrigger>
        <TabsTrigger value="calendar">📅 ForexFactory Economic Calendar</TabsTrigger>
      </TabsList>

      <TabsContent value="news" className="mt-4 flex flex-col gap-4">
        {news.isLoading && <LoadingPanel label="Loading sentiment feed…" />}
        {news.isError && <ErrorPanel message={(news.error as Error).message} />}
        {news.data && (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatTile label="Articles" value={total} />
            <StatTile label="Positive" value={positive} tone="buy" />
            <StatTile label="Negative" value={negative} tone="sell" />
            <StatTile label="Sentiment" value={avgScore} />
          </div>
        )}
        {news.data && news.data.crypto_news.length === 0 && (
          <EmptyPanel title="No fresh news for this ticker" hint="Try the economic calendar tab, or check back later." />
        )}
        <div className="flex flex-col gap-2">
          {news.data?.crypto_news.map((item, i) => {
            const score = typeof item.score === "number" ? item.score : 0;
            const border = score > 0.1 ? "border-l-signal-buy" : score < -0.1 ? "border-l-signal-sell" : "border-l-signal-hold";
            return (
              <div key={i} className={`surface-panel rounded-2xl border-l-4 p-4 ${border}`}>
                <div className="flex items-start justify-between gap-3">
                  <div className="text-sm font-medium">{String(item.title ?? "Untitled")}</div>
                  <Badge variant={score > 0.1 ? "default" : score < -0.1 ? "destructive" : "secondary"}>
                    {score > 0.1 ? "Bullish" : score < -0.1 ? "Bearish" : "Neutral"}
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </TabsContent>

      <TabsContent value="calendar" className="mt-4">
        {calendar.isLoading && <LoadingPanel label="Loading economic calendar…" />}
        {calendar.isError && <ErrorPanel message={(calendar.error as Error).message} />}
        {calendar.data && calendar.data.events.length === 0 && <EmptyPanel title="No upcoming events" />}
        {calendar.data && calendar.data.events.length > 0 && (
          <div className="surface-panel overflow-hidden rounded-2xl">
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
                {calendar.data.events.slice(0, 100).map((e, i) => (
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
        )}
      </TabsContent>
    </Tabs>
  );
}
