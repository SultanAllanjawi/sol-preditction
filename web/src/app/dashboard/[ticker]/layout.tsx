import { DashboardTabs } from "@/components/dashboard-tabs";
import { LivePriceBadge } from "@/components/live-price-badge";

export default async function TickerLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ ticker: string }>;
}) {
  const { ticker } = await params;
  const decoded = decodeURIComponent(ticker);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{decoded}</div>
          <LivePriceBadge ticker={decoded} />
        </div>
      </div>
      <DashboardTabs ticker={decoded} />
      <div>{children}</div>
    </div>
  );
}
