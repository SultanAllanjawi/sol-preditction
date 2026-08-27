"use client";

import { use } from "react";
import { usePredict } from "@/hooks/use-predict";
import { StatTile } from "@/components/stat-tile";
import { LineChart } from "@/components/line-chart";
import { LoadingPanel, ErrorPanel, EmptyPanel } from "@/components/state-views";

export default function PredictionsPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const predict = usePredict(ticker);

  if (predict.isLoading) return <LoadingPanel label="Loading price predictions…" />;
  if (predict.isError) return <ErrorPanel message={(predict.error as Error).message} />;
  const { price_prediction: pp } = predict.data!;

  if (!pp.dates.length) return <EmptyPanel title="No price predictions yet" hint="Try refreshing this ticker." />;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="RMSE" value={pp.rmse.toFixed(4)} hint="Root mean squared error" />
        <StatTile label="MAE" value={pp.mae.toFixed(4)} hint="Mean absolute error" />
        <StatTile label="Test Points" value={pp.dates.length} />
        <StatTile
          label="Last Predicted Close"
          value={`$${pp.predicted[pp.predicted.length - 1]?.toFixed(4) ?? "—"}`}
          hint={`Actual: $${pp.actual[pp.actual.length - 1]?.toFixed(4) ?? "—"}`}
        />
      </div>
      <div className="surface-panel rounded-2xl p-4">
        <div className="mb-3 text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">
          Predicted vs Actual Close
        </div>
        <LineChart
          categories={pp.dates}
          series={[
            { label: "Actual", color: "#5EEAD4", values: pp.actual },
            { label: "Predicted", color: "#A78BFA", values: pp.predicted },
          ]}
        />
      </div>
    </div>
  );
}
