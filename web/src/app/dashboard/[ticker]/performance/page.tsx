"use client";

import { use } from "react";
import { usePredict } from "@/hooks/use-predict";
import { StatTile } from "@/components/stat-tile";
import { LoadingPanel, ErrorPanel } from "@/components/state-views";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

function Bar({ value, tone }: { value: number; tone: "buy" | "muted" }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={cn("h-full rounded-full", tone === "buy" ? "bg-signal-buy" : "bg-muted-foreground/40")}
        style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}

export default function PerformancePage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);
  const predict = usePredict(ticker);

  if (predict.isLoading) return <LoadingPanel label="Loading model performance…" rows={5} />;
  if (predict.isError) return <ErrorPanel message={(predict.error as Error).message} />;
  const p = predict.data!;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Ensemble Accuracy" value={`${(p.ensemble.accuracy * 100).toFixed(1)}%`} />
        <StatTile label="Filtered Accuracy" value={`${(p.ensemble.filtered_accuracy * 100).toFixed(1)}%`} hint="High-confidence signals only" />
        <StatTile label="F1 Score" value={p.ensemble.f1.toFixed(3)} />
        <StatTile label="ROC AUC" value={p.ensemble.auc.toFixed(3)} />
      </div>

      <div className="surface-panel rounded-2xl p-4">
        <div className="mb-4 flex items-center justify-between">
          <div className="text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">Per-Model Accuracy</div>
          <div className="text-xs text-muted-foreground">
            Ensemble excludes models scoring below 55% accuracy
          </div>
        </div>
        <div className="flex flex-col gap-4">
          {Object.entries(p.models).map(([name, m]) => {
            const excluded = p.ensemble.models_excluded.includes(name);
            return (
              <div key={name} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 font-medium">
                    {name}
                    {name === p.ensemble.best_model && <Badge variant="secondary">Best</Badge>}
                    {excluded && (
                      <Badge variant="outline" className="text-signal-sell border-signal-sell/30">
                        Excluded
                      </Badge>
                    )}
                  </div>
                  <div className="font-mono tabular-nums text-muted-foreground">
                    acc {(m.accuracy * 100).toFixed(1)}% · f1 {m.f1.toFixed(2)} · auc {m.auc.toFixed(2)}
                  </div>
                </div>
                <Bar value={m.accuracy} tone={excluded ? "muted" : "buy"} />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
