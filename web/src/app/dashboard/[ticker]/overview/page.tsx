"use client";

import { use } from "react";
import { motion } from "framer-motion";
import { usePredict, useOhlcv } from "@/hooks/use-predict";
import { SignalBadge } from "@/components/signal-badge";
import { StatTile } from "@/components/stat-tile";
import { AnimatedNumber } from "@/components/animated-number";
import { PriceChart } from "@/components/price-chart";
import { LoadingPanel, ErrorPanel } from "@/components/state-views";
import { Card } from "@/components/ui/card";

export default function OverviewPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);

  const predict = usePredict(ticker);
  const ohlcv = useOhlcv(ticker, 400);

  if (predict.isLoading) return <LoadingPanel label={`Training the ensemble on ${ticker}…`} rows={6} />;
  if (predict.isError) return <ErrorPanel message={(predict.error as Error).message} />;
  const p = predict.data!;

  const toneTone = p.last_signal === "BUY" ? "buy" : p.last_signal === "SELL" ? "sell" : "default";

  return (
    <div className="flex flex-col gap-5">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="grid grid-cols-1 gap-4 md:grid-cols-[1.3fr_1fr_1fr_1fr]"
      >
        <Card
          className={
            "surface-panel flex flex-col justify-between gap-2 rounded-xl border-0 p-5 " +
            (p.last_signal === "BUY" ? "glow-buy" : p.last_signal === "SELL" ? "glow-sell" : "")
          }
        >
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Current Signal
            </span>
            <SignalBadge signal={p.last_signal} size="sm" />
          </div>
          <div className="font-mono text-4xl font-extrabold tabular-nums">
            <AnimatedNumber value={p.last_confidence} decimals={1} suffix="%" />
          </div>
          <div className="text-xs text-muted-foreground">
            Confidence · P(up) {(p.last_prob_up * 100).toFixed(1)}%
          </div>
        </Card>

        <StatTile
          label="Ensemble Accuracy"
          value={<AnimatedNumber value={p.ensemble.accuracy * 100} decimals={1} suffix="%" />}
          hint={`Filtered: ${(p.ensemble.filtered_accuracy * 100).toFixed(1)}%`}
        />
        <StatTile
          label="Best Model"
          value={p.ensemble.best_model}
          hint={`${p.ensemble.models_used.length} models in ensemble`}
        />
        <StatTile
          label="Signals Fired"
          value={<AnimatedNumber value={p.n_signals} />}
          hint={`${p.rows_used.toLocaleString()} rows trained`}
          tone={toneTone === "default" ? "default" : toneTone}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
        className="surface-panel rounded-xl p-4"
      >
        <div className="mb-3 flex items-center justify-between">
          <div className="text-sm font-semibold">Price &amp; Signals</div>
          <div className="text-xs text-muted-foreground">
            Trained {p.trained_at ? new Date(p.trained_at).toLocaleString() : "—"}
          </div>
        </div>
        {ohlcv.isLoading && <LoadingPanel label="Loading chart…" rows={3} />}
        {ohlcv.isError && <ErrorPanel message={(ohlcv.error as Error).message} />}
        {ohlcv.data && <PriceChart data={ohlcv.data} />}
      </motion.div>
    </div>
  );
}
