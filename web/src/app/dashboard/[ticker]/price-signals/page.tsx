"use client";

import { use } from "react";
import { motion } from "framer-motion";
import { usePredict, useOhlcv } from "@/hooks/use-predict";
import { SignalBadge, SignalHeroCard } from "@/components/signal-badge";
import { StatTile } from "@/components/stat-tile";
import { AnimatedNumber } from "@/components/animated-number";
import { PriceChart } from "@/components/price-chart";
import { LoadingPanel, ErrorPanel } from "@/components/state-views";

export default function PriceSignalsPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = decodeURIComponent(raw);

  const predict = usePredict(ticker);
  const ohlcv = useOhlcv(ticker, 400);

  if (predict.isLoading) return <LoadingPanel label={`Training the ensemble on ${ticker}…`} rows={6} />;
  if (predict.isError) return <ErrorPanel message={(predict.error as Error).message} />;
  const p = predict.data!;

  return (
    <div className="flex flex-col gap-5">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="grid grid-cols-1 gap-4 md:grid-cols-[1.4fr_1fr_1fr_1fr]"
      >
        <SignalHeroCard signal={p.last_signal}>
          <div className="flex h-full flex-col justify-between gap-3">
            <span className="text-[0.7rem] font-semibold uppercase tracking-[0.05em] text-muted-foreground">
              Trading Signal
            </span>
            <SignalBadge signal={p.last_signal} size="lg" />
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-xl font-bold">
                <AnimatedNumber value={p.last_confidence} decimals={1} suffix="%" />
              </span>
              <span className="text-xs text-muted-foreground">confidence · P(up) {(p.last_prob_up * 100).toFixed(1)}%</span>
            </div>
            <div className="text-[0.7rem] text-signal-hold">Not financial advice</div>
          </div>
        </SignalHeroCard>

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
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
        className="surface-panel rounded-2xl p-4"
      >
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[0.82rem] font-bold uppercase tracking-[0.04em] text-signal-buy">Price &amp; Signals</div>
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
