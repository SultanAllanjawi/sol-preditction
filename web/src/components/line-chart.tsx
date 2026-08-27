"use client";

import { useEffect, useRef } from "react";
import { createChart, LineSeries, type IChartApi, type UTCTimestamp } from "lightweight-charts";

export interface LineSeriesSpec {
  label: string;
  color: string;
  values: number[];
}

export function LineChart({
  categories,
  series,
  height = 320,
  asDates = true,
}: {
  categories: string[] | number[];
  series: LineSeriesSpec[];
  height?: number;
  asDates?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart: IChartApi = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "#8b93a7",
        fontFamily: "var(--font-geist-sans), sans-serif",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.05)" },
        horzLines: { color: "rgba(255,255,255,0.05)" },
      },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
      timeScale: {
        borderColor: "rgba(255,255,255,0.08)",
        timeVisible: asDates,
        secondsVisible: false,
        tickMarkFormatter: asDates ? undefined : (time: number) => `#${time}`,
      },
      localization: {
        timeFormatter: asDates ? undefined : (time: number) => `Trade #${time}`,
      },
    });

    series.forEach((s) => {
      const line = chart.addSeries(LineSeries, { color: s.color, lineWidth: 2, title: s.label });
      line.setData(
        categories.map((c, i) => ({
          time: (asDates ? Math.floor(new Date(c as string).getTime() / 1000) : (i as number)) as UTCTimestamp,
          value: s.values[i],
        }))
      );
    });

    chart.timeScale().fitContent();

    const resize = () => chart.applyOptions({ width: el.clientWidth });
    const observer = new ResizeObserver(resize);
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [categories, series, height, asDates]);

  return <div ref={containerRef} className="w-full" />;
}
