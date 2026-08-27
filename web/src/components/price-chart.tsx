"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  type IChartApi,
  type UTCTimestamp,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import type { OhlcvResponse } from "@/lib/api";

function toUnixSeconds(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

export function PriceChart({ data, height = 420 }: { data: OhlcvResponse; height?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
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
      timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
    });
    chartRef.current = chart;

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#5EEAD4",
      downColor: "#FB7185",
      borderVisible: false,
      wickUpColor: "#5EEAD4",
      wickDownColor: "#FB7185",
    });

    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      color: "rgba(148,163,184,0.35)",
    });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });

    const sma20 = chart.addSeries(LineSeries, { color: "#38BDF8", lineWidth: 1, title: "SMA20" });
    const sma50 = chart.addSeries(LineSeries, { color: "#A78BFA", lineWidth: 1, title: "SMA50" });

    const times = data.dates.map(toUnixSeconds);

    candles.setData(
      times.map((time, i) => ({
        time,
        open: data.open[i],
        high: data.high[i],
        low: data.low[i],
        close: data.close[i],
      }))
    );

    volume.setData(
      times.map((time, i) => ({
        time,
        value: data.volume[i],
        color: data.close[i] >= data.open[i] ? "rgba(94,234,212,0.35)" : "rgba(251,113,133,0.35)",
      }))
    );

    sma20.setData(
      times.map((time, i) => ({ time, value: data.sma20[i] })).filter((d) => d.value !== null) as {
        time: UTCTimestamp;
        value: number;
      }[]
    );
    sma50.setData(
      times.map((time, i) => ({ time, value: data.sma50[i] })).filter((d) => d.value !== null) as {
        time: UTCTimestamp;
        value: number;
      }[]
    );

    const markers: SeriesMarker<Time>[] = [];
    data.signal.forEach((s, i) => {
      if (s === 1) {
        markers.push({ time: times[i], position: "belowBar", color: "#5EEAD4", shape: "arrowUp", text: "BUY" });
      } else if (s === -1) {
        markers.push({ time: times[i], position: "aboveBar", color: "#FB7185", shape: "arrowDown", text: "SELL" });
      }
    });
    createSeriesMarkers(candles, markers);

    chart.timeScale().fitContent();

    const resize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height]);

  return <div ref={containerRef} className="w-full" />;
}
