import { cn } from "@/lib/utils";

const EMOJI = { BUY: "🟢", SELL: "🔴", HOLD: "⚪" } as const;

const TEXT_COLOR = {
  BUY: "text-signal-buy",
  SELL: "text-signal-sell",
  HOLD: "text-signal-hold",
} as const;

export function SignalBadge({
  signal,
  size = "md",
  className,
}: {
  signal: "BUY" | "SELL" | "HOLD";
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <span
      className={cn(
        "stat-pop inline-flex items-center gap-1.5 font-extrabold leading-none",
        TEXT_COLOR[signal],
        size === "sm" && "text-xs gap-1",
        size === "md" && "text-base",
        size === "lg" && "text-[2.6rem]",
        className
      )}
    >
      <span className={cn("status-dot", signal.toLowerCase())} />
      {signal}
    </span>
  );
}

/** The full hero card treatment from the original (§6): tinted bg + 2px colored border. */
export function SignalHeroCard({ signal, children }: { signal: "BUY" | "SELL" | "HOLD"; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-2xl p-5",
        signal === "BUY" && "glow-buy",
        signal === "SELL" && "glow-sell",
        signal === "HOLD" && "bg-card border-2 border-border"
      )}
    >
      {children}
    </div>
  );
}

export { EMOJI as SIGNAL_EMOJI };
