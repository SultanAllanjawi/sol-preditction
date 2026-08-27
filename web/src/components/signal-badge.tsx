import { cn } from "@/lib/utils";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

const CONFIG = {
  BUY: {
    label: "BUY",
    icon: ArrowUpRight,
    className: "bg-signal-buy/12 text-signal-buy border-signal-buy/25",
  },
  SELL: {
    label: "SELL",
    icon: ArrowDownRight,
    className: "bg-signal-sell/12 text-signal-sell border-signal-sell/25",
  },
  HOLD: {
    label: "HOLD",
    icon: Minus,
    className: "bg-signal-hold/10 text-signal-hold border-signal-hold/20",
  },
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
  const c = CONFIG[signal];
  const Icon = c.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border font-semibold tracking-wide",
        c.className,
        size === "sm" && "px-2 py-0.5 text-[11px] gap-0.5",
        size === "md" && "px-2.5 py-1 text-xs",
        size === "lg" && "px-3.5 py-1.5 text-sm",
        className
      )}
    >
      <Icon className={cn(size === "sm" ? "size-3" : size === "lg" ? "size-4" : "size-3.5")} />
      {c.label}
    </span>
  );
}
