import { cn } from "@/lib/utils";

export function StatTile({
  label,
  value,
  hint,
  tone = "default",
  className,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: "default" | "buy" | "sell";
  className?: string;
}) {
  return (
    <div className={cn("surface-panel rounded-xl px-4 py-3.5", className)}>
      <div className="text-[0.7rem] font-semibold uppercase tracking-[0.05em] text-muted-foreground">{label}</div>
      <div
        className={cn(
          "stat-pop mt-1 font-mono text-xl font-extrabold tabular-nums",
          tone === "default" && "text-white",
          tone === "buy" && "text-signal-buy",
          tone === "sell" && "text-signal-sell"
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}
