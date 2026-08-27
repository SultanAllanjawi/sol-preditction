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
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div
        className={cn(
          "mt-1 font-mono text-xl font-bold tabular-nums",
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
