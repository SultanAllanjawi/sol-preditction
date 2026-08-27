import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

export function LoadingPanel({ label = "Loading real-time data…", rows = 4 }: { label?: string; rows?: number }) {
  return (
    <div className="surface-panel flex flex-col gap-3 rounded-xl p-6">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        {label}
      </div>
      <div className="flex flex-col gap-2">
        {Array.from({ length: rows }).map((_, i) => (
          <Skeleton key={i} className="h-9 w-full" style={{ opacity: 1 - i * 0.12 }} />
        ))}
      </div>
    </div>
  );
}

export function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-signal-sell/20 bg-signal-sell/5 p-8 text-center">
      <AlertTriangle className="size-6 text-signal-sell" />
      <div className="text-sm font-medium">Couldn&apos;t load this data</div>
      <div className="max-w-md text-xs text-muted-foreground">{message}</div>
    </div>
  );
}

export function EmptyPanel({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border p-10 text-center">
      <Inbox className="size-6 text-muted-foreground" />
      <div className="text-sm font-medium">{title}</div>
      {hint && <div className="max-w-md text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}
