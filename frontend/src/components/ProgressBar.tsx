import * as React from "react";

type Props = {
  /** Number of processed tracks so far */
  processedCount: number;
  /** Total tracks if known; when null/undefined shows indeterminate bar */
  totalTracks?: number | null;
  /** "file" (authoritative) or "synthesized" (best-effort) */
  source?: "file" | "synthesized" | string;
  /** Optional compact size */
  size?: "sm" | "md";
  /** Optional label (overrides default) */
  label?: string;
  className?: string;
};

export default function ProgressBar({
  processedCount,
  totalTracks,
  source,
  size = "md",
  label,
  className,
}: Props) {
  const determinate = Number.isFinite(totalTracks as number) && (totalTracks as number) > 0;
  const pct = determinate ? Math.max(0, Math.min(100, Math.round((processedCount / (totalTracks as number)) * 100))) : 0;

  const h = size === "sm" ? "h-2" : "h-3";
  const textSize = size === "sm" ? "text-[11px]" : "text-xs";

  return (
    <div className={`w-full ${className || ""}`}>
      <div className="flex items-center justify-between mb-1">
        <div className={`text-slate-300 ${textSize}`}>
          {label ??
            (determinate
              ? `Processed ${processedCount.toLocaleString()} of ${Number(totalTracks).toLocaleString()}`
              : `Processed ${processedCount.toLocaleString()} (estimating…)`)}
        </div>
        {source && <div className={`text-slate-500 ${textSize}`}>source: {source}</div>}
      </div>

      <div className={`w-full ${h} rounded-md bg-slate-800/80 overflow-hidden border border-slate-700/60`}>
        {determinate ? (
          <div
            className="h-full bg-emerald-500/70"
            style={{ width: `${pct}%`, transition: "width 200ms ease-out" }}
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            role="progressbar"
          />
        ) : (
          <div className="relative h-full">
            {/* simple indeterminate shimmer using two moving blocks */}
            <div className="absolute inset-y-0 left-0 w-1/3 bg-sky-500/60 animate-pulse" />
            <div className="absolute inset-y-0 left-1/3 w-1/3 bg-sky-500/40 animate-pulse" style={{ animationDelay: "200ms" }} />
          </div>
        )}
      </div>

      {determinate && (
        <div className={`mt-1 text-right text-slate-400 ${textSize}`}>{pct}%</div>
      )}
    </div>
  );
}
