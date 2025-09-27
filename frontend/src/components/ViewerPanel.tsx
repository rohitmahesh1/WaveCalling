import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useApiBase } from "@/context/ApiContext";
import type { OverlayPayload } from "@/utils/types";
import ViewerToolbar, { ViewerOptions } from "@/components/ViewerToolbar";
import OverlayCanvas from "@/components/OverlayCanvas";
import { buildDebugImageUrl } from "@/utils/api";

interface Props {
  overlay: OverlayPayload | null;
  baseImageUrl?: string | null;
  debugImageUrl?: string | null;
  summary?: { tracks: number; points: number } | null;
  options: ViewerOptions;
  onOptionsChange: (partial: Partial<ViewerOptions>) => void;
  loading: boolean;
  onRefresh: () => void;
}

function EmptyOverlay({
  loading,
  onRefresh,
}: {
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="mt-3 flex flex-col items-center justify-center rounded-lg border border-slate-700/60 bg-slate-900/40 text-slate-300 p-8 min-h-[280px]">
      <div className="text-sm">
        {loading ? "Loading overlay…" : "No overlay loaded yet."}
      </div>
      <button
        onClick={onRefresh}
        disabled={loading}
        className="mt-3 px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
      >
        {loading ? "Fetching…" : "Refresh"}
      </button>
    </div>
  );
}

export default function ViewerPanel({
  overlay,
  baseImageUrl,
  debugImageUrl, // optional override
  summary,
  options,
  onOptionsChange,
  loading,
  onRefresh,
}: Props) {
  const navigate = useNavigate();
  const apiBase = useApiBase();

  const hasOverlay = !!overlay && (overlay as any)?.tracks && ((overlay as any).tracks.length || 0) > 0;

  const derivedDebugUrl = React.useMemo(() => {
    if (debugImageUrl) return debugImageUrl || undefined;
    const layer = (options as any)?.debugLayer as
      | "none"
      | "prob"
      | "mask_raw"
      | "mask_clean"
      | "mask_filtered"
      | "skeleton"
      | "mask_hysteresis"
      | undefined;
    const runId = (overlay as any)?.run_id as string | undefined;
    if (!apiBase || !runId || !layer || layer === "none") return undefined;
    return buildDebugImageUrl(apiBase, runId, layer);
  }, [apiBase, overlay, options, debugImageUrl]);

  const statusNode = (() => {
    if (loading && !hasOverlay) return <span className="italic">Loading overlay…</span>;
    if (loading && hasOverlay) return <span className="italic">Refreshing overlay…</span>;
    if (hasOverlay && summary) {
      return (
        <>
          tracks:{" "}
          <span className="text-slate-300">
            {summary.tracks.toLocaleString()}
          </span>{" "}
          · points:{" "}
          <span className="text-slate-300">
            {summary.points.toLocaleString()}
          </span>
        </>
      );
    }
    return <span className="italic">No overlay loaded yet</span>;
  })();

  return (
    <section className="rounded-xl border border-slate-700/50 bg-console-700 p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-slate-200 font-semibold">Viewer</h2>
        <ViewerToolbar
          options={options}
          onChange={onOptionsChange}
          onRefresh={onRefresh}
          loading={loading}
          onOpenAdvanced={() => navigate("/viewer/advanced")}
          advancedLabel="Advanced"
        />
      </div>

      <div className="mt-2 text-xs text-slate-400">
        {statusNode}
        {options.showBase && baseImageUrl && (
          <span className="ml-3">
            base:{" "}
            <a
              href={baseImageUrl}
              className="underline decoration-slate-600 hover:text-slate-300"
              target="_blank"
              rel="noreferrer"
            >
              base.png
            </a>
          </span>
        )}
        {((options as any)?.debugLayer !== "none") && derivedDebugUrl && (
          <span className="ml-3">
            debug:{" "}
            <a
              href={derivedDebugUrl}
              className="underline decoration-slate-600 hover:text-slate-300"
              target="_blank"
              rel="noreferrer"
            >
              {(options as any)?.debugLayer}.png
            </a>
          </span>
        )}
      </div>

      {hasOverlay ? (
        <OverlayCanvas
          payload={overlay as any}
          baseImageUrl={options.showBase ? baseImageUrl ?? undefined : undefined}
          debugImageUrl={derivedDebugUrl}
          options={options}
          padding={20}
          style={{ height: "min(70vh, 720px)" }}
          className="mt-3"
        />
      ) : (
        <EmptyOverlay loading={loading} onRefresh={onRefresh} />
      )}
    </section>
  );
}
