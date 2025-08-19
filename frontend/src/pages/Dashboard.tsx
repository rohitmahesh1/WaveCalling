import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { DashboardProvider, useDashboard } from "@/context/DashboardContext";
import { useSSE } from "@/hooks/useSSE";

import RunsPanel from "@/components/RunsPanel";
import UploadPanel from "@/components/UploadPanel";
import ViewerPanel from "@/components/ViewerPanel";
import LiveLogPanel from "@/components/LiveLogPanel";
import ArtifactsPanel from "@/components/ArtifactsPanel";
import ProgressBar from "@/components/ProgressBar";
import RunStatusBadge from "@/components/RunStatusBadge";
import EmptyState from "@/components/EmptyState";

import type { JobEvent } from "@/utils/types";

const RUN_ID_RE = /^[a-f0-9]{6,32}$/i;

function sanitizeRunId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  if (s.startsWith("http://") || s.startsWith("https://")) return null;
  if (s.includes("/")) return null;
  if (!RUN_ID_RE.test(s)) return null;
  return s;
}

function DashboardInner() {
  const apiBase = useApiBase();
  const {
    selectedRunId,
    setSelectedRunId,
    selectedInfo,
    artifacts,

    overlay,
    overlaySummary,
    overlayLoading,
    baseImageUrl,
    refreshOverlay,

    progress,
    refreshProgress,

    logs,
    appendLog,
    clearLog,
    downloadLog,

    viewerOptions,
    setViewerOptions,
  } = useDashboard();

  // Use a sanitized run id for *all* network calls & conditionals
  const safeRunId = React.useMemo(() => sanitizeRunId(selectedRunId), [selectedRunId]);

  // SSE connection to the current run (final-or-partial replay)
  const sseUrl = React.useMemo(
    () => (safeRunId ? `${apiBase}/api/runs/${encodeURIComponent(safeRunId)}/events?replay=1` : null),
    [apiBase, safeRunId]
  );

  const [logPaused, setLogPaused] = React.useState(false);
  const { status: sseStatus, last, reconnect, close } = useSSE<JobEvent>({
    url: sseUrl,
    withCredentials: false,
    autoReconnect: true,
    reconnectDelayMs: 800,
    maxReconnectDelayMs: 10_000,
  });

  // When the *safe* run id changes, pull a fresh overlay/progress snapshot
  React.useEffect(() => {
    if (!safeRunId) return;
    void refreshProgress();
    void refreshOverlay();
  }, [safeRunId, refreshProgress, refreshOverlay]);

  // Handle incoming SSE events
  const lastPartialRef = React.useRef<number>(-1);
  React.useEffect(() => {
    if (!last || !last.data) return;
    const evt = last.data as JobEvent;
    const phase = evt.phase ?? "(event)";
    const msg = evt.message ?? "";

    // Always record; "pause" only affects the viewer, not ingestion.
    appendLog(`[${phase}] ${msg}`);

    // Surface any server-rewritten artifact URLs
    if (evt.extra && typeof evt.extra === "object") {
      for (const [k, v] of Object.entries(evt.extra)) {
        if (typeof v === "string" && v.startsWith("/runs/")) {
          appendLog(`[${phase}] ${k}: ${v}`);
        }
      }
    }

    // Keep progress fresh while work is ongoing
    if (phase === "PROCESS" || phase === "KYMO" || phase === "WRITE_PARTIAL") {
      void refreshProgress();
    }

    // Refresh overlay when we know new data landed
    const partialIdx =
      (evt.extra && typeof evt.extra === "object" && (evt.extra as any).partial_index) ?? null;

    const shouldRefreshOverlay =
      phase === "WRITE_PARTIAL" || phase === "OVERLAY" || phase === "PROCESS" || phase === "WRITE";

    if (shouldRefreshOverlay) {
      if (typeof partialIdx === "number") {
        if (partialIdx !== lastPartialRef.current) {
          lastPartialRef.current = partialIdx;
          void refreshOverlay();
        }
      } else {
        void refreshOverlay();
      }
    }

    // Final snapshot on terminal phases
    if (phase === "DONE" || phase === "ERROR" || phase === "CANCELLED") {
      void refreshProgress();
      void refreshOverlay();
    }
  }, [last, appendLog, refreshOverlay, refreshProgress]);

  React.useEffect(() => {
    lastPartialRef.current = -1;
  }, [safeRunId]);

  // Header right side info
  const titleRight = React.useMemo(() => {
    if (!selectedInfo) return null;
    return (
      <div className="flex items-center gap-3 text-sm">
        <div className="text-slate-400">SSE:</div>
        <div className="text-slate-300">{sseStatus}</div>
        <div className="text-slate-400">Status:</div>
        <RunStatusBadge status={selectedInfo.status} />
        {selectedInfo.error && (
          <span className="text-rose-300 truncate max-w-[280px]">— {selectedInfo.error}</span>
        )}
      </div>
    );
  }, [selectedInfo, sseStatus]);

  return (
    <div className="min-h-[calc(100vh-56px)] p-4">
      {/* Top heading */}
      <div className="mb-3 flex items-center justify-between">
        <h1 className="text-slate-100 font-semibold text-lg">Run Console</h1>
        {titleRight}
      </div>

      {/* 2-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[360px,1fr] gap-4">
        {/* Left column: Runs */}
        <div className="min-h-[300px]">
          <RunsPanel />
        </div>

        {/* Right column: everything else */}
        <div className="flex flex-col gap-4">
          {/* Upload */}
          <UploadPanel
            onStarted={(rid /*, info */) => {
              setSelectedRunId(rid); // context can also sanitize; this is the source of truth
            }}
          />

          {/* Selected run summary & progress */}
          <div className="rounded-xl border border-slate-700/50 bg-console-700 p-4">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="text-slate-200 font-semibold">Selected</div>
                <div className="text-xs text-slate-400 mt-1">
                  run_id:{" "}
                  <span className="text-slate-300 font-mono">{safeRunId ?? "(none)"}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {safeRunId && (
                  <>
                    <button
                      className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
                      onClick={() => reconnect()}
                      title="Reconnect SSE"
                    >
                      Reconnect SSE
                    </button>
                    <button
                      className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
                      onClick={() => close()}
                      title="Close SSE"
                    >
                      Close SSE
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="mt-3">
              <ProgressBar
                processedCount={progress?.processedCount ?? 0}
                totalTracks={progress?.totalTracks ?? null}
                source={progress?.source ?? "synthesized"}
                size="md"
              />
            </div>
          </div>

          {/* Viewer + Artifacts + Logs */}
          {safeRunId ? (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div className="order-1 xl:order-none">
                <ViewerPanel
                  overlay={overlay}
                  baseImageUrl={baseImageUrl ?? undefined}
                  summary={overlaySummary ?? undefined}
                  options={viewerOptions}
                  onOptionsChange={setViewerOptions}
                  loading={overlayLoading}
                  onRefresh={() => void refreshOverlay()}
                />
              </div>

              <div className="flex flex-col gap-4">
                <ArtifactsPanel artifacts={artifacts} />
                <LiveLogPanel
                  logs={logs}
                  paused={logPaused}
                  setPaused={setLogPaused}
                  onClear={clearLog}
                  onDownload={downloadLog}
                />
              </div>
            </div>
          ) : (
            <EmptyState
              title="No run selected"
              subtitle="Start a new run or choose one from the left."
              className="mt-2"
              actionLabel="Start new run"
              onAction={() => {
                // Could focus the file input in UploadPanel via ref if desired.
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  return (
    <DashboardProvider>
      <DashboardInner />
    </DashboardProvider>
  );
}
