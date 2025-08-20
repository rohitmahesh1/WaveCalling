// frontend/src/pages/Dashboard.tsx
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
import { cancelRun, resumeRun } from "@/utils/api";

/**
 * Configure phases here.
 * - EXCLUDED_PHASES: phases hidden from the log (still allowed to refresh UI).
 * - PROGRESS_REFRESH_PHASES / OVERLAY_REFRESH_PHASES: phases that trigger throttled refreshes.
 */
const EXCLUDED_PHASES = new Set<string>([
  "WRITE_PARTIAL",
  "DISCOVER",
]);

const PROGRESS_REFRESH_PHASES = new Set<string>([
  "KYMO",
  "WRITE_PARTIAL",
  "DISCOVER",
]);

const OVERLAY_REFRESH_PHASES = new Set<string>([
  "WRITE_PARTIAL",
  "OVERLAY",
  "WRITE",
  "DISCOVER",
]);

const RUN_ID_RE = /^[a-f0-9]{6,32}$/i;
const pauseKey = (runId: string) => `log:paused:${runId}`;

function sanitizeRunId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  if (s.startsWith("http://") || s.startsWith("https://")) return null;
  if (s.includes("/")) return null;
  if (!RUN_ID_RE.test(s)) return null;
  return s;
}

// throttle helper
function useThrottle(fn: () => void, minMs: number) {
  const lastRef = React.useRef(0);
  return React.useCallback(() => {
    const now = Date.now();
    if (now - lastRef.current >= minMs) {
      lastRef.current = now;
      fn();
    }
  }, [fn, minMs]);
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

  const safeRunId = React.useMemo(() => sanitizeRunId(selectedRunId), [selectedRunId]);

  // Toggle SSE on/off to prevent reconnect after terminal
  const [sseEnabled, setSseEnabled] = React.useState(true);

  const sseUrl = React.useMemo(
    () =>
      sseEnabled && safeRunId
        ? `${apiBase}/api/runs/${encodeURIComponent(safeRunId)}/events?replay=10`
        : null,
    [apiBase, safeRunId, sseEnabled]
  );

  // --- Paused state: persisted per run, with smart default ---
  const [logPaused, _setLogPaused] = React.useState<boolean>(true); // default true until we decide
  // Wrap setter to persist for current run id
  const setLogPaused = React.useCallback(
    (v: boolean) => {
      _setLogPaused(v);
      if (safeRunId) {
        try {
          window.localStorage.setItem(pauseKey(safeRunId), v ? "1" : "0");
        } catch {}
      }
    },
    [safeRunId]
  );

  // On run change or status change, initialize paused:
  // 1) use saved value if present
  // 2) else derive from status (RUNNING/QUEUED => not paused, otherwise paused)
  React.useEffect(() => {
    if (!safeRunId) return;
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(pauseKey(safeRunId));
    } catch {}
    if (stored !== null) {
      _setLogPaused(stored === "1");
      return;
    }
    const status = selectedInfo?.status;
    const active = status === "RUNNING" || status === "QUEUED";
    _setLogPaused(!active);
  }, [safeRunId, selectedInfo?.status]);

  const { status: sseStatus, last, reconnect, close } = useSSE<JobEvent>({
    url: sseUrl,
    withCredentials: false,
    autoReconnect: true,
    reconnectDelayMs: 800,
    maxReconnectDelayMs: 10_000,
  });

  React.useEffect(() => {
    if (!safeRunId) return;
    // re-enable SSE for the new selection
    setSseEnabled(true);
    void refreshProgress();
    void refreshOverlay();
  }, [safeRunId, refreshProgress, refreshOverlay]);

  const refreshOverlayThrottled = useThrottle(() => void refreshOverlay(), 1500);
  const refreshProgressThrottled = useThrottle(() => void refreshProgress(), 4000);

  // de-dupe & terminal guards
  const lastLogRef = React.useRef<string>("");
  const lastProcessLogAtRef = React.useRef<number>(0);
  const terminalSeenRef = React.useRef<boolean>(false);

  React.useEffect(() => {
    if (!last || !last.data) return;
    const evt = last.data as JobEvent;
    const phase = evt.phase ?? "(event)";
    const isTerminal =
      phase === "DONE" || phase === "ERROR" || phase === "CANCELLED";
    const msgText = `[${phase}] ${evt.message ?? ""}`.trim();

    // Terminal handling once: stop SSE reconnection + log only the main line
    if (isTerminal) {
      if (!terminalSeenRef.current) {
        terminalSeenRef.current = true;
        if (msgText && msgText !== lastLogRef.current) {
          appendLog(msgText);
          lastLogRef.current = msgText;
        }
        // stop SSE (prevents replay of terminal on reconnect)
        setSseEnabled(false);
        close();
        // one last snapshot (throttled)
        refreshProgressThrottled();
        refreshOverlayThrottled();
      }
      return; // ignore extras for terminal
    }

    const excluded = EXCLUDED_PHASES.has(phase);

    // If excluded, skip logging but still do throttled UI refreshes if configured
    if (excluded) {
      if (PROGRESS_REFRESH_PHASES.has(phase)) refreshProgressThrottled();
      if (OVERLAY_REFRESH_PHASES.has(phase)) refreshOverlayThrottled();
      return;
    }

    // Normal logging (with rate-limit for PROCESS)
    if (phase === "PROCESS") {
      const now = Date.now();
      if (now - lastProcessLogAtRef.current >= 1500) {
        if (msgText && msgText !== lastLogRef.current) {
          appendLog(msgText);
          lastLogRef.current = msgText;
        }
        lastProcessLogAtRef.current = now;
      }
    } else {
      if (msgText && msgText !== lastLogRef.current) {
        appendLog(msgText);
        lastLogRef.current = msgText;
      }
    }

    // Extras (only when not excluded)
    if (evt.extra && typeof evt.extra === "object") {
      for (const [k, v] of Object.entries(evt.extra)) {
        if (typeof v === "string" && v.startsWith("/runs/")) {
          const urlLine = `[${phase}] ${k}: ${v}`;
          if (urlLine !== lastLogRef.current) {
            appendLog(urlLine);
            lastLogRef.current = urlLine;
          }
        }
      }
    }

    if (PROGRESS_REFRESH_PHASES.has(phase)) refreshProgressThrottled();
    if (OVERLAY_REFRESH_PHASES.has(phase)) refreshOverlayThrottled();
  }, [
    last,
    appendLog,
    close,
    refreshOverlayThrottled,
    refreshProgressThrottled,
  ]);

  // Reset guards when run changes
  React.useEffect(() => {
    lastProcessLogAtRef.current = 0;
    lastLogRef.current = "";
    terminalSeenRef.current = false;
  }, [safeRunId]);

  const titleRight = React.useMemo(() => {
    if (!selectedInfo) return null;
    return (
      <div className="flex items-center gap-3 text-sm">
        <div className="text-slate-400">SSE:</div>
        <div className="text-slate-300">{sseStatus}</div>
        <div className="text-slate-400">Status:</div>
        <RunStatusBadge status={selectedInfo.status} />
        {selectedInfo.error && (
          <span className="text-rose-300 truncate max-w-[280px]">
            — {selectedInfo.error}
          </span>
        )}
      </div>
    );
  }, [selectedInfo, sseStatus]);

  return (
    <div className="min-h-[calc(100vh-56px)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h1 className="text-slate-100 font-semibold text-lg">Run Console</h1>
        {titleRight}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[360px,1fr] gap-4">
        <div className="min-h-[300px]">
          <RunsPanel />
        </div>

        <div className="flex flex-col gap-4">
          <UploadPanel onStarted={(rid) => setSelectedRunId(rid)} />

          <div className="rounded-xl border border-slate-700/50 bg-console-700 p-4">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="text-slate-200 font-semibold">Selected</div>
                <div className="text-xs text-slate-400 mt-1">
                  run_id:{" "}
                  <span className="text-slate-300 font-mono">
                    {safeRunId ?? "(none)"}
                  </span>
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
                  setPaused={setLogPaused} // persists per run
                  onClear={clearLog}
                  onDownload={downloadLog}
                  onPauseCancel={async () => {
                    if (!safeRunId) return;
                    try {
                      await cancelRun(apiBase, safeRunId);
                      appendLog(`[CANCEL] requested ${safeRunId}`);
                    } catch (e: any) {
                      appendLog(`[CANCEL] error: ${String(e)}`);
                    }
                  }}
                  onResume={async () => {
                    if (!safeRunId) return;
                    try {
                      await resumeRun(apiBase, safeRunId);
                      appendLog(`[RESUME] requested ${safeRunId}`);
                    } catch (e: any) {
                      appendLog(`[RESUME] error: ${String(e)}`);
                    }
                  }}
                />
              </div>
            </div>
          ) : (
            <EmptyState
              title="No run selected"
              subtitle="Start a new run or choose one from the left."
              className="mt-2"
              actionLabel="Start new run"
              onAction={() => {}}
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
