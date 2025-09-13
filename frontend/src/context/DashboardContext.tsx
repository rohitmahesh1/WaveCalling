// frontend/src/context/DashboardContext.tsx
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { getRun } from "@/utils/api";
import type {
  ArtifactMap,
  OverlayPayload,
  ProgressResponse,
  RunInfo,
} from "@/utils/types";
import { useOverlay } from "@/hooks/useOverlay";
import { useProgress } from "@/hooks/useProgress";
import type { ViewerOptions } from "@/components/ViewerToolbar";

const RUN_ID_RE = /^[a-f0-9]{6,32}$/i;
const LS_LAST_RUN = "wc:lastRunId";
const LS_LOG_PREFIX = "wc:logs:"; // per-run logs

// Normalize to lower-case so equality checks (and URLs) are consistent. // <<<
function sanitizeRunId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  if (s.startsWith("http://") || s.startsWith("https://")) return null;
  if (s.includes("/")) return null;
  if (!RUN_ID_RE.test(s)) return null;
  return s.toLowerCase(); // <<<
}

type Ctx = {
  selectedRunId: string | null;
  safeRunId: string | null;
  setSelectedRunId: (id: string | null) => void;

  selectedInfo: RunInfo | null;
  setSelectedInfo: (r: RunInfo | null) => void;

  /** True when selectedInfo matches safeRunId — safe to open per-run SSE. */
  verified: boolean;

  artifacts: ArtifactMap | null;
  setArtifacts: (a: ArtifactMap | null) => void;

  overlay: OverlayPayload | null;
  overlaySummary: { tracks: number; points: number } | null;
  overlayLoading: boolean;

  /** Same-origin base image URL (/runs/:id/output/base.png) or null. */
  baseImageUrl: string | null;

  /** Same-origin debug image URL (/runs/:id/output/<layer>.png) or null. */
  debugImageUrl: string | null;

  /** Returns true iff the overlay actually changed (cheap signature check). */
  refreshOverlay: (opts?: { force?: boolean }) => Promise<boolean>;

  progress: ProgressResponse | null;
  refreshProgress: () => Promise<void>;

  logs: string[];
  appendLog: (line: string) => void;
  clearLog: () => void;
  downloadLog: () => void;

  viewerOptions: ViewerOptions;
  setViewerOptions: (
    partial:
      | Partial<ViewerOptions>
      | ((prev: ViewerOptions) => Partial<ViewerOptions>)
  ) => void;
};

const DashboardContext = React.createContext<Ctx | null>(null);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const apiBase = useApiBase();

  // --- selection (with sanitize + localStorage heal) ---
  const [selectedRunId, _setSelectedRunId] = React.useState<string | null>(() => {
    let stored: string | null = null;
    try { stored = window.localStorage.getItem(LS_LAST_RUN); } catch {}
    const safe = sanitizeRunId(stored);
    if (stored && !safe) {
      try { window.localStorage.removeItem(LS_LAST_RUN); } catch {}
    }
    return safe;
  });

  const safeRunId = React.useMemo(() => sanitizeRunId(selectedRunId), [selectedRunId]);

  const setSelectedRunId = React.useCallback((id: string | null) => {
    const safe = sanitizeRunId(id);
    _setSelectedRunId(safe);
    try {
      if (safe) window.localStorage.setItem(LS_LAST_RUN, safe);
      else window.localStorage.removeItem(LS_LAST_RUN);
    } catch {}
  }, []);

  // --- logs (per run) ---
  const [logs, setLogs] = React.useState<string[]>([]);
  const maxLogLines = 1500;

  React.useEffect(() => {
    if (!safeRunId) { setLogs([]); return; }
    const key = `${LS_LOG_PREFIX}${safeRunId}`;
    let raw: string | null = null;
    try { raw = window.localStorage.getItem(key); } catch {}
    setLogs(raw ? JSON.parse(raw) : []);
  }, [safeRunId]);

  const persistLogs = React.useCallback((next: string[]) => {
    if (!safeRunId) return;
    const key = `${LS_LOG_PREFIX}${safeRunId}`;
    try { window.localStorage.setItem(key, JSON.stringify(next)); } catch {}
  }, [safeRunId]);

  const appendLog = React.useCallback((line: string) => {
    setLogs((prev) => {
      const next = [...prev, line].slice(-maxLogLines);
      persistLogs(next);
      return next;
    });
  }, [persistLogs]);

  const clearLog = React.useCallback(() => {
    setLogs([]);
    if (safeRunId) {
      try { window.localStorage.removeItem(`${LS_LOG_PREFIX}${safeRunId}`); } catch {}
    }
  }, [safeRunId]);

  const downloadLog = React.useCallback(() => {
    const blob = new Blob([logs.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const name = safeRunId ? `run_${safeRunId}_log.txt` : "log.txt";
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }, [logs, safeRunId]);

  // --- selected run info & artifacts (refreshed alongside overlay fetch) ---
  const [selectedInfo, setSelectedInfo] = React.useState<RunInfo | null>(null);
  const [artifacts, setArtifacts] = React.useState<ArtifactMap | null>(null);

  const refreshInfo = React.useCallback(async () => {
    if (!safeRunId) {
      setSelectedInfo(null);
      setArtifacts(null);
      return;
    }
    try {
      const rs = await getRun(apiBase, safeRunId);
      setSelectedInfo(rs.info);
      setArtifacts(rs.artifacts);
    } catch (err: any) {
      const isAbort =
        err?.name === "AbortError" || /aborted/i.test(String(err?.message));
      const status = err?.status ?? err?.response?.status ?? err?.code;

      if (isAbort) {
        if (import.meta.env.MODE !== "production") {
          console.warn("[DC] getRun aborted — keeping selection");
        }
        // Do NOT clear anything on aborts; just bail out.
        return;
      }

      if (status === 404) {
        if (import.meta.env.MODE !== "production") {
          console.warn("[DC] getRun 404 — clearing selection");
        }
        setSelectedInfo(null);
        setArtifacts(null);
        try { localStorage.removeItem(LS_LAST_RUN); } catch {}
        setSelectedRunId(null);
        return;
      }

      // Any other transient error: keep the selection; just drop info/artifacts for now.
      if (import.meta.env.MODE !== "production") {
        console.warn("[DC] getRun failed (transient) — keeping selection", err);
      }
      setSelectedInfo(null);
      setArtifacts(null);
    }
  }, [apiBase, safeRunId, setSelectedRunId]);

  React.useEffect(() => { void refreshInfo(); }, [refreshInfo]);

  // --- viewer options (include debugLayer; blind-merge setter) ---
  const [viewerOptions, _setViewerOptions] = React.useState<ViewerOptions>({
    timeDirection: "down",
    colorBy: "none",
    showBase: true,
    debugLayer: "none",
  });

  const setViewerOptions = React.useCallback((
    partial:
      | Partial<ViewerOptions>
      | ((prev: ViewerOptions) => Partial<ViewerOptions>)
  ) => {
    _setViewerOptions((prev) => {
      const patch = typeof partial === "function" ? partial(prev) : partial;
      return { ...prev, ...patch };
    });
  }, []);

  // --- overlay/progress hooks driven by *safeRunId* ---
  const {
    overlay,
    baseImageUrl,      // `/runs/:id/output/base.png`
    overlayLoading,
    overlaySummary,
    refreshOverlay: refreshOverlayRaw,
  } = useOverlay(safeRunId);

  const { progress, refreshProgress } = useProgress(safeRunId);

  // When overlay actually changes, refresh info/artifacts (cheap + in-memory)
  const refreshOverlay = React.useCallback(async (opts?: { force?: boolean }) => {
    const changed = await refreshOverlayRaw(opts);
    if (changed) {
      try { await refreshInfo(); } catch {}
    }
    return changed;
  }, [refreshOverlayRaw, refreshInfo]);

  // Single source of truth for debug image URL
  const debugImageUrl = React.useMemo(() => {
    const dl = viewerOptions.debugLayer;
    if (!safeRunId || !dl || dl === "none") return null;
    return `/runs/${encodeURIComponent(safeRunId)}/output/${dl}.png`;
  }, [safeRunId, viewerOptions.debugLayer]);

  // Verified flag: case-insensitive match (ids are normalized to lowercase). // <<<
  const verified = React.useMemo(() => {
    const a = selectedInfo?.run_id?.toLowerCase?.();
    const b = safeRunId ?? null;
    return !!a && !!b && a === b;
  }, [selectedInfo, safeRunId]);

  // ===== DEBUG: log state changes (noise-gated) =====
  React.useEffect(() => {
    if (import.meta.env.MODE === "production") return;
    console.log("[DC] run", { safeRunId, verified });
  }, [safeRunId, verified]);

  React.useEffect(() => {
    if (import.meta.env.MODE === "production") return;
    console.log("[DC] viewerOptions", viewerOptions);
  }, [viewerOptions]);

  React.useEffect(() => {
    if (import.meta.env.MODE === "production") return;
    console.log("[DC] overlay", {
      tracks: overlay?.tracks?.length ?? 0,
      summary: overlaySummary,
      baseImageUrl,
      debugImageUrl,
    });
  }, [overlay, overlaySummary, baseImageUrl, debugImageUrl]);

  // ===== DEBUG: HEAD preflight to base/debug images so Network shows activity =====
  React.useEffect(() => {
    if (import.meta.env.MODE === "production") return;

    const head = async (url: string, label: string) => {
      try {
        const r = await fetch(url, { method: "HEAD", cache: "no-store" });
        console.log(`[DC] ${label} HEAD ${r.status}`, url);
      } catch (e) {
        console.warn(`[DC] ${label} HEAD failed`, url, e);
      }
    };

    if (baseImageUrl) head(`${baseImageUrl}?t=${Date.now()}`, "base.png"); // <<<
    if (debugImageUrl) head(`${debugImageUrl}?t=${Date.now()}`, "debug");   // <<<
  }, [baseImageUrl, debugImageUrl]);

  const value: Ctx = {
    selectedRunId,
    safeRunId,
    setSelectedRunId,

    selectedInfo,
    setSelectedInfo,
    verified,

    artifacts,
    setArtifacts,

    overlay,
    overlaySummary,
    overlayLoading,
    baseImageUrl,
    debugImageUrl,
    refreshOverlay,

    progress,
    refreshProgress,

    logs,
    appendLog,
    clearLog,
    downloadLog,

    viewerOptions,
    setViewerOptions,
  };

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard(): Ctx {
  const ctx = React.useContext(DashboardContext);
  if (!ctx) throw new Error("useDashboard must be used within DashboardProvider");
  return ctx;
}
