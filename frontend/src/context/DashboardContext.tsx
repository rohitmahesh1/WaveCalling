// src/context/DashboardContext.tsx
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

function sanitizeRunId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  if (s.startsWith("http://") || s.startsWith("https://")) return null;
  if (s.includes("/")) return null;
  if (!RUN_ID_RE.test(s)) return null;
  return s;
}

type Ctx = {
  selectedRunId: string | null;   // raw value
  safeRunId: string | null;       // sanitized or null
  setSelectedRunId: (id: string | null) => void;

  selectedInfo: RunInfo | null;
  setSelectedInfo: (r: RunInfo | null) => void;

  artifacts: ArtifactMap | null;
  setArtifacts: (a: ArtifactMap | null) => void;

  overlay: OverlayPayload | null;
  overlaySummary: { tracks: number; points: number } | null;
  overlayLoading: boolean;
  baseImageUrl: string | null;
  /** Returns true iff the overlay actually changed (cheap signature check). */
  refreshOverlay: (opts?: { force?: boolean }) => Promise<boolean>;

  progress: ProgressResponse | null;
  refreshProgress: () => Promise<void>;

  logs: string[];
  appendLog: (line: string) => void;
  clearLog: () => void;
  downloadLog: () => void;

  viewerOptions: ViewerOptions;
  setViewerOptions: (partial: Partial<ViewerOptions> | ViewerOptions) => void;
};

const DashboardContext = React.createContext<Ctx | null>(null);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const apiBase = useApiBase();

  // --- selection (with sanitize + localStorage heal) ---
  const [selectedRunId, _setSelectedRunId] = React.useState<string | null>(() => {
    const stored = localStorage.getItem(LS_LAST_RUN);
    const safe = sanitizeRunId(stored);
    if (stored && !safe) {
      // purge bad leftover (e.g., "http://localhost:5173")
      localStorage.removeItem(LS_LAST_RUN);
    }
    return safe;
  });
  const safeRunId = React.useMemo(() => sanitizeRunId(selectedRunId), [selectedRunId]);

  const setSelectedRunId = React.useCallback((id: string | null) => {
    const safe = sanitizeRunId(id);
    _setSelectedRunId(safe);
    if (safe) localStorage.setItem(LS_LAST_RUN, safe);
    else localStorage.removeItem(LS_LAST_RUN);
  }, []);

  // --- logs (per run) ---
  const [logs, setLogs] = React.useState<string[]>([]);
  const maxLogLines = 1500;

  // load logs when safeRunId changes
  React.useEffect(() => {
    if (!safeRunId) {
      setLogs([]);
      return;
    }
    const key = `${LS_LOG_PREFIX}${safeRunId}`;
    const raw = localStorage.getItem(key);
    setLogs(raw ? JSON.parse(raw) : []);
  }, [safeRunId]);

  const persistLogs = React.useCallback(
    (next: string[]) => {
      if (!safeRunId) return;
      const key = `${LS_LOG_PREFIX}${safeRunId}`;
      localStorage.setItem(key, JSON.stringify(next));
    },
    [safeRunId]
  );

  const appendLog = React.useCallback(
    (line: string) => {
      setLogs((prev) => {
        const next = [...prev, line].slice(-maxLogLines);
        persistLogs(next);
        return next;
      });
    },
    [persistLogs]
  );

  const clearLog = React.useCallback(() => {
    setLogs([]);
    if (safeRunId) localStorage.removeItem(`${LS_LOG_PREFIX}${safeRunId}`);
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
    } catch {
      // ignore transient failures
    }
  }, [apiBase, safeRunId]);

  React.useEffect(() => {
    void refreshInfo();
  }, [refreshInfo]);

  // --- viewer options ---
  const [viewerOptions, _setViewerOptions] = React.useState<ViewerOptions>({
    timeDirection: "up",
    colorBy: "none",
    showBase: true,
  });
  const setViewerOptions = React.useCallback(
    (partial: Partial<ViewerOptions> | ViewerOptions) => {
      _setViewerOptions((prev) =>
        typeof partial === "function" ? (partial as any)(prev) : { ...prev, ...partial }
      );
    },
    []
  );

  // --- overlay/progress hooks driven by *safeRunId* ---
  const {
    overlay,
    baseImageUrl,
    overlayLoading,
    overlaySummary,
    // NOTE: hook returns Promise<boolean> (true = overlay changed)
    refreshOverlay: refreshOverlayRaw,
  } = useOverlay(safeRunId);

  const { progress, refreshProgress } = useProgress(safeRunId);

  // When overlay refresh *changes*, also refresh info/artifacts (cheap and in-memory)
  const refreshOverlay = React.useCallback(async (opts?: { force?: boolean }) => {
    const changed = await refreshOverlayRaw(opts);
    if (changed) {
      try { await refreshInfo(); } catch {}
    }
    return changed;
  }, [refreshOverlayRaw, refreshInfo]);

  const value: Ctx = {
    selectedRunId,
    safeRunId,
    setSelectedRunId,

    selectedInfo,
    setSelectedInfo,

    artifacts,
    setArtifacts,

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
  };

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard(): Ctx {
  const ctx = React.useContext(DashboardContext);
  if (!ctx) throw new Error("useDashboard must be used within DashboardProvider");
  return ctx;
}
