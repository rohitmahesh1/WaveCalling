// frontend/src/pages/AdvancedViewerPage.tsx
import * as React from "react";
import { useNavigate } from "react-router-dom";

import OverlayCanvas from "@/components/OverlayCanvas";
import WaveMiniView from "@/components/viewer/WaveMiniView";
import CollapsiblePane from "@/components/ui/CollapsiblePane";
import AdvancedSidePanel from "@/components/viewer/AdvancedSidePanel";

import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { useSSE, type BackendEvent } from "@/hooks/useSSE";

import { useTrackQuery } from "@/hooks/useTrackQuery";
import { useColorRules } from "@/hooks/useColorRules";
import { useRafThrottle } from "@/hooks/useRafThrottle";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { useWaveMetrics } from "@/hooks/useWaveMetrics";

import { downloadJSON, downloadPNGFromCanvas, downloadCSV } from "@/utils/download";
import type { DebugLayer } from "@/utils/api";

import type { OverlayPayload, OverlayTrack as Track } from "@/utils/types";
import type { FilterPredicate, ColorRule, SortSpec } from "@/utils/viewerTypes";

const PANEL_STORAGE_KEY = "advanced_viewer:pane";
const CANVAS_DOWNLOAD_NAME = "overlay.png";
const FILTERED_JSON_NAME = "filtered_tracks.json";
const FILTERED_CSV_NAME = "filtered_tracks.csv";

export default function AdvancedViewerPage() {
  const navigate = useNavigate();
  const apiBase = useApiBase();

  // Pull everything from context so Advanced stays in lockstep with Dashboard
  const {
    safeRunId,
    verified,
    overlay,
    overlayLoading,
    baseImageUrl,    // /runs/:id/output/base.png
    debugImageUrl,   // /runs/:id/output/<layer>.png or null
    viewerOptions,
    setViewerOptions,
    refreshOverlay,
    artifacts,
  } = useDashboard();

  // ===== DEBUG: log inputs whenever they change =====
  React.useEffect(() => {
    console.log("[AV] inputs", {
      safeRunId,
      verified,
      baseImageUrl,
      debugImageUrl,
      showBase: viewerOptions.showBase,
      debugLayer: (viewerOptions as any)?.debugLayer,
      tracks: overlay?.tracks?.length ?? 0,
    });
  }, [safeRunId, verified, baseImageUrl, debugImageUrl, viewerOptions, overlay]);

  // ---- CSV metrics (augment each track with a `csv` object) ----
  const { rowByTrackId } = useWaveMetrics(safeRunId || undefined, artifacts, { allowGuesses: false });
  const allBaseTracks: Track[] = overlay?.tracks ?? [];
  const allTracksAugmented: (Track & { csv?: Record<string, any> })[] = React.useMemo(
    () =>
      allBaseTracks.map((t) => {
        const csv = rowByTrackId[String(t.id)];
        return csv ? { ...t, csv } : t;
      }),
    [allBaseTracks, rowByTrackId]
  );

  // ---- Persisted viewer state (per-run key) ----
  const lsKey = (suffix: string) => `adv_view:${safeRunId ?? "global"}:${suffix}`;
  const [filters, setFilters] = useLocalStorage<FilterPredicate[]>(lsKey("filters:v1"), []);
  const [sort, setSort] = useLocalStorage<SortSpec[]>(lsKey("sort:v1"), []);
  const [colorRules, setColorRules] = useLocalStorage<ColorRule[]>(lsKey("color:v1"), []);
  const [showCursorCoords, setShowCursorCoords] = useLocalStorage<boolean>(lsKey("cursor:v1"), true);

  React.useEffect(() => {
    if (!safeRunId) {
      setFilters([]);
      setSort([]);
      setColorRules([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeRunId]);

  const [selectedTrackId, setSelectedTrackId] = React.useState<string | number | null>(null);
  const [canvasEl, setCanvasEl] = React.useState<HTMLCanvasElement | null>(null);

  const { filteredTracks } = useTrackQuery(allTracksAugmented as Track[], filters, sort);
  const { colorOverrideFn } = useColorRules(colorRules);

  const filteredPayload: OverlayPayload | null = React.useMemo(() => {
    if (!overlay) return null;
    return { ...overlay, tracks: filteredTracks as Track[] };
  }, [overlay, filteredTracks]);

  React.useEffect(() => {
    if (selectedTrackId == null) return;
    const stillVisible = filteredTracks.some((t) => String(t.id) === String(selectedTrackId));
    if (!stillVisible) setSelectedTrackId(null);
  }, [filteredTracks, selectedTrackId]);

  const stats = React.useMemo(() => {
    if (!filteredTracks.length) {
      return { count: 0, points: 0, avgAmplitude: null as number | null, avgFrequency: null as number | null };
    }
    let sumAmp = 0, cntAmp = 0, sumFreq = 0, cntFreq = 0, pts = 0;
    for (const t of filteredTracks) {
      const a = Number(t.metrics?.mean_amplitude);
      const f = Number(t.metrics?.dominant_frequency);
      if (Number.isFinite(a)) { sumAmp += a; cntAmp++; }
      if (Number.isFinite(f)) { sumFreq += f; cntFreq++; }
      pts += Array.isArray(t.poly) ? t.poly.length : 0;
    }
    return {
      count: filteredTracks.length,
      points: pts,
      avgAmplitude: cntAmp ? sumAmp / cntAmp : null,
      avgFrequency: cntFreq ? sumFreq / cntFreq : null,
    };
  }, [filteredTracks]);

  const selectedTrack: Track | null = React.useMemo(() => {
    if (selectedTrackId == null) return null;
    return allBaseTracks.find((t) => String(t.id) === String(selectedTrackId)) ?? null;
  }, [selectedTrackId, allBaseTracks]);

  // ---- Cursor coords tooltip ----
  const wrapperRef = React.useRef<HTMLDivElement | null>(null);
  const [cursorTip, setCursorTip] = React.useState({ visible: false, x: 0, y: 0, dataX: 0, dataY: 0 });
  const updateCursor = useRafThrottle((p: { canvasX: number; canvasY: number; dataX: number; dataY: number }) => {
    const roundedX = Math.round(p.dataX);
    const roundedY = Math.round(p.dataY);
    setCursorTip((prev) => {
      if (
        prev.visible &&
        Math.round(prev.dataX) === roundedX &&
        Math.round(prev.dataY) === roundedY &&
        Math.round(prev.x) === Math.round(p.canvasX + 12) &&
        Math.round(prev.y) === Math.round(p.canvasY + 12)
      ) {
        return prev; // nothing changed; skip update
      }
      return { visible: true, x: p.canvasX + 12, y: p.canvasY + 12, dataX: p.dataX, dataY: p.dataY };
    });
  });
  const clearCursor = React.useCallback(() => setCursorTip((c) => ({ ...c, visible: false })), []);

  // ---- Downloads ----
  const handleDownloadPNG = React.useCallback(() => { if (canvasEl) downloadPNGFromCanvas(canvasEl, CANVAS_DOWNLOAD_NAME); }, [canvasEl]);
  const handleDownloadJSON = React.useCallback(() => { downloadJSON(FILTERED_JSON_NAME, filteredTracks); }, [filteredTracks]);
  const handleDownloadCSV = React.useCallback(() => {
    const rows = filteredTracks.map((t: any) => ({
      id: t.id,
      sample: t.sample,
      points: t.poly?.length ?? 0,
      mean_amplitude: t.metrics?.mean_amplitude ?? "",
      dominant_frequency: t.metrics?.dominant_frequency ?? "",
      num_peaks: t.metrics?.num_peaks ?? "",
      period: t.metrics?.period ?? "",
      csv_waves_count: t.csv?.waves_count ?? "",
      csv_fit_freq_hz_mean: t.csv?.waves_fit_freq_hz_mean ?? "",
      csv_amplitude_pixels_mean: t.csv?.waves_amplitude_pixels_mean ?? "",
    }));
    downloadCSV(FILTERED_CSV_NAME, rows);
  }, [filteredTracks]);

  // ---- OverlayCanvas handlers ----
  const handleClickTrack = React.useCallback((track: Track | null) => {
    setSelectedTrackId(track?.id ?? null);
  }, []);

  const hasOverlay = (overlay?.tracks?.length ?? 0) > 0;

  // Remember last non-"none" debug layer for the toggle UX
  const lastDebugLayerRef = React.useRef<DebugLayer>("prob");
  React.useEffect(() => {
    const dl = (viewerOptions as any)?.debugLayer as DebugLayer | "none" | undefined;
    if (dl && dl !== "none") lastDebugLayerRef.current = dl;
  }, [viewerOptions]);

  // ================== SSE for overlay updates ==================
  const [sseEnabled, setSseEnabled] = React.useState(false);

  // Always do one forced fetch as soon as we know the run id
  React.useEffect(() => {
    if (!safeRunId) return;
    void refreshOverlay({ force: true });
  }, [safeRunId, refreshOverlay]);

  // Only start SSE after the run has been verified to exist
  React.useEffect(() => { setSseEnabled(!!verified); }, [verified]);

  const sseUrl = React.useMemo(
    () => (sseEnabled && safeRunId ? `${apiBase}/api/runs/${encodeURIComponent(safeRunId)}/events?replay=1` : null),
    [apiBase, sseEnabled, safeRunId]
  );

  useSSE<BackendEvent>({
    url: sseUrl,
    autoReconnect: false, // avoid looping on 404 for stale/missing runs
    onStatus: (st) => { if (/^(DONE|ERROR|CANCELLED)$/i.test(st)) setSseEnabled(false); },
    onDirty: async (flags) => { if (flags.overlay) await refreshOverlay(); },
    coalesceWindowMs: 80,
  });
  // =============================================================

  const filtersHideAll =
    hasOverlay && filteredTracks.length === 0 && (filters.length > 0 || sort.length > 0);

  return (
    <div className="min-h-[calc(100vh-56px)] p-4">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="min-w-0">
          <h1 className="text-slate-100 font-semibold text-lg">Advanced Viewer</h1>

          <div className="text-xs text-slate-400 mt-1 flex items-center flex-wrap">
            {/* status / counts */}
            {overlayLoading
              ? "Refreshing overlay…"
              : (overlay?.tracks?.length ?? 0) > 0
              ? `${overlay!.tracks.length.toLocaleString()} tracks`
              : "No overlay loaded"}

            {/* Base/debug links come from context; valid even if no tracks */}
            {viewerOptions.showBase && baseImageUrl && (
              <span className="ml-2">
                base:{" "}
                <a
                  href={baseImageUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="underline decoration-slate-600 hover:text-slate-300"
                >
                  base.png
                </a>
              </span>
            )}

            {(viewerOptions as any)?.debugLayer !== "none" && !!debugImageUrl && (
              <span className="ml-2">
                debug:{" "}
                <a
                  href={debugImageUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="underline decoration-slate-600 hover:text-slate-300"
                >
                  {(viewerOptions as any).debugLayer}.png
                </a>
              </span>
            )}
          </div>

          {/* ===== DEBUG: show raw state for quick visual confirmation ===== */}
          <div className="text-[11px] text-slate-500 mt-1">
            run: <span className="font-mono">{safeRunId || "—"}</span> · verified: {String(verified)}
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <label className="inline-flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={viewerOptions.showBase}
              onChange={(e) => setViewerOptions({ showBase: e.target.checked })}
            />
            Show base
          </label>

          {/* Debug layer toggle + selector */}
          <label className="inline-flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={(viewerOptions as any)?.debugLayer && (viewerOptions as any).debugLayer !== "none"}
              onChange={(e) => {
                const on = e.target.checked;
                if (on) {
                  setViewerOptions({ debugLayer: lastDebugLayerRef.current } as any);
                } else {
                  const dl = (viewerOptions as any)?.debugLayer as DebugLayer | "none" | undefined;
                  if (dl && dl !== "none") lastDebugLayerRef.current = dl;
                  setViewerOptions({ debugLayer: "none" } as any);
                }
              }}
            />
            Debug
          </label>
          <select
            className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
            value={(viewerOptions as any)?.debugLayer ?? "none"}
            onChange={(e) => {
              const dl = e.target.value as DebugLayer | "none";
              setViewerOptions({ debugLayer: dl } as any);
              if (dl !== "none") lastDebugLayerRef.current = dl as DebugLayer;
            }}
          >
            <option value="none">none</option>
            <option value="prob">prob</option>
            <option value="mask_raw">mask_raw</option>
            <option value="mask_clean">mask_clean</option>
            <option value="mask_filtered">mask_filtered</option>
            <option value="skeleton">skeleton</option>
            <option value="mask_hysteresis">mask_hysteresis</option>
          </select>

          <div className="inline-flex items-center gap-1 text-sm">
            <span className="text-slate-300">Time</span>
            <select
              className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600"
              value={viewerOptions.timeDirection}
              onChange={(e) => setViewerOptions({ timeDirection: e.target.value as typeof viewerOptions.timeDirection })}
            >
              <option value="down">↓ down</option>
              <option value="up">↑ up</option>
            </select>
          </div>

          <div className="inline-flex items-center gap-1 text-sm">
            <span className="text-slate-300">Color by</span>
            <select
              className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600"
              value={viewerOptions.colorBy}
              onChange={(e) => setViewerOptions({ colorBy: e.target.value as typeof viewerOptions.colorBy })}
            >
              <option value="none">none</option>
              <option value="dominant_frequency">dominant_frequency</option>
              <option value="amplitude">amplitude</option>
            </select>
          </div>

          <label className="inline-flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showCursorCoords}
              onChange={(e) => setShowCursorCoords(e.target.checked)}
            />
            Coords
          </label>

          <button
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            onClick={() => void refreshOverlay({ force: true })}
            disabled={overlayLoading}
          >
            {overlayLoading ? "Refreshing…" : "Refresh"}
          </button>

          <button
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-300 hover:bg-slate-800"
            onClick={() => navigate(-1)}
          >
            Back
          </button>
        </div>
      </div>

      {/* Main */}
      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr),auto] gap-4 items-start">
        {/* Canvas column */}
        <div ref={wrapperRef} className="relative min-w-0">
          <div className="rounded-xl border border-slate-700/50 bg-console-700 p-3">
            <OverlayCanvas
              payload={filteredPayload as OverlayPayload}
              // Pass URLs unconditionally; OverlayCanvas decides via options whether to draw them
              baseImageUrl={baseImageUrl ?? undefined}
              debugImageUrl={debugImageUrl ?? undefined}
              options={viewerOptions}
              padding={20}
              // ===== DEBUG: force visible size and outline so you can see the canvas box =====
              style={{ height: "min(75vh, 760px)", minHeight: 320, outline: "2px dashed #ff4d4f" }}
              className="block"
              onPointerMove={showCursorCoords ? updateCursor : undefined}
              onPointerLeave={showCursorCoords ? clearCursor : undefined}
              onClickTrack={handleClickTrack}
              highlightTrackId={selectedTrackId ?? undefined}
              colorOverrideFn={colorOverrideFn}
              onCanvasReady={setCanvasEl}
            />

            {/* Cursor tooltip */}
            {showCursorCoords && cursorTip.visible && (
              <div
                className="pointer-events-none absolute z-10 px-2 py-1 rounded bg-slate-900/90 border border-slate-700 text-[11px] text-slate-200 shadow"
                style={{ left: cursorTip.x, top: cursorTip.y }}
              >
                x: {Math.round(cursorTip.dataX)}, y: {Math.round(cursorTip.dataY)}
              </div>
            )}
          </div>

          {/* Selection mini-view */}
          {selectedTrack && (
            <div className="mt-3 rounded-xl border border-slate-700/50 bg-console-700 p-3">
              <div className="mb-2 text-slate-300 text-sm">
                Selected track: <span className="font-mono text-slate-100">{String(selectedTrack.id)}</span>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-[360px,1fr] gap-3">
                <WaveMiniView
                  track={selectedTrack}
                  baseImageUrl={viewerOptions.showBase ? baseImageUrl ?? undefined : undefined}
                  className="rounded-lg border border-slate-700/50 bg-slate-900/50"
                  options={{
                    showBase: viewerOptions.showBase,
                    invertY: viewerOptions.timeDirection === "up",
                    showBaseline: true,
                    showSineFit: true,
                    showAxes: false,
                    showPeaks: true,
                    requestIncludeResidual: false,
                  }}
                />
                <div className="text-xs text-slate-300 leading-6">
                  <div>points: <span className="text-slate-100">{selectedTrack.poly?.length ?? 0}</span></div>
                  <div>mean amplitude: <span className="text-slate-100">{selectedTrack.metrics?.mean_amplitude ?? "—"}</span></div>
                  <div>dominant freq: <span className="text-slate-100">{selectedTrack.metrics?.dominant_frequency ?? "—"}</span></div>
                  {Array.isArray(selectedTrack.peaks) && (
                    <div>peaks: <span className="text-slate-100">{selectedTrack.peaks.length}</span></div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right panel */}
        <CollapsiblePane title="Controls & Stats" storageKey={PANEL_STORAGE_KEY} defaultWidth={600} minWidth={340} maxWidth={900}>
          <div className="overflow-x-auto pr-2">
            <div className="min-w-[520px]">
              <AdvancedSidePanel
                tracks={allTracksAugmented as any}
                onDownloadPNG={handleDownloadPNG}
                onDownloadJSON={handleDownloadJSON}
                onDownloadCSV={handleDownloadCSV}
                filters={filters}
                onFiltersChange={setFilters}
                sort={sort}
                onSortChange={setSort}
                colorRules={colorRules}
                onColorRulesChange={setColorRules}
                filteredStats={{
                  count: stats.count,
                  points: stats.points,
                  avgAmplitude: stats.avgAmplitude,
                  avgFrequency: stats.avgFrequency,
                }}
                selectedTrack={selectedTrack}
              />
            </div>
          </div>
        </CollapsiblePane>
      </div>
    </div>
  );
}
