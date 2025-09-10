// frontend/src/components/viewer/WaveMiniView.tsx
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import type { OverlayTrack as Track } from "@/utils/types";

export type WaveMiniViewOptions = {
  showAxes?: boolean;
  showBaseline?: boolean;
  showSineFit?: boolean;
  showPeaks?: boolean;
  showBase?: boolean;
  invertY?: boolean;
  requestIncludeResidual?: boolean;
  initialWindow?: "full" | "peak";
  initialWindowPts?: number;
  baselineDegree?: number;
};

type Props = {
  track: Track;
  className?: string;
  baseImageUrl?: string;
  options?: WaveMiniViewOptions;
};

type TrackDetailResponse = {
  id: string | number;
  time_index?: number[];
  baseline?: number[];  // x per index
  residual?: number[];
  sine_fit?: number[];  // baseline + phase-anchored sine (x per index)
  regression?: { method?: string; degree?: number; params?: Record<string, any> };
};

const MIN_H = 180;
const MAX_H = 520;

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function niceTicks(min: number, max: number, maxTicks: number): number[] {
  if (!isFinite(min) || !isFinite(max) || max <= min) return [min, max];
  const range = max - min;
  const rough = range / Math.max(1, maxTicks);
  const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
  const fr = rough / pow10;
  const step =
    fr < 1.5 ? 1 * pow10 :
    fr < 3   ? 2 * pow10 :
    fr < 7   ? 5 * pow10 :
               10 * pow10;
  const t0 = Math.ceil(min / step) * step;
  const arr: number[] = [];
  for (let v = t0; v <= max + 1e-9; v += step) arr.push(v);
  return arr.slice(0, Math.max(2, maxTicks + 1));
}

export default function WaveMiniView({ track, className, baseImageUrl: baseUrlProp, options }: Props) {
  const apiBase = useApiBase();
  const { selectedRunId, viewerOptions, baseImageUrl: baseUrlCtx } = useDashboard();

  const resolvedBaseUrl = baseUrlProp ?? baseUrlCtx ?? undefined;
  const resolvedInvertY = options?.invertY ?? (viewerOptions.timeDirection === "up");

  // ---- Persisted UI state ----
  type Prefs = {
    showAxes: boolean;
    showBaseline: boolean;
    showSineFit: boolean;
    showPeaks: boolean;
    showBase: boolean;
    mode: "full" | "peak";
    windowPts: number;
  };
  const [prefs, setPrefs] = useLocalStorage<Prefs>("wmv:prefs:v1", {
    showAxes: options?.showAxes ?? true,
    showBaseline: options?.showBaseline ?? true,
    showSineFit: options?.showSineFit ?? false,
    showPeaks: options?.showPeaks ?? true,
    showBase: options?.showBase ?? false,
    mode: options?.initialWindow ?? "peak",
    windowPts: options?.initialWindowPts ?? 120,
  });

  const showAxes = prefs.showAxes;
  const showBaseline = prefs.showBaseline;
  const showSineFit = prefs.showSineFit;
  const showPeaks = prefs.showPeaks;
  const showBase = prefs.showBase;
  const mode = prefs.mode;
  const windowPts = prefs.windowPts;

  // Canvas + image + auto height
  const wrapRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = React.useState<HTMLImageElement | null>(null);
  const [autoHeight, setAutoHeight] = React.useState<number>(260);

  // Detail (baseline/sine) fetched on demand
  const [detail, setDetail] = React.useState<TrackDetailResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = React.useState(false);

  // Load / clear base image
  React.useEffect(() => {
    let cancelled = false;
    if (!showBase || !resolvedBaseUrl) { setImg(null); return () => { cancelled = true; }; }
    const im = new Image();
    im.crossOrigin = "anonymous";
    const u = new URL(resolvedBaseUrl, window.location.origin);
    u.searchParams.set("t", Date.now().toString());
    im.onload = () => { if (!cancelled) setImg(im); };
    im.onerror = () => { if (!cancelled) setImg(null); };
    im.src = u.toString();
    return () => { cancelled = true; };
  }, [showBase, resolvedBaseUrl]);

  // Compute current window bounds (used for both drawing + auto height)
  const windowBounds = React.useMemo(() => {
    const pts = Array.isArray(track.poly) ? track.poly : [];
    if (pts.length < 2) return null;

    let lo = 0, hi = pts.length - 1;
    if (mode === "peak") {
      const peaks = Array.isArray(track.peaks) ? track.peaks : [];
      const center = peaks?.length ? Math.max(0, Math.min(pts.length - 1, peaks[0])) : Math.floor(pts.length / 2);
      const half = Math.max(5, Math.floor(windowPts / 2));
      lo = Math.max(0, center - half);
      hi = Math.min(pts.length - 1, center + half);
    }

    let minX = +Infinity, maxX = -Infinity, minY = +Infinity, maxY = -Infinity;
    for (let i = lo; i <= hi; i++) {
      const y = pts[i][0];
      const x = pts[i][1];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }

    if (showBase && img) {
      minX = Math.max(0, minX);
      minY = Math.max(0, minY);
      maxX = Math.min(img.naturalWidth,  Math.max(maxX, minX + 1));
      maxY = Math.min(img.naturalHeight, Math.max(maxY, minY + 1));
    }

    if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) return null;
    return { lo, hi, minX, maxX, minY, maxY };
  }, [track, mode, windowPts, showBase, img]);

  // Auto-height from aspect ratio
  React.useEffect(() => {
    const w = wrapRef.current?.clientWidth ?? 360;
    if (!windowBounds) {
      setAutoHeight(260);
      return;
    }
    const spanX = Math.max(1, windowBounds.maxX - windowBounds.minX);
    const spanY = Math.max(1, windowBounds.maxY - windowBounds.minY);
    const ar = spanX / spanY; // width / height
    const h = clamp(Math.round(w / Math.max(0.001, ar)), MIN_H, MAX_H);
    setAutoHeight(h);
  }, [windowBounds]);

  // Fetch baseline/sine/residual as needed; endpoint matches refactored FastAPI
  React.useEffect(() => {
    const need = showBaseline || showSineFit || options?.requestIncludeResidual;
    if (!need) { setDetail(null); return; }
    if (!selectedRunId || track?.id == null) return;

    let cancelled = false;
    (async () => {
      try {
        setLoadingDetail(true);
        const params = new URLSearchParams();
        if (showSineFit) params.set("include_sine", "1");
        if (options?.requestIncludeResidual) params.set("include_residual", "1");
        const qs = params.toString() ? `?${params.toString()}` : "";

        const r = await fetch(
          `${apiBase}/api/runs/${encodeURIComponent(String(selectedRunId))}/tracks/${encodeURIComponent(String(track.id))}${qs}`,
          { cache: "no-cache" }
        );
        if (!r.ok) throw new Error(`detail ${r.status}`);
        const data = (await r.json()) as TrackDetailResponse;
        if (!cancelled) setDetail(data);
      } catch {
        if (!cancelled) setDetail(null);
      } finally {
        if (!cancelled) setLoadingDetail(false);
      }
    })();
    return () => { cancelled = true; };
  }, [apiBase, selectedRunId, track?.id, showBaseline, showSineFit, options?.requestIncludeResidual]);

  // ----- Drawing -----
  const draw = React.useCallback(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapRef.current;
    if (!canvas || !wrapper || !track || !windowBounds) return;

    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
    const cssW = Math.max(240, wrapper.clientWidth);
    const cssH = Math.max(MIN_H, wrapper.clientHeight);
    const w = Math.floor(cssW * dpr);
    const h = Math.floor(cssH * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    canvas.style.width = `${cssW}px`; canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, cssW, cssH);

    const pts = Array.isArray(track.poly) ? track.poly : [];
    if (pts.length < 2) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("No data", 10, 18);
      ctx.restore(); return;
    }

    const { lo, hi, minX, maxX, minY, maxY } = windowBounds;
    const padding = 16;
    const drawW = Math.max(1, cssW - 2 * padding);
    const drawH = Math.max(1, cssH - 2 * padding);

    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);

    // Aspect-ratio preserving letterbox
    const sx = drawW / spanX;
    const sy = drawH / spanY;
    const scale = Math.min(sx, sy);
    const effW = spanX * scale;
    const effH = spanY * scale;
    const offX = padding + (drawW - effW) / 2;
    const offY = padding + (drawH - effH) / 2;

    const toCanvas = (x: number, y: number) => {
      const yNorm = resolvedInvertY ? maxY - y : y - minY;
      return { cx: offX + (x - minX) * scale, cy: offY + yNorm * scale };
    };

    // Base image
    if (showBase && img) {
      ctx.save();
      ctx.translate(offX, offY);
      ctx.scale(scale, scale);
      if (resolvedInvertY) { ctx.translate(-minX, -maxY); ctx.scale(1, -1); }
      else { ctx.translate(-minX, -minY); }
      ctx.globalAlpha = 0.65;
      ctx.drawImage(img, 0, 0);
      ctx.globalAlpha = 1;
      ctx.restore();
    }

    // Raw track
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "rgba(80, 200, 120, 0.95)";
    ctx.beginPath();
    const p0 = toCanvas(pts[lo][1], pts[lo][0]);
    ctx.moveTo(p0.cx, p0.cy);
    for (let i = lo + 1; i <= hi; i++) {
      const p = toCanvas(pts[i][1], pts[i][0]);
      ctx.lineTo(p.cx, p.cy);
    }
    ctx.stroke();

    // Peaks
    if (showPeaks && Array.isArray(track.peaks) && track.peaks.length) {
      ctx.fillStyle = "rgba(255, 120, 120, 0.95)";
      for (const idx of track.peaks) {
        if (idx < lo || idx > hi) continue;
        const c = toCanvas(pts[idx][1], pts[idx][0]);
        ctx.beginPath();
        ctx.arc(c.cx, c.cy, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Overlays
    if ((showBaseline || showSineFit) && detail) {
      const series = showSineFit && detail.sine_fit ? detail.sine_fit : detail.baseline;
      if (Array.isArray(series) && series.length >= pts.length) {
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = showSineFit ? "rgba(80,160,255,0.95)" : "rgba(255,210,90,0.95)";
        ctx.beginPath();
        const c0 = toCanvas(series[lo], pts[lo][0]);
        ctx.moveTo(c0.cx, c0.cy);
        for (let i = lo + 1; i <= hi; i++) {
          const c = toCanvas(series[i], pts[i][0]);
          ctx.lineTo(c.cx, c.cy);
        }
        ctx.stroke();
      }
    }

    // Axes (nice ticks, outside the box)
    if (showAxes) {
      ctx.strokeStyle = "#344053";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, cssW - 1, cssH - 1);

      ctx.fillStyle = "#9aa4bf";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

      const maxXTicks = clamp(Math.floor(effW / 80), 2, 8);
      const maxYTicks = clamp(Math.floor(effH / 60), 2, 8);
      const xt = niceTicks(minX, maxX, maxXTicks);
      const yt = niceTicks(minY, maxY, maxYTicks);

      // X ticks along bottom of effective box
      const baseY = offY + effH;
      for (const tx of xt) {
        const cx = offX + (tx - minX) * scale;
        ctx.beginPath(); ctx.moveTo(cx, baseY); ctx.lineTo(cx, baseY + 4); ctx.stroke();
        const s = Math.round(tx).toString();
        const tw = ctx.measureText(s).width;
        ctx.fillText(s, cx - tw / 2, baseY + 14);
      }

      // Y ticks along left of effective box
      for (const ty of yt) {
        const cy = offY + (resolvedInvertY ? (maxY - ty) * scale : (ty - minY) * scale);
        ctx.beginPath(); ctx.moveTo(offX - 4, cy); ctx.lineTo(offX, cy); ctx.stroke();
        const s = Math.round(ty).toString();
        const tw = ctx.measureText(s).width;
        ctx.fillText(s, Math.max(4, offX - 6 - tw), cy - 2);
      }
    }

    ctx.restore();
  }, [track, windowBounds, img, showBase, showAxes, showBaseline, showSineFit, showPeaks, detail, resolvedInvertY]);

  // Redraw on size / deps
  React.useEffect(() => {
    draw();
    const wrapper = wrapRef.current;
    if (!wrapper) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(wrapper);
    return () => ro.disconnect();
  }, [draw]);

  React.useEffect(() => { draw(); }, [detail, img, draw]);

  return (
    <div className={className}>
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <div className="inline-flex items-center gap-2 text-xs text-slate-300">
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showAxes}
              onChange={(e) => setPrefs(p => ({ ...p, showAxes: e.target.checked }))}
            />
            Axes
          </label>
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showBaseline}
              onChange={(e) => setPrefs(p => ({ ...p, showBaseline: e.target.checked }))}
            />
            Regression
          </label>
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showSineFit}
              onChange={(e) => setPrefs(p => ({ ...p, showSineFit: e.target.checked }))}
            />
            Sine fit
          </label>
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showPeaks}
              onChange={(e) => setPrefs(p => ({ ...p, showPeaks: e.target.checked }))}
            />
            Peaks
          </label>
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showBase}
              onChange={(e) => setPrefs(p => ({ ...p, showBase: e.target.checked }))}
            />
            Base img
          </label>
        </div>

        <div className="ml-auto inline-flex items-center gap-2 text-xs">
          <select
            className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600"
            value={mode}
            onChange={(e) => setPrefs(p => ({ ...p, mode: e.target.value as "full" | "peak" }))}
          >
            <option value="peak">around peak</option>
            <option value="full">full track</option>
          </select>
          {mode === "peak" && (
            <>
              <span className="text-slate-400">± pts</span>
              <input
                type="number"
                className="w-20 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600"
                value={windowPts}
                min={20}
                step={10}
                onChange={(e) =>
                  setPrefs(p => ({ ...p, windowPts: Math.max(20, Number(e.target.value) || 120) }))
                }
              />
            </>
          )}
        </div>
      </div>

      {/* Canvas wrapper — height auto from aspect ratio */}
      <div
        ref={wrapRef}
        className="w-full rounded-lg border border-slate-700/50 bg-slate-950/50"
        style={{ height: `${autoHeight}px` }}
      >
        <canvas ref={canvasRef} className="block w-full h-full" />
      </div>

      {/* Metrics */}
      <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-300">
        <div><div className="text-slate-400">points</div><div className="text-slate-100">{Array.isArray(track.poly) ? track.poly.length : 0}</div></div>
        <div><div className="text-slate-400">peaks</div><div className="text-slate-100">{Array.isArray(track.peaks) ? track.peaks.length : 0}</div></div>
        <div><div className="text-slate-400">mean amplitude</div><div className="text-slate-100">{track.metrics?.mean_amplitude ?? "—"}</div></div>
        <div><div className="text-slate-400">dominant freq (Hz)</div><div className="text-slate-100">{track.metrics?.dominant_frequency ?? "—"}</div></div>
      </div>

      {loadingDetail && <div className="mt-1 text-[11px] text-slate-400 italic">loading regression…</div>}
    </div>
  );
}
