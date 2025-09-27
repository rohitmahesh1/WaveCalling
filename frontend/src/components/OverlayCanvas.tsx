// frontend/src/components/OverlayCanvas.tsx
import * as React from "react";
import type { OverlayPayload, OverlayTrack as Track } from "@/utils/types";
import type { ViewerOptions } from "@/components/ViewerToolbar";

export type OverlayCanvasProps = {
  payload: OverlayPayload | null;
  baseImageUrl?: string | null;
  debugImageUrl?: string | null;
  options: ViewerOptions;

  padding?: number;
  className?: string;
  style?: React.CSSProperties;

  onPointerMove?: (p: { canvasX: number; canvasY: number; dataX: number; dataY: number }) => void;
  onPointerLeave?: () => void;
  onClickTrack?: (track: Track | null) => void;

  highlightTrackId?: string | number;

  filterFn?: (t: Track) => boolean;
  colorOverrideFn?: (t: Track) => string | undefined;
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;

  preserveAspect?: boolean;

  /** === Hover UX === */
  hoverHighlight?: boolean;
  hoveredTrackId?: string | number;
  onHoverTrack?: (track: Track | null) => void;
  hoverThresholdPx?: number;

  /** Redraw keys so changes to rules force a draw, even if function identities are stable. */
  filterKey?: string;
  colorKey?: string;
  sortKey?: string;

  /** Show small "base/debug" link badges inside the canvas wrapper (off by default). */
  showSourceBadges?: boolean;
};

type Viewport = {
  minX: number; maxX: number; minY: number; maxY: number;
  scaleX: number; scaleY: number; invertY: boolean;
  padding: number;
  cssW: number; cssH: number;
  offsetX: number; offsetY: number;
};

type HitSample = { x: number; y: number };
type HitEntry = {
  id: string | number;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  samples: HitSample[];
  track: Track;
};

const OverlayCanvas = React.memo(function OverlayCanvasInner({
  payload,
  baseImageUrl,
  debugImageUrl,
  options,
  padding = 20,
  className,
  style,

  onPointerMove,
  onPointerLeave,
  onClickTrack,
  highlightTrackId,
  filterFn,
  colorOverrideFn,
  onCanvasReady,

  preserveAspect = true,

  hoverHighlight = true,
  hoveredTrackId,
  onHoverTrack,
  hoverThresholdPx = 8,

  filterKey,
  colorKey,
  sortKey,

  showSourceBadges = false,
}: OverlayCanvasProps) {
  const wrapperRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = React.useState<HTMLImageElement | null>(null);
  const [debugImg, setDebugImg] = React.useState<HTMLImageElement | null>(null);

  const viewerDebugLayer = (options as any)?.debugLayer as (string | undefined);

  // Let parent capture the <canvas> (for downloads, etc.)
  React.useEffect(() => {
    onCanvasReady?.(canvasRef.current);
    return () => onCanvasReady?.(null);
  }, [onCanvasReady]);

  // base image effect
  React.useEffect(() => {
    let cancelled = false;
    if (!options.showBase || !baseImageUrl) {
      console.log("[Canvas] skip base", { showBase: options.showBase, baseImageUrl });
      setImg(null);
      return () => { cancelled = true; };
    }
    console.log("[Canvas] load base", baseImageUrl);
    const im = new Image();
    const u = new URL(baseImageUrl, window.location.origin);
    if (u.origin !== window.location.origin) im.crossOrigin = "anonymous";
    u.searchParams.set("t", Date.now().toString());
    im.onload = () => { if (!cancelled) { console.log("[Canvas] base onload"); setImg(im); } };
    im.onerror = () => { if (!cancelled) { console.warn("[Canvas] base onerror", u.toString()); setImg(null); } };
    im.decoding = "async";
    im.src = u.toString();
    return () => { cancelled = true; };
  }, [options.showBase, baseImageUrl]);

  // debug image effect
  React.useEffect(() => {
    let cancelled = false;
    if (!debugImageUrl || !viewerDebugLayer || viewerDebugLayer === "none") {
      console.log("[Canvas] skip debug", { debugImageUrl, viewerDebugLayer });
      setDebugImg(null);
      return () => { cancelled = true; };
    }
    console.log("[Canvas] load debug", debugImageUrl);
    const im = new Image();
    const u = new URL(debugImageUrl, window.location.origin);
    if (u.origin !== window.location.origin) im.crossOrigin = "anonymous";
    u.searchParams.set("t", Date.now().toString());
    im.onload = () => { if (!cancelled) { console.log("[Canvas] debug onload"); setDebugImg(im); } };
    im.onerror = () => { if (!cancelled) { console.warn("[Canvas] debug onerror", u.toString()); setDebugImg(null); } };
    im.decoding = "async";
    im.src = u.toString();
    return () => { cancelled = true; };
  }, [debugImageUrl, viewerDebugLayer]);

  // Current viewport mapping (for conversions + hit cache)
  const viewportRef = React.useRef<Viewport | null>(null);

  // ===== Hit-test cache (canvas space) =====
  const hitCacheRef = React.useRef<HitEntry[] | null>(null);

  // Local hovered id (uncontrolled mode)
  const [hoverLocal, setHoverLocal] = React.useState<string | number | null>(null);
  const effectiveHoverId =
    hoveredTrackId != null ? hoveredTrackId : (hoverHighlight ? hoverLocal : null);

  // rAF throttle for hover hit-testing
  const hoverRAF = React.useRef<number | null>(null);
  const pendingHoverPoint = React.useRef<{ cx: number; cy: number } | null>(null);

  // Helper: convert data (image) coords → canvas CSS px
  const toCanvas = React.useCallback((v: Viewport, x: number, y: number) => {
    const yNorm = v.invertY ? (v.maxY - y) : (y - v.minY);
    const cx = v.offsetX + (x - v.minX) * v.scaleX;
    const cy = v.offsetY + yNorm * v.scaleY;
    return { cx, cy };
  }, []);

  const rebuildHitCache = React.useCallback(
    (tracks: Track[], v: Viewport) => {
      const entries: HitEntry[] = [];
      for (const t of tracks) {
        const pts = t.poly || [];
        if (pts.length < 2) continue;

        // downsample so hover is cheap
        const step = Math.max(1, Math.floor(pts.length / 500));
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        const samples: HitSample[] = new Array(Math.ceil(pts.length / step));
        let si = 0;

        for (let i = 0; i < pts.length; i += step) {
          const { cx, cy } = toCanvas(v, pts[i][1], pts[i][0]);
          samples[si++] = { x: cx, y: cy };
          if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
        }
        if (!Number.isFinite(minX + minY + maxX + maxY)) continue;

        entries.push({ id: t.id!, bbox: { minX, minY, maxX, maxY }, samples, track: t });
      }
      hitCacheRef.current = entries;
    },
    [toCanvas]
  );

  const draw = React.useCallback(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
    const cssW = Math.max(200, wrapper.clientWidth);
    const cssH = Math.max(160, wrapper.clientHeight);
    const w = Math.floor(cssW * dpr);
    const h = Math.floor(cssH * dpr);

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr); // draw in CSS pixels
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, cssW, cssH);

    // Tracks to render (filtered); ensure highlighted/hovered track included even if filtered out
    const allTracks = payload?.tracks ?? [];
    const filtered = filterFn ? allTracks.filter(filterFn) : allTracks.slice();

    const explicitHighlight =
      highlightTrackId != null
        ? allTracks.find((t) => String(t.id) === String(highlightTrackId)) || null
        : null;
    const hoverHighlightTrack =
      effectiveHoverId != null
        ? allTracks.find((t) => String(t.id) === String(effectiveHoverId)) || null
        : null;

    const drawTracks: Track[] = [...filtered];
    const pushUnique = (t: Track | null) => {
      if (!t) return;
      if (!drawTracks.some((x) => String(x.id) === String(t.id))) {
        drawTracks.push(t);
      }
    };
    pushUnique(explicitHighlight);
    pushUnique(hoverHighlightTrack);

    const noTracks = drawTracks.length === 0;

    // Compute bounds from tracks
    let minX = +Infinity, maxX = -Infinity, minY = +Infinity, maxY = -Infinity;
    for (const t of drawTracks) {
      const pts = t.poly || [];
      for (let i = 0; i < pts.length; i++) {
        const y = pts[i][0];
        const x = pts[i][1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }

    // Prefer image bounds when present so overlay aligns perfectly
    if ((options.showBase && img) || debugImg) {
      minX = 0;
      minY = 0;
      const w1 = img?.naturalWidth ?? 0;
      const h1 = img?.naturalHeight ?? 0;
      const w2 = debugImg?.naturalWidth ?? 0;
      const h2 = debugImg?.naturalHeight ?? 0;
      maxX = Math.max(maxX, w1, w2);
      maxY = Math.max(maxY, h1, h2);
    }

    if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("No tracks to render", 12, 20);
      ctx.restore();
      viewportRef.current = null;
      hitCacheRef.current = null;
      return;
    }

    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);
    const drawW = Math.max(1, cssW - 2 * padding);
    const drawH = Math.max(1, cssH - 2 * padding);

    const rawScaleX = drawW / spanX;
    const rawScaleY = drawH / spanY;

    // Aspect-preserving scale (uniform) with letterboxing
    let scaleX = rawScaleX;
    let scaleY = rawScaleY;
    let offsetX = padding;
    let offsetY = padding;

    if (preserveAspect) {
      const s = Math.min(rawScaleX, rawScaleY);
      scaleX = s;
      scaleY = s;
      const effW = spanX * s;
      const effH = spanY * s;
      offsetX = padding + (drawW - effW) / 2;
      offsetY = padding + (drawH - effH) / 2;
    }

    const invertY = options.timeDirection === "up";

    const viewport: Viewport = {
      minX, maxX, minY, maxY,
      scaleX, scaleY, invertY,
      padding, cssW, cssH,
      offsetX, offsetY,
    };
    viewportRef.current = viewport;

    // Helper to draw any image aligned to current data bounds (handles Y origin)
    const drawAlignedImage = (image: HTMLImageElement, alpha: number) => {
      const sx = Math.max(0, Math.floor(minX));
      const syDataMin = Math.max(0, Math.floor(minY));
      const sw = Math.max(1, Math.min(image.naturalWidth  - sx, Math.ceil(maxX - minX)));
      const sh = Math.max(1, Math.min(image.naturalHeight - syDataMin, Math.ceil(maxY - minY)));
      const syTL = invertY ? (image.naturalHeight - (syDataMin + sh)) : syDataMin;

      const dx = offsetX;
      const dy = offsetY;
      const dw = (maxX - minX) * scaleX;
      const dh = (maxY - minY) * scaleY;

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.drawImage(image, sx, syTL, sw, sh, dx, dy, dw, dh);
      ctx.restore();
    };

    // Base image
    if (options.showBase && img) drawAlignedImage(img, 0.6);

    // Debug image
    if (debugImg) drawAlignedImage(debugImg, 0.7);

    // ----- Color logic -----
    const colorForTrack = (t: Track) => {
      const override = colorOverrideFn?.(t);
      if (override) return override;

      if (options.colorBy === "dominant_frequency") {
        const f = Number(t.metrics?.dominant_frequency ?? 0);
        const hue = 200 + Math.max(0, Math.min(1, f / 2)) * 140;
        return `hsl(${hue}deg 70% 70% / 0.9)`;
      }
      if (options.colorBy === "amplitude") {
        const a = Number(t.metrics?.mean_amplitude ?? 0);
        const tval = Math.max(0, Math.min(1, a / 10));
        const hue = 140 - tval * 100;
        return `hsl(${hue}deg 70% 70% / 0.9)`;
      }
      return "rgba(80, 200, 120, 0.85)";
    };

    // ----- Draw tracks: non-highlight first, then highlights on top -----
    const isExplicit = (t: Track) => String(t.id) === String(highlightTrackId);
    const isHover = (t: Track) => effectiveHoverId != null && String(t.id) === String(effectiveHoverId);

    const normal = drawTracks.filter((t) => !isExplicit(t) && !isHover(t));
    const hovered = drawTracks.filter(isHover);
    const highlighted = drawTracks.filter(isExplicit);

    const drawOne = (t: Track, style: { width: number; stroke: string; glow?: boolean }) => {
      const pts = t.poly;
      if (!pts || pts.length < 2) return;

      ctx.lineWidth = style.width;
      ctx.strokeStyle = style.stroke;
      ctx.shadowColor = style.glow ? style.stroke : "transparent";
      ctx.shadowBlur = style.glow ? 6 : 0;

      ctx.beginPath();
      const p0 = toCanvas(viewport, pts[0][1], pts[0][0]);
      ctx.moveTo(p0.cx, p0.cy);
      for (let i = 1; i < pts.length; i++) {
        const p = toCanvas(viewport, pts[i][1], pts[i][0]);
        ctx.lineTo(p.cx, p.cy);
      }
      ctx.stroke();
    };

    ctx.globalCompositeOperation = "source-over";
    for (const t of normal) drawOne(t, { width: 1.2, stroke: colorForTrack(t) });
    for (const t of hovered) drawOne(t, { width: 2.0, stroke: colorForTrack(t), glow: true });
    for (const t of highlighted) drawOne(t, { width: 2.4, stroke: colorForTrack(t), glow: true });

    // Peaks
    ctx.fillStyle = "rgba(255, 120, 120, 0.9)";
    for (const t of drawTracks) {
      if (!t.peaks || !t.poly) continue;
      for (const idx of t.peaks) {
        if (idx < 0 || idx >= t.poly.length) continue;
        const y = t.poly[idx][0];
        const x = t.poly[idx][1];
        const p = toCanvas(viewport, x, y);
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, 2.0, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Border
    ctx.strokeStyle = "#2a2f3a";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, cssW - 1, cssH - 1);

    // Hint when truly empty
    if (noTracks && !(options.showBase && img) && !debugImg) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("No tracks to render", 12, 20);
    }

    ctx.restore();

    rebuildHitCache(drawTracks, viewport);
  }, [
    payload,
    img,
    debugImg,
    options.showBase,
    options.timeDirection,
    options.colorBy,
    viewerDebugLayer,       // <- react to debug layer changes safely
    padding,
    filterFn,
    colorOverrideFn,
    highlightTrackId,
    preserveAspect,
    toCanvas,
    rebuildHitCache,
    effectiveHoverId,
    filterKey,
    colorKey,
    sortKey,
  ]);

  // Redraw when size changes (responsive)
  React.useEffect(() => {
    draw(); // initial
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(wrapper);
    return () => ro.disconnect();
  }, [draw]);

  // Redraw on payload/base/debug image changes
  React.useEffect(() => { draw(); }, [payload, img, debugImg, draw]);

  // If external hover id changes, redraw to reflect it
  React.useEffect(() => { if (hoveredTrackId != null) draw(); }, [hoveredTrackId, draw]);

  // ----- Pointer interactions -----
  const toDataCoords = React.useCallback((canvasX: number, canvasY: number) => {
    const v = viewportRef.current;
    if (!v) return { dataX: NaN, dataY: NaN };
    const x = v.minX + (canvasX - v.offsetX) / v.scaleX;
    const y = v.invertY
      ? v.maxY - (canvasY - v.offsetY) / v.scaleY
      : v.minY + (canvasY - v.offsetY) / v.scaleY;
    return { dataX: x, dataY: y };
  }, []);

  const runHoverHitTest = React.useCallback(() => {
    const canvas = canvasRef.current;
    const v = viewportRef.current;
    const cache = hitCacheRef.current;
    const pt = pendingHoverPoint.current;
    pendingHoverPoint.current = null;
    if (!canvas || !v || !cache || !pt) return;

    const thresh2 = hoverThresholdPx * hoverThresholdPx;

    let best: { entry: HitEntry; d2: number } | null = null;
    for (const e of cache) {
      if (
        pt.cx < e.bbox.minX - hoverThresholdPx ||
        pt.cx > e.bbox.maxX + hoverThresholdPx ||
        pt.cy < e.bbox.minY - hoverThresholdPx ||
        pt.cy > e.bbox.maxY + hoverThresholdPx
      ) continue;

      for (let i = 0; i < e.samples.length; i++) {
        const dx = e.samples[i].x - pt.cx;
        const dy = e.samples[i].y - pt.cy;
        const d2 = dx * dx + dy * dy;
        if (!best || d2 < best.d2) best = { entry: e, d2 };
      }
    }

    const newHover = best && best.d2 <= thresh2 ? String(best.entry.id) : null;
    const prevHover = hoverLocal != null ? String(hoverLocal) : null;

    if (newHover !== prevHover) {
      if (hoveredTrackId == null && hoverHighlight) setHoverLocal(newHover);
      onHoverTrack?.(
        newHover ? (cache.find((e) => String(e.id) === newHover)?.track ?? null) : null
      );
      draw();
      const el = canvasRef.current;
      if (el) el.style.cursor = newHover ? "pointer" : "default";
    }
  }, [hoverThresholdPx, hoveredTrackId, hoverHighlight, onHoverTrack, hoverLocal, draw]);

  const scheduleHoverHitTest = React.useCallback(() => {
    if (hoverRAF.current != null) return;
    hoverRAF.current = requestAnimationFrame(() => {
      hoverRAF.current = null;
      runHoverHitTest();
    });
  }, [runHoverHitTest]);

  const handlePointerMove = React.useCallback(
    (ev: React.PointerEvent<HTMLCanvasElement>) => {
      const rect = (ev.currentTarget as HTMLCanvasElement).getBoundingClientRect();
      const canvasX = ev.clientX - rect.left;
      const canvasY = ev.clientY - rect.top;

      // public callback first
      if (onPointerMove) {
        const { dataX, dataY } = toDataCoords(canvasX, canvasY);
        onPointerMove({ canvasX, canvasY, dataX, dataY });
      }

      // Hover testing (if enabled)
      if (hoverHighlight || onHoverTrack) {
        pendingHoverPoint.current = { cx: canvasX, cy: canvasY };
        scheduleHoverHitTest();
      }
    },
    [onPointerMove, toDataCoords, hoverHighlight, onHoverTrack, scheduleHoverHitTest]
  );

  const handlePointerLeave = React.useCallback(() => {
    onPointerLeave?.();
    if (hoveredTrackId == null && hoverHighlight) {
      if (hoverLocal != null) {
        setHoverLocal(null);
        draw();
      }
    }
    const el = canvasRef.current;
    if (el) el.style.cursor = "default";
  }, [onPointerLeave, hoveredTrackId, hoverHighlight, hoverLocal, draw]);

  const handleClick = React.useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onClickTrack) return;
      const cache = hitCacheRef.current;
      if (!cache) { onClickTrack(null); return; }

      const rect = (ev.currentTarget as HTMLCanvasElement).getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;

      const thresh2 = hoverThresholdPx * hoverThresholdPx;
      let best: { entry: HitEntry; d2: number } | null = null;

      for (const e of cache) {
        if (
          cx < e.bbox.minX - hoverThresholdPx ||
          cx > e.bbox.maxX + hoverThresholdPx ||
          cy < e.bbox.minY - hoverThresholdPx ||
          cy > e.bbox.maxY + hoverThresholdPx
        ) continue;

        for (let i = 0; i < e.samples.length; i++) {
          const dx = e.samples[i].x - cx;
          const dy = e.samples[i].y - cy;
          const d2 = dx * dx + dy * dy;
          if (!best || d2 < best.d2) best = { entry: e, d2 };
        }
      }

      onClickTrack(best && best.d2 <= thresh2 ? best.entry.track : null);
    },
    [onClickTrack, hoverThresholdPx]
  );

  const showBaseBadge = Boolean(showSourceBadges && options.showBase && baseImageUrl);
  const showDebugBadge = Boolean(showSourceBadges && debugImageUrl && viewerDebugLayer && viewerDebugLayer !== "none");

  return (
    <div ref={wrapperRef} className="relative w-full h-full" style={style}>
      <canvas
        ref={canvasRef}
        className={`block w-full h-full ${className || ""}`}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onClick={handleClick}
      />

      {(showBaseBadge || showDebugBadge) && (
        <div className="pointer-events-none absolute left-2 bottom-2 flex gap-3 text-xs text-slate-400">
          {showBaseBadge && (
            <span className="pointer-events-auto">
              base:{" "}
              <a
                href={baseImageUrl!}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-slate-600 hover:text-slate-300"
                aria-label="Open base image"
              >
                base.png
              </a>
            </span>
          )}
          {showDebugBadge && (
            <span className="pointer-events-auto">
              debug:{" "}
              <a
                href={debugImageUrl!}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-slate-600 hover:text-slate-300"
                aria-label="Open debug layer image"
              >
                {String(viewerDebugLayer)}.png
              </a>
            </span>
          )}
        </div>
      )}
    </div>
  );
});

export default OverlayCanvas;
