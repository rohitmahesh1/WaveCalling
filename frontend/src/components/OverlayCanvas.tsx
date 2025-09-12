// frontend/src/components/OverlayCanvas.tsx
import * as React from "react";
import type { OverlayPayload, OverlayTrack as Track } from "@/utils/types";
import type { ViewerOptions } from "@/components/ViewerToolbar";

export type OverlayCanvasProps = {
  payload: OverlayPayload | null;
  baseImageUrl?: string | null;
  options: ViewerOptions;
  /** Optional padding around drawing area in CSS pixels. */
  padding?: number;
  /** Extra className for the canvas element. */
  className?: string;
  /** Inline style for the canvas wrapper. */
  style?: React.CSSProperties;

  onPointerMove?: (p: { canvasX: number; canvasY: number; dataX: number; dataY: number }) => void;
  onPointerLeave?: () => void;
  onClickTrack?: (track: Track | null) => void;

  /** Explicit highlight (e.g., selected). */
  highlightTrackId?: string | number;

  /** Return true to keep the track; if omitted, draw all tracks. */
  filterFn?: (t: Track) => boolean;
  /** If returns a CSS color string, overrides the normal color for that track. */
  colorOverrideFn?: (t: Track) => string | undefined;
  /** Called once the canvas element is ready (and on re-mount). */
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;

  /** Keep physical aspect ratio of the data/image (letterbox if needed). */
  preserveAspect?: boolean;

  /** === NEW: hover UX === */
  hoverHighlight?: boolean;
  /** Optionally control the hovered id externally. */
  hoveredTrackId?: string | number;
  /** Receive hover changes (null when leaving). */
  onHoverTrack?: (track: Track | null) => void;
  /** Hit-test tolerance in CSS px (default 8). */
  hoverThresholdPx?: number;

  /** === NEW: redraw keys so changes to rules force a draw, even if functions are stable === */
  filterKey?: string;
  colorKey?: string;
  sortKey?: string;
};

type Viewport = {
  minX: number; maxX: number; minY: number; maxY: number;
  scaleX: number; scaleY: number; invertY: boolean;
  padding: number;
  cssW: number; cssH: number;
  /** Offsets to center the content when aspect is preserved. */
  offsetX: number; offsetY: number;
};

type HitSample = { x: number; y: number }; // canvas-space
type HitEntry = {
  id: string | number;
  bbox: { minX: number; minY: number; maxX: number; maxY: number }; // canvas-space bbox
  samples: HitSample[]; // downsampled points in canvas space
  track: Track;
};

const OverlayCanvas = React.memo(function OverlayCanvasInner({
  payload,
  baseImageUrl,
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
}: OverlayCanvasProps) {
  const wrapperRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = React.useState<HTMLImageElement | null>(null);

  // Let parent capture the <canvas> for downloads
  React.useEffect(() => {
    onCanvasReady?.(canvasRef.current);
    return () => onCanvasReady?.(null);
  }, [onCanvasReady]);

  // Load (or clear) base image when toggled / URL changes
  React.useEffect(() => {
    let cancelled = false;
    if (!options.showBase || !baseImageUrl) {
      setImg(null);
      return () => { cancelled = true; };
    }
    const im = new Image();
    im.crossOrigin = "anonymous";
    // cache-bust keeps PNG in sync; OK to keep until you switch to ETag drawing
    const u = new URL(baseImageUrl, window.location.origin);
    u.searchParams.set("t", Date.now().toString());
    im.onload = () => { if (!cancelled) setImg(im); };
    im.onerror = () => { if (!cancelled) setImg(null); };
    im.src = u.toString();
    return () => { cancelled = true; };
  }, [options.showBase, baseImageUrl]);

  // Current viewport mapping (for conversions + hit cache)
  const viewportRef = React.useRef<Viewport | null>(null);

  // ===== Hit-test cache (canvas space) =====
  const hitCacheRef = React.useRef<HitEntry[] | null>(null);

  // Local hovered id (uncontrolled mode)
  const [hoverLocal, setHoverLocal] = React.useState<string | number | null>(null);
  const effectiveHoverId =
    hoveredTrackId != null ? hoveredTrackId : hoverHighlight ? hoverLocal : null;

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
      // Build downsampled points + bbox in canvas space for each visible track
      const entries: HitEntry[] = [];
      for (const t of tracks) {
        const pts = t.poly || [];
        if (pts.length < 2) continue;

        // Choose a step so we cap samples roughly ~500 per long track
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
        // Guard for degenerate
        if (!Number.isFinite(minX + minY + maxX + maxY)) continue;

        entries.push({
          id: t.id!,
          bbox: { minX, minY, maxX, maxY },
          samples,
          track: t,
        });
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

    // merge highlights into draw list (unique)
    const drawTracks: Track[] = [...filtered];
    const pushUnique = (t: Track | null) => {
      if (!t) return;
      if (!drawTracks.some((x) => String(x.id) === String(t.id))) {
        drawTracks.push(t);
      }
    };
    pushUnique(explicitHighlight);
    pushUnique(hoverHighlightTrack);

    if (!drawTracks.length) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("No tracks to render", 12, 20);
      ctx.restore();
      viewportRef.current = null;
      hitCacheRef.current = null;
      return;
    }

    // Compute bounds
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

    // If base image is present, prefer its natural bounds so overlay aligns perfectly
    if (options.showBase && img) {
      minX = 0;
      minY = 0;
      maxX = Math.max(maxX, img.naturalWidth);
      maxY = Math.max(maxY, img.naturalHeight);
    }

    if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("Invalid bounds", 12, 20);
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

    // ----- Base image (optional) -----
    if (options.showBase && img) {
      // Current data bounds and effective content box
      const sx = Math.max(0, Math.floor(minX));
      const syUp = Math.max(0, Math.floor(minY));
      const sw = Math.max(1, Math.min(img.naturalWidth  - sx, Math.ceil(maxX - minX)));
      const sh = Math.max(1, Math.min(img.naturalHeight - syUp, Math.ceil(maxY - minY)));

      const syTL = img.naturalHeight - (syUp + sh);

      const dx = offsetX;
      const dy = offsetY;
      const dw = (maxX - minX) * scaleX;
      const dh = (maxY - minY) * scaleY;

      ctx.save();
      ctx.globalAlpha = 0.6;
      ctx.drawImage(img, sx, syTL, sw, sh, dx, dy, dw, dh);
      ctx.restore();
    }

    // ----- Color logic -----
    const colorForTrack = (t: Track) => {
      const override = colorOverrideFn?.(t);
      if (override) return override;

      if (options.colorBy === "dominant_frequency") {
        const f = Number(t.metrics?.dominant_frequency ?? 0);
        const hue = 200 + Math.max(0, Math.min(1, f / 2)) * 140; // 0..2Hz → 200..340
        return `hsl(${hue}deg 70% 70% / 0.9)`;
      }
      if (options.colorBy === "amplitude") {
        const a = Number(t.metrics?.mean_amplitude ?? 0);
        const tval = Math.max(0, Math.min(1, a / 10)); // heuristic
        const hue = 140 - tval * 100; // green → yellow/red
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
      if (style.glow) {
        ctx.shadowColor = style.stroke;
        ctx.shadowBlur = 6;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      const p0 = toCanvas(viewport, pts[0][1], pts[0][0]);
      ctx.moveTo(p0.cx, p0.cy);
      for (let i = 1; i < pts.length; i++) {
        const p = toCanvas(viewport, pts[i][1], pts[i][0]);
        ctx.lineTo(p.cx, p.cy);
      }
      ctx.stroke();
    };

    // Normal layer
    ctx.globalCompositeOperation = "source-over";
    for (const t of normal) {
      drawOne(t, { width: 1.2, stroke: colorForTrack(t) });
    }
    // Hover layer (slightly thicker)
    for (const t of hovered) {
      const c = colorForTrack(t);
      drawOne(t, { width: 2.0, stroke: c, glow: true });
    }
    // Explicit highlight layer (top-most)
    for (const t of highlighted) {
      const c = colorForTrack(t);
      drawOne(t, { width: 2.4, stroke: c, glow: true });
    }

    // ----- Peaks -----
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

    ctx.restore();

    // Build hit cache for hover from what we just drew
    rebuildHitCache(drawTracks, viewport);
  }, [
    payload,
    img,
    options.showBase,
    options.timeDirection,
    options.colorBy,
    padding,
    filterFn,
    colorOverrideFn,
    highlightTrackId,
    preserveAspect,
    toCanvas,
    rebuildHitCache,
    effectiveHoverId,
    // NEW: when rules change, force a redraw even if filterFn/colorOverrideFn are stable
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

  // Redraw on payload/base image changes
  React.useEffect(() => {
    draw();
  }, [payload, img, draw]);

  // If external hover id changes, redraw to reflect it
  React.useEffect(() => {
    if (hoveredTrackId != null) draw();
  }, [hoveredTrackId, draw]);

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

    // Quick pass: bbox rejection
    let best: { entry: HitEntry; d2: number } | null = null;
    for (const e of cache) {
      if (
        pt.cx < e.bbox.minX - hoverThresholdPx ||
        pt.cx > e.bbox.maxX + hoverThresholdPx ||
        pt.cy < e.bbox.minY - hoverThresholdPx ||
        pt.cy > e.bbox.maxY + hoverThresholdPx
      ) {
        continue;
      }
      // Coarse nearest on samples
      for (let i = 0; i < e.samples.length; i++) {
        const dx = e.samples[i].x - pt.cx;
        const dy = e.samples[i].y - pt.cy;
        const d2 = dx * dx + dy * dy;
        if (!best || d2 < best.d2) {
          best = { entry: e, d2 };
        }
      }
    }

    const newHover =
      best && best.d2 <= thresh2 ? String(best.entry.id) : null;
    const prevHover = hoverLocal != null ? String(hoverLocal) : null;

    if (newHover !== prevHover) {
      // Update local state only if uncontrolled
      if (hoveredTrackId == null && hoverHighlight) {
        setHoverLocal(newHover);
      }
      // Notify parent
      if (onHoverTrack) {
        onHoverTrack(
          newHover
            ? (cache.find((e) => String(e.id) === newHover)?.track ?? null)
            : null
        );
      }
      // Redraw to show hover layer + cursor
      draw();
      const el = canvasRef.current;
      if (el) el.style.cursor = newHover ? "pointer" : "default";
    }
  }, [
    hoverThresholdPx,
    hoveredTrackId,
    hoverHighlight,
    onHoverTrack,
    hoverLocal,
    draw,
  ]);

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
    // Clear hover when leaving (uncontrolled mode)
    if (hoveredTrackId == null && hoverHighlight) {
      if (hoverLocal != null) {
        setHoverLocal(null);
        draw();
      }
    }
    const el = canvasRef.current;
    if (el) el.style.cursor = "default";
  }, [onPointerLeave, hoveredTrackId, hoverHighlight, hoverLocal, draw]);

  // Simple click hit-test: choose nearest cached sample (cheap & good UX)
  const handleClick = React.useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onClickTrack) return;
      const cache = hitCacheRef.current;
      if (!cache) {
        onClickTrack(null);
        return;
      }

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

      if (best && best.d2 <= thresh2) onClickTrack(best.entry.track);
      else onClickTrack(null);
    },
    [onClickTrack, hoverThresholdPx]
  );

  return (
    <div ref={wrapperRef} className="w-full h-full" style={style}>
      <canvas
        ref={canvasRef}
        className={`block w-full h-full ${className || ""}`}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onClick={handleClick}
      />
    </div>
  );
});

export default OverlayCanvas;
