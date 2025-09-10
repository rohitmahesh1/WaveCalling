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
  highlightTrackId?: string | number;
  /** Return true to keep the track; if omitted, draw all tracks. */
  filterFn?: (t: Track) => boolean;
  /** If returns a CSS color string, overrides the normal color for that track. */
  colorOverrideFn?: (t: Track) => string | undefined;
  /** Called once the canvas element is ready (and on re-mount). */
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;

  /** Keep physical aspect ratio of the data/image (letterbox if needed). */
  preserveAspect?: boolean; // NEW
};

type Viewport = {
  minX: number; maxX: number; minY: number; maxY: number;
  scaleX: number; scaleY: number; invertY: boolean;
  padding: number;
  cssW: number; cssH: number;
  /** Offsets to center the content when aspect is preserved. */
  offsetX: number; offsetY: number; // NEW
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

  preserveAspect = true, // NEW
}: OverlayCanvasProps) {
  const wrapperRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = React.useState<HTMLImageElement | null>(null);

  // Let parent capture the <canvas> for downloads
  React.useEffect(() => {
    if (onCanvasReady) onCanvasReady(canvasRef.current);
    return () => {
      if (onCanvasReady) onCanvasReady(null); // keep v1 cleanup behavior
    };
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
    const u = new URL(baseImageUrl, window.location.origin);
    u.searchParams.set("t", Date.now().toString()); // cache-bust
    im.onload = () => { if (!cancelled) setImg(im); };
    im.onerror = () => { if (!cancelled) setImg(null); };
    im.src = u.toString();

    return () => { cancelled = true; };
  }, [options.showBase, baseImageUrl]);

  // Current viewport mapping (for pointer coord conversions)
  const viewportRef = React.useRef<Viewport | null>(null);

  // Helper: convert data (image) coords → canvas CSS px
  const toCanvas = React.useCallback((v: Viewport, x: number, y: number) => {
    const yNorm = v.invertY ? (v.maxY - y) : (y - v.minY);
    const cx = v.offsetX + (x - v.minX) * v.scaleX; // use offsetX instead of raw padding
    const cy = v.offsetY + yNorm * v.scaleY;        // use offsetY instead of raw padding
    return { cx, cy };
  }, []);

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
    ctx.scale(dpr, dpr); // draw in CSS pixels for simplicity
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, cssW, cssH);

    // Tracks to render (filtered); ensure highlighted track is included even if filtered out
    const allTracks = payload?.tracks ?? [];
    const filtered = filterFn ? allTracks.filter(filterFn) : allTracks.slice();
    const highlight =
      highlightTrackId != null
        ? allTracks.find((t) => String(t.id) === String(highlightTrackId)) || null
        : null;
    const drawTracks: Track[] = highlight && !filtered.includes(highlight)
      ? [...filtered, highlight]
      : filtered;

    if (!drawTracks.length) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("No tracks to render", 12, 20);
      ctx.restore();
      viewportRef.current = null;
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
      return;
    }

    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);

    const drawW = Math.max(1, cssW - 2 * padding);
    const drawH = Math.max(1, cssH - 2 * padding);

    // Raw scale factors to map image coords (px) → canvas CSS px
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
      offsetX, offsetY, // NEW
    };
    viewportRef.current = viewport;

    // ----- Base image (optional) -----
    if (options.showBase && img) {
      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scaleX, scaleY);
      if (invertY) {
        ctx.translate(-minX, -(maxY));
        ctx.scale(1, -1);
      } else {
        ctx.translate(-minX, -minY);
      }
      ctx.globalAlpha = 0.6;
      ctx.drawImage(img, 0, 0);
      ctx.globalAlpha = 1.0;
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

    // ----- Draw tracks: non-highlight first, then highlight on top -----
    const normal = drawTracks.filter((t) => String(t.id) !== String(highlightTrackId));
    const highlighted = drawTracks.filter((t) => String(t.id) === String(highlightTrackId));

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
      const p0 = toCanvas(viewport, pts[0][1], pts[0][0]); // [y,x] → (x,y)
      ctx.moveTo(p0.cx, p0.cy);
      for (let i = 1; i < pts.length; i++) {
        const p = toCanvas(viewport, pts[i][1], pts[i][0]);
        ctx.lineTo(p.cx, p.cy);
      }
      ctx.stroke();
    };

    // Normal
    ctx.globalCompositeOperation = "source-over";
    for (const t of normal) {
      drawOne(t, { width: 1.2, stroke: colorForTrack(t) });
    }
    // Highlighted
    for (const t of highlighted) {
      const c = colorForTrack(t);
      drawOne(t, { width: 2.2, stroke: c, glow: true });
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
    preserveAspect, // NEW
    toCanvas,
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

  const handlePointerMove = React.useCallback(
    (ev: React.PointerEvent<HTMLCanvasElement>) => {
      if (!onPointerMove) return;
      const rect = (ev.currentTarget as HTMLCanvasElement).getBoundingClientRect();
      const canvasX = ev.clientX - rect.left;
      const canvasY = ev.clientY - rect.top;
      const { dataX, dataY } = toDataCoords(canvasX, canvasY);
      onPointerMove({ canvasX, canvasY, dataX, dataY });
    },
    [onPointerMove, toDataCoords]
  );

  const handlePointerLeave = React.useCallback(() => {
    onPointerLeave?.();
  }, [onPointerLeave]);

  // Simple click hit-test: find nearest segment/vertex in canvas space
  const handleClick = React.useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onClickTrack) return;
      const allTracks = payload?.tracks ?? [];
      const filtered = filterFn ? allTracks.filter(filterFn) : allTracks;
      const rect = (ev.currentTarget as HTMLCanvasElement).getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;

      const v = viewportRef.current;
      if (!v) {
        onClickTrack(null);
        return;
      }

      // helper: squared distance to segment
      const d2Segment = (ax: number, ay: number, bx: number, by: number, px: number, py: number) => {
        const abx = bx - ax, aby = by - ay;
        const apx = px - ax, apy = py - ay;
        const ab2 = abx * abx + aby * aby || 1e-6;
        let t = (apx * abx + apy * aby) / ab2;
        t = Math.max(0, Math.min(1, t));
        const qx = ax + t * abx, qy = ay + t * aby;
        const dx = px - qx, dy = py - qy;
        return dx * dx + dy * dy;
      };

      const threshold = 8 * 8; // 8px tolerance (squared)
      let best: { track: Track; d2: number } | null = null;

      const toC = (x: number, y: number) => {
        const p = toCanvas(v, x, y);
        return [p.cx, p.cy] as const;
      };

      for (const t of filtered) {
        const pts = t.poly || [];
        if (pts.length < 2) continue;

        // sample every N points for speed; still check segments locally
        const step = Math.max(1, Math.floor(pts.length / 400)); // heuristic
        // first check vertices
        for (let i = 0; i < pts.length; i += step) {
          const [vx, vy] = toC(pts[i][1], pts[i][0]);
          const dx = vx - cx, dy = vy - cy;
          const d2 = dx * dx + dy * dy;
          if (d2 < (best?.d2 ?? Infinity)) best = { track: t, d2 };
        }
        // then check segments (coarse)
        for (let i = 1; i < pts.length; i += step) {
          const [ax, ay] = toC(pts[i - 1][1], pts[i - 1][0]);
          const [bx, by] = toC(pts[i][1], pts[i][0]);
          const d2 = d2Segment(ax, ay, bx, by, cx, cy);
          if (d2 < (best?.d2 ?? Infinity)) best = { track: t, d2 };
        }
      }

      if (best && best.d2 <= threshold) {
        onClickTrack(best.track);
      } else {
        onClickTrack(null);
      }
    },
    [onClickTrack, payload, filterFn, toCanvas]
  );

  return (
    <div ref={wrapperRef} className="w-full h-full" style={style}>
      <canvas
        ref={canvasRef}
        className={`block w-full h-full ${className || ""}`}
        onPointerMove={onPointerMove ? handlePointerMove : undefined}
        onPointerLeave={onPointerLeave ? handlePointerLeave : undefined}
        onClick={onClickTrack ? handleClick : undefined}
      />
    </div>
  );
});

export default OverlayCanvas;
