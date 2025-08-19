import * as React from "react";
import type { OverlayPayload } from "@/utils/types";
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
};

const OverlayCanvas = React.memo(function OverlayCanvasInner({
  payload,
  baseImageUrl,
  options,
  padding = 20,
  className,
  style,
}: OverlayCanvasProps) {
  const wrapperRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = React.useState<HTMLImageElement | null>(null);

  // Load (or clear) base image when toggled / URL changes
  React.useEffect(() => {
    let cancelled = false;
    if (!options.showBase || !baseImageUrl) {
      setImg(null);
      return () => {
        cancelled = true;
      };
    }

    const im = new Image();
    im.crossOrigin = "anonymous";
    const u = new URL(baseImageUrl, window.location.origin);
    u.searchParams.set("t", Date.now().toString()); // cache-bust
    im.onload = () => {
      if (!cancelled) setImg(im);
    };
    im.onerror = () => {
      if (!cancelled) setImg(null);
    };
    im.src = u.toString();

    return () => {
      cancelled = true;
    };
  }, [options.showBase, baseImageUrl]);

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
    // keep CSS size in CSS pixels
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear + background
    ctx.save();
    ctx.scale(dpr, dpr); // draw in CSS pixels for simplicity
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, cssW, cssH);

    // If no data, show message
    if (!payload || !payload.tracks || payload.tracks.length === 0) {
      ctx.fillStyle = "#9aa4bf";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("No tracks to render", 12, 20);
      ctx.restore();
      return;
    }

    // Compute bounds
    let minX = +Infinity, maxX = -Infinity, minY = +Infinity, maxY = -Infinity;
    for (const t of payload.tracks) {
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
      return;
    }

    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);

    const drawW = Math.max(1, cssW - 2 * padding);
    const drawH = Math.max(1, cssH - 2 * padding);

    // Scale factors to map image coords (px) → canvas CSS px
    const scaleX = drawW / spanX;
    const scaleY = drawH / spanY;

    const invertY = options.timeDirection === "up";

    const toCanvas = (x: number, y: number) => {
      const yNorm = invertY ? (maxY - y) : (y - minY);
      const cx = padding + (x - minX) * scaleX;
      const cy = padding + yNorm * scaleY;
      return { cx, cy };
    };

    // ----- Base image (optional) -----
    if (options.showBase && img) {
      ctx.save();
      // Establish the drawing plane so that (minX,minY) maps to (padding,padding)
      ctx.translate(padding, padding);
      // Map image pixels into the draw area with non-uniform scale (toCanvas uses separate scales)
      ctx.scale(scaleX, scaleY);

      // Crop/offset so that top-left of domain is (minX, minY)
      if (invertY) {
        // Flip vertically so image aligns with inverted Y axis
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

    // ----- Tracks -----
    const colorForTrack = (t: any) => {
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

    ctx.lineWidth = 1.2;
    ctx.globalCompositeOperation = "source-over";
    for (const t of payload.tracks) {
      const pts = t.poly;
      if (!pts || pts.length < 2) continue;
      ctx.strokeStyle = colorForTrack(t);
      ctx.beginPath();
      let p0 = toCanvas(pts[0][1], pts[0][0]); // [y,x] → (x,y)
      ctx.moveTo(p0.cx, p0.cy);
      for (let i = 1; i < pts.length; i++) {
        const p = toCanvas(pts[i][1], pts[i][0]);
        ctx.lineTo(p.cx, p.cy);
      }
      ctx.stroke();
    }

    // ----- Peaks -----
    ctx.fillStyle = "rgba(255, 120, 120, 0.9)";
    for (const t of payload.tracks) {
      if (!t.peaks || !t.poly) continue;
      for (const idx of t.peaks) {
        if (idx < 0 || idx >= t.poly.length) continue;
        const y = t.poly[idx][0];
        const x = t.poly[idx][1];
        const p = toCanvas(x, y);
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
  }, [payload, img, options.showBase, options.timeDirection, options.colorBy, padding]);

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

  return (
    <div ref={wrapperRef} className="w-full h-full" style={style}>
      <canvas ref={canvasRef} className={`block w-full h-full ${className || ""}`} />
    </div>
  );
});

export default OverlayCanvas;
