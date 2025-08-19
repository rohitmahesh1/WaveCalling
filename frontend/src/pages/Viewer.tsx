import * as React from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useApiBase } from "@/context/ApiContext";
import { getRun } from "@/utils/api";
import type { RunStatusResponse, OverlayPayload } from "@/utils/types";

function useRunId(): string | null {
  const params = useParams<{ runId?: string }>();
  const [sp] = useSearchParams();
  return params.runId ?? sp.get("runId");
}

export default function Viewer() {
  const apiBase = useApiBase();
  const runId = useRunId();

  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [overlayUrl, setOverlayUrl] = React.useState<string | null>(null);
  const [payload, setPayload] = React.useState<OverlayPayload | null>(null);
  const [summary, setSummary] = React.useState<{ tracks: number; points: number } | null>(null);

  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const holderRef = React.useRef<HTMLDivElement | null>(null);

  // fetch run artifacts + overlay json
  const fetchOverlay = React.useCallback(async () => {
    if (!runId) {
      setError("No runId provided.");
      return;
    }
    setLoading(true);
    setError(null);
    setPayload(null);
    setOverlayUrl(null);

    try {
      const rs: RunStatusResponse = await getRun(apiBase, runId);
      const url = rs.artifacts?.overlay_json;
      if (!url) {
        setError("overlay_json not available yet. If the run is still processing, try Refresh.");
        setLoading(false);
        return;
      }
      setOverlayUrl(url);

      const resp = await fetch(url, { cache: "no-cache" });
      if (!resp.ok) {
        setError(`Failed to load overlay: ${resp.status} ${resp.statusText}`);
        setLoading(false);
        return;
      }
      const data: OverlayPayload = await resp.json();
      setPayload(data);

      // quick summary
      let pts = 0;
      for (const t of data.tracks) pts += t.poly.length;
      setSummary({ tracks: data.tracks.length, points: pts });
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase, runId]);

  React.useEffect(() => {
    void fetchOverlay();
  }, [fetchOverlay]);

  // draw when payload or size changes
  React.useEffect(() => {
    if (!payload || !canvasRef.current || !holderRef.current) return;

    const canvas = canvasRef.current;
    const holder = holderRef.current;

    // Helper: draw everything for a given payload
    function draw(data: OverlayPayload) {
      const pad = 20;
      const w = Math.max(400, holder.clientWidth);
      const h = Math.max(300, Math.round(holder.clientWidth * 0.6));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // compute bounds from all points (data.tracks[].poly is [[y,x],...])
      let minX = +Infinity, maxX = -Infinity, minY = +Infinity, maxY = -Infinity;
      for (const t of data.tracks) {
        for (const [y, x] of t.poly) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
      if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#999";
        ctx.fillText("No points to render", 12, 20);
        return;
      }
      const spanX = Math.max(1, maxX - minX);
      const spanY = Math.max(1, maxY - minY);

      const drawW = canvas.width - 2 * pad;
      const drawH = canvas.height - 2 * pad;

      const scaleX = drawW / spanX;
      const scaleY = drawH / spanY;

      function toCanvas(x: number, y: number) {
        // image coords (y downward). For inverted Y (time ↑), use maxY - y in place of (y - minY).
        const cx = pad + (x - minX) * scaleX;
        const cy = pad + (y - minY) * scaleY;
        return { cx, cy };
      }

      // background
      ctx.fillStyle = "#0b0e14";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // tracks
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(80, 200, 120, 0.8)";
      ctx.globalCompositeOperation = "source-over";

      for (const t of data.tracks) {
        const pts = t.poly;
        if (pts.length < 2) continue;
        ctx.beginPath();
        let { cx: x0, cy: y0 } = toCanvas(pts[0][1], pts[0][0]); // [y,x] -> (x,y)
        ctx.moveTo(x0, y0);
        for (let i = 1; i < pts.length; i++) {
          const { cx, cy } = toCanvas(pts[i][1], pts[i][0]);
          ctx.lineTo(cx, cy);
        }
        ctx.stroke();
      }

      // peaks
      ctx.fillStyle = "rgba(255, 120, 120, 0.9)";
      for (const t of data.tracks) {
        for (const idx of t.peaks || []) {
          if (idx < 0 || idx >= t.poly.length) continue;
          const [y, x] = t.poly[idx];
          const { cx, cy } = toCanvas(x, y);
          ctx.beginPath();
          ctx.arc(cx, cy, 2.0, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // border
      ctx.strokeStyle = "#2a2f3a";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);

      // legend
      ctx.fillStyle = "#c8d3f5";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("Tracks", pad, canvas.height - pad + 4);
      ctx.fillStyle = "rgba(80, 200, 120, 0.8)";
      ctx.fillRect(pad + 50, canvas.height - pad - 5, 22, 2);
      ctx.fillStyle = "#c8d3f5";
      ctx.fillText("Peaks", pad + 90, canvas.height - pad + 4);
      ctx.fillStyle = "rgba(255, 120, 120, 0.9)";
      ctx.beginPath();
      ctx.arc(pad + 130, canvas.height - pad, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // initial draw + responsive redraw
    draw(payload);
    const ro = new ResizeObserver(() => draw(payload));
    ro.observe(holder);
    return () => ro.disconnect();
  }, [payload]);

  return (
    <div className="stack">
      <h2>Viewer</h2>

      <div className="card">
        <div className="row">
          <div>run_id:&nbsp;</div>
          <code>{runId ?? "(none)"}</code>
          <button className="ml" onClick={() => void fetchOverlay()} disabled={!runId || loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {overlayUrl && (
          <div className="mt" style={{ color: "var(--muted)", fontSize: 13 }}>
            overlay:{" "}
            <a href={overlayUrl} target="_blank" rel="noreferrer">
              {overlayUrl}
            </a>
          </div>
        )}

        {error && (
          <div className="mt" style={{ color: "#ffb4a9" }}>
            {error}
          </div>
        )}

        {summary && (
          <div className="mt" style={{ color: "var(--muted)" }}>
            tracks: {summary.tracks.toLocaleString()} &middot; points:{" "}
            {summary.points.toLocaleString()}
          </div>
        )}

        <div ref={holderRef} className="mt" style={{ width: "100%", height: "min(70vh, 720px)" }}>
          <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
        </div>
      </div>

      <div className="card mt">
        <h3>How to use</h3>
        <ol style={{ marginLeft: 16 }}>
          <li>Start a run from the Upload page and wait for it to finish (DONE).</li>
          <li>
            Open this page at <code>/viewer/&lt;runId&gt;</code> or{" "}
            <code>/viewer?runId=&lt;runId&gt;</code>.
          </li>
          <li>Click Refresh if the overlay hasn’t been generated yet.</li>
        </ol>
      </div>
    </div>
  );
}
