// src/hooks/useOverlay.ts
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import type { OverlayPayload } from "@/utils/types";

const RUN_ID_RE = /^[a-f0-9]{6,32}$/i;
function sanitizeRunId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  if (s.startsWith("http://") || s.startsWith("https://")) return null;
  if (s.includes("/")) return null;
  if (!RUN_ID_RE.test(s)) return null;
  return s;
}

export function useOverlay(runId: string | null) {
  const apiBase = useApiBase();

  const [overlay, setOverlay] = React.useState<OverlayPayload | null>(null);
  const [overlayLoading, setOverlayLoading] = React.useState(false);
  const [overlaySummary, setOverlaySummary] = React.useState<{ tracks: number; points: number } | null>(null);
  const [baseImageUrl, setBaseImageUrl] = React.useState<string | null>(null);

  // track in-flight to avoid overlapping fetches
  const inflightRef = React.useRef<Promise<void> | null>(null);

  // Reset state when run changes
  React.useEffect(() => {
    const id = sanitizeRunId(runId);
    if (!id) {
      setOverlay(null);
      setOverlaySummary(null);
      setBaseImageUrl(null);
      return;
    }
    // deterministic base image url (even if it 404s early)
    setBaseImageUrl(`/runs/${id}/output/base.png`);
    // don’t clear overlay here; keep it sticky across retries
  }, [runId]);

  const refreshOverlay = React.useCallback(async () => {
    const id = sanitizeRunId(runId);
    if (!id) {
      // no valid id → nothing to do
      return;
    }
    // coalesce if a fetch is already running
    if (inflightRef.current) {
      await inflightRef.current;
      return;
    }

    const task = (async () => {
      // Only show “Loading…” if we don’t have any overlay yet.
      if (!overlay) setOverlayLoading(true);
      try {
        // unified endpoint chooses final or partial if available
        const url = `${apiBase}/api/runs/${encodeURIComponent(id)}/overlay?t=${Date.now()}`;
        const resp = await fetch(url, { cache: "no-cache" });
        if (!resp.ok) {
          // Do NOT clear overlay on failure; keep last good view
          return;
        }
        const data: OverlayPayload = await resp.json();
        setOverlay(data);

        let pts = 0;
        for (const t of data.tracks || []) pts += (t.poly?.length || 0);
        setOverlaySummary({ tracks: data.tracks?.length || 0, points: pts });
      } finally {
        setOverlayLoading(false);
      }
    })();

    inflightRef.current = task;
    try {
      await task;
    } finally {
      inflightRef.current = null;
    }
  }, [apiBase, runId, overlay]);

  return { overlay, overlayLoading, overlaySummary, baseImageUrl, refreshOverlay };
}
