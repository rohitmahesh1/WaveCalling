// src/hooks/useOverlay.ts
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { getRun } from "@/utils/api";
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

  const refreshOverlay = React.useCallback(async () => {
    const id = sanitizeRunId(runId);
    if (!id) {
      setOverlay(null);
      setOverlaySummary(null);
      setBaseImageUrl(null);
      return;
    }
    setOverlayLoading(true);
    try {
      const rs = await getRun(apiBase, id);
      const art = rs.artifacts || {};
      setBaseImageUrl(art.base_image ?? null);

      const url = art.overlay_json || art.overlay_json_partial;
      if (!url) {
        setOverlay(null);
        setOverlaySummary(null);
        return;
      }
      const resp = await fetch(url, { cache: "no-cache" });
      if (!resp.ok) {
        setOverlay(null);
        setOverlaySummary(null);
        return;
      }
      const data: OverlayPayload = await resp.json();
      setOverlay(data);

      let pts = 0;
      for (const t of data.tracks || []) pts += (t.poly?.length || 0);
      setOverlaySummary({ tracks: data.tracks?.length || 0, points: pts });
    } catch {
      // swallow transient errors
    } finally {
      setOverlayLoading(false);
    }
  }, [apiBase, runId]);

  return { overlay, overlayLoading, overlaySummary, baseImageUrl, refreshOverlay };
}
