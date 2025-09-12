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

type RefreshOpts = { force?: boolean };

export function useOverlay(runId: string | null) {
  const apiBase = useApiBase();

  const [overlay, setOverlay] = React.useState<OverlayPayload | null>(null);
  const [overlayLoading, setOverlayLoading] = React.useState(false);
  const [overlaySummary, setOverlaySummary] = React.useState<{ tracks: number; points: number } | null>(null);
  const [baseImageUrl, setBaseImageUrl] = React.useState<string | null>(null);

  // Track in-flight + min interval gating
  const inflightRef = React.useRef<Promise<boolean> | null>(null);
  const lastFetchedAtRef = React.useRef<number>(0);
  const pendingTimerRef = React.useRef<number | null>(null);
  const pendingRequestedRef = React.useRef<boolean>(false);

  // Signature of last applied overlay to skip redundant setState + to tell callers if it changed
  const lastSigRef = React.useRef<string>("");

  // Minimum spacing between network fetches (ms)
  const MIN_INTERVAL_MS = 1200;

  // Reset state when run changes
  React.useEffect(() => {
    const id = sanitizeRunId(runId);
    if (!id) {
      setOverlay(null);
      setOverlaySummary(null);
      setBaseImageUrl(null);
      lastSigRef.current = "";
      return;
    }
    setBaseImageUrl(`/runs/${id}/output/base.png`);
    // keep previous overlay sticky while new one loads
  }, [runId]);

  const computeSig = (data: OverlayPayload | null) => {
    if (!data || !Array.isArray(data.tracks)) return "empty";
    let pts = 0;
    for (const t of data.tracks) pts += Array.isArray(t.poly) ? t.poly.length : 0;
    // tracks count + total points is a cheap, stable proxy for change without deep hashing
    return `${data.tracks.length}|${pts}`;
  };

  const actuallyFetch = React.useCallback(async (id: string): Promise<boolean> => {
    let changed = false;
    // Only show loading if we truly have nothing rendered yet
    if (!overlay) setOverlayLoading(true);
    try {
      // Unified endpoint (final/partial/ndjson fallback). No cache-buster; rely on caller throttling.
      const url = `${apiBase}/api/runs/${encodeURIComponent(id)}/overlay`;
      const resp = await fetch(url, { cache: "no-cache" });
      if (!resp.ok) {
        // keep last known overlay on errors
        return false;
      }
      const data: OverlayPayload = await resp.json();

      // compute signature and avoid redundant setState
      const nextSig = computeSig(data);
      const prevSig = lastSigRef.current;
      changed = nextSig !== prevSig;

      if (changed) {
        lastSigRef.current = nextSig;
        setOverlay(data);

        let pts = 0;
        for (const t of data.tracks || []) pts += (t.poly?.length || 0);
        setOverlaySummary({ tracks: data.tracks?.length || 0, points: pts });
      }
      return changed;
    } finally {
      setOverlayLoading(false);
      lastFetchedAtRef.current = Date.now();
    }
  }, [apiBase, overlay]);

  const refreshOverlay = React.useCallback(
    async (opts?: RefreshOpts): Promise<boolean> => {
      const id = sanitizeRunId(runId);
      if (!id) return false;

      // If a fetch is already running, piggyback on it
      if (inflightRef.current) {
        try {
          return await inflightRef.current;
        } catch {
          return false;
        }
      }

      const now = Date.now();
      const elapsed = now - lastFetchedAtRef.current;
      const force = !!opts?.force;

      if (!force && elapsed < MIN_INTERVAL_MS) {
        // Coalesce: schedule exactly one trailing fetch after the remaining window
        pendingRequestedRef.current = true;
        if (pendingTimerRef.current == null) {
          const delay = Math.max(0, MIN_INTERVAL_MS - elapsed);
          pendingTimerRef.current = window.setTimeout(async () => {
            pendingTimerRef.current = null;
            if (pendingRequestedRef.current) {
              pendingRequestedRef.current = false;
              inflightRef.current = actuallyFetch(id);
              try {
                await inflightRef.current;
              } finally {
                inflightRef.current = null;
              }
            }
          }, delay) as unknown as number;
        }
        return false;
      }

      const task = actuallyFetch(id);
      inflightRef.current = task;
      try {
        return await task;
      } finally {
        inflightRef.current = null;
      }
    },
    [runId, actuallyFetch]
  );

  return { overlay, overlayLoading, overlaySummary, baseImageUrl, refreshOverlay };
}
