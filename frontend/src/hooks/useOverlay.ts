// frontend/src/hooks/useOverlay.ts
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

  // in-flight + gating
  const inflightRef = React.useRef<Promise<boolean> | null>(null);
  const lastFetchedAtRef = React.useRef<number>(0);
  const pendingTimerRef = React.useRef<number | null>(null);
  const pendingRequestedRef = React.useRef<boolean>(false);

  // last applied overlay signature
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
    // Keep previous overlay sticky while new one loads
  }, [runId]);

  const computeSig = (data: OverlayPayload | null) => {
    if (!data || !Array.isArray(data.tracks)) return "empty";
    let pts = 0;
    for (const t of data.tracks) pts += Array.isArray(t.poly) ? t.poly.length : 0;
    return `${data.tracks.length}|${pts}`;
  };

  // Fetch helper with optional cache-busting
  const fetchOverlayOnce = React.useCallback(
    async (id: string, bust: boolean): Promise<Response> => {
      const u = new URL(`${apiBase}/api/runs/${encodeURIComponent(id)}/overlay`, window.location.origin);
      if (bust) u.searchParams.set("t", String(Date.now())); // avoid 304 on first/forced load
      return await fetch(u.toString(), { cache: bust ? "no-store" : "no-cache" });
    },
    [apiBase]
  );

  const actuallyFetch = React.useCallback(
    async (id: string, bust: boolean): Promise<boolean> => {
      let changed = false;
      // Only show spinner if nothing rendered yet
      if (!overlay) setOverlayLoading(true);
      try {
        // First attempt
        let resp = await fetchOverlayOnce(id, bust);

        // If server replies 304 and we don't have an overlay in memory yet,
        // do a hard refetch with cache-buster to get a 200 + body.
        if (resp.status === 304 && !overlay) {
          resp = await fetchOverlayOnce(id, true);
        }

        if (!resp.ok) {
          // keep last known overlay on errors
          return false;
        }

        const data: OverlayPayload = await resp.json();

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
    },
    [overlay, fetchOverlayOnce]
  );

  const refreshOverlay = React.useCallback(
    async (opts?: RefreshOpts): Promise<boolean> => {
      const id = sanitizeRunId(runId);
      if (!id) return false;

      // piggyback if there’s already a fetch in flight
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

      // Decide when we must bypass cache:
      //  - first time (no signature yet) or no overlay in memory
      //  - explicit force request
      const mustBust = force || lastSigRef.current === "" || !overlay;

      if (!force && elapsed < MIN_INTERVAL_MS) {
        // Coalesce: schedule one trailing fetch after remaining window
        pendingRequestedRef.current = true;
        if (pendingTimerRef.current == null) {
          const delay = Math.max(0, MIN_INTERVAL_MS - elapsed);
          pendingTimerRef.current = window.setTimeout(async () => {
            pendingTimerRef.current = null;
            if (pendingRequestedRef.current) {
              pendingRequestedRef.current = false;
              inflightRef.current = actuallyFetch(id, mustBust);
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

      const task = actuallyFetch(id, mustBust);
      inflightRef.current = task;
      try {
        return await task;
      } finally {
        inflightRef.current = null;
      }
    },
    [runId, overlay, actuallyFetch]
  );

  return { overlay, overlayLoading, overlaySummary, baseImageUrl, refreshOverlay };
}
