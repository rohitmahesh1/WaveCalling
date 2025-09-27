// frontend/src/hooks/useOverlay.ts
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import type { OverlayPayload } from "@/utils/types";
import { getOverlay, isFresh } from "@/utils/api";

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
    // FIX: use /files gateway + apiBase
    setBaseImageUrl(`${apiBase}/files/${id}/output/base.png`);
    // keep previous overlay sticky while new one loads
  }, [runId, apiBase]);

  const computeSig = (data: OverlayPayload | null) => {
    if (!data || !Array.isArray((data as any).tracks)) return "empty";
    let pts = 0;
    for (const t of (data as any).tracks) pts += Array.isArray((t as any).poly) ? t.poly.length : 0;
    return `${(data as any).tracks.length}|${pts}`;
  };

  const actuallyFetch = React.useCallback(
    async (id: string, force: boolean): Promise<boolean> => {
      let changed = false;
      // only show spinner if nothing rendered yet
      if (!overlay) setOverlayLoading(true);
      try {
        // Use the ETag-aware helper (handles 204/304/aborts internally)
        const { data, notModified } = await getOverlay(apiBase, id, {
          key: `GET /api/runs/${id}/overlay`,
        });

        // If server said "not modified" or we got nothing new, don't change UI
        if (notModified || !data) {
          return false;
        }

        const nextSig = computeSig(data);
        const prevSig = lastSigRef.current;
        changed = nextSig !== prevSig;

        if (changed) {
          lastSigRef.current = nextSig;
          setOverlay(data);

          let pts = 0;
          const tracks = (data as any).tracks || [];
          for (const t of tracks) pts += (t.poly?.length || 0);
          setOverlaySummary({ tracks: tracks.length, points: pts });
        }
        return changed;
      } finally {
        setOverlayLoading(false);
        lastFetchedAtRef.current = Date.now();
      }
    },
    [overlay, apiBase]
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

      // decide when we must fetch immediately
      const mustFetchNow = force || lastSigRef.current === "" || !overlay;

      if (!mustFetchNow && elapsed < MIN_INTERVAL_MS) {
        // Coalesce: schedule one trailing fetch after remaining window
        pendingRequestedRef.current = true;
        if (pendingTimerRef.current == null) {
          const delay = Math.max(0, MIN_INTERVAL_MS - elapsed);
          pendingTimerRef.current = window.setTimeout(async () => {
            pendingTimerRef.current = null;
            if (pendingRequestedRef.current) {
              pendingRequestedRef.current = false;
              inflightRef.current = actuallyFetch(id, false);
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

      const task = actuallyFetch(id, force);
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
