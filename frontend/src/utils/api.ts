// frontend/src/utils/api.ts
import type {
  RunInfo,
  RunStatusResponse,
  CreateRunResponse,
  ProgressResponse,
  TracksListResponse,
  TrackDetailResponse,
} from "./types";

/* ----------------------------- Types (new) ----------------------------- */
export interface SnapshotResponse {
  run_id: string;
  status: "QUEUED" | "RUNNING" | "DONE" | "ERROR" | "CANCELLED";
  error?: string | null;
  overlay_version: number;
  artifacts: Record<string, boolean>;
  progress?: any | null; // tiny pass-through (optional)
}

export type OverlayPayload =
  | { version?: number; tracks?: any[] } // NDJSON-assembled or structured
  | Record<string, any>;

export interface CondResult<T> {
  data: T | null;
  notModified: boolean;
  etag?: string | null;
}

/* --------------------- Small fetch helper (new) ----------------------- */
/** De-dupe + abort identical in-flight requests by key (method+url by default). */
const inflight = new Map<string, AbortController>();
async function fetchx(
  url: string,
  init: RequestInit = {},
  opts: { key?: string; signal?: AbortSignal; timeoutMs?: number } = {}
): Promise<Response> {
  const key = opts.key || `${(init.method || "GET").toUpperCase()} ${url}`;
  // Abort previous identical
  inflight.get(key)?.abort();
  const ac = new AbortController();

  // Bridge external signal → abort our controller
  if (opts.signal) {
    if (opts.signal.aborted) ac.abort();
    else opts.signal.addEventListener("abort", () => ac.abort(), { once: true });
  }

  inflight.set(key, ac);

  let timeoutId: number | null = null;
  if (opts.timeoutMs && opts.timeoutMs > 0) {
    timeoutId = window.setTimeout(() => ac.abort(), opts.timeoutMs) as unknown as number;
  }

  try {
    const res = await fetch(url, { ...init, signal: ac.signal });
    return res;
  } finally {
    if (timeoutId) window.clearTimeout(timeoutId);
    if (inflight.get(key) === ac) inflight.delete(key);
  }
}

function okOrThrow(r: Response, label: string) {
  if (!r.ok) throw new Error(`${label} failed: ${r.status} ${r.statusText}`);
}

/* ------------------------- ETag cache (new) --------------------------- */
const etagCache = new Map<string, string>();

async function conditionalGet<T>(
  url: string,
  init: RequestInit = {},
  opts: { key?: string; signal?: AbortSignal } = {}
): Promise<CondResult<T>> {
  const headers = new Headers(init.headers || {});
  const cached = etagCache.get(url);
  if (cached) headers.set("If-None-Match", cached);

  const res = await fetchx(url, { ...init, headers }, { key: opts.key, signal: opts.signal });
  const et = res.headers.get("ETag");
  if (res.status === 304) {
    // Not modified—keep existing state
    return { data: null, notModified: true, etag: et || cached || null };
  }
  okOrThrow(res, `GET ${url}`);
  const data = (await res.json()) as T;
  if (et) etagCache.set(url, et);
  return { data, notModified: false, etag: et };
}

/* ----------------------------- API calls ------------------------------ */
// Note: all functions accept optional opts for AbortSignal + de-dupe key.
// Existing call sites that pass only (apiBase, ...) still work.

export async function listRuns(apiBase: string, opts: { signal?: AbortSignal; key?: string } = {}): Promise<RunInfo[]> {
  const url = `${apiBase}/api/runs`;
  const r = await fetchx(url, { cache: "no-cache" }, opts);
  okOrThrow(r, "listRuns");
  return r.json();
}

// FormData should contain files[] + optional fields
export async function startRun(apiBase: string, form: FormData, opts: { signal?: AbortSignal; key?: string } = {}) {
  const url = `${apiBase}/api/runs`;
  const r = await fetchx(url, { method: "POST", body: form }, opts);
  okOrThrow(r, "startRun");
  return r.json() as Promise<CreateRunResponse>;
}

export async function getRun(apiBase: string, runId: string, opts: { signal?: AbortSignal; key?: string } = {}) {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}`;
  const r = await fetchx(url, { cache: "no-cache" }, opts);
  okOrThrow(r, "getRun");
  return r.json() as Promise<RunStatusResponse>;
}

export async function cancelRun(apiBase: string, runId: string, opts: { signal?: AbortSignal; key?: string } = {}) {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/cancel`;
  const r = await fetchx(url, { method: "POST" }, opts);
  okOrThrow(r, "cancelRun");
  return r.json();
}

export async function resumeRun(
  apiBase: string,
  runId: string,
  verbose = false,
  opts: { signal?: AbortSignal; key?: string } = {}
) {
  const body = verbose ? new URLSearchParams({ verbose: "true" }) : undefined;
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/resume`;
  const r = await fetchx(
    url,
    { method: "POST", body, headers: body ? { "Content-Type": "application/x-www-form-urlencoded" } : undefined },
    opts
  );
  okOrThrow(r, "resumeRun");
  return r.json();
}

export async function deleteRun(
  apiBase: string,
  runId: string,
  opts: { force?: boolean; signal?: AbortSignal; key?: string } = {}
) {
  const qs = opts.force ? "?force=1" : "";
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}${qs}`;
  const r = await fetchx(url, { method: "DELETE" }, opts);
  okOrThrow(r, "deleteRun");
  return r.json();
}

/** Plain (non-conditional) progress getter — keep current behavior for now. */
export async function getProgress(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<ProgressResponse> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/progress`;
  const r = await fetchx(url, { cache: "no-cache" }, opts);
  okOrThrow(r, "getProgress");
  return r.json();
}

/** Optional conditional progress (use when server adds ETag to /progress). */
export async function getProgressCond(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<CondResult<ProgressResponse>> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/progress`;
  return conditionalGet<ProgressResponse>(url, { cache: "no-cache" }, opts);
}

export async function listTracks(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<TracksListResponse> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/tracks`;
  const r = await fetchx(url, { cache: "no-cache" }, opts);
  okOrThrow(r, "listTracks");
  return r.json();
}

// >>> detail endpoint for exact Python regression arrays, peaks, etc.
export async function getTrackDetail(
  apiBase: string,
  runId: string,
  trackId: string,
  params: { includeSine?: boolean; includeResidual?: boolean; range?: string; freqSource?: "auto" | "metrics" } = {},
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<TrackDetailResponse> {
  const qp = new URLSearchParams();
  if (params.includeSine) qp.set("include_sine", "1");
  if (params.includeResidual) qp.set("include_residual", "1");
  if (params.range) qp.set("range", params.range);
  if (params.freqSource) qp.set("freq_source", params.freqSource);
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/tracks/${encodeURIComponent(trackId)}${
    qp.toString() ? `?${qp.toString()}` : ""
  }`;
  const r = await fetchx(url, { cache: "no-cache" }, opts);
  okOrThrow(r, "getTrackDetail");
  return r.json();
}

/* -------------------- New conditional endpoints ---------------------- */
export async function getSnapshot(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<CondResult<SnapshotResponse>> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/snapshot`;
  return conditionalGet<SnapshotResponse>(url, { cache: "no-cache" }, opts);
}

export async function getOverlay(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<CondResult<OverlayPayload>> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/overlay`;
  // Note: server returns 304 when unchanged; we use weak file etag.
  return conditionalGet<OverlayPayload>(url, { cache: "no-cache" }, opts);
}

/* ---------------------- Helpers for consumers ------------------------ */
/** Convenience: truthy when `result` contains fresh data. */
export function isFresh<T>(r: CondResult<T>): r is { data: T; notModified: false; etag?: string | null } {
  return !r.notModified && r.data != null;
}
