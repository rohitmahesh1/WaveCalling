// frontend/src/utils/api.ts
import type {
  RunInfo,
  RunStatusResponse,
  CreateRunResponse,
  ProgressResponse,
  TracksListResponse,
  TrackDetailResponse,
} from "./types";

/* ----------------------------- Debug toggle ----------------------------- */
const DEBUG = (() => {
  try {
    const qs = new URLSearchParams(location.search);
    if (qs.get("debug") === "1") return true;
    return localStorage.getItem("waves_debug") === "1";
  } catch {
    return false;
  }
})();

function d(...args: any[]) {
  if (DEBUG) console.debug(...args);
}

// Simple request-id for correlating server logs
function rid(): string {
  try {
    const a = new Uint8Array(8);
    crypto.getRandomValues(a);
    return Array.from(a, x => x.toString(16).padStart(2, "0")).join("");
  } catch {
    return Math.random().toString(16).slice(2, 10);
  }
}

function headersToObject(h: Headers): Record<string, string> {
  const o: Record<string, string> = {};
  h.forEach((v, k) => (o[k] = v));
  return o;
}

/* ----------------------------- Types ----------------------------- */
export interface SnapshotResponse {
  run_id: string;
  status: "QUEUED" | "RUNNING" | "DONE" | "ERROR" | "CANCELLED";
  error?: string | null;
  overlay_version: number;
  artifacts: Record<string, boolean>;
  progress?: any | null;
}

export type OverlayPayload =
  | { version?: number; tracks?: any[] }
  | Record<string, any>;

export interface CondResult<T> {
  data: T | null;
  notModified: boolean;
  etag?: string | null;
}

/** Minimal event shape; server may send more fields. */
export type RunEvent = Record<string, any> & {
  phase?: string;
  ts?: number | string;
  type?: string; // e.g., "log", "progress", "overlay", "ping"
  message?: string;
};

/* --------------------- Small fetch helper ----------------------- */
/** De-dupe + abort identical in-flight requests by key (method+url by default). */
const inflight = new Map<string, AbortController>();

async function fetchx(
  url: string,
  init: RequestInit = {},
  opts: { key?: string; signal?: AbortSignal; timeoutMs?: number } = {}
): Promise<Response> {
  const method = (init.method || "GET").toUpperCase();
  const key = opts.key || `${method} ${url}`;
  const reqId = rid();
  const t0 = performance.now();

  // Abort the previous identical request
  const prev = inflight.get(key);
  if (prev) {
    prev.abort();
    d(`[API] ${key} — aborted previous inflight`);
  }

  const ac = new AbortController();

  // Propagate external signal
  if (opts.signal) {
    if (opts.signal.aborted) ac.abort();
    else opts.signal.addEventListener("abort", () => ac.abort(), { once: true });
  }

  inflight.set(key, ac);

  let timeoutId: number | null = null;
  if (opts.timeoutMs && opts.timeoutMs > 0) {
    timeoutId = window.setTimeout(() => ac.abort(), opts.timeoutMs) as unknown as number;
  }

  // Inject a client request id for server log correlation
  const headers = new Headers(init.headers || {});
  headers.set("X-Client-Request-Id", reqId);

  // Log request start (include If-None-Match if present)
  const ine = headers.get("If-None-Match");
  if (DEBUG) {
    console.groupCollapsed(
      `%c→ ${key}`,
      "color:#09f",
      JSON.stringify({ reqId, ine }, null, 0)
    );
    console.log("init", { ...init, headers: headersToObject(headers) });
    console.groupEnd();
  }

  try {
    const res = await fetch(url, { ...init, credentials: "include", headers, signal: ac.signal });

    // Snapshot interesting headers for debugging
    const meta = {
      status: res.status,
      statusText: res.statusText,
      etag: res.headers.get("ETag"),
      servedFrom: res.headers.get("X-Served-From"),
      storage: res.headers.get("X-Storage"),
      revision: res.headers.get("X-Revision"),
      hostname: res.headers.get("X-Hostname"),
      cache: res.headers.get("Cache-Control"),
      age: res.headers.get("Age"),
      date: res.headers.get("Date"),
    };
    d(`[API] ← ${key} (${Math.round(performance.now() - t0)}ms)`, meta);

    // Paint a single-line summary when debugging
    if (DEBUG) {
      const tag = meta.etag ? ` etag=${meta.etag}` : "";
      const src = meta.servedFrom ? ` src=${meta.servedFrom}` : "";
      const host = meta.hostname ? ` host=${meta.hostname}` : "";
      console.log(
        `%c[API]%c ${method} ${url} %c${res.status}%c${tag}${src}${host}`,
        "color:#999",
        "color:inherit",
        res.ok ? "color:green" : "color:red",
        "color:#555"
      );
    }

    return res;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      d(`[API] ✕ ${key} aborted (${Math.round(performance.now() - t0)}ms)`);
      // Response-like sentinel so callers can no-op without throwing
      return new Response(null, { status: 499, statusText: "Client Closed Request" });
    }
    console.error(`[API] ${key} error`, err);
    throw err;
  } finally {
    if (timeoutId) window.clearTimeout(timeoutId);
    if (inflight.get(key) === ac) inflight.delete(key);
  }
}

async function okOrThrow(r: Response, label: string): Promise<void> {
  // Treat client-abort sentinel as a no-op
  if (r.status === 499) return;

  if (!r.ok) {
    let body = "";
    try {
      body = await r.text(); // consume once on error path
      try {
        const j = JSON.parse(body);
        if (j && typeof j === "object" && "detail" in j) body = String((j as any).detail);
      } catch {
        /* keep raw */
      }
    } catch {
      /* ignore */
    }
    console.error(`[API] ${label} failed`, { status: r.status, statusText: r.statusText, body });
    throw new Error(`${label} failed: ${r.status} ${r.statusText}${body ? ` — ${body}` : ""}`);
  }
}

/* ------------------------- ETag + data cache --------------------------- */
const etagCache = new Map<string, string>();
const dataCache = new Map<string, unknown>();

async function conditionalGet<T>(
  url: string,
  init: RequestInit = {},
  opts: { key?: string; signal?: AbortSignal } = {}
): Promise<CondResult<T>> {
  const headers = new Headers(init.headers || {});
  const cachedEtag = etagCache.get(url);
  if (cachedEtag) headers.set("If-None-Match", cachedEtag);

  const res = await fetchx(url, { ...init, headers }, { key: opts.key, signal: opts.signal });
  const et = res.headers.get("ETag");

  // Aborted: act like "no change" and return last data if we have it
  if (res.status === 499) {
    const prev = (dataCache.get(url) as T | undefined) ?? null;
    d(`[API] 499 (aborted) ${url} — returning cached=${Boolean(prev)} etag=${cachedEtag || ""}`);
    return { data: prev, notModified: true, etag: cachedEtag || null };
  }

  // Not modified: return cached data if available
  if (res.status === 304) {
    const prev = (dataCache.get(url) as T | undefined) ?? null;
    d(`[API] 304 ${url} — cached=${Boolean(prev)} etag=${et || cachedEtag || ""}`);
    return { data: prev, notModified: true, etag: et || cachedEtag || null };
  }

  // No content (e.g., overlay/progress "not ready yet")
  if (res.status === 204) {
    const prev = (dataCache.get(url) as T | undefined) ?? null;
    d(`[API] 204 ${url} — previous cached=${Boolean(prev)}`);
    // notModified=false (state changed: "still cooking"), but data may be null
    return { data: prev, notModified: false, etag: et || cachedEtag || null };
  }

  await okOrThrow(res, `GET ${url}`);

  // Success with body
  let data: T | null = null;
  const ct = (res.headers.get("Content-Type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    data = (await res.json()) as T;
    d(`[API] 200 ${url} — JSON ok etag=${et || ""}`);
  } else if (res.status !== 204) {
    d(`[API] 200 ${url} — non-JSON Content-Type=${ct}`);
  }

  if (et) etagCache.set(url, et);
  if (data !== null) dataCache.set(url, data);

  return { data, notModified: false, etag: et || null };
}

/* ----------------------------- API calls ------------------------------ */

export async function listRuns(
  apiBase: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<RunInfo[]> {
  const url = `${apiBase}/api/runs`;
  const { data } = await conditionalGet<RunInfo[]>(
    url,
    { cache: "no-cache" },
    { key: opts.key ?? "GET /api/runs", signal: opts.signal }
  );
  // When not modified / aborted and we have a cache, data will be the last value
  return data ?? [];
}

export async function startRun(
  apiBase: string,
  form: FormData,
  opts: { signal?: AbortSignal; key?: string } = {}
) {
  const url = `${apiBase}/api/runs`;
  const r = await fetchx(url, { method: "POST", body: form }, { key: opts.key ?? "POST /api/runs", signal: opts.signal });
  await okOrThrow(r, "startRun");
  return r.json() as Promise<CreateRunResponse>;
}

export async function getRun(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
) {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}`;
  const { data } = await conditionalGet<RunStatusResponse>(
    url,
    { cache: "no-cache" },
    { key: opts.key ?? `GET /api/runs/${runId}`, signal: opts.signal }
  );
  if (!data) {
    // When aborted or 304 without prior cache, noop by throwing AbortError-like
    throw new DOMException("Aborted", "AbortError");
  }
  return data;
}

export async function cancelRun(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
) {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/cancel`;
  const r = await fetchx(url, { method: "POST" }, { key: opts.key ?? `POST /api/runs/${runId}/cancel`, signal: opts.signal });
  await okOrThrow(r, "cancelRun");
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
    { key: opts.key ?? `POST /api/runs/${runId}/resume`, signal: opts.signal }
  );
  await okOrThrow(r, "resumeRun");
  return r.json();
}

export async function deleteRun(
  apiBase: string,
  runId: string,
  opts: { force?: boolean; signal?: AbortSignal; key?: string } = {}
) {
  const qs = opts.force ? "?force=1" : "";
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}${qs}`;
  const r = await fetchx(url, { method: "DELETE" }, { key: opts.key ?? `DELETE /api/runs/${runId}${qs}`, signal: opts.signal });
  await okOrThrow(r, "deleteRun");
  return r.json();
}

export async function getProgress(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<ProgressResponse> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/progress`;
  const r = await fetchx(url, { cache: "no-cache" }, { key: opts.key ?? `GET /api/runs/${runId}/progress`, signal: opts.signal });
  await okOrThrow(r, "getProgress");
  if (r.status === 204) return (dataCache.get(url) as ProgressResponse) ?? ({} as any);
  const data = (await r.json()) as ProgressResponse;
  dataCache.set(url, data);
  const et = r.headers.get("ETag");
  if (et) etagCache.set(url, et);
  return data;
}

export async function getProgressCond(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<CondResult<ProgressResponse>> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/progress`;
  return conditionalGet<ProgressResponse>(
    url,
    { cache: "no-cache" },
    { key: opts.key ?? `GET /api/runs/${runId}/progress`, signal: opts.signal }
  );
}

export async function listTracks(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<TracksListResponse> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/tracks`;
  const r = await fetchx(url, { cache: "no-cache" }, { key: opts.key ?? `GET /api/runs/${runId}/tracks`, signal: opts.signal });
  await okOrThrow(r, "listTracks");
  return r.json();
}

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
  const r = await fetchx(url, { cache: "no-cache" }, { key: opts.key ?? `GET /api/runs/${runId}/tracks/${trackId}?${qp}`, signal: opts.signal });
  await okOrThrow(r, "getTrackDetail");
  return r.json();
}

/* -------------------- Conditional endpoints ---------------------- */
export async function getSnapshot(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<CondResult<SnapshotResponse>> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/snapshot`;
  return conditionalGet<SnapshotResponse>(
    url,
    { cache: "no-cache" },
    { key: opts.key ?? `GET /api/runs/${runId}/snapshot`, signal: opts.signal }
  );
}

export async function getOverlay(
  apiBase: string,
  runId: string,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<CondResult<OverlayPayload>> {
  const url = `${apiBase}/api/runs/${encodeURIComponent(runId)}/overlay`;
  return conditionalGet<OverlayPayload>(
    url,
    { cache: "no-cache" },
    { key: opts.key ?? `GET /api/runs/${runId}/overlay`, signal: opts.signal }
  );
}

/* ---------------------- Helpers for consumers ------------------------ */
export function isFresh<T>(r: CondResult<T>): r is { data: T; notModified: false; etag?: string | null } {
  return !r.notModified && r.data != null;
}

export const DEBUG_LAYERS = [
  "prob",
  "mask_raw",
  "mask_clean",
  "mask_filtered",
  "skeleton",
  "mask_hysteresis",
] as const;

export type DebugLayer = (typeof DEBUG_LAYERS)[number];

export function buildDebugImageUrl(apiBase: string, runId: string, layer: DebugLayer): string {
  return `${apiBase}/api/runs/${encodeURIComponent(runId)}/debug/${encodeURIComponent(layer)}`;
}

/* ---------------------- Upload helpers ------------------------ */
export async function startUpload(
  apiBase: string,
  file: File,
  opts: { signal?: AbortSignal; key?: string } = {}
): Promise<{ upload_url: string; object: string }> {
  const fd = new FormData();
  fd.append("file_name", file.name);
  fd.append("content_type", file.type || "application/octet-stream");

  const url = `${apiBase}/api/uploads/start`;
  const r = await fetchx(url, { method: "POST", body: fd }, { key: opts.key ?? "POST /api/uploads/start", signal: opts.signal });
  await okOrThrow(r, "startUpload");
  return r.json();
}

/** One-shot resumable upload to GCS signed URL (no credentials to API). */
export async function uploadResumable(uploadUrl: string, file: File): Promise<void> {
  const r = await fetch(uploadUrl, {
    method: "PUT",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
  if (!r.ok) {
    let msg = "";
    try {
      msg = await r.text();
    } catch {}
    throw new Error(`upload failed: ${r.status}${msg ? ` — ${msg}` : ""}`);
  }
}

export async function startRunFromGcs(
  apiBase: string,
  objects: string[],
  opts: { runName?: string; overrides?: any; verbose?: boolean } = {}
): Promise<CreateRunResponse> {
  const url = `${apiBase}/api/runs/from_gcs`;
  const r = await fetchx(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        objects,
        run_name: opts.runName ?? null,
        config_overrides: opts.overrides ?? null,
        verbose: !!opts.verbose,
      }),
    },
    { key: `POST /api/runs/from_gcs`, signal: undefined }
  );
  await okOrThrow(r, "startRunFromGcs");
  return r.json();
}

/* ---------------------- SSE helpers (NEW) ------------------------ */
/**
 * Open a per-run Server-Sent Events stream with credentials (cookies),
 * so session-based authorization works cross-origin.
 */
export function openRunEvents(
  apiBase: string,
  runId: string,
  handlers: {
    onEvent?: (data: RunEvent, raw: MessageEvent<string>) => void;
    onOpen?: (ev: Event) => void;
    onError?: (ev: Event) => void;
    /** If persisted the last event id, can pass it here to resume. */
    lastEventId?: string | null;
  } = {}
): EventSource {
  const url = new URL(`${apiBase}/api/runs/${encodeURIComponent(runId)}/events`);
  // Optional: help caches/proxies treat each stream as unique
  url.searchParams.set("v", Date.now().toString());
  if (handlers.lastEventId) url.searchParams.set("last_event_id", handlers.lastEventId);

  const es = new EventSource(url.toString(), { withCredentials: true });

  es.addEventListener("open", (ev) => {
    d(`[SSE] open run=${runId}`);
    handlers.onOpen?.(ev);
  });

  // Default channel
  es.addEventListener("message", (ev: MessageEvent<string>) => {
    try {
      const data = ev.data ? (JSON.parse(ev.data) as RunEvent) : ({} as RunEvent);
      if (data?.type !== "ping") handlers.onEvent?.(data, ev);
    } catch {
      // If server sometimes sends plain text
      handlers.onEvent?.({ type: "raw", message: ev.data }, ev);
    }
  });

  // Common named events (optional; harmless if server doesn't send them)
  for (const name of ["log", "progress", "overlay", "phase", "ping"]) {
    es.addEventListener(name, (ev: MessageEvent<string>) => {
      try {
        const data = ev.data ? (JSON.parse(ev.data) as RunEvent) : ({} as RunEvent);
        if (data?.type !== "ping") handlers.onEvent?.(data, ev);
      } catch {
        handlers.onEvent?.({ type: name, message: ev.data }, ev);
      }
    });
  }

  es.addEventListener("error", (ev) => {
    if (DEBUG) console.warn("[SSE] error (run)", ev);
    handlers.onError?.(ev);
  });

  return es;
}

/** Open the global events bus (if backend exposes /api/runs/events). */
export function openGlobalEvents(
  apiBase: string,
  handlers: {
    onEvent?: (data: RunEvent, raw: MessageEvent<string>) => void;
    onOpen?: (ev: Event) => void;
    onError?: (ev: Event) => void;
  } = {}
): EventSource {
  const url = new URL(`${apiBase}/api/runs/events`);
  url.searchParams.set("v", Date.now().toString());

  const es = new EventSource(url.toString(), { withCredentials: true });

  es.addEventListener("open", (ev) => {
    d(`[SSE] open global`);
    handlers.onOpen?.(ev);
  });

  es.addEventListener("message", (ev: MessageEvent<string>) => {
    try {
      const data = ev.data ? (JSON.parse(ev.data) as RunEvent) : ({} as RunEvent);
      if (data?.type !== "ping") handlers.onEvent?.(data, ev);
    } catch {
      handlers.onEvent?.({ type: "raw", message: ev.data }, ev);
    }
  });

  for (const name of ["log", "progress", "overlay", "phase", "ping"]) {
    es.addEventListener(name, (ev: MessageEvent<string>) => {
      try {
        const data = ev.data ? (JSON.parse(ev.data) as RunEvent) : ({} as RunEvent);
        if (data?.type !== "ping") handlers.onEvent?.(data, ev);
      } catch {
        handlers.onEvent?.({ type: name, message: ev.data }, ev);
      }
    });
  }

  es.addEventListener("error", (ev) => {
    if (DEBUG) console.warn("[SSE] error (global)", ev);
    handlers.onError?.(ev);
  });

  return es;
}

/* ---------------------- Debug helpers ------------------------ */
declare global {
  interface Window {
    wavesApiDebug: {
      enable: () => void;
      disable: () => void;
      etags: Map<string, string>;
      cache: Map<string, unknown>;
      inflight: Map<string, AbortController>;
    };
  }
}

if (typeof window !== "undefined") {
  window.wavesApiDebug = {
    enable: () => {
      localStorage.setItem("waves_debug", "1");
      location.reload();
    },
    disable: () => {
      localStorage.removeItem("waves_debug");
      location.reload();
    },
    etags: etagCache,
    cache: dataCache,
    inflight,
  };
}
