// frontend/src/utils/api.ts
import type {
  RunInfo,
  RunStatusResponse,
  CreateRunResponse,
  ProgressResponse,
  TracksListResponse,
  TrackDetailResponse,
} from "./types";

function okOrThrow(r: Response, label: string) {
  if (!r.ok) throw new Error(`${label} failed: ${r.status} ${r.statusText}`);
}

export async function listRuns(apiBase: string): Promise<RunInfo[]> {
  const r = await fetch(`${apiBase}/api/runs`, { cache: "no-cache" });
  okOrThrow(r, "listRuns");
  return r.json();
}

// FormData should contain files[] + optional fields
export async function startRun(apiBase: string, form: FormData) {
  const r = await fetch(`${apiBase}/api/runs`, { method: "POST", body: form });
  okOrThrow(r, "startRun");
  return r.json() as Promise<CreateRunResponse>;
}

export async function getRun(apiBase: string, runId: string) {
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}`, { cache: "no-cache" });
  okOrThrow(r, "getRun");
  return r.json() as Promise<RunStatusResponse>;
}

export async function cancelRun(apiBase: string, runId: string) {
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
  okOrThrow(r, "cancelRun");
  return r.json();
}

export async function resumeRun(apiBase: string, runId: string, verbose = false) {
  const body = verbose ? new URLSearchParams({ verbose: "true" }) : undefined;
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}/resume`, {
    method: "POST",
    body,
    headers: body ? { "Content-Type": "application/x-www-form-urlencoded" } : undefined,
  });
  okOrThrow(r, "resumeRun");
  return r.json();
}

export async function deleteRun(apiBase: string, runId: string, opts: { force?: boolean } = {}) {
  const qs = opts.force ? "?force=1" : "";
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}${qs}`, { method: "DELETE" });
  okOrThrow(r, "deleteRun");
  return r.json();
}

export async function getProgress(apiBase: string, runId: string): Promise<ProgressResponse> {
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}/progress`, { cache: "no-cache" });
  okOrThrow(r, "getProgress");
  return r.json();
}

export async function listTracks(apiBase: string, runId: string): Promise<TracksListResponse> {
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}/tracks`, { cache: "no-cache" });
  okOrThrow(r, "listTracks");
  return r.json();
}

// >>> NEW: detail endpoint for exact Python regression arrays, peaks, etc.
export async function getTrackDetail(
  apiBase: string,
  runId: string,
  trackId: string,
  opts: { includeSine?: boolean } = {}
): Promise<TrackDetailResponse> {
  const qs = opts.includeSine ? "?include_sine=1" : "";
  const r = await fetch(
    `${apiBase}/api/runs/${encodeURIComponent(runId)}/tracks/${encodeURIComponent(trackId)}${qs}`,
    { cache: "no-cache" }
  );
  okOrThrow(r, "getTrackDetail");
  return r.json();
}