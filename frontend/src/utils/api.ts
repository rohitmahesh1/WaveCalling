import { RunInfo } from "./types";

export async function listRuns(apiBase: string): Promise<RunInfo[]> {
  const r = await fetch(`${apiBase}/api/runs`);
  if (!r.ok) throw new Error(`listRuns failed: ${r.status}`);
  return r.json();
}

// FormData should contain files[] (one or more) + optional fields
export async function startRun(apiBase: string, form: FormData) {
  const r = await fetch(`${apiBase}/api/runs`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`startRun failed: ${r.status}`);
  return r.json() as Promise<{ run_id: string; status: string; info: RunInfo }>;
}

export async function getRun(apiBase: string, runId: string) {
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}`);
  if (!r.ok) throw new Error(`getRun failed: ${r.status}`);
  return r.json() as Promise<{ info: RunInfo; artifacts: Record<string, string> }>;
}

export async function cancelRun(apiBase: string, runId: string) {
  const r = await fetch(`${apiBase}/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
  if (!r.ok) throw new Error(`cancelRun failed: ${r.status}`);
  return r.json();
}
