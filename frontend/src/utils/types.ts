// frontend/src/utils/types.ts

export type RunPhase =
  | "QUEUED" | "RUNNING" | "DONE" | "ERROR" | "CANCELLED"
  | "INIT" | "DISCOVER" | "TABLE2HEATMAP" | "KYMO" | "PROCESS" | "OVERLAY"
  | "WRITE" | "WRITE_PARTIAL";

export interface JobEvent {
  phase: RunPhase;
  message: string;
  progress: number;                  // 0..1
  // May contain server-rewritten URLs for partial/final artifacts
  // or other small payloads for the UI.
  extra?: Record<string, unknown>;
}

export interface RunInfo {
  run_id: string;
  name: string;
  created_at: string;                // ISO8601
  status: RunPhase;
  error?: string | null;
  input_dir: string;
  output_dir: string;
  plots_dir?: string | null;
  config_path: string;
}

/** Map of key -> URL for artifacts the API exposes. */
export interface ArtifactMap {
  tracks_csv?: string;
  waves_csv?: string;
  overlay_json?: string;
  manifest_json?: string;
  plots_dir?: string;
  output_dir?: string;
  // allow future keys without breaking types
  [k: string]: string | undefined;
}

/** Responses from the API (used by utils/api.ts) */
export interface CreateRunResponse {
  run_id: string;
  status: RunPhase;
  info: RunInfo;
}

export interface RunStatusResponse {
  info: RunInfo;
  artifacts: ArtifactMap;
}

/** Optional: share the SSE connection status type app-wide */
export type SSEStatus = "idle" | "connecting" | "open" | "closed" | "error";

/** Overlay JSON schema produced by the backend for the viewer */
export interface OverlayTrackMetrics {
  dominant_frequency: number;
  period: number;
  num_peaks: number;
  mean_amplitude: number;
}

export interface OverlayTrack {
  id: string;                        // track filename stem, e.g. "102"
  sample: string;                    // sample name from path heuristic
  poly: [number, number][];          // [[y, x], ...] image coords (row, col)
  peaks: number[];                   // indices into poly for detected peaks
  metrics: OverlayTrackMetrics;
}


export interface OverlayPayload {
  version: number;
  tracks: OverlayTrack[];
}
