// frontend/src/utils/types.ts

export type RunPhase =
  | "QUEUED" | "RUNNING" | "DONE" | "ERROR" | "CANCELLED"
  | "INIT" | "DISCOVER" | "TABLE2HEATMAP" | "KYMO" | "PROCESS" | "OVERLAY"
  | "WRITE" | "WRITE_PARTIAL";

export interface JobEventExtra {
  // Server may include these for partial/final artifacts and progress hints.
  partial_index?: number;
  tracks_partial?: string;
  waves_partial?: string;
  overlay_partial?: string;
  total_tracks?: number;  // KYMO phase “total_tracks”
  total?: number;         // resume hint “total”
  // Allow future keys without breaking types:
  [k: string]: unknown;
}

export interface JobEvent {
  phase: RunPhase;
  message: string;
  progress: number;                  // 0..1
  extra?: JobEventExtra;             // typed-but-flexible
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
  overlay_json_partial?: string;
  manifest_json?: string;
  run_json?: string;
  events_ndjson?: string;
  progress_json?: string;
  base_image?: string;
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

/** Progress endpoint shape */
export interface ProgressResponse {
  totalTracks: number | null;
  processedCount: number;
  skippedCount?: number | null;
  lastUpdatedAt?: string | null;
  source: "file" | "synthesized";
}

export interface RegressionInfo {
  method: string;                      // e.g. "ransac_poly"
  degree?: number | null;              // polynomial degree if applicable
  params?: Record<string, any> | null; // RANSAC kwargs, etc.
}

export interface OverlayTrackMetrics {
  dominant_frequency: number;
  period: number;
  num_peaks: number;
  mean_amplitude: number;
}

export interface OverlayTrack {
  id: string;
  sample: string;
  poly: [number, number][];            // [[y,x], ...]
  peaks: number[];
  metrics: OverlayTrackMetrics;

  regression?: RegressionInfo | null;
  baseline?: (number | null)[];
  residual?: (number | null)[];
  sine_fit?: (number | null)[] | null; // only if you choose to return it
}

export interface OverlayPayload {
  version: number;
  tracks: OverlayTrack[];
}

export type Track = OverlayTrack;
export type TrackDetailResponse = OverlayTrack;

/** Tracks listing (for download / per-track actions) */
export type TracksListItem = { id: string; url: string };
export type TracksListResponse = { count: number; tracks: TracksListItem[] };
