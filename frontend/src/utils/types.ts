export type RunPhase =
  | "QUEUED" | "RUNNING" | "DONE" | "ERROR" | "CANCELLED"
  | "INIT" | "DISCOVER" | "TABLE2HEATMAP" | "KYMO" | "PROCESS" | "OVERLAY" | "WRITE" | "WRITE_PARTIAL";

export interface JobEvent {
  phase: RunPhase;
  message: string;
  progress: number;
  extra?: Record<string, unknown>;
}

export interface RunInfo {
  run_id: string;
  name: string;
  created_at: string;
  status: RunPhase;
  error?: string | null;
  input_dir: string;
  output_dir: string;
  plots_dir?: string | null;
  config_path: string;
}
