// frontend/src/pages/UploadRun.tsx
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { startRun, getRun, cancelRun } from "@/utils/api";
import { useSSE } from "@/hooks/useSSE"

import type { RunInfo, JobEvent } from "@/utils/types";


export default function UploadRun() {
  const apiBase = useApiBase();

  // form state
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const [overrides, setOverrides] = React.useState<string>("");
  const [verbose, setVerbose] = React.useState<boolean>(false);
  const [runName, setRunName] = React.useState<string>("");

  // run state
  const [runId, setRunId] = React.useState<string | null>(null);
  const [info, setInfo] = React.useState<RunInfo | null>(null);
  const [running, setRunning] = React.useState<boolean>(false);

  // UI state
  const [log, setLog] = React.useState<string[]>([]);
  const [progressPct, setProgressPct] = React.useState<number>(0);
  const [artifacts, setArtifacts] = React.useState<Record<string, string>>({});

  // SSE stream
  const sseUrl = React.useMemo(
    () => (runId ? `${apiBase}/api/runs/${encodeURIComponent(runId)}/events` : null),
    [apiBase, runId]
  );
  const { status: sseStatus, last, close } = useSSE<JobEvent>({ url: sseUrl });

  React.useEffect(() => {
    if (!last || !last.data) return;         // guard
    const evt: JobEvent = last.data;         // keep the type

    const phase = evt.phase ?? "(event)";
    const msg = evt.message ?? "";
    setLog((L) => [...L, `[${phase}] ${msg}`]);

    if (typeof evt.progress === "number" && !Number.isNaN(evt.progress)) {
        setProgressPct(Math.max(0, Math.min(100, Math.round(evt.progress * 100))));
    }

    if (evt.extra && typeof evt.extra === "object") {
        const entries = Object.entries(evt.extra)
        .filter(([, v]) => typeof v === "string" && (v as string).startsWith("/runs/")) as [string, string][];
        if (entries.length) {
        setLog((L) => [...L, ...entries.map(([k, v]) => `[${phase}] ${k}: ${v}`)]);
        }
    }

    if (phase === "DONE" || phase === "ERROR") {
        setRunning(false);
        if (runId) {
        getRun(apiBase, runId)
            .then((r) => {
            setInfo(r.info);
            setArtifacts(r.artifacts || {});
            })
            .catch(() => void 0);
        }
        close();
    }
    }, [last, close, apiBase, runId]);

  async function onStart(e: React.FormEvent) {
    e.preventDefault();
    setLog([]);
    setProgressPct(0);
    setArtifacts({});
    setInfo(null);

    // Build FormData explicitly so we append all files
    const fd = new FormData();
    if (fileInputRef.current?.files && fileInputRef.current.files.length > 0) {
      Array.from(fileInputRef.current.files).forEach((f) => fd.append("files", f));
    } else {
      setLog((L) => [...L, "No files selected."]);
      return;
    }
    if (runName.trim()) fd.append("run_name", runName.trim());
    if (overrides.trim()) fd.append("config_overrides", overrides.trim());
    if (verbose) fd.append("verbose", "true");

    try {
      const res = await startRun(apiBase, fd);
      setRunId(res.run_id);
      setInfo(res.info);
      setRunning(true);
      setLog((L) => [...L, `run_id=${res.run_id}`, `[${res.status}] subscribed`]);
    } catch (err: any) {
      setLog((L) => [...L, `startRun error: ${String(err)}`]);
    }
  }

  async function onCancel() {
    if (!runId) return;
    try {
      await cancelRun(apiBase, runId);
      setLog((L) => [...L, "[CANCEL] requested"]);
      setRunning(false);
      close();
    } catch (err: any) {
      setLog((L) => [...L, `[CANCEL] error: ${String(err)}`]);
    }
  }

  const disabled = running;

  return (
    <div className="container" style={{ maxWidth: 860, margin: "0 auto", padding: "1rem" }}>
      <h2>Upload & Run</h2>

      <form onSubmit={onStart} className="card" style={{ padding: 12, marginBottom: 16 }}>
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <label style={{ display: "block", marginBottom: 6 }}>Files (CSV/XLS/PNG/JPG)</label>
            <input type="file" name="files" ref={fileInputRef} multiple disabled={disabled} />
          </div>

          <div>
            <label style={{ display: "block", marginBottom: 6 }}>Run name (optional)</label>
            <input
              type="text"
              value={runName}
              onChange={(e) => setRunName(e.target.value)}
              placeholder="e.g. Test run"
              disabled={disabled}
              style={{ width: "100%" }}
            />
          </div>

          <div>
            <label style={{ display: "block", marginBottom: 6 }}>Config overrides (JSON, optional)</label>
            <textarea
              value={overrides}
              onChange={(e) => setOverrides(e.target.value)}
              placeholder='{"peaks":{"prominence":4},"kymo":{"onnx":{"thresholds":{"thr_bi":0.17}}}}'
              rows={4}
              spellCheck={false}
              disabled={disabled}
              style={{ width: "100%", fontFamily: "monospace" }}
            />
          </div>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={verbose}
              onChange={(e) => setVerbose(e.target.checked)}
              disabled={disabled}
            />
            verbose
          </label>

          <div style={{ display: "flex", gap: 8 }}>
            <button type="submit" disabled={disabled}>
              Start run
            </button>
            <button type="button" onClick={onCancel} disabled={!running}>
              Cancel
            </button>
          </div>
        </div>
      </form>

      <div className="card" style={{ padding: 12, marginBottom: 16 }}>
        <div style={{ marginBottom: 8 }}>
          <strong>Run:</strong> <code>{runId ?? "-"}</code>
        </div>
        <div style={{ marginBottom: 8 }}>
          <strong>SSE:</strong> {sseStatus}
        </div>
        <div style={{ marginBottom: 8 }}>
          <strong>Progress:</strong>{" "}
          <progress value={progressPct} max={100} style={{ verticalAlign: "middle" }} />{" "}
          {progressPct}%
        </div>
        {info && (
          <div style={{ marginBottom: 8 }}>
            <strong>Status:</strong> {info.status} {info.error ? `— ${info.error}` : ""}
          </div>
        )}
        {Object.keys(artifacts).length > 0 && (
          <div style={{ marginTop: 8 }}>
            <strong>Artifacts:</strong>
            <ul style={{ marginTop: 6 }}>
              {Object.entries(artifacts).map(([k, v]) => (
                <li key={k}>
                  {k}:{" "}
                  <a href={v} target="_blank" rel="noreferrer">
                    {v}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="card" style={{ padding: 12 }}>
        <strong>Log</strong>
        <pre
          style={{
            marginTop: 8,
            maxHeight: 260,
            overflow: "auto",
            background: "#0b0e14",
            color: "#c8d3f5",
            padding: 10,
            borderRadius: 6,
          }}
        >
{log.join("\n")}
        </pre>
      </div>
    </div>
  );
}
