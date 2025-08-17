import React from "react";
import FileDrop from "@/components/FileDrop";
import { useApiBase } from "@/context/ApiContext";
import { startRun, cancelRun } from "@/utils/api";
import { useSSE } from "@/hooks/useSSE";
import JSONPreview from "@/components/JSONPreview";
import RunStatusBadge from "@/components/RunStatusBadge";
import type { JobEvent, RunInfo } from "@/utils/types";

export default function UploadRun() {
  const apiBase = useApiBase();
  const [files, setFiles] = React.useState<FileList | null>(null);
  const [runId, setRunId] = React.useState<string | null>(null);
  const [runInfo, setRunInfo] = React.useState<RunInfo | null>(null);
  const [events, setEvents] = React.useState<JobEvent[]>([]);

  const sseUrl = runId ? `${apiBase}/api/runs/${encodeURIComponent(runId)}/events` : null;
  useSSE(sseUrl, (evt) => setEvents((prev) => [...prev, evt]));

  const onStart = async () => {
    if (!files || files.length === 0) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    // optional: fd.append("run_name", "my-run");
    const resp = await startRun(apiBase, fd);
    setRunId(resp.run_id);
    setRunInfo(resp.info);
    setEvents([{ phase: "RUNNING", message: "subscribed", progress: 0 } as JobEvent]);
  };

  const onCancel = async () => {
    if (!runId) return;
    await cancelRun(apiBase, runId);
  };

  return (
    <div className="stack">
      <h2>New run</h2>
      <FileDrop onFiles={setFiles} />
      <div className="row mt">
        <button className="primary" onClick={onStart} disabled={!files || files.length === 0}>
          Start run
        </button>
        <button onClick={onCancel} disabled={!runId}>Cancel</button>
        {runInfo && (
          <div className="row">
            <div>run_id: <code>{runInfo.run_id}</code></div>
            <RunStatusBadge status={runInfo.status} />
          </div>
        )}
      </div>

      <div className="card mt">
        <h3>Events</h3>
        <div style={{ maxHeight: 320, overflow: "auto", fontFamily: "ui-monospace, SFMono-Regular, Menlo" }}>
          {events.map((e, i) => (
            <div key={i}>
              [{e.phase}] {e.message}{typeof e.progress === "number" ? ` (${Math.round(e.progress * 100)}%)` : ""}
              {e.extra && (
                <div style={{ marginLeft: 12, color: "var(--muted)" }}>
                  <JSONPreview data={e.extra} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="card mt">
        <h3>Debug</h3>
        <JSONPreview data={{ apiBase, runId }} />
      </div>
    </div>
  );
}
