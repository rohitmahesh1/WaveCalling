import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { startRun, cancelRun } from "@/utils/api";
import type { RunInfo } from "@/utils/types";

export default function UploadPanel({
  disabled,
  onStarted,
}: {
  disabled?: boolean;
  onStarted?: (runId: string, info: RunInfo) => void;
}) {
  const apiBase = useApiBase();
  const { selectedRunId, setSelectedRunId, appendLog } = useDashboard();

  // form state
  const fileRef = React.useRef<HTMLInputElement | null>(null);
  const [overrides, setOverrides] = React.useState("");
  const [verbose, setVerbose] = React.useState(false);
  const [runName, setRunName] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  const canSubmit = !disabled && !busy;

  async function onStart(e: React.FormEvent) {
    e.preventDefault();
    if (!fileRef.current?.files || fileRef.current.files.length === 0) {
      appendLog("No files selected.");
      return;
    }
    const fd = new FormData();
    Array.from(fileRef.current.files).forEach((f) => fd.append("files", f));
    if (runName.trim()) fd.append("run_name", runName.trim());
    if (overrides.trim()) fd.append("config_overrides", overrides.trim());
    if (verbose) fd.append("verbose", "true");

    setBusy(true);
    try {
      const res = await startRun(apiBase, fd);
      appendLog(`[RUN] started run_id=${res.run_id}`);
      setSelectedRunId(res.run_id);
      onStarted?.(res.run_id, res.info);
      // Clear form (keep overrides for convenience)
      setRunName("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (e: any) {
      appendLog(`[RUN] start error: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function onCancel() {
    if (!selectedRunId) return;
    try {
      await cancelRun(apiBase, selectedRunId);
      appendLog(`[CANCEL] requested ${selectedRunId}`);
    } catch (e: any) {
      appendLog(`[CANCEL] error: ${String(e)}`);
    }
  }

  return (
    <section className="rounded-xl border border-slate-700/50 bg-console-700 p-4">
      <h2 className="text-slate-200 font-semibold">Upload & Run</h2>

      <form onSubmit={onStart} className="mt-3 grid gap-3">
        <div>
          <label className="block text-sm text-slate-300 mb-1">Files (CSV/XLS/PNG/JPG)</label>
          <input
            ref={fileRef}
            type="file"
            multiple
            disabled={!canSubmit}
            className="block w-full text-sm text-slate-300 cursor-pointer file:cursor-pointer file:mr-3 file:px-3 file:py-1.5 file:rounded-md file:border file:border-slate-600 file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
          />
        </div>

        <div>
          <label className="block text-sm text-slate-300 mb-1">Run name (optional)</label>
          <input
            className="w-full bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder="e.g. Test run"
            disabled={!canSubmit}
          />
        </div>

        <div>
          <label className="block text-sm text-slate-300 mb-1">Config overrides (JSON, optional)</label>
          <textarea
            className="w-full bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-slate-400"
            value={overrides}
            onChange={(e) => setOverrides(e.target.value)}
            rows={4}
            spellCheck={false}
            placeholder='{"peaks":{"prominence":4},"kymo":{"onnx":{"thresholds":{"thr_bi":0.17}}}}'
            disabled={!canSubmit}
          />
        </div>

        <label className="inline-flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} disabled={!canSubmit} />
          verbose
        </label>

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
          >
            {busy ? "Starting…" : "Start run"}
          </button>
          <button
            type="button"
            onClick={() => void onCancel()}
            disabled={!selectedRunId}
            className="px-3 py-1.5 rounded-md border border-rose-600 text-rose-300 hover:bg-rose-600/10 disabled:opacity-50"
          >
            Cancel selected
          </button>
        </div>
      </form>
    </section>
  );
}
