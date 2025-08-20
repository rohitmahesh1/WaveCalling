import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { startRun, cancelRun } from "@/utils/api";
import type { RunInfo } from "@/utils/types";

const STORAGE_KEY = "config:overrides";
const HEIGHT_KEY = "config:overrides:height";

export default function UploadPanel({
  disabled,
  onStarted,
}: {
  disabled?: boolean;
  onStarted?: (runId: string, info: RunInfo) => void;
}) {
  const navigate = useNavigate();
  const apiBase = useApiBase();
  const { selectedRunId, setSelectedRunId, appendLog } = useDashboard();

  // form state
  const fileRef = React.useRef<HTMLInputElement | null>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);
  const [overrides, setOverrides] = React.useState("");
  const [runName, setRunName] = React.useState("");
  const [verbose, setVerbose] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [startHeight, setStartHeight] = React.useState<number | undefined>(undefined);

  // hydrate overrides + height
  React.useEffect(() => {
    try {
      const v = window.localStorage.getItem(STORAGE_KEY);
      if (typeof v === "string") setOverrides(v);
      const h = window.localStorage.getItem(HEIGHT_KEY);
      if (h) setStartHeight(Math.max(96, Math.min(0.7 * window.innerHeight, parseInt(h, 10))));
    } catch {}
  }, []);

  // persist on manual edit
  const updateOverrides = React.useCallback((val: string) => {
    setOverrides(val);
    try { window.localStorage.setItem(STORAGE_KEY, val); } catch {}
  }, []);

  // persist height after user drags the native resize handle
  const persistHeight = React.useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const h = el.offsetHeight;
    setStartHeight(h);
    try { window.localStorage.setItem(HEIGHT_KEY, String(h)); } catch {}
  }, []);

  // reflect changes from /config page (cross-tab/window)
  React.useEffect(() => {
    function onStorage(e: StorageEvent) {
      if (e.key === STORAGE_KEY) setOverrides(e.newValue || "");
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // live JSON validation
  const { jsonError, jsonWarning } = React.useMemo(() => {
    const txt = overrides.trim();
    if (!txt) return { jsonError: null as string | null, jsonWarning: null as string | null };
    try {
      const parsed = JSON.parse(txt);
      const warn =
        parsed && typeof parsed === "object" && !Array.isArray(parsed)
          ? null
          : "Overrides should be a JSON object (not an array or primitive).";
      return { jsonError: null, jsonWarning: warn };
    } catch (e: any) {
      return { jsonError: String(e?.message || "Invalid JSON"), jsonWarning: null };
    }
  }, [overrides]);

  const canSubmit = !disabled && !busy && !jsonError;

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
      <div className="flex items-center justify-between">
        <h2 className="text-slate-200 font-semibold">Upload & Run</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            onClick={() => navigate("/config")}
            title="Open full configuration editor"
          >
            Open Config Editor
          </button>
          <button
            type="button"
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-300 hover:bg-slate-800"
            onClick={() => {
              try {
                const v = window.localStorage.getItem(STORAGE_KEY) || "";
                setOverrides(v);
              } catch {}
            }}
            title="Reload overrides from editor"
          >
            Pull Latest
          </button>
        </div>
      </div>

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
          <div className="flex items-center justify-between mb-1">
            <label className="block text-sm text-slate-300">Config overrides (JSON)</label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
                onClick={() => {
                  try {
                    if (!overrides.trim()) return;
                    const pretty = JSON.stringify(JSON.parse(overrides), null, 2);
                    updateOverrides(pretty);
                  } catch {}
                }}
                title="Format / prettify JSON"
              >
                Format
              </button>
            </div>
          </div>

          <textarea
            ref={textareaRef}
            className={[
              "w-full bg-slate-900/60 text-slate-200 px-2 py-2 rounded border font-mono text-xs focus:outline-none",
              "resize-y overflow-auto min-h-[96px] max-h-[70vh]",
              jsonError ? "border-rose-600 focus:ring-1 focus:ring-rose-400" : "border-slate-600 focus:ring-1 focus:ring-slate-400",
            ].join(" ")}
            value={overrides}
            onChange={(e) => updateOverrides(e.target.value)}
            onMouseUp={persistHeight}
            onTouchEnd={persistHeight}
            spellCheck={false}
            placeholder='{"peaks":{"prominence":4},"kymo":{"onnx":{"thresholds":{"thr_bi":0.17}}}}'
            style={startHeight ? { height: startHeight } as React.CSSProperties : undefined}
            disabled={!canSubmit && !jsonError /* allow edits even when busy if error present */}
            aria-invalid={Boolean(jsonError)}
          />

          <div className="mt-1 text-xs">
            {jsonError ? (
              <span className="text-rose-400">JSON error: {jsonError}</span>
            ) : jsonWarning ? (
              <span className="text-amber-400">{jsonWarning}</span>
            ) : overrides.trim() ? (
              <span className="text-emerald-400">JSON is valid.</span>
            ) : (
              <span className="text-slate-400">Paste or edit overrides (optional).</span>
            )}
          </div>
        </div>

        <label className="inline-flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={verbose}
            onChange={(e) => setVerbose(e.target.checked)}
            disabled={!canSubmit}
          />
          verbose
        </label>

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
            title={jsonError ? "Fix JSON errors before starting" : undefined}
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
