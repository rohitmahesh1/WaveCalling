// frontend/src/components/UploadPanel.tsx
import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { cancelRun } from "@/utils/api";
import type { RunInfo } from "@/utils/types";

import {
  startUpload,        // POST /api/uploads/start -> { upload_url, object }
  uploadResumable,    // PUT file to upload_url
  startRunFromGcs,    // POST /api/runs/from_gcs with {objects:[...]}
} from "@/utils/api";

const STORAGE_KEY = "config:overrides";
const HEIGHT_KEY = "config:overrides:height";

type StagedObj = { object: string; name: string; size?: number };

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

  const fileRef = React.useRef<HTMLInputElement | null>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);

  const [overrides, setOverrides] = React.useState("");
  const [runName, setRunName] = React.useState("");
  const [verbose, setVerbose] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [startHeight, setStartHeight] = React.useState<number | undefined>(undefined);

  const [staged, setStaged] = React.useState<StagedObj[]>([]);
  const stagedCount = staged.length;

  // ---------- persisted UI state ----------
  React.useEffect(() => {
    try {
      const v = window.localStorage.getItem(STORAGE_KEY);
      if (typeof v === "string") setOverrides(v);
      const h = window.localStorage.getItem(HEIGHT_KEY);
      if (h) setStartHeight(Math.max(96, Math.min(0.7 * window.innerHeight, parseInt(h, 10))));
    } catch {}
  }, []);

  const updateOverrides = React.useCallback((val: string) => {
    setOverrides(val);
    try { window.localStorage.setItem(STORAGE_KEY, val); } catch {}
  }, []);

  const persistHeight = React.useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const h = el.offsetHeight;
    setStartHeight(h);
    try { window.localStorage.setItem(HEIGHT_KEY, String(h)); } catch {}
  }, []);

  // ---------- JSON validation ----------
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

  const canInteract = !disabled && !busy;
  const canUpload = canInteract && !jsonError;
  const canStart = canInteract && !jsonError && stagedCount > 0;

  // ---------- staging to GCS ----------
  async function stageSelectedToGcs() {
    if (!fileRef.current?.files || fileRef.current.files.length === 0) {
      appendLog("No files selected.");
      return;
    }
    setBusy(true);
    try {
      const files = Array.from(fileRef.current.files);
      for (const f of files) {
        appendLog(`[UPLOAD] preparing ${f.name} (${f.size.toLocaleString()} bytes)`);
        // 1) ask API for a resumable upload session
        const { upload_url, object } = await startUpload(apiBase, f);
        // 2) upload directly to GCS
        await uploadResumable(upload_url, f);
        // 3) record the staged object locally
        setStaged((prev) => [...prev, { object, name: f.name, size: f.size }]);
        appendLog(`[UPLOAD] done -> gs://${object}`);
      }
      // keep file input value (so user can re-upload same filename if desired, optional)
    } catch (e: any) {
      appendLog(`[UPLOAD] error: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  // Stage an already-existing GCS sample (no bytes through Cloud Run)
  async function stageSampleGcs(filename: "ripple.csv" | "vertical.csv") {
    if (!canUpload) return;
    const object = `samples/${filename}`;
    setStaged((prev) => {
      if (prev.some((s) => s.object === object)) return prev;
      return [...prev, { object, name: filename }];
    });
    appendLog(`[SAMPLE] staged ${object}`);
  }

  function clearStaged() {
    setStaged([]);
    appendLog("[UPLOADS] cleared staged list (GCS objects remain; bucket lifecycle will clean up).");
  }

  // ---------- start run ----------
  async function onStart(e: React.FormEvent) {
    e.preventDefault();
    if (!stagedCount) {
      appendLog("No staged files. Upload or stage a sample first.");
      return;
    }

    let parsedOverrides: any = null;
    if (overrides.trim()) {
      try { parsedOverrides = JSON.parse(overrides); }
      catch (err: any) {
        appendLog(`[RUN] overrides JSON error: ${String(err?.message || err)}`);
        return;
      }
    }

    setBusy(true);
    try {
      const objects = staged.map((s) => s.object);
      const res = await startRunFromGcs(apiBase, objects, {
        runName: runName.trim() || undefined,
        overrides: parsedOverrides || undefined,
        verbose,
      });
      appendLog(`[RUN] started run_id=${res.run_id}`);
      setSelectedRunId(res.run_id);
      onStarted?.(res.run_id, res.info);
      setRunName("");
      setStaged([]);
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

      {/* Samples (GCS-based) */}
      <div className="mt-3">
        <div className="text-sm text-slate-300 mb-1">Quick start with samples</div>
        <div className="flex gap-2">
          <button
            type="button"
            disabled={!canUpload}
            onClick={() => void stageSampleGcs("ripple.csv")}
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
          >
            Stage ripple.csv
          </button>
          <button
            type="button"
            disabled={!canUpload}
            onClick={() => void stageSampleGcs("vertical.csv")}
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
          >
            Stage vertical.csv
          </button>
        </div>
      </div>

      <form onSubmit={onStart} className="mt-4 grid gap-3">
        <div>
          <label className="block text-sm text-slate-300 mb-1">Files (CSV/XLS/PNG/JPG)</label>
          <input
            ref={fileRef}
            type="file"
            multiple
            disabled={!canUpload}
            className="block w-full text-sm text-slate-300 cursor-pointer file:cursor-pointer file:mr-3 file:px-3 file:py-1.5 file:rounded-md file:border file:border-slate-600 file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
          />
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={() => void stageSelectedToGcs()}
              disabled={!canUpload}
              className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
              title="Uploads go directly to Cloud Storage; run starts later"
            >
              Upload (don’t start)
            </button>
            <button
              type="button"
              onClick={() => void clearStaged()}
              disabled={!canInteract || stagedCount === 0}
              className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-300 hover:bg-slate-800 disabled:opacity-50"
              title="Clear staged list (does not delete GCS objects)"
            >
              Clear staged
            </button>
          </div>
        </div>

        {/* staged list */}
        <div className="text-xs text-slate-300">
          <div className="mb-1">Staged files ({stagedCount}):</div>
          {stagedCount ? (
            <ul className="list-disc pl-5 space-y-0.5">
              {staged.map((s) => (
                <li key={s.object} className="text-slate-200">
                  <span className="font-mono">{s.name}</span>
                  <span className="text-slate-400"> — {s.size?.toLocaleString() ?? "?"} bytes</span>
                  <span className="ml-2 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 border border-slate-600">
                    GCS: {s.object}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-slate-400">None staged yet. Upload or stage a sample.</div>
          )}
        </div>

        <div>
          <label className="block text-sm text-slate-300 mb-1">Run name (optional)</label>
          <input
            className="w-full bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder="e.g. Test run"
            disabled={!canInteract}
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
            disabled={!canInteract && !jsonError}
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
            disabled={!canInteract}
          />
          verbose
        </label>

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={!canStart}
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
            title={!stagedCount ? "Upload or stage a sample first" : undefined}
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
