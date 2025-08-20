// frontend/src/components/RunActions.tsx
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { cancelRun, resumeRun, deleteRun } from "@/utils/api";
import type { RunInfo } from "@/utils/types";

export default function RunActions({
  run,
  onOpen,
  onChanged,
}: {
  run: RunInfo;
  onOpen: () => void;
  onChanged?: () => void;
}) {
  const apiBase = useApiBase();
  const {
    selectedRunId,
    setSelectedRunId,
    appendLog,
    clearLog,            // NEW
  } = useDashboard();

  const active = run.status === "RUNNING" || run.status === "QUEUED";
  const terminal = run.status === "DONE" || run.status === "ERROR" || run.status === "CANCELLED";

  async function handleCancel() {
    try {
      await cancelRun(apiBase, run.run_id);
      appendLog(`[CANCEL] requested for ${run.run_id}`);
      onChanged?.();
    } catch (e: any) {
      appendLog(`[CANCEL] error: ${String(e)}`);
    }
  }

  async function handleResume() {
    try {
      await resumeRun(apiBase, run.run_id);
      appendLog(`[RESUME] requested for ${run.run_id}`);
      setSelectedRunId(run.run_id);
      onChanged?.();
    } catch (e: any) {
      appendLog(`[RESUME] error: ${String(e)}`);
    }
  }

  async function handleDelete() {
    if (!confirm(`Delete run ${run.run_id}? This removes files on disk.`)) return;
    try {
      await deleteRun(apiBase, run.run_id);
      appendLog(`[DELETE] ${run.run_id}`);
      // If we just deleted the selected run, clear selection and logs
      if (selectedRunId === run.run_id) {
        setSelectedRunId(null);
        clearLog();                               // NEW
      }
      onChanged?.();
    } catch (e: any) {
      appendLog(`[DELETE] error: ${String(e)}`);
    }
  }

  return (
    <div className="flex items-center gap-2">
      <button
        className="px-2.5 py-1 text-xs rounded border border-slate-600 hover:bg-slate-800"
        onClick={() => {
          setSelectedRunId(run.run_id);
          onOpen();
        }}
        title="Open"
      >
        Open
      </button>

      {!active && (
        <button
          className="px-2.5 py-1 text-xs rounded border border-slate-600 hover:bg-slate-800"
          onClick={() => {
            setSelectedRunId(run.run_id);
            onOpen();
          }}
          title="Rejoin (attach)"
        >
          Rejoin
        </button>
      )}

      {active ? (
        <button
          className="px-2.5 py-1 text-xs rounded border border-rose-600 text-rose-300 hover:bg-rose-600/10"
          onClick={() => void handleCancel()}
          title="Cancel run"
        >
          Cancel
        </button>
      ) : (
        <button
          className="px-2.5 py-1 text-xs rounded border border-emerald-600 text-emerald-300 hover:bg-emerald-600/10 disabled:opacity-50"
          onClick={() => void handleResume()}
          disabled={run.status === "DONE"}
          title="Resume run"
        >
          Resume
        </button>
      )}

      {terminal && (
        <button
          className="px-2.5 py-1 text-xs rounded border border-slate-600 hover:bg-slate-800"
          onClick={() => void handleDelete()}
          title="Delete run"
        >
          Delete
        </button>
      )}
    </div>
  );
}
