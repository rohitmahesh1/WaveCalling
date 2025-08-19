import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { useDashboard } from "@/context/DashboardContext";
import { listRuns } from "@/utils/api";
import type { RunInfo } from "@/utils/types";
import RunRow from "@/components/RunRow";

export default function RunsPanel() {
  const { selectedRunId, setSelectedRunId } = useDashboard();
  const apiBase = useApiBase();

  const [runs, setRuns] = React.useState<RunInfo[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [q, setQ] = React.useState("");

  const refresh = React.useCallback(async () => {
    setLoading(true);
    try {
      const data = await listRuns(apiBase);
      setRuns(data);
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  // Optional polling (pause later if SSE on active run)
  React.useEffect(() => {
    const t = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(t);
  }, [refresh]);

  const filtered = React.useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return runs;
    return runs.filter((r) =>
      [r.run_id, r.name, r.status, r.created_at].some((s) => String(s).toLowerCase().includes(needle))
    );
  }, [q, runs]);

  return (
    <section className="h-full flex flex-col rounded-xl border border-slate-700/50 bg-console-700 p-4">
      <header className="flex items-center justify-between gap-2">
        <h2 className="text-slate-200 font-semibold">Runs</h2>
        <div className="flex items-center gap-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search runs…"
            className="text-sm bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
          />
          <button
            onClick={() => void refresh()}
            disabled={loading}
            className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      <div className="mt-3 border-t border-slate-700/40" />

      {filtered.length === 0 ? (
        <div className="text-sm text-slate-400 py-6">No runs yet.</div>
      ) : (
        <div className="mt-3 overflow-auto">
          <table className="w-full text-sm">
            <thead className="text-slate-400">
              <tr className="text-left">
                <th className="py-2 pr-3 font-medium">Run ID</th>
                <th className="py-2 pr-3 font-medium">Name</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 pr-3 font-medium">Created</th>
                <th className="py-2 pr-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="text-slate-200">
              {filtered.map((r) => (
                <RunRow
                  key={r.run_id}
                  run={r}
                  selected={selectedRunId === r.run_id}
                  onOpen={() => setSelectedRunId(r.run_id)}
                  onChanged={refresh}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
