import React from "react";
import { useApiBase } from "@/context/ApiContext";
import { listRuns } from "@/utils/api";
import type { RunInfo } from "@/utils/types";
import RunStatusBadge from "@/components/RunStatusBadge";

export default function RunsList() {
  const apiBase = useApiBase();
  const [runs, setRuns] = React.useState<RunInfo[]>([]);
  const [loading, setLoading] = React.useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const r = await listRuns(apiBase);
      setRuns(r);
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => { void refresh(); }, []);

  return (
    <div>
      <div className="row">
        <h2>Your runs</h2>
        <button onClick={refresh} disabled={loading}>Refresh</button>
      </div>
      <div className="card mt">
        {runs.length === 0 ? (
          <div>No runs yet.</div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left" }}>
                <th>Run ID</th><th>Name</th><th>Status</th><th>Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td><code>{r.run_id}</code></td>
                  <td>{r.name}</td>
                  <td><RunStatusBadge status={r.status} /></td>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
