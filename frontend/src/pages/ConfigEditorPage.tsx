// frontend/src/pages/ConfigEditorPage.tsx
import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useDashboard } from "@/context/DashboardContext";
import ConfigEditorPanel from "@/components/config/ConfigEditorPanel";

const STORAGE_KEY = "config:overrides";

export default function ConfigEditorPage() {
  const navigate = useNavigate();
  const { selectedRunId } = useDashboard();
  const [overridesJson, setOverridesJson] = React.useState<string>("");

  React.useEffect(() => {
    try {
      const v = window.localStorage.getItem(STORAGE_KEY);
      if (typeof v === "string") setOverridesJson(v);
    } catch {}
  }, []);

  const handleOverridesChange = React.useCallback((json: string) => {
    setOverridesJson(json);
    try {
      window.localStorage.setItem(STORAGE_KEY, json || "");
    } catch {}
  }, []);

  return (
    <div className="min-h-[calc(100vh-56px)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h1 className="text-slate-100 font-semibold text-lg">Config Editor</h1>
        <div className="flex items-center gap-2">
          <button
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            onClick={() => navigate(-1)}
          >
            Back
          </button>
          <button
            className="px-3 py-1.5 rounded-md border border-emerald-600 text-emerald-200 hover:bg-emerald-600/10"
            onClick={() => {
              try {
                window.localStorage.setItem(STORAGE_KEY, overridesJson || "");
              } catch {}
            }}
            title="Save overrides to Upload panel"
          >
            Save
          </button>
          <button
            className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-300 hover:bg-slate-800"
            onClick={() => {
              setOverridesJson("");
              try {
                window.localStorage.removeItem(STORAGE_KEY);
              } catch {}
            }}
            title="Clear overrides"
          >
            Clear
          </button>
        </div>
      </div>

      <ConfigEditorPanel
        runId={selectedRunId ?? null}
        onOverridesChange={handleOverridesChange}
        className="rounded-xl border border-slate-700/50 bg-console-700 p-4"
      />
    </div>
  );
}
