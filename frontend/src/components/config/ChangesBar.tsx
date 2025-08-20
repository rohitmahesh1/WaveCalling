import * as React from "react";

export default function ChangesBar({
  dirty,
  changesCount = 0,
  saving = false,
  onSave,
  onReset,
  onCopyJSON,
  onDownloadJSON,
  className,
}: {
  dirty: boolean;
  changesCount?: number;
  saving?: boolean;
  onSave: () => void;
  onReset?: () => void;
  onCopyJSON?: () => void;
  onDownloadJSON?: () => void;
  className?: string;
}) {
  if (!dirty && !saving) return null;

  return (
    <div
      className={`sticky bottom-3 z-10 mx-auto w-full max-w-5xl rounded-xl border border-emerald-700/40 bg-emerald-900/30 px-3 py-2 backdrop-blur ${className || ""}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm text-emerald-300">
          {saving ? "Saving changes…" : `${changesCount} change${changesCount === 1 ? "" : "s"} pending`}
        </div>
        <div className="flex items-center gap-2">
          {onCopyJSON && (
            <button
              type="button"
              onClick={onCopyJSON}
              className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800"
              title="Copy overrides JSON to clipboard"
            >
              Copy JSON
            </button>
          )}
          {onDownloadJSON && (
            <button
              type="button"
              onClick={onDownloadJSON}
              className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800"
              title="Download overrides.json"
            >
              Download
            </button>
          )}
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="px-2.5 py-1 text-xs rounded border border-rose-600 text-rose-300 hover:bg-rose-600/10"
              title="Discard all edits"
            >
              Reset
            </button>
          )}
          <button
            type="button"
            onClick={onSave}
            disabled={saving}
            className="px-3 py-1.5 text-sm rounded-md border border-emerald-600 text-emerald-200 hover:bg-emerald-600/10 disabled:opacity-60"
            title="Apply changes"
          >
            {saving ? "Saving…" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
