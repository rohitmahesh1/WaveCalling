import * as React from "react";

export type ViewerOptions = {
  showBase: boolean;
  timeDirection: "up" | "down";
  colorBy: "none" | "dominant_frequency" | "amplitude";
};

interface Props {
  options: ViewerOptions;
  onChange: (partial: Partial<ViewerOptions>) => void;
  onRefresh: () => void;
  loading?: boolean;

  /** Optional: show an “Advanced” action if provided */
  onOpenAdvanced?: () => void;
  advancedLabel?: string;
}

export default function ViewerToolbar({
  options,
  onChange,
  onRefresh,
  loading,
  onOpenAdvanced,
  advancedLabel = "Advanced",
}: Props) {
  return (
    <div className="flex items-center gap-3">
      {/* Show base toggle */}
      <label className="inline-flex items-center gap-2 text-sm text-slate-300">
        <input
          type="checkbox"
          checked={options.showBase}
          onChange={(e) => onChange({ showBase: e.target.checked })}
        />
        Show base
      </label>

      {/* Time direction selector */}
      <div className="inline-flex items-center gap-2 text-sm">
        <span className="text-slate-300">Time</span>
        <select
          className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600"
          value={options.timeDirection}
          onChange={(e) =>
            onChange({
              timeDirection: e.target.value as ViewerOptions["timeDirection"],
            })
          }
        >
          <option value="down">↓ down</option>
          <option value="up">↑ up</option>
        </select>
      </div>

      {/* Color mode selector */}
      <div className="inline-flex items-center gap-2 text-sm">
        <span className="text-slate-300">Color by</span>
        <select
          className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600"
          value={options.colorBy}
          onChange={(e) =>
            onChange({
              colorBy: e.target.value as ViewerOptions["colorBy"],
            })
          }
        >
          <option value="none">none</option>
          <option value="dominant_frequency">dominant_frequency</option>
          <option value="amplitude">amplitude</option>
        </select>
      </div>

      {/* Optional advanced action */}
      {onOpenAdvanced && (
        <button
          onClick={onOpenAdvanced}
          className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
        >
          {advancedLabel}
        </button>
      )}

      {/* Refresh action */}
      <button
        onClick={onRefresh}
        disabled={loading}
        className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
      >
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  );
}
