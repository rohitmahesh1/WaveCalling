import * as React from "react";

export type ViewerOptions = {
  showBase: boolean;
  timeDirection: "up" | "down";
  colorBy: "none" | "dominant_frequency" | "amplitude";
  debugLayer:
    | "none"
    | "prob"
    | "mask_raw"
    | "mask_clean"
    | "mask_filtered"
    | "skeleton"
    | "mask_hysteresis";
};

interface Props {
  options: ViewerOptions;
  onChange: (partial: Partial<ViewerOptions>) => void;
  onRefresh: () => void;
  loading?: boolean;

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
    // Allow wrapping + small horizontal scroll fallback if it still overflows.
    <div className="flex flex-wrap items-center justify-end gap-x-3 gap-y-2 w-full overflow-x-auto sm:overflow-visible">
      {/* Show base toggle */}
      <label className="inline-flex items-center gap-2 text-sm text-slate-300 whitespace-nowrap">
        <input
          type="checkbox"
          checked={options.showBase}
          onChange={(e) => onChange({ showBase: e.target.checked })}
        />
        <span>Show base</span>
      </label>

      {/* Time direction selector */}
      <div className="inline-flex items-center gap-2 text-sm whitespace-nowrap">
        {/* Hide the label on xs to save space */}
        <span className="text-slate-300 hidden sm:inline">Time</span>
        <select
          className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 w-[120px] sm:w-auto"
          value={options.timeDirection}
          onChange={(e) =>
            onChange({
              timeDirection: e.target.value as ViewerOptions["timeDirection"],
            })
          }
          title="Time direction"
        >
          <option value="down">↓ down</option>
          <option value="up">↑ up</option>
        </select>
      </div>

      {/* Color mode selector */}
      <div className="inline-flex items-center gap-2 text-sm whitespace-nowrap">
        <span className="text-slate-300 hidden md:inline">Color by</span>
        <select
          className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 w-[180px] md:w-auto"
          value={options.colorBy}
          onChange={(e) =>
            onChange({
              colorBy: e.target.value as ViewerOptions["colorBy"],
            })
          }
          title="Color by"
        >
          <option value="none">none</option>
          <option value="dominant_frequency">dominant_frequency</option>
          <option value="amplitude">amplitude</option>
        </select>
      </div>

      {/* Debug layer selector */}
      <div className="inline-flex items-center gap-2 text-sm whitespace-nowrap">
        <span className="text-slate-300 hidden md:inline">Debug</span>
        <select
          className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 w-[200px] lg:w-auto"
          value={options.debugLayer}
          onChange={(e) =>
            onChange({
              debugLayer: e.target.value as ViewerOptions["debugLayer"],
            })
          }
          title="Debug layer"
        >
          <option value="none">none</option>
          <option value="prob">prob</option>
          <option value="mask_raw">mask_raw</option>
          <option value="mask_clean">mask_clean</option>
          <option value="mask_filtered">mask_filtered</option>
          <option value="skeleton">skeleton</option>
          <option value="mask_hysteresis">mask_hysteresis</option>
        </select>
      </div>

      {/* Optional advanced action */}
      {onOpenAdvanced && (
        <button
          onClick={onOpenAdvanced}
          className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 whitespace-nowrap"
        >
          {advancedLabel}
        </button>
      )}

      {/* Refresh action – keep at the end, allow it to wrap to next line */}
      <button
        onClick={onRefresh}
        disabled={loading}
        className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60 whitespace-nowrap"
      >
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  );
}
