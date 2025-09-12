// frontend/src/components/viewer/AdvancedSidePanel.tsx
import * as React from "react";
import type { Track } from "@/utils/types";
import type { FilterPredicate, ColorRule, SortSpec } from "@/utils/viewerTypes";

import { buildFieldCatalog, FieldDef } from "@/utils/fields";
import StatsSummary from "@/components/viewer/StatsSummary";
import FilterRuleEditor from "@/components/viewer/FilterRuleEditor";
import ColorRuleEditor from "@/components/viewer/ColorRuleEditor";
import SortRuleEditor from "@/components/viewer/SortRuleEditor";
import { downloadJSON } from "@/utils/download";
import { useLocalStorage } from "@/hooks/useLocalStorage";

type Props = {
  /** Full (unfiltered) track list so we can infer fields for editors */
  tracks: Track[];

  // Downloads for the whole (filtered) set / canvas
  onDownloadPNG: () => void;
  onDownloadJSON: () => void;
  onDownloadCSV?: () => void;

  // Filters
  filters: FilterPredicate[];
  onFiltersChange: (next: FilterPredicate[]) => void;

  // Color rules
  colorRules: ColorRule[];
  onColorRulesChange: (next: ColorRule[]) => void;

  // Sorting
  sort?: SortSpec[];
  onSortChange?: (next: SortSpec[]) => void;

  // Global stats (computed on filtered set)
  filteredStats: {
    count: number;
    points: number;
    avgAmplitude: number | null;
    avgFrequency: number | null;
  };

  // Optional selection echo (shown compactly here)
  selectedTrack?: Track | null;

  // (Optional) clear selection handler — only render the button if provided
  onClearSelection?: () => void;
};

const RECOMMENDED_PATHS = new Set<string>([
  "metrics.mean_amplitude",
  "metrics.dominant_frequency",
  "metrics.num_peaks",
  "metrics.period",
  "points",
  "peaks_count",
  "sample",
  "id",
]);

export default function AdvancedSidePanel({
  tracks,

  onDownloadPNG,
  onDownloadJSON,
  onDownloadCSV,

  filters,
  onFiltersChange,

  colorRules,
  onColorRulesChange,

  sort = [],
  onSortChange,

  filteredStats,

  selectedTrack,
  onClearSelection,
}: Props) {
  // Build full catalog from tracks (flattened fields with types/enums)
  const fullCatalog: FieldDef[] = React.useMemo(() => buildFieldCatalog(tracks ?? []), [tracks]);

  // Toggle: show concise set vs all fields (persist this choice)
  const [showAllFields, setShowAllFields] = useLocalStorage<boolean>("viewer:moreFields", false);

  // Recommended subset (fall back to full if we somehow filter everything out)
  const recommendedCatalog = React.useMemo(() => {
    const picked = fullCatalog.filter((f) => RECOMMENDED_PATHS.has(f.path));
    return picked.length ? picked : fullCatalog;
  }, [fullCatalog]);

  const catalog = showAllFields ? fullCatalog : recommendedCatalog;

  // ---- Presets ----
  const addPresetAmplitude = React.useCallback(() => {
    onFiltersChange([
      ...filters,
      { field: "metrics.mean_amplitude", op: ">", value: 30 },
    ]);
  }, [filters, onFiltersChange]);

  const addPresetLowFreqBlue = React.useCallback(() => {
    onColorRulesChange([
      ...colorRules,
      {
        when: { field: "metrics.dominant_frequency", op: "<", value: 20 },
        color: "#3b82f6", // Tailwind blue-500
      },
    ]);
  }, [colorRules, onColorRulesChange]);

  const clearFilters = React.useCallback(() => onFiltersChange([]), [onFiltersChange]);
  const clearRules = React.useCallback(() => onColorRulesChange([]), [onColorRulesChange]);

  // Selected track quick-download (handy for debugging)
  const downloadSelectedJSON = React.useCallback(() => {
    if (!selectedTrack) return;
    downloadJSON(`track_${String(selectedTrack.id)}.json`, selectedTrack);
  }, [selectedTrack]);

  return (
    <div className="flex flex-col gap-4 min-w-[500px]">
      {/* Downloads */}
      <section className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 overflow-x-auto">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <h4 className="text-slate-200 font-semibold">Download</h4>
        </div>
        <div className="mt-2 grid grid-cols-2 gap-2">
          <button
            className="cursor-pointer px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            onClick={onDownloadPNG}
          >
            Canvas PNG
          </button>
          <button
            className="cursor-pointer px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            onClick={onDownloadJSON}
          >
            Tracks JSON
          </button>
          {onDownloadCSV && (
            <button
              className="cursor-pointer px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
              onClick={onDownloadCSV}
            >
              Tracks CSV
            </button>
          )}
        </div>
      </section>

      {/* Filters */}
      <section className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 overflow-x-auto">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <h4 className="text-slate-200 font-semibold">Filters</h4>
          <div className="flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showAllFields}
                onChange={(e) => setShowAllFields(e.target.checked)}
              />
              More fields
            </label>
            <button
              className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
              onClick={addPresetAmplitude}
              title="Add preset: metrics.mean_amplitude > 30"
            >
              + amp &gt; 30
            </button>
            <button
              className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
              onClick={clearFilters}
              title="Clear all filters"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="mt-2">
          <FilterRuleEditor
            value={filters}
            onChange={onFiltersChange}
            catalog={catalog}
          />
        </div>
      </section>

      {/* Color rules */}
      <section className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 overflow-x-auto">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <h4 className="text-slate-200 font-semibold">Color rules</h4>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
              onClick={addPresetLowFreqBlue}
              title="Add preset: metrics.dominant_frequency < 20 → blue"
            >
              + freq &lt; 20 → blue
            </button>
            <button
              className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
              onClick={clearRules}
              title="Clear all rules"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="mt-2">
          <ColorRuleEditor
            value={colorRules}
            onChange={onColorRulesChange}
            catalog={catalog}
          />
        </div>
      </section>

      {/* Sorting */}
      {onSortChange && (
        <section className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 overflow-x-auto">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <h4 className="text-slate-200 font-semibold">Sorting</h4>
            <div className="flex items-center gap-2 flex-wrap">
              <button
                className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
                onClick={() => onSortChange([])}
                title="Clear all sort criteria"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="mt-2">
            <SortRuleEditor
              value={sort}
              onChange={onSortChange}
              catalog={catalog}
            />
          </div>
        </section>
      )}

      {/* Stats */}
      <section className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 overflow-x-auto">
        <h4 className="text-slate-200 font-semibold">Stats</h4>
        <div className="mt-2">
          <StatsSummary
            count={filteredStats.count}
            points={filteredStats.points}
            avgAmplitude={filteredStats.avgAmplitude}
            avgFrequency={filteredStats.avgFrequency}
          />
        </div>
      </section>

      {/* Selection echo (compact) */}
      {selectedTrack && (
        <section className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 overflow-x-auto">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <h4 className="text-slate-200 font-semibold">Selection</h4>
            <div className="flex items-center gap-2 flex-wrap">
              <button
                className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
                onClick={downloadSelectedJSON}
              >
                Save JSON
              </button>
              {onClearSelection && (
                <button
                  className="cursor-pointer px-2 py-1 rounded border text-xs border-slate-600 text-slate-300 hover:bg-slate-800"
                  onClick={onClearSelection}
                >
                  Clear
                </button>
              )}
            </div>
          </div>
          <div className="mt-2 text-xs text-slate-300 leading-6">
            <div>ID: <span className="font-mono text-slate-100">{String(selectedTrack.id)}</span></div>
            <div>points: <span className="text-slate-100">{selectedTrack.poly?.length ?? 0}</span></div>
            <div>mean amplitude: <span className="text-slate-100">{selectedTrack.metrics?.mean_amplitude ?? "—"}</span></div>
            <div>dominant freq: <span className="text-slate-100">{selectedTrack.metrics?.dominant_frequency ?? "—"}</span></div>
            {Array.isArray(selectedTrack.peaks) && (
              <div>peaks: <span className="text-slate-100">{selectedTrack.peaks.length}</span></div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
