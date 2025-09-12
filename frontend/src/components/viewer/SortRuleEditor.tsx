// frontend/src/components/viewer/SortRuleEditor.tsx
import * as React from "react";
import type { SortSpec } from "@/utils/viewerTypes";
import type { FieldDef } from "@/utils/fields";

type Props = {
  catalog?: FieldDef[];
  value: SortSpec[];
  onChange: (next: SortSpec[]) => void;
  className?: string;
};

const FALLBACK_FIELDS: FieldDef[] = [
  { path: "metrics.mean_amplitude", label: "mean_amplitude", kind: "number" },
  { path: "metrics.dominant_frequency", label: "dominant_frequency", kind: "number" },
  { path: "metrics.num_peaks", label: "num_peaks", kind: "number" },
  { path: "metrics.period", label: "period", kind: "number" },
  { path: "points", label: "points", kind: "number" },
  { path: "peaks_count", label: "peaks_count", kind: "number" },
  { path: "id", label: "id", kind: "string" },
  { path: "sample", label: "sample", kind: "string" },
];

export default function SortRuleEditor({ catalog, value, onChange, className }: Props) {
  const fields = (catalog?.length ? catalog : FALLBACK_FIELDS);
  const sort = value ?? [];

  const add = () => onChange([...(sort ?? []), { field: fields[0]?.path ?? "id", dir: "asc" }]);
  const remove = (i: number) => onChange(sort.filter((_, idx) => idx !== i));
  const update = (i: number, patch: Partial<SortSpec>) =>
    onChange(sort.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));

  const move = (i: number, delta: -1 | 1) => {
    const j = i + delta;
    if (j < 0 || j >= sort.length) return;
    const next = sort.slice();
    [next[i], next[j]] = [next[j], next[i]];
    onChange(next);
  };

  return (
    <div className={`flex flex-col gap-2 ${className || ""}`}>
      {sort.length === 0 && <div className="text-xs text-slate-400">No sort applied.</div>}

      {sort.map((r, i) => (
        <div
          key={`${r.field}:${i}`}
          className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr),110px,auto,auto,auto] items-center gap-2"
        >
          <select
            className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
            value={r.field}
            onChange={(e) => update(i, { field: e.target.value })}
          >
            {fields.map((f) => <option key={f.path} value={f.path}>{f.label}</option>)}
          </select>

          <select
            className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
            value={r.dir}
            onChange={(e) => update(i, { dir: e.target.value as "asc" | "desc" })}
          >
            <option value="asc">ascending</option>
            <option value="desc">descending</option>
          </select>

          <button
            type="button"
            className="justify-self-start md:justify-self-auto px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-800 text-xs"
            onClick={() => move(i, -1)}
            disabled={i === 0}
            title="Move up"
          >
            ↑
          </button>
          <button
            type="button"
            className="justify-self-start md:justify-self-auto px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-800 text-xs"
            onClick={() => move(i, +1)}
            disabled={i === sort.length - 1}
            title="Move down"
          >
            ↓
          </button>

          <button
            type="button"
            className="justify-self-start md:justify-self-auto px-2 py-1 rounded border border-rose-600 text-rose-300 hover:bg-rose-600/10 text-xs"
            onClick={() => remove(i)}
          >
            Remove
          </button>
        </div>
      ))}

      <div className="pt-1">
        <button
          type="button"
          className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 text-sm"
          onClick={add}
        >
          + Add sort
        </button>
      </div>
    </div>
  );
}
