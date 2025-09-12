// frontend/src/components/viewer/FilterRuleEditor.tsx
import * as React from "react";
import type { FilterPredicate } from "@/utils/viewerTypes";
import type { FieldDef } from "@/utils/fields";

type Props = {
  catalog?: FieldDef[];
  value: FilterPredicate[];
  onChange: (next: FilterPredicate[]) => void;
  className?: string;
};

const OPS_NUM = [">", "<", ">=", "<=", "==", "!=", "between"] as const;
const OPS_STR = ["contains", "startsWith", "endsWith", "==", "!=", "in", "notIn"] as const;
const OPS_BOOL = ["isTrue", "isFalse"] as const;

const parseList = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);

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

export default function FilterRuleEditor({ catalog, value, onChange, className }: Props) {
  const fields = (catalog?.length ? catalog : FALLBACK_FIELDS);
  const rules = value ?? [];

  const addRule = () => {
    const f = fields[0];
    onChange([
      ...rules,
      {
        field: f?.path ?? "id",
        op: f?.kind === "number" ? ">" : f?.kind === "boolean" ? "isTrue" : "contains",
        value: f?.kind === "number" ? 0 : "",
      } as FilterPredicate,
    ]);
  };

  const removeRule = (idx: number) => onChange(rules.filter((_, i) => i !== idx));
  const update = (idx: number, patch: Partial<FilterPredicate>) =>
    onChange(rules.map((r, i) => (i === idx ? { ...r, ...patch } : r)));

  const opsForKind = (kind: FieldDef["kind"]) =>
    kind === "number" ? OPS_NUM : kind === "boolean" ? OPS_BOOL : OPS_STR;

  return (
    <div className={`flex flex-col gap-2 ${className || ""}`}>
      {rules.length === 0 && <div className="text-xs text-slate-400">No filters. Add one below.</div>}

      {rules.map((r, idx) => {
        const field = fields.find((f) => f.path === r.field) ?? fields[0];
        const ops = opsForKind(field.kind);
        const op = (ops as readonly string[]).includes(r.op) ? r.op : ops[0];

        return (
          <div
            key={idx}
            className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr),110px,minmax(0,1fr),minmax(0,1fr),auto] items-center gap-2"
          >
            {/* Field */}
            <select
              className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
              value={r.field}
              onChange={(e) => update(idx, { field: e.target.value })}
            >
              {fields.map((f) => (
                <option key={f.path} value={f.path}>{f.label}</option>
              ))}
            </select>

            {/* Op */}
            <select
              className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
              value={op as string}
              onChange={(e) => update(idx, { op: e.target.value as any })}
            >
              {ops.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>

            {/* Values */}
            {field.kind === "number" && op === "between" ? (
              <>
                <input
                  type="number"
                  step="any"
                  className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
                  value={String(r.value ?? "")}
                  onChange={(e) => update(idx, { value: Number(e.target.value) })}
                  placeholder="min"
                />
                <input
                  type="number"
                  step="any"
                  className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
                  value={String(r.value2 ?? "")}
                  onChange={(e) => update(idx, { value2: Number(e.target.value) })}
                  placeholder="max"
                />
              </>
            ) : field.kind === "boolean" ? (
              <div className="text-xs text-slate-400 md:col-span-2">No value needed</div>
            ) : op === "in" || op === "notIn" ? (
              <input
                type="text"
                className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm md:col-span-2"
                value={Array.isArray(r.value) ? r.value.join(", ") : String(r.value ?? "")}
                onChange={(e) => update(idx, { value: parseList(e.target.value) })}
                placeholder="a, b, c"
              />
            ) : (
              <input
                type="text"
                className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm md:col-span-2"
                value={String(r.value ?? "")}
                onChange={(e) => update(idx, { value: e.target.value })}
                placeholder="value…"
              />
            )}

            {/* Remove */}
            <button
              type="button"
              className="justify-self-start md:justify-self-auto px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-800 text-xs"
              onClick={() => removeRule(idx)}
            >
              Remove
            </button>
          </div>
        );
      })}

      <div className="pt-1">
        <button
          type="button"
          className="px-3 py-1.5 rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 text-sm"
          onClick={addRule}
        >
          + Add filter
        </button>
      </div>
    </div>
  );
}
