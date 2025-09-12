import * as React from "react";
import type { ColorRule, FilterPredicate } from "@/utils/viewerTypes";
import type { FieldDef } from "@/utils/fields";

type Props = {
  /** Optional field catalog; if omitted we show a sensible default set. */
  catalog?: FieldDef[];
  value: ColorRule[];
  onChange: (next: ColorRule[]) => void;
  className?: string;
};

const OPS_NUM = [">", "<", ">=", "<=", "==", "!=", "between"] as const;
const OPS_STR = ["contains", "startsWith", "endsWith", "==", "!=", "in", "notIn"] as const;
const OPS_BOOL = ["isTrue", "isFalse"] as const;

const parseList = (s: string) =>
  s.split(",").map((x) => x.trim()).filter(Boolean);

/** Fallback fields if no catalog provided */
const FALLBACK_FIELDS: FieldDef[] = [
  { path: "metrics.mean_amplitude", label: "mean_amplitude", kind: "number" },
  { path: "metrics.dominant_frequency", label: "dominant_frequency", kind: "number" },
  { path: "metrics.num_peaks", label: "num_peaks", kind: "number" },
  { path: "metrics.period", label: "period", kind: "number" },
  { path: "id", label: "id", kind: "string" },
  { path: "sample", label: "sample", kind: "string" },
];

export default function ColorRuleEditor({
  catalog,
  value,
  onChange,
  className,
}: Props) {
  const fields = (catalog?.length ? catalog : FALLBACK_FIELDS);

  const rules = value ?? [];
  const add = () => {
    const f = fields[0];
    onChange([
      ...rules,
      {
        when: {
          field: f?.path ?? "id",
          op: f?.kind === "number" ? ">" : f?.kind === "boolean" ? "isTrue" : "contains",
          value: f?.kind === "number" ? 0 : "",
        },
        color: "#3b82f6",
      },
    ]);
  };
  const remove = (i: number) => onChange(rules.filter((_, idx) => idx !== i));
  const update = (i: number, patch: Partial<ColorRule>) =>
    onChange(rules.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  const updateWhen = (i: number, patch: Partial<FilterPredicate>) =>
    onChange(rules.map((r, idx) => (idx === i ? { ...r, when: { ...r.when, ...patch } } : r)));

  const opsForKind = (kind: FieldDef["kind"]) =>
    kind === "number" ? OPS_NUM : kind === "boolean" ? OPS_BOOL : OPS_STR;

  return (
    <div className={`flex flex-col gap-2 ${className || ""}`}>
      {rules.length === 0 && <div className="text-xs text-slate-400">No color rules. Add one below.</div>}

      {rules.map((r, idx) => {
        const field = fields.find((f) => f.path === r.when.field) ?? fields[0];
        const ops = opsForKind(field.kind);
        const op = (ops as readonly string[]).includes(r.when.op) ? r.when.op : ops[0];

        return (
          <div
            key={idx}
            className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr),110px,minmax(0,1fr),minmax(0,1fr),minmax(0,150px),auto] items-center gap-2"
          >
            {/* field */}
            <select
              className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
              value={r.when.field}
              onChange={(e) => updateWhen(idx, { field: e.target.value })}
              title={field.path}
            >
              {fields.map((f) => <option key={f.path} value={f.path}>{f.label}</option>)}
            </select>

            {/* op */}
            <select
              className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
              value={op as string}
              onChange={(e) => updateWhen(idx, { op: e.target.value as any })}
            >
              {ops.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>

            {/* values */}
            {field.kind === "number" && op === "between" ? (
              <>
                <input
                  type="number" step="any"
                  className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
                  value={String(r.when.value ?? "")}
                  onChange={(e) => updateWhen(idx, { value: Number(e.target.value) })}
                  placeholder="min"
                />
                <input
                  type="number" step="any"
                  className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
                  value={String(r.when.value2 ?? "")}
                  onChange={(e) => updateWhen(idx, { value2: Number(e.target.value) })}
                  placeholder="max"
                />
              </>
            ) : field.kind === "boolean" ? (
              <div className="text-xs text-slate-400 md:col-span-2">No value</div>
            ) : op === "in" || op === "notIn" ? (
              <input
                type="text"
                className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm md:col-span-2"
                value={Array.isArray(r.when.value) ? r.when.value.join(", ") : String(r.when.value ?? "")}
                onChange={(e) => updateWhen(idx, { value: parseList(e.target.value) })}
                placeholder="a, b, c"
              />
            ) : (
              <input
                type="text"
                className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm md:col-span-2"
                value={String(r.when.value ?? "")}
                onChange={(e) => updateWhen(idx, { value: e.target.value })}
                placeholder="value…"
              />
            )}

            {/* color */}
            <div className="flex items-center gap-2 min-w-0">
              <input
                type="color"
                className="h-8 w-8 rounded border border-slate-600 bg-slate-900/60"
                value={/^#([0-9a-f]{6})$/i.test(r.color) ? r.color : "#3b82f6"}
                onChange={(e) => update(idx, { color: e.target.value })}
                title="Pick color"
              />
              <input
                type="text"
                className="w-full min-w-0 bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 text-sm"
                value={r.color}
                onChange={(e) => update(idx, { color: e.target.value })}
                placeholder="#3b82f6 or hsl(...)"
              />
            </div>

            <button
              type="button"
              className="justify-self-start md:justify-self-auto px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-800 text-xs"
              onClick={() => remove(idx)}
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
          onClick={add}
        >
          + Add color rule
        </button>
      </div>
    </div>
  );
}
