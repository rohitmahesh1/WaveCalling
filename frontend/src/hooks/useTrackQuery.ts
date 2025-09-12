// hooks/useTrackQuery.ts
import * as React from "react";
import { buildFieldCatalog, flattenTrack, FieldDef } from "@/utils/fields";
import type { Track } from "@/utils/types";
import type { FilterPredicate, SortSpec } from "@/utils/viewerTypes";

/** Coerce to number if possible; otherwise return NaN. */
function toNum(v: unknown): number {
  if (typeof v === "number") return Number.isFinite(v) ? v : NaN;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }
  return NaN;
}

function cmpAsc(a: any, b: any): number {
  // Nulls last
  const an = a == null;
  const bn = b == null;
  if (an && bn) return 0;
  if (an) return 1;
  if (bn) return -1;

  // Prefer numeric compare when both look numeric
  const na = toNum(a);
  const nb = toNum(b);
  if (Number.isFinite(na) && Number.isFinite(nb)) {
    if (na < nb) return -1;
    if (na > nb) return 1;
    return 0;
  }

  // Fallback to string compare (case-sensitive by default)
  const sa = String(a);
  const sb = String(b);
  if (sa < sb) return -1;
  if (sa > sb) return 1;
  return 0;
}

function cmpDir(dir: "asc" | "desc") {
  return dir === "asc"
    ? (a: any, b: any) => cmpAsc(a, b)
    : (a: any, b: any) => -cmpAsc(a, b);
}

/**
 * Compiles a single filter predicate into a fast function.
 * Uses case-insensitive matching for string containment ops, numeric coercion for numeric ops.
 */
function compilePredicate(fp: FilterPredicate): (row: Record<string, any>) => boolean {
  const field = fp.field;
  const op = fp.op;

  // Normalize values once
  const v1n = toNum(fp.value as any);
  const v2n = toNum(fp.value2 as any);
  const v1s = fp.value != null ? String(fp.value) : "";
  const vLower = v1s.toLowerCase();

  let setStr: Set<string> | null = null;
  let setNum: Set<number> | null = null;

  if (op === "in" || op === "notIn") {
    const raw = Array.isArray(fp.value) ? fp.value : [fp.value];
    setStr = new Set(raw.map((x) => String(x)));
    const nums = raw.map((x) => toNum(x)).filter((n) => Number.isFinite(n));
    setNum = nums.length ? new Set(nums) : null;
  }

  switch (op) {
    case ">":
      return (row) => {
        const nv = toNum(row[field]);
        return Number.isFinite(nv) && nv > v1n;
      };
    case "<":
      return (row) => {
        const nv = toNum(row[field]);
        return Number.isFinite(nv) && nv < v1n;
      };
    case ">=":
      return (row) => {
        const nv = toNum(row[field]);
        return Number.isFinite(nv) && nv >= v1n;
      };
    case "<=":
      return (row) => {
        const nv = toNum(row[field]);
        return Number.isFinite(nv) && nv <= v1n;
      };
    case "==":
      return (row) => {
        const a = row[field];
        const an = toNum(a);
        if (Number.isFinite(an) && Number.isFinite(v1n)) return an === v1n;
        return String(a) === v1s;
      };
    case "!=":
      return (row) => {
        const a = row[field];
        const an = toNum(a);
        if (Number.isFinite(an) && Number.isFinite(v1n)) return an !== v1n;
        return String(a) !== v1s;
      };
    case "between": {
      const lo = Math.min(v1n, v2n);
      const hi = Math.max(v1n, v2n);
      return (row) => {
        const nv = toNum(row[field]);
        return Number.isFinite(nv) && nv >= lo && nv <= hi;
      };
    }
    case "contains":
      return (row) => String(row[field] ?? "").toLowerCase().includes(vLower);
    case "startsWith":
      return (row) => String(row[field] ?? "").toLowerCase().startsWith(vLower);
    case "endsWith":
      return (row) => String(row[field] ?? "").toLowerCase().endsWith(vLower);
    case "in":
      return (row) => {
        const val = row[field];
        const nv = toNum(val);
        if (Number.isFinite(nv) && setNum) return setNum.has(nv);
        return setStr!.has(String(val));
      };
    case "notIn":
      return (row) => {
        const val = row[field];
        const nv = toNum(val);
        if (Number.isFinite(nv) && setNum) return !setNum.has(nv);
        return !setStr!.has(String(val));
      };
    case "isTrue":
      return (row) => Boolean(row[field]) === true;
    case "isFalse":
      return (row) => Boolean(row[field]) === false;
    default:
      return () => true;
  }
}

type Options = {
  /** Resolve values for non-track fields (e.g. "csv.Column"). */
  extraFieldResolver?: (track: Track, path: string) => any;
  /** Merge additional fields into the returned catalog (e.g. CSV-driven FieldDefs). */
  externalCatalog?: FieldDef[];
};

/**
 * Hook that builds a field catalog, compiles filter/sort, and returns filtered tracks.
 * It memoizes a flattened representation of each track for speed and (optionally)
 * augments only the *referenced* fields via `extraFieldResolver`.
 */
export function useTrackQuery(
  tracks: Track[] | any[],
  filters: FilterPredicate[] = [],
  sort: SortSpec[] = [],
  options?: Options
) {
  // Merge catalog: native track fields + any external FieldDefs (e.g., csv.*)
  const catalog: FieldDef[] = React.useMemo(() => {
    const base = buildFieldCatalog(tracks);
    const extra = options?.externalCatalog ?? [];
    if (!extra.length) return base;
    const seen = new Set(base.map((d) => d.path));
    const merged = base.slice();
    for (const d of extra) {
      if (!seen.has(d.path)) {
        merged.push(d);
        seen.add(d.path);
      }
    }
    return merged;
  }, [tracks, options?.externalCatalog]);

  // Which fields are needed right now (for fast augmenting via extraFieldResolver)
  const usedFields = React.useMemo(() => {
    const s = new Set<string>();
    for (const f of filters ?? []) s.add(f.field);
    for (const srt of sort ?? []) s.add(srt.field);
    return s;
  }, [filters, sort]);

  // Flatten all tracks once (memoized) + augment used fields via resolver
  const flatRows = React.useMemo(() => {
    const rows = tracks.map((t) => flattenTrack(t));
    if (options?.extraFieldResolver && usedFields.size) {
      for (let i = 0; i < tracks.length; i++) {
        const t = tracks[i] as Track;
        const row = rows[i];
        for (const f of usedFields) {
          if (row[f] === undefined) {
            try {
              row[f] = options.extraFieldResolver(t, f);
            } catch {
              row[f] = undefined;
            }
          }
        }
      }
    }
    return rows;
  }, [tracks, usedFields, options?.extraFieldResolver]);

  // Track → index lookups (for a quick ‘filterFnTrack’)
  const indexByRef = React.useMemo(() => {
    const wm = new WeakMap<object, number>();
    for (let i = 0; i < tracks.length; i++) wm.set(tracks[i] as object, i);
    return wm;
  }, [tracks]);

  // Compile filters once
  const filterFn = React.useMemo(() => {
    if (!filters?.length) return (_row: Record<string, any>) => true;
    const compiled = filters.map(compilePredicate);
    return (row: Record<string, any>) => compiled.every((fn) => fn(row));
  }, [filters]);

  // Track-level predicate (for OverlayCanvas)
  const filterFnTrack = React.useMemo(() => {
    if (!filters?.length) return (_t: Track) => true;
    const compiled = filters.map(compilePredicate);
    return (t: Track) => {
      const idx = indexByRef.get(t as unknown as object);
      if (idx == null) return true; // if not found, don't filter it out
      const row = flatRows[idx];
      for (const fn of compiled) {
        if (!fn(row)) return false;
      }
      return true;
    };
  }, [filters, flatRows, indexByRef]);

  // Compile sort comparator once
  const sortFn = React.useMemo(() => {
    if (!sort?.length) return null as null | ((aIdx: number, bIdx: number) => number);
    const cmps = sort.map((s) => {
      const dirCmp = cmpDir(s.dir);
      const f = s.field;
      return (a: Record<string, any>, b: Record<string, any>) => dirCmp(a[f], b[f]);
    });
    return (ai: number, bi: number) => {
      const a = flatRows[ai];
      const b = flatRows[bi];
      for (const cmp of cmps) {
        const r = cmp(a, b);
        if (r !== 0) return r;
      }
      return 0;
    };
  }, [sort, flatRows]);

  // Filter + sort: operate on indices to avoid copying large objects repeatedly
  const filteredTracks = React.useMemo(() => {
    const idxs: number[] = [];
    for (let i = 0; i < tracks.length; i++) {
      if (filterFn(flatRows[i])) idxs.push(i);
    }
    if (sortFn) idxs.sort(sortFn);
    return idxs.map((i) => tracks[i]);
  }, [tracks, flatRows, filterFn, sortFn]);

  return {
    catalog,
    filteredTracks,
    filterFn,       // row-based (Record<string, any>) → boolean
    filterFnTrack,  // track-based (Track) → boolean  ← pass this to OverlayCanvas
    sortFn,
  };
}

export default useTrackQuery;
