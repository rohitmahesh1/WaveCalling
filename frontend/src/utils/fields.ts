// utils/fields.ts

export type FieldKind = "number" | "string" | "boolean";
export type FieldDef = {
  /** Dotted path into a flattened track object, e.g. "metrics.mean_amplitude" */
  path: string;
  /** Nice label for UI */
  label: string;
  /** Primitive kind used for operators/widgets */
  kind: FieldKind;
  /** Optional finite set of allowed values for string fields (used by in/notIn UIs) */
  enumValues?: string[];
};

// Safe getter for dotted paths
export function getAtPath(obj: any, path: string): any {
  try {
    return path.split(".").reduce((a, k) => (a == null ? a : (a as any)[k]), obj);
  } catch {
    return undefined;
  }
}

const toLabel = (key: string) =>
  key
    .replace(/\./g, " ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());

/**
 * Flatten a track into key/value pairs suitable for filtering/sorting/CSV.
 * Adds a few convenience fields:
 *  - points: poly length
 *  - peaks_count: peaks length
 */
export function flattenTrack(track: any): Record<string, any> {
  const out: Record<string, any> = {};

  // Only recurse into plain objects, not arrays
  const isPlainObject = (v: any) =>
    v != null && typeof v === "object" && !Array.isArray(v);

  function walk(prefix: string, v: any) {
    if (isPlainObject(v)) {
      for (const [k, val] of Object.entries(v)) {
        walk(prefix ? `${prefix}.${k}` : k, val);
      }
    } else {
      out[prefix] = v;
    }
  }

  // Seed a shallow object that includes convenience fields
  const seed = {
    id: track?.id,
    sample: track?.sample,
    points: Array.isArray(track?.poly) ? track.poly.length : undefined,
    peaks_count: Array.isArray(track?.peaks) ? track.peaks.length : undefined,
    metrics: track?.metrics ?? {},
  };

  walk("", seed);
  return out;
}

/**
 * Infer field definitions from up to `sampleCount` tracks.
 * Skips arrays/objects when deciding kinds; collects enumValues for low-card strings.
 */
export function buildFieldCatalog(tracks: any[], sampleCount = 1000): FieldDef[] {
  const samples = tracks.slice(0, sampleCount).map(flattenTrack);

  // Collect all keys present
  const keys = new Set<string>();
  for (const s of samples) for (const k of Object.keys(s)) keys.add(k);

  // Helper to infer kind
  function inferKind(values: any[]): FieldKind {
    const scalars = values.filter(
      (v) => v !== null && v !== undefined && typeof v !== "object"
    );
    if (scalars.length === 0) return "string";
    const allNum = scalars.every((v) => typeof v === "number" && Number.isFinite(v));
    if (allNum) return "number";
    const allBool = scalars.every((v) => typeof v === "boolean");
    if (allBool) return "boolean";
    return "string";
  }

  const defs: FieldDef[] = [];

  for (const path of keys) {
    // Skip obviously non-useful keys if you want (e.g. raw arrays). Since we seeded only primitives, we’re fine.
    const vals = samples
      .map((s) => s[path])
      .filter((v) => v !== null && v !== undefined && typeof v !== "object");

    if (vals.length === 0) continue;

    const kind = inferKind(vals);

    // Build enumValues for low-cardinality strings
    let enumValues: string[] | undefined;
    if (kind === "string") {
      const uniq = Array.from(new Set(vals.map(String)));
      if (uniq.length > 0 && uniq.length <= 24) enumValues = uniq.sort();
    }

    defs.push({
      path,
      label: toLabel(path),
      kind,
      enumValues,
    });
  }

  // Prefer common/interesting fields first
  const priority = new Map<string, number>([
    ["metrics.mean_amplitude", 0],
    ["metrics.dominant_frequency", 1],
    ["metrics.num_peaks", 2],
    ["metrics.period", 3],
    ["points", 4],
    ["peaks_count", 5],
    ["sample", 6],
    ["id", 7],
  ]);

  defs.sort((a, b) => {
    const pa = priority.has(a.path) ? priority.get(a.path)! : 999;
    const pb = priority.has(b.path) ? priority.get(b.path)! : 999;
    if (pa !== pb) return pa - pb;
    return a.label.localeCompare(b.label);
  });

  return defs;
}
