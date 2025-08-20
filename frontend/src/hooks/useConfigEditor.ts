// hooks/useConfigEditor.ts
import * as React from "react";
import { deepDiff, deepGet, deepMerge, deepSet, isEmptyObject } from "@/utils/configDiff";
import type { SectionSpec, FieldSpec } from "@/utils/configSchema";
import { isFieldVisible as _isFieldVisible } from "@/utils/configSchema";

function deepEqual(a: any, b: any) {
  if (a === b) return true;
  // quick structural compare; good enough for config primitives/arrays/objects
  try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; }
}

export type ConfigEditorApi = {
  /** Working draft (current config being edited) */
  value: any;
  /** Convenience alias used by UI: set dotted-path value immutably */
  setValue: (path: string, v: any) => void;
  /** Low-level setter used by some callsites */
  setValueAt: (path: string, v: any) => void;
  /** Replace the whole object (rare) */
  setAll: (v: any) => void;
  /** Reset to defaults (not initial/base) */
  reset: () => void;

  /** Minimal overrides relative to defaults (object, `{}` if none) */
  overrides: any;
  /** Return the current diff (same as `overrides`) */
  getDiff: () => any;
  /** True if any difference vs defaults exists */
  isDirty: boolean;

  /** Per-field changed flags (by dotted path) */
  changed: Record<string, boolean>;
  /** Count of changed fields */
  changedCount: number;

  /** Map of path -> error message (only visible fields validated) */
  errors: Record<string, string>;
  /** Any visible error present? */
  hasErrors: boolean;
  /** Count of visible errors */
  errorCount: number;

  /** Visible field paths (based on visibleIf rules) */
  visiblePaths: Set<string>;

  /** Import overrides (deep-merge into current) */
  importOverrides: (overrides: any) => void;
  /** Import overrides from JSON string (no-throw helper) */
  applyOverridesJson: (text: string) => void;

  /** Schema passthrough for convenience */
  schema: SectionSpec[];
};

/**
 * Config editor state/logic. Provide:
 *  - `initial`: current config (e.g., run's config.yaml merged with overrides) — optional
 *  - `defaults`: baseline defaults for diffing (from schema or default file)
 *  - `schema`: UI metadata (sections/groups/fields with labels/validation/visibility)
 */
export function useConfigEditor({
  initial,
  defaults,
  schema,
}: {
  initial?: any;
  defaults: any;
  schema: SectionSpec[];
}): ConfigEditorApi {
  const [value, setValue] = React.useState<any>(initial ?? {});
  const defaultsRef = React.useRef<any>(defaults ?? {});
  const schemaRef = React.useRef<SectionSpec[]>(schema);
  const [schemaKey, setSchemaKey] = React.useState(0);

  React.useEffect(() => {
    setValue(initial ?? {});
    defaultsRef.current = defaults ?? {};
    schemaRef.current = schema;
    setSchemaKey((k) => k + 1);
  }, [initial, defaults, schema]);

  const setAll = React.useCallback((v: any) => {
    setValue(v ?? {});
  }, []);

  const setValueAt = React.useCallback((path: string, v: any) => {
    setValue((curr: any) => deepSet(curr ?? {}, path, v));
  }, []);

  // Alias used by UI components
  const setValueAlias = setValueAt;

  const reset = React.useCallback(() => {
    setValue(defaultsRef.current ?? {});
  }, []);

  // Flatten all FieldSpecs from schema.sections[].groups[].fields[]
  const allFieldSpecs: FieldSpec[] = React.useMemo(() => {
    const list: FieldSpec[] = [];
    for (const sec of schemaRef.current) {
      const groups = sec.groups ?? [];
      for (const g of groups) {
        for (const f of g.fields) list.push(f);
      }
    }
    return list;
  }, [schemaKey]);

  // Visible paths
  const visiblePaths = React.useMemo(() => {
    const set = new Set<string>();
    for (const f of allFieldSpecs) {
      if (_isFieldVisible(f, value)) set.add(f.path);
    }
    return set;
  }, [allFieldSpecs, value]);

  // Validation (only for visible fields)
  const errors = React.useMemo(() => {
    const out: Record<string, string> = {};
    for (const f of allFieldSpecs) {
      if (!_isFieldVisible(f, value)) continue;
      if (typeof f.validate === "function") {
        try {
          const v = deepGet(value, f.path);
          const msg = f.validate(v, value);
          if (msg) out[f.path] = String(msg);
        } catch (e: any) {
          out[f.path] = String(e?.message || e || "Invalid value");
        }
      }
    }
    return out;
  }, [allFieldSpecs, value]);

  const errorCount = React.useMemo(() => Object.keys(errors).length, [errors]);
  const hasErrors = errorCount > 0;

  // Overrides (diff) and per-field "changed" map vs defaults
  const overrides = React.useMemo(() => deepDiff(value, defaultsRef.current) ?? {}, [value]);
  const isDirty = React.useMemo(() => !isEmptyObject(overrides), [overrides]);

  const changed = React.useMemo(() => {
    const m: Record<string, boolean> = {};
    for (const f of allFieldSpecs) {
      const cur = deepGet(value, f.path);
      const def = deepGet(defaultsRef.current, f.path);
      if (!deepEqual(cur, def)) m[f.path] = true;
    }
    return m;
  }, [allFieldSpecs, value]);

  const changedCount = React.useMemo(() => Object.keys(changed).length, [changed]);

  const getDiff = React.useCallback(() => overrides, [overrides]);

  // Import overrides: deep-merge into current value
  const importOverrides = React.useCallback((o: any) => {
    if (!o || typeof o !== "object") return;
    setValue((curr: any) => deepMerge(curr ?? {}, o));
  }, []);

  const applyOverridesJson = React.useCallback((text: string) => {
    try {
      const parsed = JSON.parse(text);
      importOverrides(parsed);
    } catch {
      // ignore invalid JSON
    }
  }, [importOverrides]);

  return {
    value,
    setValue: setValueAlias,
    setValueAt,
    setAll,
    reset,

    overrides,
    getDiff,
    isDirty,

    changed,
    changedCount,

    errors,
    hasErrors,
    errorCount,

    visiblePaths,

    importOverrides,
    applyOverridesJson,

    schema: schemaRef.current,
  };
}
