// frontend/src/hooks/useColorRules.ts
import * as React from "react";
import { flattenTrack } from "@/utils/fields";
import type { ColorRule, FilterPredicate } from "@/utils/viewerTypes";

/** Return true if a flattened record matches a single predicate. */
function matchPredicate(flat: Record<string, any>, fp: FilterPredicate): boolean {
  const v = flat[fp.field];

  switch (fp.op) {
    // numeric
    case ">":  return Number(v) >  Number(fp.value);
    case "<":  return Number(v) <  Number(fp.value);
    case ">=": return Number(v) >= Number(fp.value);
    case "<=": return Number(v) <= Number(fp.value);
    case "==": return String(v) === String(fp.value);
    case "!=": return String(v) !== String(fp.value);
    case "between": {
      const n = Number(v);
      return n >= Number(fp.value) && n <= Number(fp.value2);
    }

    // strings
    case "contains":   return String(v ?? "").toLowerCase().includes(String(fp.value ?? "").toLowerCase());
    case "startsWith": return String(v ?? "").toLowerCase().startsWith(String(fp.value ?? "").toLowerCase());
    case "endsWith":   return String(v ?? "").toLowerCase().endsWith(String(fp.value ?? "").toLowerCase());
    case "in": {
      const arr = Array.isArray(fp.value) ? fp.value : [fp.value];
      const set = new Set(arr.map((x) => String(x)));
      return set.has(String(v));
    }
    case "notIn": {
      const arr = Array.isArray(fp.value) ? fp.value : [fp.value];
      const set = new Set(arr.map((x) => String(x)));
      return !set.has(String(v));
    }

    // booleans
    case "isTrue":  return Boolean(v) === true;
    case "isFalse": return Boolean(v) === false;

    default:
      return true;
  }
}

/**
 * Compiles color rules into a single function used by the canvas renderer.
 * First matching rule wins; returns its color, otherwise undefined.
 */
export function useColorRules(rules: ColorRule[]) {
  const colorOverrideFn = React.useMemo(() => {
    if (!rules || rules.length === 0) return undefined as
      | ((track: any) => string | undefined)
      | undefined;

    // Precompile rule matchers for speed
    const compiled = rules.map((r) => ({
      color: r.color,
      test: (flat: Record<string, any>) => matchPredicate(flat, r.when),
    }));

    return (track: any): string | undefined => {
      const flat = flattenTrack(track);
      for (const r of compiled) {
        if (r.test(flat)) return r.color;
      }
      return undefined;
    };
  }, [rules]);

  return { colorOverrideFn };
}
