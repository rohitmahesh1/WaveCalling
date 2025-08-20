// utils/configDiff.ts

export type AnyObj = Record<string, any>;

export function isPlainObject(v: any): v is AnyObj {
  return Object.prototype.toString.call(v) === "[object Object]";
}

export function deepClone<T>(v: T): T {
  if (Array.isArray(v)) return v.map((x) => deepClone(x)) as unknown as T;
  if (isPlainObject(v)) {
    const out: AnyObj = {};
    for (const k of Object.keys(v)) out[k] = deepClone(v[k]);
    return out as T;
  }
  return v;
}

export function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (Number.isNaN(a) && Number.isNaN(b)) return true;

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  if (isPlainObject(a) && isPlainObject(b)) {
    const ak = Object.keys(a);
    const bk = Object.keys(b);
    if (ak.length !== bk.length) return false;
    for (const k of ak) if (!deepEqual(a[k], b[k])) return false;
    return true;
  }
  return false;
}

export function deepGet<T = any>(obj: AnyObj | undefined, path: string, fallback?: T): T | undefined {
  if (!obj) return fallback;
  const parts = path.split(".");
  let cur: any = obj;
  for (const p of parts) {
    if (cur == null) return fallback;
    cur = cur[p];
  }
  return (cur === undefined ? fallback : cur) as T | undefined;
}

export function deepSet<T = any>(obj: AnyObj, path: string, value: T): AnyObj {
  const parts = path.split(".");
  const out = Array.isArray(obj) ? [...obj] : { ...obj };
  let cur: any = out;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i]!;
    const next = cur[k];
    if (!isPlainObject(next) && !Array.isArray(next)) {
      cur[k] = {};
    } else {
      // clone on write
      cur[k] = Array.isArray(next) ? [...next] : { ...next };
    }
    cur = cur[k];
  }
  cur[parts[parts.length - 1]!] = value;
  return out;
}

export function deepMerge<A extends AnyObj, B extends AnyObj>(a: A, b: B): A & B {
  if (a === b) return a as any;
  if (!isPlainObject(a)) return deepClone(b) as any;
  if (!isPlainObject(b)) return deepClone(b) as any;

  const out: AnyObj = deepClone(a);
  for (const k of Object.keys(b)) {
    const av = (a as AnyObj)[k];
    const bv = (b as AnyObj)[k];
    if (isPlainObject(av) && isPlainObject(bv)) {
      out[k] = deepMerge(av, bv);
    } else {
      out[k] = deepClone(bv);
    }
  }
  return out as A & B;
}

/**
 * Produce a minimal object of overrides: fields where `current` differs from `base`.
 * - Objects are diffed recursively
 * - Arrays: replaced wholesale if not deeply equal
 * - `undefined` values are omitted (not written)
 */
export function deepDiff(current: any, base: any): any {
  if (deepEqual(current, base)) return undefined;

  // Primitive difference or array difference → replace whole subtree
  if (!isPlainObject(current) || !isPlainObject(base)) {
    if (Array.isArray(current) && Array.isArray(base)) {
      if (deepEqual(current, base)) return undefined;
      return deepClone(current);
    }
    return deepClone(current);
  }

  // Both plain objects: recurse per key
  const out: AnyObj = {};
  const keys = new Set([...Object.keys(current), ...Object.keys(base)]);
  for (const k of keys) {
    const curV = current[k];
    const baseV = base[k];
    const d = deepDiff(curV, baseV);
    if (d !== undefined) out[k] = d;
  }
  return Object.keys(out).length ? out : undefined;
}

export function isEmptyObject(v: any): boolean {
  return isPlainObject(v) && Object.keys(v).length === 0;
}

/** Utility: pretty-print overrides (returns "{}" when empty). */
export function formatOverridesJSON(overrides: any): string {
  if (overrides === undefined || isEmptyObject(overrides)) return "{}";
  return JSON.stringify(overrides, null, 2);
}
