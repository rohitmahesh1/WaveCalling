// utils/viewerTypes.ts

/** All supported operators across field types. */
export type GenericOp =
  | ">" | "<" | ">=" | "<=" | "==" | "!=" | "between"
  | "contains" | "startsWith" | "endsWith"
  | "in" | "notIn"
  | "isTrue" | "isFalse";

/**
 * A single filter condition against a flattened track field.
 * - For numbers: >, <, >=, <=, ==, !=, between (use value + value2)
 * - For strings: contains, startsWith, endsWith, ==, !=, in/notIn (value = string or string[])
 * - For booleans: isTrue / isFalse (no value required)
 */
export type FilterPredicate = {
  field: string;       // dotted path (e.g., "metrics.mean_amplitude")
  op: GenericOp;
  value?: number | string | string[] | boolean | null;
  value2?: number | null;  // only used for "between"
};

/** Coloring rule: if `when` matches, stroke/fill with `color`. */
export type ColorRule = {
  when: FilterPredicate;
  color: string;       // e.g. "#3b82f6" or "hsl(220 90% 60%)"
};

/** Sorting spec */
export type SortSpec = { field: string; dir: "asc" | "desc" };

/* shared operator buckets for importing in editors */
export const NUMERIC_OPS: GenericOp[] = [">", "<", ">=", "<=", "==", "!=", "between"];
export const STRING_OPS: GenericOp[]  = ["contains", "startsWith", "endsWith", "==", "!=", "in", "notIn"];
export const BOOLEAN_OPS: GenericOp[] = ["isTrue", "isFalse"];
