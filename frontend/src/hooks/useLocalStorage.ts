// frontend/src/hooks/useLocalStorage.ts
import * as React from "react";

/** JSON localStorage with SSR safety. Returns [value, setValue]. */
export function useLocalStorage<T>(key: string, initialValue: T) {
  const read = React.useCallback((): T => {
    if (typeof window === "undefined") return initialValue;
    try {
      const raw = window.localStorage.getItem(key);
      return raw != null ? (JSON.parse(raw) as T) : initialValue;
    } catch {
      return initialValue;
    }
  }, [key, initialValue]);

  const [state, setState] = React.useState<T>(read);

  const set = React.useCallback(
    (val: T | ((prev: T) => T)) => {
      setState(prev => {
        const next = typeof val === "function" ? (val as (p: T) => T)(prev) : val;
        try { window.localStorage.setItem(key, JSON.stringify(next)); } catch {}
        return next;
      });
    },
    [key]
  );

  // Reload from storage if key changes at runtime
  React.useEffect(() => { setState(read()); }, [read]);

  return [state, set] as const;
}
