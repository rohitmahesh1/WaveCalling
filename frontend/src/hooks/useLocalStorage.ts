// frontend/src/hooks/useLocalStorage.ts
import * as React from "react";

export function useLocalStorage<T>(key: string, initial: T) {
  // lazy init from current key
  const [value, setValue] = React.useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw != null ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });

  // Track the active key; when the key changes, load once without looping
  const keyRef = React.useRef(key);
  React.useEffect(() => {
    if (keyRef.current === key) return;
    keyRef.current = key;
    let next = initial;
    try {
      const raw = localStorage.getItem(key);
      next = raw != null ? (JSON.parse(raw) as T) : initial;
    } catch {}
    // Only update if different to avoid unnecessary renders
    setValue((prev) => (Object.is(prev, next) ? prev : next));
  }, [key, initial]);

  // Persist on value/key change
  React.useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
  }, [key, value]);

  return [value, setValue] as const;
}
