// frontend/src/hooks/useRafThrottle.ts
import * as React from "react";

/**
 * Returns a throttled function that invokes `fn` at most once per animation frame.
 * - Always calls with the latest arguments.
 * - Cleans up a pending rAF on unmount.
 */
export function useRafThrottle<T extends any[]>(
  fn: (...args: T) => void
): (...args: T) => void {
  const fnRef = React.useRef(fn);
  React.useEffect(() => { fnRef.current = fn; }, [fn]);

  const rafRef = React.useRef<number | null>(null);
  const lastArgsRef = React.useRef<T | null>(null);

  const throttled = React.useCallback((...args: T) => {
    lastArgsRef.current = args;
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        const a = lastArgsRef.current;
        if (a) fnRef.current(...a);
      });
    }
  }, []);

  React.useEffect(() => {
    return () => {
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  return throttled;
}
