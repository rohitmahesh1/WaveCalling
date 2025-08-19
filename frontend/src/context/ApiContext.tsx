// context/ApiContext.tsx
import * as React from "react";

function normalizeBase(base: string) {
  // Remove trailing slash, ensure absolute
  const u = new URL(base, typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
  let s = u.toString();
  if (s.endsWith("/")) s = s.slice(0, -1);
  return s;
}

const sameOrigin = typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";

const ENV_BASE = (import.meta as any)?.env?.VITE_API_URL || sameOrigin;

// Allow overrides: ?api=https://foo OR localStorage.apiBase
function initialBase(): string {
  try {
    const urlApi = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "").get("api");
    const persisted = typeof window !== "undefined" ? window.localStorage.getItem("apiBase") : null;
    return normalizeBase(urlApi || persisted || ENV_BASE);
  } catch {
    return normalizeBase(ENV_BASE);
  }
}

type ApiCtx = { base: string; setBase: (b: string) => void };

const ApiContext = React.createContext<ApiCtx>({ base: normalizeBase(ENV_BASE), setBase: () => {} });

export function ApiProvider({ base, children }: { base?: string; children: React.ReactNode }) {
  const [stateBase, setStateBase] = React.useState<string>(normalizeBase(base || initialBase()));
  const setBase = React.useCallback((b: string) => {
    const v = normalizeBase(b);
    setStateBase(v);
    try { window.localStorage.setItem("apiBase", v); } catch {}
  }, []);
  const value = React.useMemo(() => ({ base: stateBase, setBase }), [stateBase, setBase]);
  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

// Convenience hook used throughout the app
export function useApiBase() {
  return React.useContext(ApiContext).base;
}

// Optional: expose setter when we build Dashboard
export function useApiControls() {
  return React.useContext(ApiContext); // { base, setBase }
}
