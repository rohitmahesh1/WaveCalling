import * as React from "react";

// Default to same-origin (useful when UI is served by FastAPI in prod)
const sameOrigin = typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";

// Prefer build-time Vite env, else fallback
const DEFAULT_API_BASE =
  (import.meta as any)?.env?.VITE_API_URL || sameOrigin;

const ApiContext = React.createContext<string>(DEFAULT_API_BASE);

export function ApiProvider({
  base,
  children,
}: {
  base?: string;
  children: React.ReactNode;
}) {
  const value = base ?? DEFAULT_API_BASE;
  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

export function useApiBase() {
  return React.useContext(ApiContext);
}
