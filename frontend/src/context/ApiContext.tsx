import React, { createContext, useContext, useMemo } from "react";

type Ctx = { apiBase: string };
const ApiCtx = createContext<Ctx | null>(null);

export const ApiProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
  // Prefer env; fallback to current origin (strip /ui if present)
  const env = import.meta.env.VITE_API_URL as string | undefined;
  const fromOrigin = window.location.origin.replace(/\/$/, "");
  const apiBase = (env && env.trim()) || fromOrigin;

  const value = useMemo(() => ({ apiBase }), [apiBase]);
  return <ApiCtx.Provider value={value}>{children}</ApiCtx.Provider>;
};

export function useApiBase() {
  const v = useContext(ApiCtx);
  if (!v) throw new Error("ApiProvider missing");
  return v.apiBase;
}
