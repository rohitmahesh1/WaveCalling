// src/hooks/useProgress.ts
import * as React from "react";
import { useApiBase } from "@/context/ApiContext";
import { getProgress } from "@/utils/api";
import type { ProgressResponse } from "@/utils/types";

const RUN_ID_RE = /^[a-f0-9]{6,32}$/i;
function sanitizeRunId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  if (s.startsWith("http://") || s.startsWith("https://")) return null;
  if (s.includes("/")) return null;
  if (!RUN_ID_RE.test(s)) return null;
  return s;
}

export function useProgress(runId: string | null) {
  const apiBase = useApiBase();
  const [progress, setProgress] = React.useState<ProgressResponse | null>(null);

  const refreshProgress = React.useCallback(async () => {
    const id = sanitizeRunId(runId);
    if (!id) {
      setProgress(null);
      return;
    }
    try {
      const p = await getProgress(apiBase, id);
      setProgress(p);
    } catch {
      // ignore transient fetch issues
    }
  }, [apiBase, runId]);

  return { progress, refreshProgress };
}
