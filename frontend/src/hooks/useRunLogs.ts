import * as React from "react";

/**
 * Manages a per-run ring buffer of log lines persisted in localStorage.
 * - Keyed by `logs:<runId>`
 * - Max lines kept (default 5000)
 */
export function useRunLogs(runId?: string, maxLines = 5000) {
  const storageKey = runId ? `logs:${runId}` : null;

  const [logs, setLogs] = React.useState<string[]>([]);

  // Load from storage when runId changes
  React.useEffect(() => {
    if (!storageKey) {
      setLogs([]);
      return;
    }
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) {
        setLogs([]);
      } else {
        const arr = JSON.parse(raw);
        setLogs(Array.isArray(arr) ? arr : []);
      }
    } catch {
      setLogs([]);
    }
  }, [storageKey]);

  const persist = React.useCallback((next: string[]) => {
    if (!storageKey) return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(next));
    } catch {
      // ignore quota errors
    }
  }, [storageKey]);

  const append = React.useCallback((line: string | string[]) => {
    if (!runId) return;
    const lines = Array.isArray(line) ? line : [line];
    setLogs((prev) => {
      const next = [...prev, ...lines];
      const sliced = next.length > maxLines ? next.slice(next.length - maxLines) : next;
      persist(sliced);
      return sliced;
    });
  }, [persist, maxLines, runId]);

  const clear = React.useCallback(() => {
    setLogs([]);
    if (storageKey) {
      try { window.localStorage.removeItem(storageKey); } catch {}
    }
  }, [storageKey]);

  const download = React.useCallback((filename = (runId ? `run_${runId}_log.txt` : `log.txt`)) => {
    const blob = new Blob([logs.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }, [logs, runId]);

  return { logs, append, clear, download };
}
