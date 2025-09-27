// frontend/src/components/ArtifactsPanel.tsx
import * as React from "react";
import type { ArtifactMap } from "@/utils/types";

type Props = {
  /** Map of artifact key -> URL (from /api/runs/:id/artifacts). */
  artifactUrls?: ArtifactMap | null | undefined;
  /** Existence flags from /api/runs/:id/snapshot.artifacts (key -> bool). */
  artifactFlags?: Record<string, boolean> | null | undefined;
  className?: string;
  title?: string;
};

const LABELS: Record<string, string> = {
  overlay_json: "Overlay (final)",
  overlay_json_partial: "Overlay (partial)",
  tracks_csv: "Tracks CSV",
  waves_csv: "Waves CSV",
  manifest_json: "Manifest",
  progress_json: "Progress",
  run_json: "Run meta",
  events_ndjson: "Events log (NDJSON)",
  base_image: "Base image",
  plots_dir: "Plots dir",
  output_dir: "Output dir",
};

const ORDER = [
  "overlay_json",
  "overlay_json_partial",
  "tracks_csv",
  "waves_csv",
  "manifest_json",
  "progress_json",
  "run_json",
  "events_ndjson",
  "base_image",
  "plots_dir",
  "output_dir",
];

const DL_EXTS = new Set([".csv", ".json", ".ndjson", ".png", ".jpg", ".jpeg", ".txt"]);

function getExt(u: string) {
  try {
    const p = new URL(u, window.location.origin).pathname;
    const dot = p.lastIndexOf(".");
    return dot >= 0 ? p.slice(dot).toLowerCase() : "";
  } catch {
    const dot = u.lastIndexOf(".");
    return dot >= 0 ? u.slice(dot).toLowerCase() : "";
  }
}

function suggestFilename(u: string, labelKey: string) {
  try {
    const url = new URL(u, window.location.origin);
    const base = url.pathname.split("/").pop() || labelKey;
    return base;
  } catch {
    const base = u.split("/").pop() || labelKey;
    return base;
  }
}

export default function ArtifactsPanel({
  artifactUrls,
  artifactFlags,
  className,
  title = "Artifacts",
}: Props) {
  const [copiedKey, setCopiedKey] = React.useState<string | null>(null);

  // Build ordered list of [key, url]
  const pairs = React.useMemo(() => {
    if (!artifactUrls) return [] as [string, string][];
    const entries = Object.entries(artifactUrls).filter(([, v]) => !!v) as [string, string][];
    entries.sort((a, b) => {
      const ai = ORDER.indexOf(a[0]);
      const bi = ORDER.indexOf(b[0]);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a[0].localeCompare(b[0]);
    });
    return entries;
  }, [artifactUrls]);

  // Decide visibility: rely ONLY on flags (no HEAD probing to avoid 404 spam).
  const visible = React.useMemo(() => {
    if (!pairs.length) return [] as [string, string][];
    if (artifactFlags && Object.keys(artifactFlags).length) {
      return pairs.filter(([k]) => !!artifactFlags[k]);
    }
    // No flags yet: show nothing to avoid probing non-existent artifacts.
    return [];
  }, [pairs, artifactFlags]);

  const copy = React.useCallback(async (k: string, v: string) => {
    try {
      await navigator.clipboard.writeText(v);
      setCopiedKey(k);
      setTimeout(() => setCopiedKey((curr) => (curr === k ? null : curr)), 1200);
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <section className={`rounded-xl border border-slate-700/50 bg-console-700 p-4 ${className || ""}`}>
      <h2 className="text-slate-200 font-semibold">{title}</h2>
      {visible.length === 0 ? (
        <div className="text-sm text-slate-400 mt-3">
          No artifacts available yet.
        </div>
      ) : (
        <ul className="mt-3 space-y-2">
          {visible.map(([k, url]) => {
            const ext = getExt(url);
            const canDownload = DL_EXTS.has(ext);
            const filename = suggestFilename(url, k);
            const isPartial = k === "overlay_json_partial";

            return (
              <li key={k} className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="text-sm text-slate-300">
                    {LABELS[k] || k}
                    {isPartial && (
                      <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] bg-amber-500/10 text-amber-300 border border-amber-600/40">
                        partial
                      </span>
                    )}
                  </div>
                  <a
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="block truncate text-xs text-slate-400 hover:text-slate-300 underline decoration-slate-600"
                    title={url}
                  >
                    {url}
                  </a>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <a
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800"
                  >
                    Open
                  </a>
                  {canDownload && (
                    <a
                      href={url}
                      download={filename}
                      className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800"
                    >
                      Download
                    </a>
                  )}
                  <button
                    onClick={() => copy(k, url)}
                    className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800"
                  >
                    {copiedKey === k ? "Copied" : "Copy URL"}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
