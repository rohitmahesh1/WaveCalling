import * as React from "react";
import type { ArtifactMap } from "@/utils/types";

type Props = {
  artifacts: ArtifactMap | null | undefined;
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

// Extensions we’ll suggest as downloadable
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

export default function ArtifactsPanel({ artifacts, className, title = "Artifacts" }: Props) {
  const [copiedKey, setCopiedKey] = React.useState<string | null>(null);

  const items = React.useMemo(() => {
    if (!artifacts) return [];
    const pairs = Object.entries(artifacts).filter(([, v]) => !!v);
    return pairs.sort((a, b) => {
      const ai = ORDER.indexOf(a[0]);
      const bi = ORDER.indexOf(b[0]);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a[0].localeCompare(b[0]);
    });
  }, [artifacts]);

  const copy = React.useCallback(async (k: string, v: string) => {
    try {
      await navigator.clipboard.writeText(v);
      setCopiedKey(k);
      window.setTimeout(() => setCopiedKey((curr) => (curr === k ? null : curr)), 1200);
    } catch {
      /* no-op */
    }
  }, []);

  return (
    <section className={`rounded-xl border border-slate-700/50 bg-console-700 p-4 ${className || ""}`}>
      <h2 className="text-slate-200 font-semibold">{title}</h2>

      {(!artifacts || items.length === 0) ? (
        <div className="text-sm text-slate-400 mt-3">No artifacts yet.</div>
      ) : (
        <ul className="mt-3 space-y-2">
          {items.map(([k, url]) => {
            const u = url as string;
            const ext = getExt(u);
            const canDownload = DL_EXTS.has(ext);
            const filename = suggestFilename(u, k);

            return (
              <li key={k} className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="text-sm text-slate-300">
                    {LABELS[k] || k}
                    {k === "overlay_json_partial" && (
                      <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] bg-amber-500/10 text-amber-300 border border-amber-600/40">
                        partial
                      </span>
                    )}
                  </div>
                  <a
                    href={u}
                    target="_blank"
                    rel="noreferrer"
                    className="block truncate text-xs text-slate-400 hover:text-slate-300 underline decoration-slate-600 cursor-pointer"
                    title={u}
                  >
                    {u}
                  </a>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <a
                    href={u}
                    target="_blank"
                    rel="noreferrer"
                    className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800 cursor-pointer"
                  >
                    Open
                  </a>

                  {canDownload && (
                    <a
                      href={u}
                      download={filename}
                      className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800 cursor-pointer"
                    >
                      Download
                    </a>
                  )}

                  <button
                    onClick={() => copy(k, u)}
                    className="px-2.5 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800 cursor-pointer"
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
