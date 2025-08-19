import * as React from "react";

type Props = {
  logs: string[];
  paused: boolean;
  setPaused: (b: boolean) => void;
  onClear: () => void;
  onDownload: () => void;
  className?: string;
};

export default function LiveLogPanel({
  logs,
  paused,
  setPaused,
  onClear,
  onDownload,
  className,
}: Props) {
  const wrapKey = "log:wrap";
  const [wrap, setWrap] = React.useState<boolean>(() => {
    try {
      return window.localStorage.getItem(wrapKey) === "1";
    } catch {
      return false;
    }
  });
  const [query, setQuery] = React.useState("");
  const scrollerRef = React.useRef<HTMLDivElement | null>(null);

  // persist wrap preference
  React.useEffect(() => {
    try {
      window.localStorage.setItem(wrapKey, wrap ? "1" : "0");
    } catch {}
  }, [wrap]);

  // auto-scroll to bottom on new logs when not paused
  React.useEffect(() => {
    if (paused) return;
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [logs, paused]);

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return logs;
    return logs.filter((line) => line.toLowerCase().includes(q));
  }, [logs, query]);

  const copyFiltered = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(filtered.join("\n"));
    } catch {
      // ignore
    }
  }, [filtered]);

  return (
    <section className={`rounded-xl border border-slate-700/50 bg-console-700 p-4 ${className || ""}`}>
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-slate-200 font-semibold">Live Log</h2>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter…"
            className="text-sm bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
          />
          <label className="inline-flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={wrap} onChange={(e) => setWrap(e.target.checked)} />
            Wrap
          </label>
          <button
            onClick={() => setPaused(!paused)}
            className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={copyFiltered}
            className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            title="Copy visible lines"
          >
            Copy
          </button>
          <button
            onClick={onDownload}
            className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
          >
            Download
          </button>
          <button
            onClick={onClear}
            className="px-3 py-1.5 text-sm rounded-md border border-rose-600 text-rose-300 hover:bg-rose-600/10"
          >
            Clear
          </button>
        </div>
      </header>

      <div className="mt-2 text-xs text-slate-400">
        showing{" "}
        <span className="text-slate-300">{filtered.length.toLocaleString()}</span> /
        <span className="text-slate-300"> {logs.length.toLocaleString()}</span> lines
      </div>

      <div
        ref={scrollerRef}
        className="mt-3 h-64 md:h-80 lg:h-96 overflow-auto rounded border border-slate-800 bg-slate-950/60"
      >
        <pre
          className={`p-3 text-[12px] leading-5 text-slate-200 font-mono ${
            wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre"
          }`}
        >
          {filtered.join("\n")}
        </pre>
      </div>
    </section>
  );
}
