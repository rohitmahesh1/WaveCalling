import * as React from "react";

type Props = {
  logs: string[];
  paused: boolean;
  setPaused: (b: boolean) => void;
  onClear: () => void;
  onDownload: () => void;
  /** Called when user clicks Pause (e.g. cancel the run). Optional. */
  onPauseCancel?: () => void | Promise<void>;
  /** Called when user clicks Resume (e.g. resume the run). Optional. */
  onResume?: () => void | Promise<void>;
  className?: string;
};

export default function LiveLogPanel({
  logs,
  paused,
  setPaused,
  onClear,
  onDownload,
  onPauseCancel,
  onResume,
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
  const [follow, setFollow] = React.useState(true);
  const [busy, setBusy] = React.useState(false); // prevent double-clicks & show status
  const scrollerRef = React.useRef<HTMLDivElement | null>(null);

  // persist wrap preference
  React.useEffect(() => {
    try {
      window.localStorage.setItem(wrapKey, wrap ? "1" : "0");
    } catch {}
  }, [wrap]);

  // auto-scroll when following & not paused
  React.useEffect(() => {
    if (paused || !follow) return;
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [logs, paused, follow]);

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

  function handleScroll() {
    const el = scrollerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 10;
    if (!atBottom && follow) {
      setFollow(false);
    }
  }

  async function handlePauseResumeClick() {
    if (busy) return;
    setBusy(true);
    try {
      if (!paused) {
        // Pause: optionally cancel upstream, then pause logs
        await onPauseCancel?.();
        setPaused(true);
      } else {
        // Resume: optionally resume upstream, then resume logs & re-follow
        await onResume?.();
        setPaused(false);
        setFollow(true);
        // jump to bottom immediately on resume
        const el = scrollerRef.current;
        if (el) el.scrollTop = el.scrollHeight;
      }
    } finally {
      setBusy(false);
    }
  }

  const isPausing = busy && !paused;
  const isResuming = busy && paused;

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
            <input
              type="checkbox"
              checked={wrap}
              onChange={(e) => setWrap(e.target.checked)}
            />
            Wrap
          </label>
          <button
            onClick={handlePauseResumeClick}
            disabled={busy}
            className={`px-3 py-1.5 text-sm rounded-md border text-slate-200 disabled:opacity-60 ${
              paused
                ? "border-emerald-600 hover:bg-emerald-600/10"
                : "border-rose-600 hover:bg-rose-600/10"
            }`}
            title={
              paused
                ? "Resume run & log streaming"
                : "Pause log streaming (and cancel run if wired)"
            }
          >
            {paused ? (isResuming ? "Resuming…" : "Resume") : (isPausing ? "Pausing…" : "Pause")}
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
        <span className="text-slate-300">{filtered.length.toLocaleString()}</span>{" "}
        / <span className="text-slate-300">{logs.length.toLocaleString()}</span> lines
      </div>

      <div
        ref={scrollerRef}
        onScroll={handleScroll}
        className="mt-3 h-64 md:h-80 lg:h-96 overflow-auto rounded border border-slate-800 bg-slate-950/60 relative"
      >
        <pre
          className={`p-3 text-[12px] leading-5 text-slate-200 font-mono ${
            wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre"
          }`}
        >
          {filtered.join("\n")}
        </pre>

        {!follow && (
          <button
            onClick={() => {
              setFollow(true);
              // snap to bottom when re-following
              const el = scrollerRef.current;
              if (el) el.scrollTop = el.scrollHeight;
            }}
            className="absolute bottom-2 right-2 px-2 py-1 text-xs rounded-md border border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700"
          >
            Scroll to bottom
          </button>
        )}
      </div>
    </section>
  );
}
