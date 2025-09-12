// frontend/src/components/ui/CollapsiblePane.tsx
import * as React from "react";

type Props = {
  title: string;
  storageKey: string;          // used to persist width/open state
  defaultWidth?: number;       // px; default 360
  minWidth?: number;           // px; default 280
  maxWidth?: number;           // px; default 720
  startOpen?: boolean;         // default true
  className?: string;
  headerExtras?: React.ReactNode;
  children: React.ReactNode;
};

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

export default function CollapsiblePane({
  title,
  storageKey,
  defaultWidth = 360,
  minWidth = 280,
  maxWidth = 720,
  startOpen = true,
  className,
  headerExtras,
  children,
}: Props) {
  const WIDTH_KEY = `${storageKey}:width`;
  const OPEN_KEY = `${storageKey}:open`;

  // state
  const [open, setOpen] = React.useState<boolean>(() => {
    try {
      const v = window.localStorage.getItem(OPEN_KEY);
      return v === null ? startOpen : v === "1";
    } catch {
      return startOpen;
    }
  });

  const [width, setWidth] = React.useState<number>(() => {
    try {
      const v = window.localStorage.getItem(WIDTH_KEY);
      const parsed = v ? parseInt(v, 10) : defaultWidth;
      return clamp(Number.isFinite(parsed) ? parsed : defaultWidth, minWidth, maxWidth);
    } catch {
      return defaultWidth;
    }
  });

  // persist on change
  React.useEffect(() => {
    try { window.localStorage.setItem(OPEN_KEY, open ? "1" : "0"); } catch {}
  }, [open]);

  React.useEffect(() => {
    try { window.localStorage.setItem(WIDTH_KEY, String(width)); } catch {}
  }, [width]);

  // drag to resize (left edge)
  const paneRef = React.useRef<HTMLDivElement | null>(null);
  const dragRef = React.useRef<{ startX: number; startW: number } | null>(null);

  const onPointerDown = (e: React.PointerEvent) => {
    if (!open) return;
    const pane = paneRef.current;
    if (!pane) return;
    dragRef.current = { startX: e.clientX, startW: pane.offsetWidth };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    document.body.classList.add("select-none", "cursor-col-resize");
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d || !open) return;
    const delta = d.startX - e.clientX; // dragging left increases width
    const next = clamp(d.startW + delta, minWidth, maxWidth);
    setWidth(next);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (dragRef.current) {
      dragRef.current = null;
      (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
      document.body.classList.remove("select-none", "cursor-col-resize");
    }
  };

  const toggleOpen = React.useCallback(() => setOpen((v) => !v), []);
  const onDoubleClickRail = () => setWidth(defaultWidth);

  const paneStyle: React.CSSProperties = open
    ? { width: `${width}px` }
    : { width: "28px" }; // collapsed rail width

  // Minimum inner content width so wide grid rows can scroll instead of clipping.
  // Tweak if your editor rows get wider.
  const CONTENT_MIN_PX = 560;

  return (
    <div
      ref={paneRef}
      className={`relative h-full min-w-0 ${className || ""}`}  // min-w-0 allows the flex child to shrink
      style={paneStyle}
      aria-label={title}
      aria-expanded={open}
    >
      {/* Resize rail on the left */}
      <div
        role="separator"
        aria-orientation="vertical"
        title="Drag to resize"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={onDoubleClickRail}
        className={[
          "absolute left-0 top-0 h-full w-1.5",
          "cursor-col-resize",
          "bg-transparent hover:bg-slate-600/30 active:bg-slate-500/40",
          "rounded-l",
          open ? "" : "opacity-60",
        ].join(" ")}
      />

      {/* Collapsed tab */}
      {!open && (
        <div className="h-full pl-2 flex items-start">
          <button
            type="button"
            onClick={toggleOpen}
            aria-expanded={open}
            className="mt-2 px-2 py-1 rounded-md border border-slate-600 text-slate-200 bg-slate-800 hover:bg-slate-700"
            title={`Open ${title}`}
          >
            ⤢
          </button>
        </div>
      )}

      {/* Pane content */}
      {open && (
        <section className="h-full rounded-xl border border-slate-700/50 bg-console-700 flex flex-col min-w-0">
          <header className="flex items-center justify-between gap-2 px-3 py-2 border-b border-slate-700/50">
            <h3 className="text-slate-200 font-semibold truncate">{title}</h3>
            <div className="flex items-center gap-2">
              {headerExtras}
              <button
                type="button"
                onClick={toggleOpen}
                aria-expanded={open}
                className="px-2 py-1 rounded-md border border-slate-600 text-slate-300 hover:bg-slate-800"
                title={`Collapse ${title}`}
              >
                ⤡
              </button>
            </div>
          </header>

          {/* Scroll containers:
              - outer: vertical scroll, allow shrinking (min-w-0)
              - middle: horizontal scroll
              - inner: establish a minimum content width */}
          <div className="flex-1 min-h-0 min-w-0 overflow-y-auto">
            <div className="w-full min-w-0 overflow-x-auto">
              <div className="p-3" style={{ minWidth: `${CONTENT_MIN_PX}px` }}>
                {children}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
