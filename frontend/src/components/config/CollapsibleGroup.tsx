import * as React from "react";

export default function CollapsibleGroup({
  title,
  description,
  defaultOpen = true,
  right,
  children,
  className,
}: {
  title: string;
  description?: string;
  defaultOpen?: boolean;
  right?: React.ReactNode; // actions on the right of the header
  children: React.ReactNode;
  className?: string;
}) {
  const [open, setOpen] = React.useState<boolean>(defaultOpen);

  return (
    <div className={`rounded-lg border border-slate-700/50 bg-slate-900/40 ${className || ""}`}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-start justify-between gap-3 px-3 py-2 hover:bg-slate-800/40"
        aria-expanded={open}
      >
        <div className="flex items-start gap-2 text-left">
          <span
            className={`mt-0.5 inline-block h-4 w-4 rotate-0 transition-transform ${open ? "rotate-90" : ""}`}
          >
            {/* Caret */}
            <svg viewBox="0 0 20 20" className="h-4 w-4 fill-slate-300">
              <path d="M7 5l6 5-6 5V5z" />
            </svg>
          </span>
          <div className="min-w-0">
            <div className="text-slate-200 font-medium">{title}</div>
            {description && <div className="text-xs text-slate-400">{description}</div>}
          </div>
        </div>
        {right && <div className="shrink-0">{right}</div>}
      </button>

      {open && <div className="px-3 pb-3 pt-1">{children}</div>}
    </div>
  );
}
