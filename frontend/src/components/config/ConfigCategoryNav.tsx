import * as React from "react";
import InfoTooltip from "./InfoTooltip";

export type ConfigNavItem = {
  id: string;
  label: string;
  /** Optional short help shown in a tooltip. */
  hint?: string;
  /** Optional icon node (e.g., <SomeIcon className="h-4 w-4" />). */
  icon?: React.ReactNode;
  /** Number of changed fields within this category. */
  changed?: number;
  /** Number of validation errors within this category. */
  errors?: number;
  /** Disable navigation to this item. */
  disabled?: boolean;
};

type Props = {
  items: ConfigNavItem[];
  activeId?: string;
  onSelect: (id: string) => void;
  /** Show a small search box to filter categories. */
  searchable?: boolean;
  className?: string;
};

export default function ConfigCategoryNav({
  items,
  activeId,
  onSelect,
  searchable = true,
  className,
}: Props) {
  const [q, setQ] = React.useState("");

  const filtered = React.useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((it) => it.label.toLowerCase().includes(needle));
  }, [items, q]);

  const activeIndex = React.useMemo(
    () => filtered.findIndex((it) => it.id === activeId),
    [filtered, activeId]
  );

  // Keyboard navigation: Up/Down to move focus/selection, Enter/Space to select
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!filtered.length) return;
    const idx = Math.max(0, activeIndex);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      const next = filtered[Math.min(filtered.length - 1, idx + 1)];
      if (next && !next.disabled) onSelect(next.id);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      const prev = filtered[Math.max(0, idx - 1)];
      if (prev && !prev.disabled) onSelect(prev.id);
    } else if (e.key === "Home") {
      e.preventDefault();
      const first = filtered.find((f) => !f.disabled);
      if (first) onSelect(first.id);
    } else if (e.key === "End") {
      e.preventDefault();
      const last = [...filtered].reverse().find((f) => !f.disabled);
      if (last) onSelect(last.id);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (idx >= 0 && !filtered[idx].disabled) onSelect(filtered[idx].id);
    }
  };

  const totalChanged = React.useMemo(
    () => items.reduce((sum, it) => sum + (it.changed || 0), 0),
    [items]
  );

  return (
    <nav
      className={`rounded-xl border border-slate-700/50 bg-console-700 p-3 ${className || ""}`}
      aria-label="Configuration sections"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-slate-200 font-semibold text-sm">Configuration</h2>
        {totalChanged > 0 && (
          <span className="text-[10px] rounded-full border border-amber-600/40 bg-amber-500/10 text-amber-300 px-2 py-0.5">
            {totalChanged} change{totalChanged === 1 ? "" : "s"}
          </span>
        )}
      </header>

      {searchable && (
        <div className="mb-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search sections…"
            className="w-full text-sm bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
            aria-label="Search configuration sections"
          />
        </div>
      )}

      <ul
        className="max-h-[60vh] overflow-auto divide-y divide-slate-800/70"
        onKeyDown={onKeyDown}
        tabIndex={0}
      >
        {filtered.length === 0 ? (
          <li className="py-3 text-sm text-slate-400">No matching sections</li>
        ) : (
          filtered.map((it) => {
            const isActive = it.id === activeId;
            const hasChanges = (it.changed || 0) > 0;
            const hasErrors = (it.errors || 0) > 0;

            return (
              <li key={it.id} className="flex items-center">
                <button
                  type="button"
                  disabled={it.disabled}
                  onClick={() => onSelect(it.id)}
                  className={[
                    "flex-1 min-w-0 text-left px-3 py-2 flex items-center gap-2",
                    "hover:bg-slate-800/50 focus:outline-none focus:ring-1 focus:ring-slate-500",
                    it.disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
                    isActive ? "bg-slate-800/60" : "",
                  ].join(" ")}
                  aria-current={isActive ? "page" : undefined}
                >
                  {/* Icon */}
                  <span className="h-4 w-4 inline-flex items-center justify-center text-slate-400">
                    {it.icon ?? (
                      <span className="h-1.5 w-1.5 rounded-full bg-slate-500 inline-block" />
                    )}
                  </span>

                  {/* Label */}
                  <span className="flex-1 min-w-0 truncate text-sm text-slate-200">
                    {it.label}
                  </span>

                  {/* Badges */}
                  {hasErrors && (
                    <span className="ml-1 shrink-0 text-[10px] rounded-full border border-rose-700/50 bg-rose-600/15 text-rose-300 px-1.5 py-0.5">
                      {it.errors}
                    </span>
                  )}
                  {hasChanges && !hasErrors && (
                    <span className="ml-1 shrink-0 text-[10px] rounded-full border border-amber-600/40 bg-amber-500/10 text-amber-300 px-1.5 py-0.5">
                      {it.changed}
                    </span>
                  )}
                </button>

                {/* Tooltip as a sibling so no <a> is inside the <button> */}
                {it.hint && (
                  <InfoTooltip text={it.hint} side="left" className="ml-2 shrink-0" />
                )}
              </li>
            );
          })
        )}
      </ul>
    </nav>
  );
}
