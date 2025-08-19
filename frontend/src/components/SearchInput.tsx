import * as React from "react";

type Props = {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  onEnter?: () => void;
  onClear?: () => void;
  className?: string;
  autoFocus?: boolean;
};

export default function SearchInput({
  value,
  onChange,
  placeholder = "Search…",
  onEnter,
  onClear,
  className,
  autoFocus,
}: Props) {
  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && onEnter) onEnter();
  };
  return (
    <div className={`relative ${className || ""}`}>
      <span className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-slate-500">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M21 21l-4.35-4.35M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
      </span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKey}
        autoFocus={autoFocus}
        placeholder={placeholder}
        className="w-full pl-8 pr-8 py-1.5 text-sm bg-slate-900/60 text-slate-200 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
      />
      {value && (
        <button
          aria-label="Clear"
          className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200"
          onClick={() => {
            onChange("");
            onClear?.();
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
        </button>
      )}
    </div>
  );
}
