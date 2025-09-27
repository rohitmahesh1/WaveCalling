import * as React from "react";

type Props = {
  title: string;
  subtitle?: string | React.ReactNode;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
  icon?: React.ReactNode;
};

export default function EmptyState({ title, subtitle, actionLabel, onAction, className, icon }: Props) {
  return (
    <div
      className={`rounded-xl border border-slate-700/50 bg-console-700 p-8 text-center text-slate-300 ${className || ""}`}
    >
      <div className="mx-auto w-12 h-12 mb-3 flex items-center justify-center rounded-full bg-slate-800/70">
        {icon ?? <span className="text-xl">🌀</span>}
      </div>
      <h3 className="text-slate-100 font-semibold">{title}</h3>
      {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
      {actionLabel && onAction && (
        <div className="mt-4">
          <button
            onClick={onAction}
            className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
          >
            {actionLabel}
          </button>
        </div>
      )}
    </div>
  );
}
