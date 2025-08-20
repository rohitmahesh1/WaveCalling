import * as React from "react";
import InfoTooltip from "./InfoTooltip";

export default function FieldRow({
  label,
  description,
  docsUrl,
  error,
  children,
  className,
  inline = false,
}: {
  label: string;
  description?: string;
  docsUrl?: string;
  error?: string | null;
  children: React.ReactNode;
  className?: string;
  /** If true, put label and input on the same row (2-col). */
  inline?: boolean;
}) {
  if (inline) {
    return (
      <div className={`grid grid-cols-1 sm:grid-cols-[240px,1fr] gap-3 ${className || ""}`}>
        <div className="min-w-0">
          <label className="block text-sm text-slate-200">
            {label}
            {(description || docsUrl) && (
              <InfoTooltip className="ml-2 align-middle" text={description} docsUrl={docsUrl} />
            )}
          </label>
          {error && <div className="mt-1 text-xs text-rose-300">{error}</div>}
        </div>
        <div className="min-w-0">{children}</div>
      </div>
    );
  }

  return (
    <div className={`min-w-0 ${className || ""}`}>
      <label className="block text-sm text-slate-200">
        {label}
        {(description || docsUrl) && (
          <InfoTooltip className="ml-2 align-middle" text={description} docsUrl={docsUrl} />
        )}
      </label>
      <div className="mt-1">{children}</div>
      {error && <div className="mt-1 text-xs text-rose-300">{error}</div>}
    </div>
  );
}
