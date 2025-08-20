import * as React from "react";
import InfoTooltip from "@/components/config/InfoTooltip";

type Props = {
  label?: string;
  description?: string;
  docsUrl?: string;
  value: string | null | undefined;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
  disabled?: boolean;
  className?: string;
  error?: string;
  mono?: boolean;
};

export default function TextareaField({
  label,
  description,
  docsUrl,
  value,
  onChange,
  placeholder,
  rows = 4,
  disabled,
  className,
  error,
  mono,
}: Props) {
  const control = (
    <textarea
      value={value ?? ""}
      placeholder={placeholder}
      rows={rows}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className={`w-full bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400 ${
        mono ? "font-mono" : ""
      }`}
    />
  );

  if (!label && !description && !docsUrl && !error) {
    return <div className={className}>{control}</div>;
  }

  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        {label && <div className="text-sm text-slate-300">{label}</div>}
        {(description || docsUrl) && <InfoTooltip text={description} docsUrl={docsUrl} />}
      </div>
      <div className="mt-1">{control}</div>
      {error && <div className="mt-1 text-xs text-rose-300">{error}</div>}
    </div>
  );
}
