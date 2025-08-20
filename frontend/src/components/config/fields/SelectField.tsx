import * as React from "react";
import InfoTooltip from "@/components/config/InfoTooltip";

type Option = { value: string; label: string };

type Props = {
  label?: string;
  description?: string;
  docsUrl?: string;
  value: string | null | undefined;
  onChange: (v: string) => void;
  options: Option[];
  disabled?: boolean;
  className?: string;
  error?: string;
};

export default function SelectField({
  label,
  description,
  docsUrl,
  value,
  onChange,
  options,
  disabled,
  className,
  error,
}: Props) {
  const control = (
    <select
      value={value ?? ""}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
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
