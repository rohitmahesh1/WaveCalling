import * as React from "react";
import InfoTooltip from "@/components/config/InfoTooltip";

type Props = {
  label?: string;
  description?: string;
  docsUrl?: string;
  value?: boolean | null;
  checked?: boolean;                 // alias for bare mode
  onChange: (v: boolean) => void;
  disabled?: boolean;
  className?: string;
  error?: string;
};

export default function ToggleField({
  label,
  description,
  docsUrl,
  value,
  checked,
  onChange,
  disabled,
  className,
  error,
}: Props) {
  const isChecked = typeof checked === "boolean" ? checked : !!value;

  const control = (
    <label className={`inline-flex items-center gap-2 ${disabled ? "opacity-60" : ""}`}>
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-400"
        checked={isChecked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label && <span className="text-slate-200 text-sm select-none">{label}</span>}
    </label>
  );

  if (!label && !description && !docsUrl && !error) {
    return <div className={className}>{control}</div>;
  }

  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        {control}
        {(description || docsUrl) && (
          <InfoTooltip text={description} docsUrl={docsUrl} />
        )}
      </div>
      {error && <div className="mt-1 text-xs text-rose-300">{error}</div>}
    </div>
  );
}
