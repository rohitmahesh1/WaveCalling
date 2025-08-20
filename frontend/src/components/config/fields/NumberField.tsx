import * as React from "react";
import InfoTooltip from "@/components/config/InfoTooltip";

type Props = {
  label?: string;
  description?: string;
  docsUrl?: string;
  value: number | string | null | undefined;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  error?: string;
  width?: "auto" | "sm" | "md" | "lg";
};

export default function NumberField({
  label,
  description,
  docsUrl,
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
  disabled,
  className,
  error,
  width = "md",
}: Props) {
  const widthCls =
    width === "sm" ? "w-28" : width === "md" ? "w-40" : width === "lg" ? "w-56" : "w-auto";

  const control = (
    <input
      type="number"
      value={value ?? ""}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(null);
        const num = Number(raw);
        onChange(Number.isFinite(num) ? num : null);
      }}
      className={`bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400 ${widthCls}`}
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
