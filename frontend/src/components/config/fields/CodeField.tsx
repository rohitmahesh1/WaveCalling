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
  language?: "json" | "yaml" | "text";
  onFormat?: (raw: string) => string | Promise<string>;
  onValidate?: (raw: string) => string | null | Promise<string | null>;
};

export default function CodeField({
  label,
  description,
  docsUrl,
  value,
  onChange,
  placeholder,
  rows = 8,
  disabled,
  className,
  error,
  language = "json",
  onFormat,
  onValidate,
}: Props) {
  const [busy, setBusy] = React.useState(false);
  const [localErr, setLocalErr] = React.useState<string | null>(null);

  async function handleFormat() {
    if (!onFormat) return;
    try {
      setBusy(true);
      const next = await onFormat(value ?? "");
      onChange(next);
      setLocalErr(null);
    } catch (e: any) {
      setLocalErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleValidate() {
    if (!onValidate) return;
    try {
      setBusy(true);
      const msg = await onValidate(value ?? "");
      setLocalErr(msg);
    } catch (e: any) {
      setLocalErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const control = (
    <div>
      <textarea
        value={value ?? ""}
        placeholder={placeholder}
        rows={rows}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full font-mono text-xs bg-slate-900/60 text-slate-200 px-2 py-1 rounded border border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400"
      />
      {(onFormat || onValidate) && (
        <div className="mt-2 flex items-center gap-2">
          {onFormat && (
            <button
              type="button"
              onClick={() => void handleFormat()}
              disabled={busy}
              className="px-2 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
            >
              {busy ? "Formatting…" : "Format"}
            </button>
          )}
          {onValidate && (
            <button
              type="button"
              onClick={() => void handleValidate()}
              disabled={busy}
              className="px-2 py-1 text-xs rounded border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
            >
              {busy ? "Validating…" : "Validate"}
            </button>
          )}
        </div>
      )}
    </div>
  );

  if (!label && !description && !docsUrl && !error && !localErr) {
    return <div className={className}>{control}</div>;
  }

  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        {label && <div className="text-sm text-slate-300">{label}</div>}
        {(description || docsUrl) && <InfoTooltip text={description} docsUrl={docsUrl} />}
        {language && (
          <span className="ml-auto text-[10px] uppercase tracking-wide text-slate-400 border border-slate-700 rounded px-1 py-0.5">
            {language}
          </span>
        )}
      </div>
      <div className="mt-1">{control}</div>
      {(error || localErr) && (
        <div className="mt-1 text-xs text-rose-300">{error ?? localErr}</div>
      )}
    </div>
  );
}
