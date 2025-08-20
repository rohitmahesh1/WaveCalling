import * as React from "react";
import InfoTooltip from "@/components/config/InfoTooltip";

type Props = {
  title?: string;
  /** Number of changed fields (for a badge). */
  changedCount?: number;
  /** Number of validation errors (for a badge). */
  errorCount?: number;

  /** Current overrides JSON text (pretty). Enables Copy/Download buttons. */
  overridesJson?: string;

  /** Reset all local changes to base config. */
  onReset?: () => void;

  /** Called when user wants to apply overrides from a JSON file (or pasted text). */
  onUploadOverrides?: (text: string) => void;

  /** Custom copy handler; if omitted we copy `overridesJson` to clipboard. */
  onCopyOverrides?: () => void;

  /** Optional extra controls on the right side (e.g., Save, Apply buttons). */
  rightExtra?: React.ReactNode;

  className?: string;
  /** Optional short help shown next to the title. */
  hint?: string;
  /** Optional docs URL shown in the tooltip. */
  docsUrl?: string;
};

export default function ConfigHeader({
  title = "Configuration",
  changedCount = 0,
  errorCount = 0,
  overridesJson,
  onReset,
  onUploadOverrides,
  onCopyOverrides,
  rightExtra,
  className,
  hint,
  docsUrl,
}: Props) {
  const fileRef = React.useRef<HTMLInputElement | null>(null);

  const canCopy = !!overridesJson || !!onCopyOverrides;
  const canDownload = !!overridesJson;

  const handleCopy = async () => {
    if (onCopyOverrides) return void onCopyOverrides();
    if (!overridesJson) return;
    try {
      await navigator.clipboard.writeText(overridesJson);
    } catch {
      // no-op
    }
  };

  const handleDownload = () => {
    if (!overridesJson) return;
    const blob = new Blob([overridesJson], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.overrides.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleChooseFile = () => fileRef.current?.click();

  const onFileChange: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      onUploadOverrides?.(text);
    } finally {
      // allow re-selecting the same file next time
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <header
      className={`flex items-center justify-between gap-3 rounded-xl border border-slate-700/50 bg-console-700 px-4 py-3 ${className || ""}`}
    >
      <div className="min-w-0 flex items-center gap-2">
        <h2 className="text-slate-200 font-semibold truncate">{title}</h2>
        {(hint || docsUrl) && (
          <InfoTooltip text={hint} docsUrl={docsUrl} side="right" />
        )}
        {changedCount > 0 && (
          <span className="ml-1 text-[10px] rounded-full border border-amber-600/40 bg-amber-500/10 text-amber-300 px-2 py-0.5">
            {changedCount} change{changedCount === 1 ? "" : "s"}
          </span>
        )}
        {errorCount > 0 && (
          <span className="ml-1 text-[10px] rounded-full border border-rose-700/50 bg-rose-600/15 text-rose-300 px-2 py-0.5">
            {errorCount} error{errorCount === 1 ? "" : "s"}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={onFileChange}
        />

        <button
          type="button"
          onClick={handleChooseFile}
          className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
          title="Load overrides from a JSON file"
        >
          Upload JSON
        </button>

        <button
          type="button"
          onClick={handleCopy}
          disabled={!canCopy}
          className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-50"
          title="Copy current overrides to clipboard"
        >
          Copy JSON
        </button>

        <button
          type="button"
          onClick={handleDownload}
          disabled={!canDownload}
          className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-50"
          title="Download current overrides as a JSON file"
        >
          Download
        </button>

        <button
          type="button"
          onClick={onReset}
          disabled={!onReset || changedCount === 0}
          className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 disabled:opacity-50"
          title="Reset all local changes"
        >
          Reset
        </button>

        {rightExtra}
      </div>
    </header>
  );
}
