import * as React from "react";
import { createPortal } from "react-dom";

type Props = {
  open: boolean;
  title: string;
  message?: string | React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  /** Close when clicking outside (backdrop) */
  closeOnBackdrop?: boolean;
  children?: React.ReactNode;
};

export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  busy = false,
  onConfirm,
  onCancel,
  closeOnBackdrop = true,
  children,
}: Props) {
  const [mounted, setMounted] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const confirmRef = React.useRef<HTMLButtonElement | null>(null);

  React.useEffect(() => {
    setMounted(true);
    if (!containerRef.current) {
      containerRef.current = document.createElement("div");
      document.body.appendChild(containerRef.current);
    }
    return () => {
      if (containerRef.current) {
        document.body.removeChild(containerRef.current);
        containerRef.current = null;
      }
    };
  }, []);

  React.useEffect(() => {
    if (open) {
      const prev = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      // focus confirm by default
      setTimeout(() => confirmRef.current?.focus(), 0);
      return () => {
        document.body.style.overflow = prev;
      };
    }
  }, [open]);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!open) return;
      if (e.key === "Escape") onCancel();
      if (e.key.toLowerCase() === "y" && (e.ctrlKey || e.metaKey)) onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel, onConfirm]);

  if (!mounted || !containerRef.current || !open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      aria-modal="true"
      role="dialog"
      aria-labelledby="confirm-title"
      onMouseDown={(e) => {
        if (!closeOnBackdrop) return;
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      {/* backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-[1px]" />

      {/* dialog */}
      <div className="relative z-10 w-[92vw] max-w-md rounded-xl border border-slate-700/60 bg-console-700 p-5 shadow-xl">
        <h3 id="confirm-title" className="text-slate-100 font-semibold">
          {title}
        </h3>
        {message && <div className="mt-2 text-sm text-slate-300">{message}</div>}
        {children}

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            className="px-3 py-1.5 text-sm rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800"
            onClick={onCancel}
            disabled={busy}
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            className={`px-3 py-1.5 text-sm rounded-md border ${
              destructive
                ? "border-rose-600 text-rose-300 hover:bg-rose-600/10"
                : "border-emerald-600 text-emerald-300 hover:bg-emerald-600/10"
            } disabled:opacity-60`}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    containerRef.current
  );
}
