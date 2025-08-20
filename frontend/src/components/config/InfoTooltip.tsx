import * as React from "react";

export default function InfoTooltip({
  text,
  docsUrl,
  className,
  side = "top",
}: {
  text?: string;
  docsUrl?: string;
  className?: string;
  /** Where to place the hovercard relative to the icon. */
  side?: "top" | "bottom" | "left" | "right";
}) {
  const pos =
    side === "bottom"
      ? "top-full mt-1 left-1/2 -translate-x-1/2"
      : side === "left"
      ? "right-full mr-2 top-1/2 -translate-y-1/2"
      : side === "right"
      ? "left-full ml-2 top-1/2 -translate-y-1/2"
      : "bottom-full mb-1 left-1/2 -translate-x-1/2"; // top (default)

  const hasContent = !!text || !!docsUrl;

  return (
    <span className={`relative inline-flex group ${className || ""}`}>
      <span
        className={`inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-600 text-[10px] text-slate-300 ${hasContent ? "cursor-help" : ""}`}
        title={!hasContent ? undefined : text}
      >
        i
      </span>

      {/* Custom hovercard (hidden on mobile tap; native title still works) */}
      {hasContent && (
        <span
          className={`pointer-events-none absolute z-20 whitespace-pre-wrap rounded-md border border-slate-600 bg-slate-900 px-2 py-1 text-xs text-slate-200 opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100 ${pos} max-w-[320px]`}
          role="tooltip"
          aria-hidden="true"
        >
          {text}
          {docsUrl && (
            <>
              {text ? " " : ""}
              <a
                className="underline decoration-slate-600 hover:text-slate-300"
                href={docsUrl}
                target="_blank"
                rel="noreferrer"
              >
                Docs ↗
              </a>
            </>
          )}
        </span>
      )}
    </span>
  );
}
