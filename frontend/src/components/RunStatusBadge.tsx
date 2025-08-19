import * as React from "react";
import type { RunPhase } from "@/utils/types";

export default function RunStatusBadge({ status }: { status: RunPhase }) {
  const { bg, text } = colorFor(status);
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${bg} ${text}`}>
      {status}
    </span>
  );
}

function colorFor(s: RunPhase) {
  switch (s) {
    case "RUNNING":
    case "PROCESS":
    case "KYMO":
      return { bg: "bg-emerald-500/10", text: "text-emerald-300" };
    case "QUEUED":
    case "INIT":
    case "DISCOVER":
      return { bg: "bg-sky-500/10", text: "text-sky-300" };
    case "TABLE2HEATMAP":
    case "OVERLAY":
    case "WRITE":
    case "WRITE_PARTIAL":
      return { bg: "bg-indigo-500/10", text: "text-indigo-300" };
    case "DONE":
      return { bg: "bg-emerald-600/20", text: "text-emerald-200" };
    case "CANCELLED":
      return { bg: "bg-amber-600/20", text: "text-amber-200" };
    case "ERROR":
      return { bg: "bg-rose-600/20", text: "text-rose-200" };
    default:
      return { bg: "bg-slate-600/20", text: "text-slate-200" };
  }
}
