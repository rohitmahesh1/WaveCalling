import * as React from "react";
import type { RunInfo } from "@/utils/types";
import RunStatusBadge from "@/components/RunStatusBadge";
import RunActions from "@/components/RunActions";

export default function RunRow({
  run,
  selected,
  onOpen,
  onChanged,
}: {
  run: RunInfo;
  selected?: boolean;
  onOpen: () => void;
  onChanged?: () => void;
}) {
  return (
    <tr
      className={`border-t border-slate-800 hover:bg-slate-800/40 ${selected ? "bg-slate-800/60" : ""}`}
      onDoubleClick={onOpen}
    >
      <td className="py-2 pr-3 align-top">
        <code className="text-slate-300">{run.run_id}</code>
      </td>
      <td className="py-2 pr-3 align-top">{run.name}</td>
      <td className="py-2 pr-3 align-top">
        <RunStatusBadge status={run.status} />
      </td>
      <td className="py-2 pr-3 align-top">{new Date(run.created_at).toLocaleString()}</td>
      <td className="py-2 pr-3 align-top">
        <RunActions run={run} onOpen={onOpen} onChanged={onChanged} />
      </td>
    </tr>
  );
}
