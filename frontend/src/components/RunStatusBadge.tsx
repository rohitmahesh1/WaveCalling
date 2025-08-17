import { RunPhase } from "@/utils/types";

export default function RunStatusBadge({ status }: { status: RunPhase }) {
  const cls =
    status === "DONE" ? "badge ok" :
    status === "ERROR" ? "badge err" :
    status === "CANCELLED" ? "badge warn" :
    "badge";
  return <span className={cls}>{status}</span>;
}
