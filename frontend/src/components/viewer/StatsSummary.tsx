import * as React from "react";

type Props = {
  count: number;
  points: number;
  avgAmplitude: number | null;
  avgFrequency: number | null;
  className?: string;
  /** Show a more compact row layout */
  dense?: boolean;
};

const fmt = (n: number | null | undefined, digits = 3) =>
  n == null || !Number.isFinite(n) ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });

export default function StatsSummary({
  count,
  points,
  avgAmplitude,
  avgFrequency,
  className,
  dense = false,
}: Props) {
  const Item = ({ label, value }: { label: string; value: React.ReactNode }) => (
    <div className="flex flex-col">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className="text-slate-100">{value}</div>
    </div>
  );

  if (dense) {
    return (
      <div className={`flex flex-wrap items-center gap-x-6 gap-y-2 ${className || ""}`}>
        <Item label="tracks" value={fmt(count, 0)} />
        <Item label="points" value={fmt(points, 0)} />
        <Item label="avg amplitude" value={fmt(avgAmplitude)} />
        <Item label="avg frequency (Hz)" value={fmt(avgFrequency)} />
      </div>
    );
  }

  return (
    <div className={`grid grid-cols-2 gap-3 ${className || ""}`}>
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3">
        <Item label="tracks" value={fmt(count, 0)} />
      </div>
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3">
        <Item label="points" value={fmt(points, 0)} />
      </div>
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3">
        <Item label="avg amplitude" value={fmt(avgAmplitude)} />
      </div>
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3">
        <Item label="avg frequency (Hz)" value={fmt(avgFrequency)} />
      </div>
    </div>
  );
}
