// frontend/src/hooks/useWaveMetrics.ts
import * as React from "react";

type Row = Record<string, any>;
type RowByTrackId = Record<string, Record<string, any>>;

function normalizeKey(s: string): string {
  return s
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "") // strip accents / Greek delta, etc.
    .replace(/[^\w]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
}

function parseCSV(text: string): Row[] {
  // Robust-ish CSV parser with quoted fields + doubled quotes
  const rows: string[][] = [];
  let cur: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") {
        cur.push(field);
        field = "";
      } else if (c === "\n") {
        cur.push(field);
        rows.push(cur);
        cur = [];
        field = "";
      } else if (c === "\r") {
        // ignore
      } else {
        field += c;
      }
    }
  }
  if (field.length || cur.length) {
    cur.push(field);
    rows.push(cur);
  }
  if (!rows.length) return [];

  const headers = rows.shift()!.map((h) => normalizeKey(String(h || "")));

  const out: Row[] = [];
  for (const r of rows) {
    // skip empty lines
    if (!r.length || r.every((v) => String(v).trim() === "")) continue;
    const obj: Row = {};
    for (let k = 0; k < headers.length; k++) {
      const key = headers[k] || `col_${k}`;
      const raw = r[k] ?? "";
      // numeric coercion when possible
      const num = Number(raw);
      obj[key] = raw === "" ? null : (Number.isFinite(num) ? num : String(raw));
    }
    out.push(obj);
  }
  return out;
}

async function fetchCsvFirst(urls: string[]): Promise<Row[] | null> {
  for (const url of urls) {
    try {
      const resp = await fetch(url, { cache: "no-store" });
      if (resp.ok) {
        const text = await resp.text();
        if (!text.trim()) return [];
        return parseCSV(text);
      }
      if (resp.status === 404) continue;
    } catch {
      // ignore and try next candidate
    }
  }
  return null;
}

function aggregateNumeric(rows: Row[]) {
  const out: Row = { waves_count: rows.length };
  // Collect all numeric keys seen
  const numericKeys = new Set<string>();
  for (const r of rows) {
    for (const [k, v] of Object.entries(r)) {
      if (v != null && typeof v === "number" && Number.isFinite(v)) {
        // skip ids
        if (/^(track|track_id|wave_id|sample|sample_id|wave_number)$/.test(k)) continue;
        numericKeys.add(k);
      }
    }
  }
  for (const k of numericKeys) {
    let sum = 0, min = +Infinity, max = -Infinity, count = 0;
    for (const r of rows) {
      const v = r[k];
      if (typeof v === "number" && Number.isFinite(v)) {
        sum += v; count++;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (count) {
      out[`waves_${k}_mean`] = sum / count;
      out[`waves_${k}_min`] = min;
      out[`waves_${k}_max`] = max;
    }
  }
  return out;
}

export function useWaveMetrics(runId?: string | null) {
  const [rowByTrackId, setRowByTrackId] = React.useState<RowByTrackId>({});

  React.useEffect(() => {
    let cancelled = false;
    if (!runId) { setRowByTrackId({}); return; }

    (async () => {
      const base = `/runs/${encodeURIComponent(runId)}/output`;

      // Your pipeline names first, then fallbacks for compatibility
      const wavesRows =
        (await fetchCsvFirst([
          `${base}/metrics_waves.partial.csv`,
          `${base}/metrics_waves_partial.csv`,
          `${base}/metrics_waves.csv`,
          `${base}/waves.csv`,
        ])) ?? [];

      const metricsRows =
        (await fetchCsvFirst([
          `${base}/metrics.partial.csv`,
          `${base}/metrics.csv`,
        ])) ?? [];

      const byId: RowByTrackId = {};

      // Index metrics (per-track) with 'metrics_' prefix
      const metricsBy = new Map<string, Row>();
      for (const r of metricsRows) {
        const tid = r.track_id ?? r.track ?? r.id;
        if (tid == null) continue;
        const key = String(tid);
        const obj: Row = {};
        for (const [k, v] of Object.entries(r)) {
          if (k === "track_id" || k === "track" || k === "sample" || k === "sample_id") {
            obj[k] = v;
          } else {
            obj[`metrics_${k}`] = v;
          }
        }
        metricsBy.set(key, obj);
      }

      // Group wave rows by track and aggregate numerics
      const groups = new Map<string, Row[]>();
      for (const r of wavesRows) {
        const tid = r.track_id ?? r.track ?? r.id;
        if (tid == null) continue;
        const key = String(tid);
        const arr = groups.get(key) || [];
        arr.push(r);
        groups.set(key, arr);
      }

      // Merge metrics + wave aggregates into one flat object per track
      for (const [key, rows] of groups) {
        const agg = aggregateNumeric(rows);
        const baseRow = metricsBy.get(key) || {};
        byId[key] = { ...baseRow, ...agg };
      }
      for (const [key, mrow] of metricsBy) {
        if (!byId[key]) byId[key] = { ...mrow, waves_count: 0 };
      }

      if (!cancelled) setRowByTrackId(byId);
    })();

    return () => { cancelled = true; };
  }, [runId]);

  return { rowByTrackId };
}

export default useWaveMetrics;
