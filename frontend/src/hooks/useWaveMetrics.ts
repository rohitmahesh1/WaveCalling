// frontend/src/hooks/useWaveMetrics.ts
import * as React from "react";
import type { ArtifactMap } from "@/utils/types";

type Row = Record<string, any>;
type RowByTrackId = Record<string, Record<string, any>>;

function normalizeKey(s: string): string {
  return s
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
}

function parseCSV(text: string): Row[] {
  const rows: string[][] = [];
  let cur: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else { inQuotes = false; }
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { cur.push(field); field = ""; }
      else if (c === "\n") { cur.push(field); rows.push(cur); cur = []; field = ""; }
      else if (c !== "\r") field += c;
    }
  }
  if (field.length || cur.length) { cur.push(field); rows.push(cur); }
  if (!rows.length) return [];

  const headers = rows.shift()!.map((h) => normalizeKey(String(h || "")));
  const out: Row[] = [];
  for (const r of rows) {
    if (!r.length || r.every((v) => String(v).trim() === "")) continue;
    const obj: Row = {};
    for (let k = 0; k < headers.length; k++) {
      const key = headers[k] || `col_${k}`;
      const raw = r[k] ?? "";
      const num = Number(raw);
      obj[key] = raw === "" ? null : (Number.isFinite(num) ? num : String(raw));
    }
    out.push(obj);
  }
  return out;
}

/** Fetch a CSV and return parsed rows. `null` means “not found / optional”. */
async function fetchCsvRows(url: string): Promise<Row[] | null> {
  try {
    const resp = await fetch(url, { cache: "no-cache" });
    if (!resp.ok) {
      if (resp.status === 404) {
        if (import.meta.env.MODE !== "production") {
          console.log("[waves.csv] optional – 404", url);
        }
        return null;
      }
      throw new Error(`CSV fetch failed: ${resp.status} ${url}`);
    }
    const text = await resp.text();
    if (!text || !text.trim()) return [];
    return parseCSV(text);
  } catch (err) {
    // Network or CORS issue — treat as missing but log once in dev
    if (import.meta.env.MODE !== "production") {
      console.warn("[waves.csv] fetch error (treated as optional)", url, err);
    }
    return null;
  }
}

function aggregateNumeric(rows: Row[]) {
  const out: Row = { waves_count: rows.length };
  const numericKeys = new Set<string>();
  for (const r of rows) {
    for (const [k, v] of Object.entries(r)) {
      if (v != null && typeof v === "number" && Number.isFinite(v)) {
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

/**
 * Read per-track metrics & per-wave metrics (aggregated) for a run.
 * It ONLY fetches URLs present in the run’s `artifacts` map to avoid 404 spam.
 * If you want to fall back to guessed filenames in /runs/:id/output, pass opts.allowGuesses: true.
 */
export function useWaveMetrics(
  runId?: string | null,
  artifactUrls?: ArtifactMap | null,
  opts?: { allowGuesses?: boolean }
) {
  const allowGuesses = !!opts?.allowGuesses;
  const [rowByTrackId, setRowByTrackId] = React.useState<RowByTrackId>({});

  // Stable dep for artifacts without re-render storms
  const artifactsKey = React.useMemo(
    () => JSON.stringify(artifactUrls ?? {}),
    [artifactUrls]
  );

  React.useEffect(() => {
    let cancelled = false;
    if (!runId) { setRowByTrackId({}); return; }

    (async () => {
      const wavesCandidates: string[] = [];
      const metricsCandidates: string[] = [];

      const pushIf = (arr: string[], maybeUrl?: string) => {
        if (typeof maybeUrl === "string" && maybeUrl.length) arr.push(maybeUrl);
      };

      if (artifactUrls) {
        // Common keys we’ve seen; keep both canonical & legacy-ish names
        pushIf(wavesCandidates, artifactUrls["waves_csv"]);
        pushIf(wavesCandidates, (artifactUrls as any)["metrics_waves_csv"]);
        pushIf(metricsCandidates, artifactUrls["metrics_csv"]);
        pushIf(metricsCandidates, (artifactUrls as any)["metrics_partial_csv"]);
      }

      if (allowGuesses && (!wavesCandidates.length || !metricsCandidates.length)) {
        const base = `/runs/${encodeURIComponent(runId)}/output`;
        if (!wavesCandidates.length) {
          wavesCandidates.push(
            `${base}/metrics_waves.partial.csv`,
            `${base}/metrics_waves_partial.csv`,
            `${base}/metrics_waves.csv`,
            `${base}/waves.csv`,
          );
        }
        if (!metricsCandidates.length) {
          metricsCandidates.push(
            `${base}/metrics.partial.csv`,
            `${base}/metrics.csv`,
          );
        }
      }

      // Try candidates in order; first one that returns rows (null = not found)
      let wavesRows: Row[] = [];
      let metricsRows: Row[] = [];

      for (const u of wavesCandidates) {
        const rows = await fetchCsvRows(u);
        if (rows !== null) { wavesRows = rows; break; }
        if (cancelled) return;
      }
      for (const u of metricsCandidates) {
        const rows = await fetchCsvRows(u);
        if (rows !== null) { metricsRows = rows; break; }
        if (cancelled) return;
      }

      const byId: RowByTrackId = {};

      // Index metrics by track id
      const metricsBy = new Map<string, Row>();
      for (const r of metricsRows) {
        const tid = r.track_id ?? r.track ?? r.id;
        if (tid == null) continue;
        const key = String(tid);
        const obj: Row = {};
        for (const [k, v] of Object.entries(r)) {
          if (k === "track_id" || k === "track" || k === "sample" || k === "sample_id") obj[k] = v;
          else obj[`metrics_${k}`] = v;
        }
        metricsBy.set(key, obj);
      }

      // Group waves by track id
      const groups = new Map<string, Row[]>();
      for (const r of wavesRows) {
        const tid = r.track_id ?? r.track ?? r.id;
        if (tid == null) continue;
        const key = String(tid);
        const arr = groups.get(key) || [];
        arr.push(r);
        groups.set(key, arr);
      }

      // Merge groups + metrics
      for (const [key, rows] of groups) {
        const agg = aggregateNumeric(rows);
        const baseRow = metricsBy.get(key) || {};
        byId[key] = { ...baseRow, ...agg };
      }
      // Add pure-metrics-only tracks
      for (const [key, mrow] of metricsBy) {
        if (!byId[key]) byId[key] = { ...mrow, waves_count: 0 };
      }

      if (!cancelled) setRowByTrackId(byId);
    })();

    return () => { cancelled = true; };
  }, [runId, artifactsKey, allowGuesses]);

  return { rowByTrackId };
}

export default useWaveMetrics;
