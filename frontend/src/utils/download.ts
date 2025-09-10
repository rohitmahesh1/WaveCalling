/** Internal: trigger a browser download for a Blob. */
function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Internal: trigger a browser download for a data URL. */
function saveDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Download arbitrary text. */
export function downloadText(filename: string, text: string, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  saveBlob(blob, filename);
}

/** Download an object as pretty JSON. */
export function downloadJSON(filename: string, data: unknown) {
  const text = JSON.stringify(data, null, 2);
  downloadText(filename, text, "application/json;charset=utf-8");
}

/** Escape one CSV cell per RFC 4180-ish rules. */
function csvEscape(val: unknown): string {
  if (val === null || val === undefined) return "";
  let s = String(val);
  // Normalize line breaks
  s = s.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const mustQuote = /[",\n]/.test(s) || /^\s|\s$/.test(s);
  if (mustQuote) {
    s = '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

/**
 * Download rows as CSV.
 * - Automatically infers the union of keys across rows for the header order.
 * - Nested objects/arrays are stringified as JSON for safety.
 */
export function downloadCSV(filename: string, rows: Record<string, any>[], delimiter = ",") {
  const flat = rows.map((r) => {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(r)) {
      if (v && typeof v === "object") {
        out[k] = JSON.stringify(v);
      } else {
        out[k] = v as any;
      }
    }
    return out;
  });

  const headerSet = new Set<string>();
  for (const r of flat) for (const k of Object.keys(r)) headerSet.add(k);
  const headers = Array.from(headerSet);

  const lines: string[] = [];
  lines.push(headers.map((h) => csvEscape(h)).join(delimiter));
  for (const r of flat) {
    lines.push(headers.map((h) => csvEscape(r[h])).join(delimiter));
  }

  downloadText(filename, lines.join("\n"), "text/csv;charset=utf-8");
}

/**
 * Download a PNG snapshot of a canvas element.
 * Uses toBlob when available (preferable), falling back to toDataURL.
 */
export function downloadPNGFromCanvas(canvas: HTMLCanvasElement, filename = "canvas.png") {
  if (canvas.toBlob) {
    canvas.toBlob((blob) => {
      if (!blob) return;
      saveBlob(blob, filename);
    }, "image/png");
  } else {
    const dataUrl = canvas.toDataURL("image/png");
    saveDataUrl(dataUrl, filename);
  }
}
