// utils/configSchema.ts
import { deepGet } from "./configDiff";

export type FieldType = "toggle" | "number" | "select" | "text" | "textarea" | "code";

export type VisibleIf =
  | { path: string; equals: any }
  | { path: string; in: any[] };

export type FieldSpec = {
  path: string;
  label: string;
  description?: string;
  docsUrl?: string;
  type: FieldType;
  default?: any;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: { label: string; value: string | number }[];
  rows?: number;
  /** transform UI -> config */
  parse?: (value: any) => any;
  /** transform config -> UI */
  format?: (value: any) => any;
  /** simple, declarative visibility */
  visibleIf?: VisibleIf | VisibleIf[];
  validate?: (value: any, root: any) => string | null | undefined;
};

export type GroupSpec = {
  id: string;
  title?: string;
  description?: string;
  collapsedDefault?: boolean;
  fields: FieldSpec[];
};

export type SectionSpec = {
  id: string;
  title: string;
  description?: string;
  groups: GroupSpec[];
};

export function isFieldVisible(spec: FieldSpec, cfg: any): boolean {
  if (!spec.visibleIf) return true;
  const rules = Array.isArray(spec.visibleIf) ? spec.visibleIf : [spec.visibleIf];
  return rules.every((rule) => {
    const v = deepGet(cfg, rule.path);
    if ("equals" in rule) return v === (rule as any).equals;
    if ("in" in rule) return (rule as any).in.includes(v);
    return true;
  });
}

// Helpers for array <-> textarea round-tripping (comma/newline separated)
const parseList = (val: any) =>
  String(val ?? "")
    .split(/[\n,]+/g)
    .map((s) => s.trim())
    .filter(Boolean);

const formatList = (val: any) =>
  Array.isArray(val) ? val.join(", ") : String(val ?? "");

export const CONFIG_SECTIONS: SectionSpec[] = [
  {
    id: "logging",
    title: "Logging",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "logging.level",
            label: "Level",
            type: "select",
            description: "Verbosity of logs produced by the pipeline and submodules.",
            options: [
              { value: "DEBUG", label: "DEBUG" },
              { value: "INFO", label: "INFO" },
              { value: "WARNING", label: "WARNING" },
              { value: "ERROR", label: "ERROR" },
            ],
            default: "INFO",
          },
        ],
      },
    ],
  },
  {
    id: "io",
    title: "I/O",
    description: "Input discovery and sampling settings.",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "io.sampling_rate",
            label: "Sampling rate",
            type: "number",
            unit: "Hz",
            description:
              "Global sampling rate (Hz). Used for period estimation unless overridden under Period.",
            min: 0,
            step: 0.001,
          },
          {
            path: "io.image_globs",
            label: "Image globs",
            type: "textarea",
            description: "Comma or newline separated patterns, e.g. *.png, *.jpg",
            parse: parseList,
            format: formatList,
            default: ["*.png", "*.jpg", "*.jpeg"],
          },
          {
            path: "io.table_globs",
            label: "Table globs",
            type: "textarea",
            description: "Comma or newline separated patterns for CSV/TSV/XLS/XLSX.",
            parse: parseList,
            format: formatList,
            default: ["*.csv", "*.tsv", "*.xls", "*.xlsx"],
          },
          {
            path: "io.track_glob",
            label: "Track glob",
            type: "text",
            description: "Pattern for precomputed track arrays, e.g. *.npy",
            default: "*.npy",
          },
        ],
      },
    ],
  },
  {
    id: "heatmap",
    title: "Heatmap",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "heatmap.lower",
            label: "Lower bound",
            type: "number",
            description: "Clamp lower intensity bound prior to binarization.",
            step: 0.1,
          },
          {
            path: "heatmap.upper",
            label: "Upper bound",
            type: "number",
            description: "Clamp upper intensity bound prior to binarization.",
            step: 0.1,
            validate: (_v, full) => {
              const lo = Number(deepGet(full, "heatmap.lower"));
              const hi = Number(deepGet(full, "heatmap.upper"));
              if (Number.isFinite(lo) && Number.isFinite(hi) && lo >= hi) {
                return "Upper must be greater than Lower";
              }
              return null;
            },
          },
          {
            path: "heatmap.binarize",
            label: "Binarize",
            type: "toggle",
            description: "Threshold the image to binary prior to kymograph extraction.",
            default: true,
          },
          {
            path: "heatmap.origin",
            label: "Origin",
            type: "select",
            options: [
              { value: "lower", label: "lower" },
              { value: "upper", label: "upper" },
            ],
            description: "Matplotlib-style origin for rendering.",
            default: "lower",
          },
          {
            path: "heatmap.cmap",
            label: "Colormap",
            type: "text",
            description: "Matplotlib colormap name (e.g. hot, viridis).",
            default: "hot",
          },
        ],
      },
    ],
  },
  {
    id: "detrend",
    title: "Detrend",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "detrend.degree",
            label: "Polynomial degree",
            type: "number",
            min: 0,
            max: 5,
            step: 1,
            description: "Polynomial degree for baseline removal.",
          },
          {
            path: "detrend.min_samples",
            label: "Min samples",
            type: "number",
            min: 0,
            max: 1,
            step: 0.01,
            description: "RANSAC min samples fraction (0..1).",
          },
        ],
      },
    ],
  },
  {
    id: "peaks",
    title: "Peaks",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "peaks.prominence",
            label: "Prominence",
            type: "number",
            step: 0.1,
            description: "Minimum peak prominence (pixels/units of detrended residual).",
          },
          {
            path: "peaks.distance",
            label: "Min distance",
            type: "number",
            step: 1,
            description: "Minimum peak spacing in samples.",
          },
          {
            path: "peaks.width",
            label: "Min width",
            type: "number",
            step: 1,
            description: "Minimum peak width in samples.",
          },
        ],
      },
    ],
  },
  {
    id: "period",
    title: "Period",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "period.sampling_rate",
            label: "Sampling rate (override)",
            type: "number",
            unit: "Hz",
            description: "Override IO sampling rate just for period estimation.",
            min: 0,
            step: 0.001,
          },
          {
            path: "period.min_freq",
            label: "Min frequency",
            type: "number",
            unit: "Hz",
            step: 0.001,
            description: "Low end of search range.",
          },
          {
            path: "period.max_freq",
            label: "Max frequency",
            type: "number",
            unit: "Hz",
            step: 0.001,
            description: "High end of search range.",
            validate: (_v, full) => {
              const lo = Number(deepGet(full, "period.min_freq"));
              const hi = Number(deepGet(full, "period.max_freq"));
              if (Number.isFinite(lo) && Number.isFinite(hi) && lo >= hi) {
                return "Max frequency must be greater than Min frequency";
              }
              return null;
            },
          },
        ],
      },
    ],
  },
  {
    id: "viz",
    title: "Visualization",
    groups: [
      {
        id: "general",
        title: "General",
        fields: [
          { path: "viz.enabled", label: "Enable plots", type: "toggle", default: true },
          {
            path: "viz.hist_bins",
            label: "Histogram bins",
            type: "number",
            min: 1,
            step: 1,
            description: "Number of bins for summary histograms.",
            default: 20,
          },
          {
            path: "viz.dpi",
            label: "DPI",
            type: "number",
            min: 50,
            max: 600,
            step: 10,
            default: 180,
          },
        ],
      },
      {
        id: "per_track",
        title: "Per-track plots",
        fields: [
          {
            path: "viz.per_track.detrended_with_peaks",
            label: "Detrended + Peaks",
            type: "toggle",
            default: true,
          },
          {
            path: "viz.per_track.spectrum",
            label: "Spectrum",
            type: "toggle",
            default: true,
          },
        ],
      },
      {
        id: "summary",
        title: "Summary",
        fields: [
          {
            path: "viz.summary.histograms",
            label: "Write summary histograms",
            type: "toggle",
            default: true,
          },
        ],
      },
      {
        id: "wave_windows",
        title: "Wave windows",
        fields: [
          {
            path: "viz.wave_windows.save",
            label: "Save wave windows",
            type: "toggle",
            default: true,
          },
          {
            path: "viz.wave_windows.max_per_track",
            label: "Max per track",
            type: "number",
            min: 0,
            step: 1,
            default: 50,
            visibleIf: { path: "viz.wave_windows.save", equals: true },
          },
          {
            path: "viz.wave_windows.stride",
            label: "Stride",
            type: "number",
            min: 1,
            step: 1,
            default: 1,
            visibleIf: { path: "viz.wave_windows.save", equals: true },
          },
        ],
      },
    ],
  },
  {
    id: "kymo",
    title: "Kymograph Extraction",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "kymo.backend",
            label: "Backend",
            type: "select",
            options: [
              { value: "onnx", label: "ONNX" },
              { value: "wolfram", label: "Wolfram" },
            ],
            description: "Select the kymograph detection backend.",
            default: "onnx",
          },
        ],
      },
      {
        id: "onnx",
        title: "ONNX settings",
        fields: [
          {
            path: "kymo.onnx.seg_size",
            label: "Segmentation size",
            type: "number",
            min: 32,
            step: 1,
            default: 256,
            visibleIf: { path: "kymo.backend", equals: "onnx" },
          },
          {
            path: "kymo.onnx.thresholds.thr_bi",
            label: "Binary threshold",
            type: "number",
            min: 0,
            max: 1,
            step: 0.01,
            default: 0.17,
            visibleIf: { path: "kymo.backend", equals: "onnx" },
          },
          {
            path: "kymo.onnx.debug.enabled",
            label: "Debug output",
            type: "toggle",
            default: false,
            visibleIf: { path: "kymo.backend", equals: "onnx" },
          },
        ],
      },
      {
        id: "wolfram",
        title: "Wolfram settings",
        fields: [
          {
            path: "kymo.wolfram.scripts_dir",
            label: "Scripts directory",
            type: "text",
            visibleIf: { path: "kymo.backend", equals: "wolfram" },
          },
        ],
      },
    ],
  },
  {
    id: "service",
    title: "Service",
    description: "Runtime behavior for partial writes and resume.",
    groups: [
      {
        id: "general",
        fields: [
          {
            path: "service.partial_every_tracks",
            label: "Partial write frequency",
            type: "number",
            min: 1,
            step: 1,
            description: "How many tracks between partial CSV/overlay emissions.",
            default: 250,
          },
          {
            path: "service.write_progress_every_secs",
            label: "Progress write cadence",
            type: "number",
            min: 1,
            step: 1,
            unit: "s",
            description: "How often to update progress.json.",
            default: 10,
          },
          {
            path: "service.resume.enabled",
            label: "Enable resume",
            type: "toggle",
            default: true,
          },
          {
            path: "service.resume.marker_dir",
            label: "Marker directory",
            type: "text",
            default: "output/processed",
            visibleIf: { path: "service.resume.enabled", equals: true },
          },
          {
            path: "service.resume.progress_file",
            label: "Progress file",
            type: "text",
            default: "progress.json",
            visibleIf: { path: "service.resume.enabled", equals: true },
          },
        ],
      },
    ],
  },
];
