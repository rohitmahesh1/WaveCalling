// components/config/ConfigSections.tsx
import * as React from "react";

import ConfigSectionCard from "@/components/config/ConfigSectionCard";
import CollapsibleGroup from "@/components/config/CollapsibleGroup";
import FieldRow from "@/components/config/FieldRow";

import ToggleField from "@/components/config/fields/ToggleField";
import NumberField from "@/components/config/fields/NumberField";
import SelectField from "@/components/config/fields/SelectField";
import TextField from "@/components/config/fields/TextField";
import TextareaField from "@/components/config/fields/TextareaField";
import CodeField from "@/components/config/fields/CodeField";

// -----------------------------
// Helpers to read values safely
// -----------------------------
function getAtPath(obj: any, path: string, fallback?: any) {
  try {
    return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj) ?? fallback;
  } catch {
    return fallback;
  }
}

function parseCommaList(s: string): string[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

type Props = {
  /** Current merged config (base + overrides applied). */
  values: any;
  /** Called when a leaf value changes. You only need to write the override at this path. */
  onChange: (path: string, value: any) => void;
  /** Optional per-path error map (path -> message). */
  errors?: Record<string, string | undefined>;
  className?: string;
};

export default function ConfigSections({ values, onChange, errors, className }: Props) {
  // Small shorthands to read values
  const v = (path: string, fb?: any) => getAtPath(values, path, fb);
  const err = (path: string) => (errors ? errors[path] : undefined);

  // Preset options
  const ORIGIN_OPTIONS = [
    { value: "lower", label: "lower (y↓)" },
    { value: "upper", label: "upper (y↑)" },
  ];
  const CMAP_OPTIONS = [
    "hot",
    "magma",
    "viridis",
    "plasma",
    "inferno",
    "gray",
    "cubehelix",
    "turbo",
  ].map((x) => ({ value: x, label: x }));

  const TIME_DIR_OPTIONS = [
    { value: "down", label: "time ↓ (image default)" },
    { value: "up", label: "time ↑ (invert Y)" },
  ];

  return (
    <div className={`flex flex-col gap-4 ${className || ""}`}>
      {/* IO */}
      <ConfigSectionCard
        title="I/O"
        subtitle="Where inputs come from and key parsing parameters."
      >
        <div className="grid gap-3">
          <FieldRow
            label="Image globs"
            description="File patterns to look for images used by the kymograph extractor."
            docsUrl=""
          >
            <TextField
              value={(v("io.image_globs", ["*.png", "*.jpg", "*.jpeg"]) as string[]).join(", ")}
              placeholder="*.png, *.jpg, *.jpeg"
              onChange={(s) => onChange("io.image_globs", parseCommaList(s))}
              mono
            />
          </FieldRow>

          <FieldRow
            label="Table globs"
            description="Tabular inputs converted to heatmaps before extraction."
          >
            <TextField
              value={(v("io.table_globs", ["*.csv", "*.tsv", "*.xls", "*.xlsx"]) as string[]).join(", ")}
              placeholder="*.csv, *.tsv, *.xls, *.xlsx"
              onChange={(s) => onChange("io.table_globs", parseCommaList(s))}
              mono
            />
          </FieldRow>

          <FieldRow
            label="Track glob"
            description="When .npy tracks already exist, the extractor phase is skipped."
          >
            <TextField
              value={String(v("io.track_glob", "*.npy"))}
              placeholder="*.npy"
              onChange={(s) => onChange("io.track_glob", s)}
              mono
            />
          </FieldRow>

          <FieldRow
            label="Sampling rate (Hz)"
            description="Frames per second for converting between frames and seconds."
            inline
          >
            <NumberField
              value={Number(v("io.sampling_rate", v("period.sampling_rate", 1.0)))}
              min={0}
              step={0.01}
              onChange={(num) => onChange("io.sampling_rate", num)}
            />
          </FieldRow>
        </div>
      </ConfigSectionCard>

      {/* Heatmap */}
      <ConfigSectionCard title="Heatmap" subtitle="Controls for converting tables → heatmap images.">
        <div className="grid gap-3">
          <FieldRow label="Lower bound" inline>
            <NumberField
              value={Number(v("heatmap.lower", -1e20))}
              step={0.1}
              onChange={(n) => onChange("heatmap.lower", n)}
            />
          </FieldRow>
          <FieldRow label="Upper bound" inline>
            <NumberField
              value={Number(v("heatmap.upper", 1e16))}
              step={0.1}
              onChange={(n) => onChange("heatmap.upper", n)}
            />
          </FieldRow>
          <FieldRow
            label="Binarize"
            description="If enabled, threshold the heatmap to a binary image prior to extraction."
            inline
          >
            <ToggleField
              checked={Boolean(v("heatmap.binarize", true))}
              onChange={(b) => onChange("heatmap.binarize", b)}
            />
          </FieldRow>
          <FieldRow label="Origin" inline>
            <SelectField
              value={String(v("heatmap.origin", "lower"))}
              onChange={(val) => onChange("heatmap.origin", val)}
              options={ORIGIN_OPTIONS}
            />
          </FieldRow>
          <FieldRow label="Colormap" inline>
            <SelectField
              value={String(v("heatmap.cmap", "hot"))}
              onChange={(val) => onChange("heatmap.cmap", val)}
              options={CMAP_OPTIONS}
            />
          </FieldRow>
        </div>
      </ConfigSectionCard>

      {/* Kymo / Extractor (advanced) */}
      <ConfigSectionCard
        title="Extractor"
        subtitle="KymoButler (ONNX) and thresholds; usually safe to keep defaults."
      >
        <CollapsibleGroup title="ONNX thresholds (advanced)" defaultOpen={false}>
          <FieldRow label="Thresholds JSON" description="Fine-grained overrides for extractor thresholds.">
            <CodeField
              language="json"
              value={JSON.stringify(
                getAtPath(values, "kymo.onnx.thresholds", { thr_bi: 0.18 }),
                null,
                2
              )}
              onChange={(code) => {
                try {
                  const obj = JSON.parse(code);
                  onChange("kymo.onnx.thresholds", obj);
                } catch {
                  // ignore until valid JSON
                }
              }}
              rows={8}
            />
          </FieldRow>
        </CollapsibleGroup>
      </ConfigSectionCard>

      {/* Detrend / Peaks / Period */}
      <ConfigSectionCard title="Signal processing" subtitle="Detrending, peak detection, and frequency estimation.">
        <div className="grid gap-4">
          <CollapsibleGroup title="Detrend" description="Remove slow drift before peak detect / spectrum.">
            <div className="grid gap-3">
              <FieldRow label="Method" inline>
                <TextField
                  value={String(v("detrend.method", "poly"))}
                  onChange={(s) => onChange("detrend.method", s)}
                  mono
                />
              </FieldRow>
              <FieldRow label="Order (poly)" inline>
                <NumberField
                  value={Number(v("detrend.order", 3))}
                  min={0}
                  step={1}
                  onChange={(n) => onChange("detrend.order", n)}
                />
              </FieldRow>
              <FieldRow label="Lambda (savgol / tv)" inline>
                <NumberField
                  value={Number(v("detrend.lam", 0))}
                  step={0.1}
                  onChange={(n) => onChange("detrend.lam", n)}
                />
              </FieldRow>
            </div>
          </CollapsibleGroup>

          <CollapsibleGroup title="Peaks" description="Parameters passed to peak detector.">
            <div className="grid gap-3">
              <FieldRow label="Prominence" inline>
                <NumberField
                  value={Number(v("peaks.prominence", 4))}
                  min={0}
                  step={0.1}
                  onChange={(n) => onChange("peaks.prominence", n)}
                />
              </FieldRow>
              <FieldRow label="Distance (min)" inline>
                <NumberField
                  value={Number(v("peaks.distance", 0))}
                  min={0}
                  step={1}
                  onChange={(n) => onChange("peaks.distance", n)}
                />
              </FieldRow>
              <FieldRow label="Width (min)" inline>
                <NumberField
                  value={Number(v("peaks.width", 0))}
                  min={0}
                  step={1}
                  onChange={(n) => onChange("peaks.width", n)}
                />
              </FieldRow>
              <FieldRow label="Height (min)" inline>
                <NumberField
                  value={Number(v("peaks.height", 0))}
                  min={0}
                  step={0.1}
                  onChange={(n) => onChange("peaks.height", n)}
                />
              </FieldRow>
            </div>
          </CollapsibleGroup>

          <CollapsibleGroup title="Period / Spectrum" description="Dominant frequency detection settings.">
            <div className="grid gap-3">
              <FieldRow label="Sampling rate (Hz)" inline>
                <NumberField
                  value={Number(v("period.sampling_rate", v("io.sampling_rate", 1.0)))}
                  min={0}
                  step={0.01}
                  onChange={(n) => onChange("period.sampling_rate", n)}
                />
              </FieldRow>
              <FieldRow label="Min frequency (Hz)" inline>
                <NumberField
                  value={Number(v("period.min_freq", 0))}
                  min={0}
                  step={0.01}
                  onChange={(n) => onChange("period.min_freq", n)}
                />
              </FieldRow>
              <FieldRow label="Max frequency (Hz)" inline>
                <NumberField
                  value={Number(v("period.max_freq", 0))}
                  min={0}
                  step={0.01}
                  onChange={(n) => onChange("period.max_freq", n)}
                />
              </FieldRow>
              <FieldRow label="NFFT (0 = auto)" inline>
                <NumberField
                  value={Number(v("period.nfft", 0))}
                  min={0}
                  step={1}
                  onChange={(n) => onChange("period.nfft", n)}
                />
              </FieldRow>
            </div>
          </CollapsibleGroup>
        </div>
      </ConfigSectionCard>

      {/* Visualization */}
      <ConfigSectionCard
        title="Visualization"
        subtitle="Per-track plots, spectra, and summary histograms."
      >
        <div className="grid gap-3">
          <FieldRow label="Enable visualization" inline>
            <ToggleField
              checked={Boolean(v("viz.enabled", true))}
              onChange={(b) => onChange("viz.enabled", b)}
            />
          </FieldRow>
          <FieldRow label="Histogram bins" inline>
            <NumberField
              value={Number(v("viz.hist_bins", 20))}
              min={1}
              step={1}
              onChange={(n) => onChange("viz.hist_bins", n)}
            />
          </FieldRow>

          <CollapsibleGroup
            title="Per-track plots"
            description="Save detrended-with-peaks and power spectrum per track."
          >
            <div className="grid gap-3">
              <FieldRow label="Detrended + peaks" inline>
                <ToggleField
                  checked={Boolean(v("viz.per_track.detrended_with_peaks", true))}
                  onChange={(b) => onChange("viz.per_track.detrended_with_peaks", b)}
                />
              </FieldRow>
              <FieldRow label="Spectrum" inline>
                <ToggleField
                  checked={Boolean(v("viz.per_track.spectrum", true))}
                  onChange={(b) => onChange("viz.per_track.spectrum", b)}
                />
              </FieldRow>
            </div>
          </CollapsibleGroup>

          <CollapsibleGroup title="Summary output" description="Aggregate histogram plots.">
            <FieldRow label="Histograms" inline>
              <ToggleField
                checked={Boolean(v("viz.summary.histograms", true))}
                onChange={(b) => onChange("viz.summary.histograms", b)}
              />
            </FieldRow>
          </CollapsibleGroup>

          <CollapsibleGroup
            title="Wave windows (viewer detail)"
            description="Save small PNG windows around detected peaks for quick inspection."
            defaultOpen={false}
          >
            <div className="grid gap-3">
              <FieldRow label="Save wave windows" inline>
                <ToggleField
                  checked={Boolean(v("viz.wave_windows.save", true))}
                  onChange={(b) => onChange("viz.wave_windows.save", b)}
                />
              </FieldRow>
              <FieldRow label="Max windows per track" inline>
                <NumberField
                  value={Number(v("viz.wave_windows.max_per_track", 50))}
                  min={0}
                  step={1}
                  onChange={(n) => onChange("viz.wave_windows.max_per_track", n)}
                />
              </FieldRow>
              <FieldRow label="Stride" inline>
                <NumberField
                  value={Number(v("viz.wave_windows.stride", 1))}
                  min={1}
                  step={1}
                  onChange={(n) => onChange("viz.wave_windows.stride", n)}
                />
              </FieldRow>
            </div>
          </CollapsibleGroup>
        </div>
      </ConfigSectionCard>

      {/* Service / Resume */}
      <ConfigSectionCard
        title="Service"
        subtitle="Pipeline runtime controls: partial write cadence, resume markers, and progress I/O."
      >
        <div className="grid gap-4">
          <FieldRow label="Partial write every N tracks" inline>
            <NumberField
              value={Number(v("service.partial_every_tracks", 250))}
              min={1}
              step={1}
              onChange={(n) => onChange("service.partial_every_tracks", n)}
            />
          </FieldRow>

          <FieldRow label="Write progress every (s)" inline>
            <NumberField
              value={Number(v("service.write_progress_every_secs", 10))}
              min={1}
              step={1}
              onChange={(n) => onChange("service.write_progress_every_secs", n)}
            />
          </FieldRow>

          <CollapsibleGroup title="Resume" description="Skip processed tracks and continue where you left off.">
            <div className="grid gap-3">
              <FieldRow label="Enable resume" inline>
                <ToggleField
                  checked={Boolean(v("service.resume.enabled", true))}
                  onChange={(b) => onChange("service.resume.enabled", b)}
                />
              </FieldRow>
              <FieldRow label="Marker dir (relative to output/)" inline>
                <TextField
                  value={String(v("service.resume.marker_dir", "output/processed"))}
                  onChange={(s) => onChange("service.resume.marker_dir", s)}
                  mono
                />
              </FieldRow>
              <FieldRow label="Progress file (relative to output/)" inline>
                <TextField
                  value={String(v("service.resume.progress_file", "progress.json"))}
                  onChange={(s) => onChange("service.resume.progress_file", s)}
                  mono
                />
              </FieldRow>
            </div>
          </CollapsibleGroup>
        </div>
      </ConfigSectionCard>
    </div>
  );
}
