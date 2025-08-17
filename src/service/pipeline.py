# src/service/pipeline.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np
import pandas as pd

# Reuse your existing modules
from ..utils import (
    load_config,
    ensure_dir,
    list_files,
    save_dataframe,
    setup_logging,
    get_logger,
)
from ..io.table_to_heatmap import table_to_heatmap
from ..signal.detrend import detrend_residual
from ..signal.peaks import detect_peaks
from ..signal.period import estimate_dominant_frequency, frequency_to_period

# We’ll reuse the backend selector in extract.py to avoid duplication
from ..extract import _build_kymo_runner as _select_kymo_backend


# -----------------------
# Event / Result datatypes
# -----------------------

@dataclass
class JobEvent:
    phase: str            # "INIT" | "DISCOVER" | "TABLE2HEATMAP" | "KYMO" | "PROCESS" | "OVERLAY" | "WRITE" | "DONE" | "ERROR"
    message: str
    progress: float       # 0..1 (best effort)
    extra: Optional[dict] = None


@dataclass
class RunArtifacts:
    out_dir: Path
    plots_dir: Optional[Path]
    tracks_csv: Path
    waves_csv: Path
    overlay_json: Path
    manifest_json: Path


@dataclass
class RunResult:
    ok: bool
    error: Optional[str]
    artifacts: Optional[RunArtifacts]
    num_images: int
    num_tracks: int
    num_waves: int


# -----------------------
# Helpers
# -----------------------

def _sample_name_from_arr_path(arr_path: Path) -> str:
    """
    Heuristic: .../<SAMPLE>_heatmap/kymobutler_output/NN.npy -> SAMPLE
    If no '_heatmap' suffix, use the directory name above kymobutler_output.
    """
    base = arr_path.parent.parent.name
    if base.endswith("_heatmap"):
        return base[:-8]
    return base


def _is_overlay(p: Path) -> bool:
    s = str(p)
    return s.endswith("_overlay.png") or s.endswith("_overlay.jpg") or "_output/" in s or s.endswith("_output")


def _emit(cb: Optional[Callable[[JobEvent], None]], evt: JobEvent) -> None:
    if cb:
        try:
            cb(evt)
        except Exception:
            # don't let UI callback errors kill the job
            pass


def _sort_waves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Track"] = pd.to_numeric(df["Track"], errors="coerce").astype("Int64")
    return df.sort_values(
        ["Track", "Wave number", "Frame position 1"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_overlay_json(
    arr_paths: List[Path],
    cfg: dict,
    save_path: Path,
    *,
    log=None,
) -> Path:
    """
    Produce a compact overlay JSON for the frontend viewer.

    For each track (.npy):
      - poly: [[x,y], ...]  (raw coordinates as stored)
      - peaks: [int, ...]   (indices of peaks found in residual)
      - peak_amplitudes: [float, ...]  (residual height at each peak)
      - metrics: {dominant_frequency, period, num_peaks, mean_amplitude}
      - sample: heuristic sample name
      - id: track filename stem (e.g., "102")
    """
    detrend_cfg = cfg.get("detrend", {}) or {}
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    io_cfg = cfg.get("io", {}) or {}
    sampling_rate = io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    payload = {
        "version": 1,
        "tracks": [],
    }

    for arr in arr_paths:
        try:
            xy = np.load(arr)
            x = xy[:, 0].astype(float)
            y = xy[:, 1].astype(float)

            residual = detrend_residual(x, y, **detrend_cfg)
            peaks_idx, _ = detect_peaks(residual, **peaks_cfg)
            freq = float(estimate_dominant_frequency(residual, **period_cfg))
            period = float(frequency_to_period(freq))

            amps = residual[peaks_idx] if len(peaks_idx) > 0 else np.array([])
            metrics = {
                "dominant_frequency": freq,
                "period": period,
                "num_peaks": int(len(peaks_idx)),
                "mean_amplitude": float(np.mean(amps)) if amps.size > 0 else float("nan"),
            }

            payload["tracks"].append({
                "id": Path(arr).stem,
                "sample": _sample_name_from_arr_path(arr),
                "poly": xy.astype(float).tolist(),
                "peaks": [int(i) for i in peaks_idx.tolist()],
                "peak_amplitudes": ([float(a) for a in amps.tolist()] if amps.size else []),
                "metrics": metrics,
            })
        except Exception as e:
            if log:
                log.debug(f"overlay: failed on {arr.name}: {e}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # keep compact (no indent) to avoid bloating large projects
    with open(save_path, "w") as f:
        json.dump(payload, f)
    return save_path


# -----------------------
# Core run (generator)
# -----------------------

def iter_run_project(
    *,
    input_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    plots_out: str | Path | None = None,
    progress_cb: Optional[Callable[[JobEvent], None]] = None,
    config_overrides: Optional[dict] = None,
    verbose: bool = False,
) -> Iterator[JobEvent]:
    """
    Generator that runs the full pipeline and yields JobEvent at each step.
    Suitable for wiring to a WebSocket.

    Yields:
        JobEvent(...)
    """
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    plots_dir = ensure_dir(plots_out) if plots_out else None

    # Load/merge config
    cfg = load_config(config_path)
    if config_overrides:
        # shallow merge for simplicity; callers can provide a pre-merged dict if needed
        cfg = {**cfg, **config_overrides}

    log_level = cfg.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)
    log = get_logger("pipeline")
    if verbose:
        log.setLevel("DEBUG")

    # pick kymo backend
    run_kymo, run_kwargs, backend = _select_kymo_backend(cfg, log, verbose)
    yield JobEvent("INIT", f"Using KymoButler backend: {backend}", 0.01)

    # discover inputs
    io_cfg = cfg.get("io", {}) or {}
    image_globs = io_cfg.get("image_globs", ["*.png", "*.jpg", "*.jpeg"])
    track_glob = io_cfg.get("track_glob", "*.npy")
    table_globs = io_cfg.get("table_globs", ["*.csv", "*.tsv", "*.xls", "*.xlsx"])
    dpi = int((cfg.get("viz", {}) or {}).get("dpi", 180))

    yield JobEvent("DISCOVER", "Scanning input directory", 0.02, {"input_dir": str(input_dir)})

    # prefer existing .npy tracks
    npy_files = list(input_dir.rglob(track_glob))
    arr_paths: List[Path] = []
    image_count: int = 0  # for manifest

    if npy_files:
        arr_paths = sorted(map(Path, npy_files))
        # best-effort inference: count unique "<sample>_heatmap" bases
        bases = {p.parent.parent for p in arr_paths}
        image_count = len(bases)
        yield JobEvent("DISCOVER", f"Found {len(arr_paths)} track arrays (.npy); skipping KymoButler", 0.05)
    else:
        # gather images
        img_files = list_files(input_dir, image_globs)
        # tables → heatmaps
        table_files = list_files(input_dir, table_globs)

        generated_imgs: List[Path] = []
        if table_files:
            hm_cfg = cfg.get("heatmap", {}) or {}
            hm_lower = float(hm_cfg.get("lower", -1e20))
            hm_upper = float(hm_cfg.get("upper", 1e16))
            hm_binarize = bool(hm_cfg.get("binarize", True))
            hm_origin = str(hm_cfg.get("origin", "lower"))
            hm_cmap = str(hm_cfg.get("cmap", "hot"))

            out_dir = ensure_dir(input_dir / "generated_heatmaps")
            yield JobEvent("TABLE2HEATMAP", f"Converting {len(table_files)} table(s) → {out_dir.name}", 0.08)

            for k, tbl in enumerate(table_files, start=1):
                try:
                    out_img, _, _ = table_to_heatmap(
                        tbl,
                        out_dir=out_dir,
                        lower=hm_lower,
                        upper=hm_upper,
                        binarize=hm_binarize,
                        origin=hm_origin,
                        cmap=hm_cmap,
                        dpi=dpi,
                    )
                    generated_imgs.append(out_img)
                except Exception as e:
                    log.exception(f"Heatmap generation failed for {tbl}: {e}")

                yield JobEvent(
                    "TABLE2HEATMAP",
                    f"Converted {k}/{len(table_files)}",
                    0.08 + 0.10 * (k / max(1, len(table_files))),
                )

        all_images = list(map(Path, img_files)) + generated_imgs
        # filter visualization outputs
        all_images = [p for p in all_images if not _is_overlay(p)]
        image_count = len(all_images)

        if not all_images:
            msg = "No images, tables, or track arrays found under input directory"
            yield JobEvent("ERROR", msg, 1.0)
            return

        yield JobEvent("KYMO", f"Running KymoButler on {len(all_images)} image(s)", 0.20, {"backend": backend})

        produced = 0
        for idx, img in enumerate(all_images, start=1):
            base_dir = run_kymo(str(img), **run_kwargs)
            out_dir = base_dir / "kymobutler_output"
            found = sorted(out_dir.glob(track_glob))
            arr_paths.extend(found)
            produced += len(found)

            yield JobEvent(
                "KYMO",
                f"{img.name}: {len(found)} track(s)",
                0.20 + 0.40 * (idx / max(1, len(all_images))),
                {"total_tracks": produced},
            )

    if not arr_paths:
        yield JobEvent("ERROR", "No track arrays (.npy) found after processing", 1.0)
        return

    # analysis (tracks + waves) and plots
    from ..extract import process_track as _process_track  # reuse your function

    viz_cfg = cfg.get("viz", {}) or {}
    viz_enabled = bool(viz_cfg.get("enabled", True))
    per_track_cfg = viz_cfg.get("per_track", {}) or {}
    make_plot_detrended = bool(per_track_cfg.get("detrended_with_peaks", True))
    make_plot_spectrum = bool(per_track_cfg.get("spectrum", True))
    summary_cfg = viz_cfg.get("summary", {}) or {}
    make_summary_hists = bool(summary_cfg.get("histograms", True))
    hist_bins = int(viz_cfg.get("hist_bins", 20))

    # configs for signal processing
    detrend_cfg = cfg.get("detrend", {}) or {}
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    sampling_rate = (cfg.get("io", {}) or {}).get("sampling_rate", period_cfg.get("sampling_rate", 1.0))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    tracks_out = ensure_dir(output_dir)
    plots_dir = ensure_dir(plots_dir) if (viz_enabled and plots_dir) else None

    track_results: List[dict] = []
    wave_results: List[dict] = []

    yield JobEvent("PROCESS", f"Processing {len(arr_paths)} track(s)", 0.60)

    for i, arr in enumerate(arr_paths, start=1):
        try:
            metrics, waves = _process_track(
                arr,
                detrend_cfg,
                peaks_cfg,
                period_cfg,
                plots_dir=plots_dir,
                sampling_rate=sampling_rate,
                make_plot_detrended=make_plot_detrended,
                make_plot_spectrum=make_plot_spectrum,
                dpi=int(viz_cfg.get("dpi", 180)),
                log=get_logger("pipeline.process"),
            )
            track_results.append(metrics)
            wave_results.extend(waves)

        except Exception as e:
            get_logger("pipeline.process").exception(f"Failed on {arr}: {e}")

        yield JobEvent(
            "PROCESS",
            f"Processed {i}/{len(arr_paths)}",
            0.60 + 0.25 * (i / max(1, len(arr_paths))),
        )

    # write CSVs (+ sort waves)
    tracks_csv = tracks_out / "metrics.csv"
    waves_csv = tracks_out / "metrics_waves.csv"

    df_tracks = pd.DataFrame(track_results)
    save_dataframe(df_tracks, tracks_csv)

    df_waves = pd.DataFrame(wave_results)
    if not df_waves.empty:
        df_waves = _sort_waves(df_waves)
        df_waves.to_csv(waves_csv, index=False, na_rep="NA")
    else:
        # still create an empty file with headers (use current schema)
        df_waves = pd.DataFrame(
            columns=[
                "Sample", "Track", "Wave number", "Frame position 1", "Frame position 2",
                "Period (frames)", "Period (s)", "Frequency (Hz)", "Pixel position 1",
                "Pixel position 2", "Amplitude (pixels)", "Δposition (px)",
                "Velocity (px/s)", "Wavelength (px)"
            ]
        )
        df_waves.to_csv(waves_csv, index=False)

    # overlay JSON for the viewer
    yield JobEvent("OVERLAY", "Building overlay JSON", 0.88)
    overlay_json = tracks_out / "overlay" / "tracks.json"
    _build_overlay_json(arr_paths, cfg, overlay_json, log=get_logger("pipeline.overlay"))

    # summary histograms if requested
    if plots_dir is not None and make_summary_hists:
        from ..visualize import plot_summary_histograms as _plots
        _plots(df_tracks, plots_dir, bins=hist_bins, dpi=int(viz_cfg.get("dpi", 180)))

    # small manifest for reproducibility
    manifest = {
        "backend": backend,
        "num_images": int(image_count),
        "num_tracks": int(len(df_tracks)),
        "num_waves": int(len(df_waves)),
        "config_path": str(Path(config_path)),
        "output_dir": str(tracks_out),
        "plots_dir": str(plots_dir) if plots_dir else None,
    }
    manifest_json = tracks_out / "manifest.json"
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)

    yield JobEvent("WRITE", "Wrote outputs", 0.96, {
        "tracks_csv": str(tracks_csv),
        "waves_csv": str(waves_csv),
        "overlay_json": str(overlay_json),
        "manifest": str(manifest_json),
    })
    yield JobEvent("DONE", "Pipeline complete", 1.0)


# -----------------------
# Convenience runner (non-generator)
# -----------------------

def run_project(
    *,
    input_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    plots_out: str | Path | None = None,
    config_overrides: Optional[dict] = None,
    progress_cb: Optional[Callable[[JobEvent], None]] = None,
    verbose: bool = False,
) -> RunResult:
    """
    Fire-and-forget wrapper that runs the generator and aggregates results.
    Returns a RunResult with artifact paths. Emits events to progress_cb if provided.
    """
    tracks_csv = Path(output_dir) / "metrics.csv"
    waves_csv = Path(output_dir) / "metrics_waves.csv"
    overlay_json = Path(output_dir) / "overlay" / "tracks.json"
    manifest_json = Path(output_dir) / "manifest.json"

    num_tracks = 0
    num_waves = 0
    num_images = 0
    last_evt: Optional[JobEvent] = None
    error: Optional[str] = None

    try:
        for evt in iter_run_project(
            input_dir=input_dir,
            config_path=config_path,
            output_dir=output_dir,
            plots_out=plots_out,
            progress_cb=progress_cb,
            config_overrides=config_overrides,
            verbose=verbose,
        ):
            last_evt = evt
            _emit(progress_cb, evt)

            if evt.phase == "ERROR":
                error = evt.message
                break

        # Try to read counts from manifest
        if Path(manifest_json).exists():
            with open(manifest_json) as f:
                m = json.load(f)
            num_tracks = int(m.get("num_tracks", 0))
            num_waves = int(m.get("num_waves", 0))
            num_images = int(m.get("num_images", 0))

        ok = (error is None) and (last_evt is not None and last_evt.phase == "DONE")
        return RunResult(
            ok=ok,
            error=error,
            artifacts=RunArtifacts(
                out_dir=Path(output_dir),
                plots_dir=Path(plots_out) if plots_out else None,
                tracks_csv=tracks_csv,
                waves_csv=waves_csv,
                overlay_json=overlay_json,
                manifest_json=manifest_json,
            ) if ok else None,
            num_images=num_images,
            num_tracks=num_tracks,
            num_waves=num_waves,
        )
    except Exception as e:
        return RunResult(
            ok=False,
            error=str(e),
            artifacts=None,
            num_images=0,
            num_tracks=0,
            num_waves=0,
        )
