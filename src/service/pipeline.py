# src/service/pipeline.py
from __future__ import annotations

import json
import shutil
import time                   # NEW
import inspect                # NEW
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Set, Dict  # NEW

import numpy as np
import pandas as pd

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

from ..extract import _build_kymo_runner as _select_kymo_backend
from ..extract import process_track as _process_track


# -----------------------
# Event / Result datatypes
# -----------------------

@dataclass
class JobEvent:
    phase: str  # "INIT" | "DISCOVER" | "TABLE2HEATMAP" | "KYMO" | "PROCESS" | "OVERLAY" | "WRITE" | "DONE" | "ERROR" | "WRITE_PARTIAL" | "CANCELLED"
    message: str
    progress: float  # 0..1
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
    detrend_cfg = cfg.get("detrend", {}) or {}
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    io_cfg = cfg.get("io", {}) or {}
    sampling_rate = io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    payload = {"version": 1, "tracks": []}

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
                "poly": xy.astype(float).tolist(),      # [[y,x],...]
                "peaks": [int(i) for i in peaks_idx.tolist()],
                "metrics": metrics,
            })
        except Exception as e:
            if log:
                log.debug(f"overlay: failed on {arr.name}: {e}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f)
    return save_path


# -----------------------
# Core run (generator) — supports cancellation + partial writes + resume (NEW)
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
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Iterator[JobEvent]:
    """
    Generator that runs the full pipeline and yields JobEvent at each step.
    Suitable for wiring to SSE. Callers can cancel via cancel_cb().
    """
    def _cancelled() -> bool:
        try:
            return bool(cancel_cb and cancel_cb())
        except Exception:
            return False

    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    plots_dir = ensure_dir(plots_out) if plots_out else None

    # Load/merge config
    cfg = load_config(config_path)
    if config_overrides:
        cfg = {**cfg, **config_overrides}

    # ---- new service controls ----
    service_cfg = (cfg.get("service") or {})
    partial_every = int(service_cfg.get("partial_every_tracks", 250))
    progress_every_secs = float(service_cfg.get("write_progress_every_secs", 10.0))
    resume_cfg = (service_cfg.get("resume") or {})
    resume_enabled = bool(resume_cfg.get("enabled", True))
    marker_dir_rel = Path(resume_cfg.get("marker_dir", "output/processed"))
    progress_file_rel = Path(resume_cfg.get("progress_file", "progress.json"))

    marker_dir = ensure_dir((Path(output_dir) / marker_dir_rel).resolve())
    progress_file = (Path(output_dir) / progress_file_rel).resolve()

    def _write_progress(total: int, processed: int, skipped: int) -> None:
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_file, "w") as f:
                json.dump(
                    {
                        "totalTracks": int(total),
                        "processedCount": int(processed),
                        "skippedCount": int(skipped),
                        "lastUpdatedAt": pd.Timestamp.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    log_level = cfg.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)
    log = get_logger("pipeline")
    if verbose:
        log.setLevel("DEBUG")

    # pick kymo backend
    run_kymo, run_kwargs, backend = _select_kymo_backend(cfg, log, verbose)
    evt = JobEvent("INIT", f"Using KymoButler backend: {backend}", 0.01)
    _emit(progress_cb, evt); yield evt
    if _cancelled():
        yield JobEvent("CANCELLED", "Run cancelled", 1.0)
        return

    # discover inputs
    io_cfg = cfg.get("io", {}) or {}
    image_globs = io_cfg.get("image_globs", ["*.png", "*.jpg", "*.jpeg"])
    track_glob = io_cfg.get("track_glob", "*.npy")
    table_globs = io_cfg.get("table_globs", ["*.csv", "*.tsv", "*.xls", "*.xlsx"])
    dpi = int((cfg.get("viz", {}) or {}).get("dpi", 180))

    evt = JobEvent("DISCOVER", "Scanning input directory", 0.02, {"input_dir": str(input_dir)})
    _emit(progress_cb, evt); yield evt
    if _cancelled():
        yield JobEvent("CANCELLED", "Run cancelled", 1.0)
        return

    # prefer existing .npy tracks
    npy_files = list(input_dir.rglob(track_glob))
    arr_paths_all: List[Path] = []
    num_images_processed = 0
    base_image_written = False

    if npy_files:
        arr_paths_all = sorted(map(Path, npy_files))
        evt = JobEvent("DISCOVER", f"Found {len(arr_paths_all)} track arrays (.npy); skipping KymoButler", 0.05)
        _emit(progress_cb, evt); yield evt
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
            evt = JobEvent("TABLE2HEATMAP", f"Converting {len(table_files)} table(s) → {out_dir.name}", 0.08)
            _emit(progress_cb, evt); yield evt

            for k, tbl in enumerate(table_files, start=1):
                if _cancelled():
                    yield JobEvent("CANCELLED", "Run cancelled during table conversion", 1.0)
                    return
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

                evt = JobEvent(
                    "TABLE2HEATMAP",
                    f"Converted {k}/{len(table_files)}",
                    0.08 + 0.10 * (k / max(1, len(table_files))),
                )
                _emit(progress_cb, evt); yield evt

        all_images = list(map(Path, img_files)) + generated_imgs
        # filter visualization outputs
        all_images = [p for p in all_images if not _is_overlay(p)]
        num_images_processed = len(all_images)

        if not all_images:
            msg = "No images, tables, or track arrays found under input directory"
            yield JobEvent("ERROR", msg, 1.0)
            return

        evt = JobEvent("KYMO", f"Running KymoButler on {len(all_images)} image(s)", 0.20, {"backend": backend})
        _emit(progress_cb, evt); yield evt

        produced = 0
        for idx, img in enumerate(all_images, start=1):
            if _cancelled():
                yield JobEvent("CANCELLED", "Run cancelled before/while KymoButler", 1.0)
                return

            # Copy the first image as base.png for the viewer (once)
            if not base_image_written:
                try:
                    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)  # ensure dir exists
                    shutil.copyfile(img, output_dir / "base.png")
                    base_image_written = True
                except Exception:
                    pass

            base_dir = run_kymo(str(img), **run_kwargs)
            out_dir = base_dir / "kymobutler_output"
            found = sorted(out_dir.glob(track_glob))
            arr_paths_all.extend(found)
            produced += len(found)

            evt = JobEvent(
                "KYMO",
                f"{img.name}: {len(found)} track(s)",
                0.20 + 0.40 * (idx / max(1, len(all_images))),
                {"total_tracks": produced},
            )
            _emit(progress_cb, evt); yield evt

    if not arr_paths_all:
        yield JobEvent("ERROR", "No track arrays (.npy) found after processing", 1.0)
        return

    # -------- resume filtering (NEW) --------
    total_tracks = len(arr_paths_all)
    arr_index: Dict[Path, int] = {p: i for i, p in enumerate(arr_paths_all)}
    processed_indices: Set[int] = set()
    skipped_count = 0

    def _marker_for(p: Path) -> Path:
        return marker_dir / f"{p.stem}.done"

    if resume_enabled:
        for p in arr_paths_all:
            if _marker_for(p).exists():
                processed_indices.add(arr_index[p])
                skipped_count += 1

    # Tracks we still need to process this run:
    arr_paths = [p for p in arr_paths_all if arr_index[p] not in processed_indices]
    already_done = len(processed_indices)
    if already_done:
        evt = JobEvent(
            "PROCESS",
            f"Resuming: {already_done}/{total_tracks} already complete, {len(arr_paths)} remaining",
            0.58,
            {"already_done": already_done, "remaining": len(arr_paths), "total": total_tracks},
        )
        _emit(progress_cb, evt); yield evt

    # analysis (tracks + waves) and plots
    viz_cfg = cfg.get("viz", {}) or {}
    viz_enabled = bool(viz_cfg.get("enabled", True))
    per_track_cfg = viz_cfg.get("per_track", {}) or {}
    make_plot_detrended = bool(per_track_cfg.get("detrended_with_peaks", True))
    make_plot_spectrum = bool(per_track_cfg.get("spectrum", True))
    summary_cfg = viz_cfg.get("summary", {}) or {}
    make_summary_hists = bool(summary_cfg.get("histograms", True))
    hist_bins = int(viz_cfg.get("hist_bins", 20))
    wave_win_cfg = (viz_cfg.get("wave_windows") or {})  # NEW

    detrend_cfg = cfg.get("detrend", {}) or {}
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    sampling_rate = (cfg.get("io", {}) or {}).get("sampling_rate", period_cfg.get("sampling_rate", 1.0))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    tracks_out = ensure_dir(output_dir)
    plots_dir = ensure_dir(plots_dir) if (viz_enabled and plots_dir) else None

    track_results: List[dict] = []
    wave_results: List[dict] = []

    # conditionally pass new wave-window limits if process_track supports them
    pt_params = set(inspect.signature(_process_track).parameters.keys())
    extra_pt_kwargs = {}
    if "save_wave_windows" in pt_params:
        extra_pt_kwargs["save_wave_windows"] = bool(wave_win_cfg.get("save", True))
    if "wave_window_max" in pt_params:
        extra_pt_kwargs["wave_window_max"] = int(wave_win_cfg.get("max_per_track", 50))
    if "wave_window_stride" in pt_params:
        extra_pt_kwargs["wave_window_stride"] = int(wave_win_cfg.get("stride", 1))

    # progress bookkeeping
    partial_index = 0
    last_progress_write = 0.0

    def _write_partials() -> dict:
        """Write partial artifacts so the UI can preview them while running/cancelled."""
        tracks_csv = tracks_out / "metrics.partial.csv"
        waves_csv = tracks_out / "metrics_waves.partial.csv"
        overlay_json = tracks_out / "overlay" / "tracks.partial.json"

        df_tracks = pd.DataFrame(track_results)
        if not df_tracks.empty:
            df_tracks.to_csv(tracks_csv, index=False)
        else:
            tracks_csv = None

        df_waves = pd.DataFrame(wave_results)
        if not df_waves.empty:
            df_waves = _sort_waves(df_waves)
            df_waves.to_csv(waves_csv, index=False, na_rep="NA")
        else:
            waves_csv = None

        try:
            done_paths = [arr_paths_all[i] for i in sorted(processed_indices)]
            _build_overlay_json(done_paths, cfg, overlay_json, log=get_logger("pipeline.overlay"))
        except Exception:
            overlay_json = None

        extra = {"partial_index": partial_index}
        if tracks_csv: extra["tracks_partial"] = str(tracks_csv)
        if waves_csv:  extra["waves_partial"] = str(waves_csv)
        if overlay_json and overlay_json.exists(): extra["overlay_partial"] = str(overlay_json)
        # also write progress on partials
        _write_progress(total_tracks, len(processed_indices), skipped_count)
        return extra

    evt = JobEvent("PROCESS", f"Processing {total_tracks} track(s)", 0.60)
    _emit(progress_cb, evt); yield evt

    # initial progress snapshot
    _write_progress(total_tracks, len(processed_indices), skipped_count)
    last_progress_write = time.time()

    for i, arr in enumerate(arr_paths, start=1):
        if _cancelled():
            extra = _write_partials()
            yield JobEvent("CANCELLED", f"Run cancelled at {len(processed_indices)}/{total_tracks}", 1.0, extra=extra)
            return
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
                **extra_pt_kwargs,   # NEW (safe; only if supported)
            )
            track_results.append(metrics)
            wave_results.extend(waves)

            # mark as done for resume
            try:
                _marker_for(arr).touch()
            except Exception:
                pass

            # update processed set
            processed_indices.add(arr_index[arr])

        except Exception as e:
            get_logger("pipeline.process").exception(f"Failed on {arr}: {e}")

        # timed progress write
        now = time.time()
        if (now - last_progress_write) >= progress_every_secs:
            _write_progress(total_tracks, len(processed_indices), skipped_count)
            last_progress_write = now

        # periodic partials so UI can browse interim results
        if (len(processed_indices) % max(1, partial_every) == 0) or (len(processed_indices) == total_tracks):
            partial_index += 1
            extra = _write_partials()
            evtp = JobEvent(
                "WRITE_PARTIAL",
                f"Wrote partial artifacts at {len(processed_indices)}/{total_tracks}",
                0.60 + 0.25 * (len(processed_indices) / max(1, total_tracks)),
                extra=extra,
            )
            _emit(progress_cb, evtp); yield evtp

        evt = JobEvent(
            "PROCESS",
            f"Processed {len(processed_indices)}/{total_tracks}",
            0.60 + 0.25 * (len(processed_indices) / max(1, total_tracks)),
        )
        _emit(progress_cb, evt); yield evt

    if _cancelled():
        extra = _write_partials()
        yield JobEvent("CANCELLED", "Run cancelled after processing", 1.0, extra=extra)
        return

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
    evt = JobEvent("OVERLAY", "Building overlay JSON", 0.88)
    _emit(progress_cb, evt); yield evt
    overlay_json = tracks_out / "overlay" / "tracks.json"
    _build_overlay_json([arr_paths_all[i] for i in sorted(processed_indices)], cfg, overlay_json, log=get_logger("pipeline.overlay"))

    # summary histograms (optional)
    if plots_dir is not None and make_summary_hists:
        from ..visualize import plot_summary_histograms as _plots
        _plots(df_tracks, plots_dir, bins=hist_bins, dpi=int(viz_cfg.get("dpi", 180)))

    # manifest
    manifest = {
        "backend": backend,
        "num_images": int(num_images_processed),
        "num_tracks": int(len(df_tracks)),
        "num_waves": int(len(df_waves)),
        "config_path": str(Path(config_path)),
        "output_dir": str(tracks_out),
        "plots_dir": str(plots_dir) if plots_dir else None,
        "resume": {
            "enabled": resume_enabled,
            "skipped_count": int(skipped_count),
            "marker_dir": str(marker_dir),
            "progress_file": str(progress_file),
        },  # NEW
    }
    manifest_json = tracks_out / "manifest.json"
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)

    # final progress snapshot
    _write_progress(total_tracks, len(processed_indices), skipped_count)

    evt = JobEvent("WRITE", "Wrote outputs", 0.96, {
        "tracks_csv": str(tracks_csv),
        "waves_csv": str(waves_csv),
        "overlay_json": str(overlay_json),
        "manifest": str(manifest_json),
        "base_image": str(output_dir / "base.png") if (output_dir / "base.png").exists() else None,
    })
    _emit(progress_cb, evt); yield evt

    yield JobEvent("DONE", "Pipeline complete", 1.0)
