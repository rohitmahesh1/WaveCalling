import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# kymo interface (Mathematica/WolframScript wrappers)
# from .modules.kymo_interface import run_kymobutler  # (use for Mathematica scripts)
from .modules.kb_adapter import run_kymobutler

# signal-processing modules (package-relative)
from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks
from .signal.period import estimate_dominant_frequency, frequency_to_period

# visualization
from .visualize import (
    plot_detrended_with_peaks,
    plot_peak_windows,
    plot_spectrum,
    plot_summary_histograms,
)

# I/O for tables → heatmaps
from .io.table_to_heatmap import table_to_heatmap

# utilities
from .utils import (
    setup_logging,
    get_logger,
    load_config,
    list_files,
    save_dataframe,
    ensure_dir,
)


def _sample_name_from_arr_path(arr_path: Path) -> str:
    """
    Heuristic to recover the sample name from the track path:
      .../<SAMPLE>_heatmap/kymobutler_output/NN.npy -> SAMPLE
    If no '_heatmap' suffix, return the directory name above kymobutler_output.
    """
    base = arr_path.parent.parent.name  # e.g., <SAMPLE>_heatmap
    if base.endswith("_heatmap"):
        return base[:-8]
    return base


def _waves_from_peaks(
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    sampling_rate: float,
    sample: str,
    track_stem: str,
) -> list[dict]:
    """
    Build per-wave rows from consecutive peak pairs (k -> k+1).
    """
    rows: list[dict] = []
    p = np.asarray(peaks_idx, dtype=int)
    if p.size < 2:
        return rows

    for k in range(p.size - 1):
        i, j = int(p[k]), int(p[k + 1])

        frame1 = float(x[i])
        frame2 = float(x[j])
        period_frames = frame2 - frame1
        period_s = period_frames / sampling_rate if sampling_rate else float("nan")
        freq = (1.0 / period_s) if (period_s and period_s > 0) else float("nan")

        pos1 = float(y[i])         # raw pixel position at first peak
        pos2 = float(y[j])         # raw pixel position at second peak
        amp = float(residual[i])   # amplitude at first peak (residual height)

        dpos = pos2 - pos1
        vel = (dpos / period_s) if (period_s and period_s != 0) else float("nan")
        wavelength = abs(dpos)     # with peak-to-peak period, λ = |Δpos|

        rows.append({
            "Sample": sample,
            "Track": track_stem,
            "Wave number": k + 1,
            "Frame position 1": frame1,
            "Frame position 2": frame2,
            "Period (frames)": period_frames,
            "Period (s)": period_s,
            "Frequency (Hz)": freq,
            "Pixel position 1": pos1,
            "Pixel position 2": pos2,
            "Amplitude (pixels)": amp,
            "Δposition (px)": dpos,
            "Velocity (px/s)": vel,
            "Wavelength (px)": wavelength,
        })
    return rows


def process_track(
    arr_path: Path,
    detrend_cfg: dict,
    peaks_cfg: dict,
    period_cfg: dict,
    *,
    plots_dir: Path | None = None,
    sampling_rate: float | None = None,
    make_plot_detrended: bool = True,
    make_plot_spectrum: bool = True,
    dpi: int | None = None,
    log=None,
) -> tuple[dict, list[dict]]:
    """
    Load a track (.npy), compute residual, detect peaks, estimate frequency,
    make plots (optional), and return:
      (track_level_metrics_dict, [per-wave rows])
    """
    data = np.load(arr_path)
    # assume data shape (N, 2): columns [x, y]
    x = data[:, 0]
    y = data[:, 1]

    # detrend
    residual = detrend_residual(x, y, **detrend_cfg)

    # detect peaks on residual
    peaks_idx, _ = detect_peaks(residual, **peaks_cfg)

    # global frequency and period
    freq = estimate_dominant_frequency(residual, **period_cfg)
    period = frequency_to_period(freq)

    amps = residual[peaks_idx] if len(peaks_idx) > 0 else np.array([])

    # per-wave rows (peak -> next peak)
    sample_name = _sample_name_from_arr_path(arr_path)
    wave_rows = _waves_from_peaks(
        x, y, residual, peaks_idx,
        sampling_rate=sampling_rate if sampling_rate is not None else 1.0,
        sample=sample_name,
        track_stem=arr_path.stem,
    )

    # optional plotting controlled by viz toggles
    if plots_dir is not None:
        track_dir = ensure_dir(Path(plots_dir) / arr_path.stem)
        # Baseline + residual + peaks (re-fit baseline just for overlay consistency)
        if make_plot_detrended:
            try:
                ransac_kwargs = {k: v for k, v in detrend_cfg.items() if k != 'degree'}
                plot_detrended_with_peaks(
                    x,
                    y,
                    peaks_idx,
                    track_dir / 'detrended_with_peaks.png',
                    degree=int(detrend_cfg.get('degree', 1)),
                    ransac_kwargs=ransac_kwargs,
                    title=f'{arr_path.stem}',
                    dpi=dpi,
                    overlay_fit=True,
                    sampling_rate=sampling_rate,
                    freq=freq,
                )
                if sampling_rate is not None and freq is not None and freq > 0:
                    plot_peak_windows(
                        x, y, peaks_idx,
                        (track_dir / "peak_windows"),
                        degree=int(detrend_cfg.get('degree', 1)),
                        ransac_kwargs=ransac_kwargs,
                        sampling_rate=sampling_rate,
                        freq=freq,
                        period_frac=0.5,   # show 0.5 of a period centered on each peak
                        max_plots=12,      # cap to avoid huge directories
                        dpi=dpi,
                        title_prefix=arr_path.stem,
                        overlay_fit=True,
                    )
            except Exception as e:
                if log:
                    log.debug(f'Plot detrended_with_peaks/peak_windows failed for {arr_path.name}: {e}')
        # Spectrum
        if make_plot_spectrum and (sampling_rate is not None):
            try:
                plot_spectrum(
                    residual,
                    sampling_rate,
                    track_dir / 'spectrum.png',
                    title=f'{arr_path.stem} spectrum',
                    dpi=dpi,
                )
            except Exception as e:
                if log:
                    log.debug(f'Plot spectrum failed for {arr_path.name}: {e}')

    track_metrics = {
        'track': arr_path.stem,
        'num_peaks': int(len(peaks_idx)),
        'mean_amplitude': float(np.mean(amps)) if amps.size > 0 else float('nan'),
        'median_amplitude': float(np.median(amps)) if amps.size > 0 else float('nan'),
        'std_amplitude': float(np.std(amps)) if amps.size > 0 else float('nan'),
        'dominant_frequency': float(freq),
        'period': float(period),
    }
    return track_metrics, wave_rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract wave characteristics from KymoButler outputs"
    )
    parser.add_argument(
        '--input-dir', '-i', required=True,
        help='Directory containing kymograph images, tables (csv/tsv/xls/xlsx), or .npy outputs'
    )
    parser.add_argument(
        '--config', '-c', required=True,
        help='YAML config file for parameters'
    )
    parser.add_argument(
        '--output-csv', '-o', required=True,
        help='Path to write master CSV of track metrics'
    )
    parser.add_argument(
        '--plots-out', type=str, default=None,
        help='Directory to save plots (per-track and summary). If omitted, no plots are generated.'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Verbose logging'
    )
    args = parser.parse_args()

    # load config & set up logging
    cfg = load_config(args.config)
    log_level = cfg.get('logging', {}).get('level', 'INFO')
    setup_logging(log_level)
    log = get_logger(__name__)
    if args.verbose:
        log.setLevel('DEBUG')

    # configs
    io_cfg = cfg.get('io', {})
    image_globs = io_cfg.get('image_globs', ['*.png', '*.jpg', '*.jpeg'])
    track_glob = io_cfg.get('track_glob', '*.npy')
    table_globs = io_cfg.get('table_globs', ['*.csv', '*.tsv', '*.xls', '*.xlsx'])

    detrend_cfg = cfg.get('detrend', {})
    peaks_cfg = cfg.get('peaks', {})
    period_cfg = cfg.get('period', {}).copy()
    # ensure sampling_rate present for period estimation
    sampling_rate = io_cfg.get('sampling_rate', 1.0)
    period_cfg.setdefault('sampling_rate', sampling_rate)

    # heatmap params (for tables → images)
    heatmap_cfg = cfg.get('heatmap', {})
    hm_lower = float(heatmap_cfg.get('lower', -1e20))
    hm_upper = float(heatmap_cfg.get('upper', 1e16))
    hm_binarize = bool(heatmap_cfg.get('binarize', True))
    hm_origin = str(heatmap_cfg.get('origin', 'lower'))
    hm_cmap = str(heatmap_cfg.get('cmap', 'hot'))

    # viz toggles
    viz_cfg = cfg.get('viz', {})
    viz_enabled = bool(viz_cfg.get('enabled', True))
    per_track_cfg = viz_cfg.get('per_track', {})
    make_plot_detrended = bool(per_track_cfg.get('detrended_with_peaks', True))
    make_plot_spectrum = bool(per_track_cfg.get('spectrum', True))
    summary_cfg = viz_cfg.get('summary', {})
    make_summary_hists = bool(summary_cfg.get('histograms', True))
    hist_bins = int(viz_cfg.get('hist_bins', 20))
    dpi = int(viz_cfg.get('dpi', 180))

    # --- KymoButler (ONNX) runtime config ---
    kymo_cfg = cfg.get('kymo', {})
    kymo_export_dir = kymo_cfg.get('export_dir', None)
    kymo_force_mode = kymo_cfg.get('force_mode', 'bi')  # bi matches WL behavior
    kymo_seg_size   = int(kymo_cfg.get('seg_size', 256))

    # thresholds
    kymo_thr       = float(kymo_cfg.get('thr', 0.20))
    kymo_thr_uni   = kymo_cfg.get('thr_uni', None)
    kymo_thr_bi    = kymo_cfg.get('thr_bi', None)

    # WL-like shaping
    kymo_morph_coc     = bool(kymo_cfg.get('morph_close_open_close', True))
    kymo_min_comp_px   = int(kymo_cfg.get('min_component_px', 5))  # legacy compat
    kymo_comp_min_px   = int(kymo_cfg.get('comp_min_px', 10))
    kymo_comp_min_rows = int(kymo_cfg.get('comp_min_rows', 10))
    kymo_prune_iters   = int(kymo_cfg.get('prune_iters', 3))

    fuse_uni_into_bi = bool(kymo_cfg.get('fuse_uni_into_bi', True))
    fuse_uni_weight = float(kymo_cfg.get('fuse_uni_weight', 0.7))

    # final save gate
    kymo_min_length = int(kymo_cfg.get('min_length', 30))

    input_path = Path(args.input_dir)

    # determine plotting directory (requires CLI flag and viz.enabled)
    plots_dir = None
    if args.plots_out:
        if viz_enabled:
            plots_dir = ensure_dir(args.plots_out)
            log.info(f'Plot outputs will be written to {plots_dir}')
        else:
            log.info('viz.enabled is False in config; skipping plot generation despite --plots-out')

    # discover existing track arrays first
    npy_files = list(input_path.rglob(track_glob))
    if npy_files:
        log.info(f'Found {len(npy_files)} existing track arrays; skipping KymoButler run')
        arr_paths = sorted(npy_files)
    else:
        # gather images from disk
        img_files = list_files(input_path, image_globs)

        # gather tables and convert to heatmaps
        table_files = list_files(input_path, table_globs)
        generated_imgs = []
        if table_files:
            out_dir = ensure_dir(input_path / 'generated_heatmaps')
            log.info(f'Converting {len(table_files)} table(s) to heatmaps → {out_dir}')
            for tbl in table_files:
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
                    log.exception(f'Heatmap generation failed for {tbl}: {e}')

        all_images = list(map(Path, img_files)) + generated_imgs

        # filter out KymoButler visualization outputs
        def _is_overlay(p: Path) -> bool:
            s = str(p)
            return s.endswith("_overlay.png") or s.endswith("_overlay.jpg") or "_output/" in s or s.endswith("_output")

        all_images = [p for p in all_images if not _is_overlay(p)]

        if not all_images:
            log.error('No images, tables, or track arrays found under input directory')
            raise SystemExit(2)

        log.info(f'Found {len(all_images)} images total; running KymoButler...')
        arr_paths = []
        for img in all_images:
            log.debug(f'Running KymoButler on {img}...')
            base_dir = run_kymobutler(
                str(img),
                min_length=kymo_min_length,
                verbose=args.verbose,
                export_dir=kymo_export_dir,
                seg_size=kymo_seg_size,
                thr=kymo_thr,
                min_component_px=kymo_min_comp_px,
                force_mode=kymo_force_mode,
                thr_uni=kymo_thr_uni,
                thr_bi=kymo_thr_bi,
                morph_close_open_close=kymo_morph_coc,
                comp_min_px=kymo_comp_min_px,
                comp_min_rows=kymo_comp_min_rows,
                prune_iters=kymo_prune_iters,
                fuse_uni_into_bi=fuse_uni_into_bi,
                fuse_uni_weight=fuse_uni_weight,
            )
            out_dir = base_dir / 'kymobutler_output'
            found = sorted(out_dir.glob(track_glob))
            log.debug(f"{Path(img).name}: found {len(found)} tracks")
            arr_paths.extend(found)

    if not arr_paths:
        log.error('No track arrays (.npy) found after processing')
        raise SystemExit(2)

    # process each track array
    track_results: list[dict] = []
    wave_results: list[dict] = []
    for arr in arr_paths:
        log.debug(f'Processing track {arr.name}...')
        try:
            metrics, wave_rows = process_track(
                arr,
                detrend_cfg,
                peaks_cfg,
                period_cfg,
                plots_dir=plots_dir,
                sampling_rate=sampling_rate,
                make_plot_detrended=make_plot_detrended,
                make_plot_spectrum=make_plot_spectrum,
                dpi=dpi,
                log=log,
            )
            track_results.append(metrics)
            wave_results.extend(wave_rows)
        except Exception as e:
            log.exception(f'Failed on {arr}: {e}')

    # write track-level results
    df_tracks = pd.DataFrame(track_results)
    save_dataframe(df_tracks, args.output_csv)
    log.info(f"Track metrics saved to {args.output_csv}")

    # write per-wave results
    waves_csv = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_waves.csv")
    df_waves = pd.DataFrame(wave_results)

    if not df_waves.empty:
        # Coerce Track to numeric for a true numeric sort (handles "0"..."102", etc.)
        df_waves["Track"] = pd.to_numeric(df_waves["Track"], errors="coerce").astype("Int64")
        # Stable sort: by Track, then Wave number, then Frame position 1 (optional)
        df_waves = df_waves.sort_values(
            ["Track", "Wave number", "Frame position 1"],
            kind="mergesort",
            ignore_index=True,
        )

    df_waves.to_csv(waves_csv, index=False, na_rep="NA")
    log.info(f"Per-wave metrics saved to {waves_csv}")

    # summary plots (honor viz toggles) — based on track-level stats
    if plots_dir is not None and make_summary_hists:
        plot_summary_histograms(df_tracks, plots_dir, bins=hist_bins, dpi=dpi)
        log.info(f"Summary plots saved under {plots_dir}")


if __name__ == '__main__':
    main()
