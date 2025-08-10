import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# kymo interface (Mathematica/WolframScript wrappers)
from .kymo_interface import run_kymobutler

# signal-processing modules (package-relative)
from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks
from .signal.period import estimate_dominant_frequency, frequency_to_period

# visualization
from .visualize import (
    plot_detrended_with_peaks,
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
) -> dict:
    """
    Load a track (.npy), compute residual, detect peaks, estimate frequency,
    and optionally save diagnostic plots.

    Returns a dict of track-level metrics.
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
                )
            except Exception as e:
                if log:
                    log.debug(f'Plot detrended_with_peaks failed for {arr_path.name}: {e}')
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

    return {
        'track': arr_path.stem,
        'num_peaks': int(len(peaks_idx)),
        'mean_amplitude': float(np.mean(amps)) if amps.size > 0 else float('nan'),
        'median_amplitude': float(np.median(amps)) if amps.size > 0 else float('nan'),
        'std_amplitude': float(np.std(amps)) if amps.size > 0 else float('nan'),
        'dominant_frequency': float(freq),
        'period': float(period),
    }


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

    input_path = Path(args.input_dir)
    results = []

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
            base_dir = run_kymobutler(str(img), verbose=args.verbose)
            out_dir = base_dir / 'kymobutler_output'
            found = sorted(out_dir.glob(track_glob))
            log.debug(f"{Path(img).name}: found {len(found)} tracks")
            arr_paths.extend(found)

    if not arr_paths:
        log.error('No track arrays (.npy) found after processing')
        raise SystemExit(2)

    # process each track array
    for arr in arr_paths:
        log.debug(f'Processing track {arr.name}...')
        try:
            metrics = process_track(
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
            results.append(metrics)
        except Exception as e:
            log.exception(f'Failed on {arr}: {e}')

    # write results
    df = pd.DataFrame(results)
    save_dataframe(df, args.output_csv)
    log.info(f"Metrics saved to {args.output_csv}")

    # summary plots (honor viz toggles)
    if plots_dir is not None and make_summary_hists:
        plot_summary_histograms(df, plots_dir, bins=hist_bins, dpi=dpi)
        log.info(f"Summary plots saved under {plots_dir}")


if __name__ == '__main__':
    main()
