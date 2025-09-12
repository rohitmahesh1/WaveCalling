import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# signal-processing modules (package-relative)
from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks, detect_peaks_adaptive
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

# NEW: features (wave & peak row builders, ids, etc.)
from .features import (
    build_wave_rows,
    build_peak_rows,
    sample_id_from_name,
    coerce_track_id,
)

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
    features_cfg: dict | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    """
    Load a track (.npy), compute residual, detect peaks (+props), estimate frequency,
    make plots (optional), and return:
      (track_level_metrics_dict, [per-wave rows], [per-peak rows])
    """
    data = np.load(arr_path)
    # assume data shape (N, 2): columns [x, y]
    x = data[:, 0]
    y = data[:, 1]

    # detrend
    residual = detrend_residual(x, y, **detrend_cfg)

    # global frequency and period (does NOT depend on peaks)
    freq = estimate_dominant_frequency(residual, **period_cfg)
    period = frequency_to_period(freq)

    # Adaptive, period-aware peak detection (default).
    # Can force legacy via YAML: peaks.adaptive: false
    sampling_rate_eff = (sampling_rate if sampling_rate is not None else 1.0)
    frames_per_period = (sampling_rate_eff / freq) if (sampling_rate_eff and np.isfinite(freq) and freq > 0) else None

    use_adaptive = bool(peaks_cfg.get("adaptive", True))
    if use_adaptive:
        # Map optional adaptive knobs from YAML (with safe defaults)
        peaks_idx, props = detect_peaks_adaptive(
            residual,
            frames_per_period=frames_per_period,
            distance_frac=float(peaks_cfg.get("distance_frac", 0.6)),
            width_frac=float(peaks_cfg.get("width_frac", 0.2)),
            rel_mad_k=float(peaks_cfg.get("rel_mad_k", 2.0)),
            abs_min_prom_px=float(peaks_cfg.get("abs_min_prom_px", 1.0)),
            nms_enable=bool(peaks_cfg.get("nms_enable", True)),
            nms_dominance_frac=float(peaks_cfg.get("nms_dominance_frac", 0.55)),
        )
    else:
        # Legacy fixed thresholds
        legacy_kwargs = {
            "prominence": float(peaks_cfg.get("prominence", 1.0)),
            "width": float(peaks_cfg.get("width", 1.0)),
        }
        if peaks_cfg.get("distance", None) is not None:
            legacy_kwargs["distance"] = int(peaks_cfg["distance"])
        peaks_idx, props = detect_peaks(residual, **legacy_kwargs)

    amps = residual[peaks_idx] if len(peaks_idx) > 0 else np.array([])

    sample_name = _sample_name_from_arr_path(arr_path)

    # per-wave rows (k -> k+1) with enriched features & anchored fit params
    wave_rows = build_wave_rows(
        x=x, y=y, residual=residual,
        peaks_idx=peaks_idx, peak_props=props,
        sampling_rate=sampling_rate_eff,
        sample=sample_name, track_stem=arr_path.stem,
        features_cfg=features_cfg or {},
        freq_hz=float(freq) if np.isfinite(freq) else None,
        period_frac_for_fit=float((features_cfg or {}).get("fit_window_period_frac", 0.5)),
    )

    # NEW: per-peak rows (anchored sine fit around EACH detected peak)
    peak_rows = build_peak_rows(
        x=x, y=y, residual=residual,
        peaks_idx=peaks_idx, peak_props=props,
        sampling_rate=sampling_rate_eff,
        sample=sample_name, track_stem=arr_path.stem,
        features_cfg=features_cfg or {},
        global_freq_hz=float(freq) if np.isfinite(freq) else None,
        period_frac_for_fit=float((features_cfg or {}).get("fit_window_period_frac", 0.5)),
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
                        period_frac=float((features_cfg or {}).get("fit_window_period_frac", 0.5)),
                        max_plots=12,
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

    # Track-level metrics (+ IDs; non-breaking additions)
    track_id = coerce_track_id(arr_path.stem)
    track_metrics = {
        'sample': sample_name,
        'sample_id': int(sample_id_from_name(sample_name)),
        'track': arr_path.stem,
        'track_id': int(track_id) if track_id is not None else np.nan,
        'num_peaks': int(len(peaks_idx)),
        'mean_amplitude': float(np.mean(amps)) if amps.size > 0 else float('nan'),
        'median_amplitude': float(np.median(amps)) if amps.size > 0 else float('nan'),
        'std_amplitude': float(np.std(amps)) if amps.size > 0 else float('nan'),
        'dominant_frequency': float(freq),
        'period': float(period),
    }
    return track_metrics, wave_rows, peak_rows


# -----------------------
# Backend selection utils
# -----------------------
def _cfg_get(dct, path, default=None):
    """Safe nested get: path is list/tuple of keys."""
    cur = dct
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_kymo_runner(cfg, log, verbose):
    kymo_cfg = cfg.get('kymo', {}) or {}
    backend = str(kymo_cfg.get('backend', 'onnx')).lower()
    if backend not in ('onnx', 'wolfram'):
        log.warning(f"Unknown kymo.backend={backend!r}; defaulting to 'onnx'")
        backend = 'onnx'

    if backend == 'wolfram':
        from .modules.kymo_interface import run_kymobutler as _runner
        min_len = kymo_cfg.get('min_length', 30)
        kwargs = dict(min_length=int(min_len), verbose=bool(verbose))
        return _runner, kwargs, 'wolfram'

    # ---- ONNX backend ----
    from .modules.kb_adapter import run_kymobutler as _runner
    onnx_cfg = kymo_cfg.get('onnx', {}) or {}

    export_dir   = onnx_cfg.get('export_dir', None)
    seg_size     = int(onnx_cfg.get('seg_size', 256))
    force_mode   = onnx_cfg.get('force_mode', None)
    fuse_uni_into_bi = bool(onnx_cfg.get('fuse_uni_into_bi', True))
    fuse_uni_weight  = float(onnx_cfg.get('fuse_uni_weight', 0.7))

    thr_cfg  = onnx_cfg.get('thresholds', {}) or {}
    thr      = float(thr_cfg.get('thr_default', 0.20))
    thr_bi   = thr_cfg.get('thr_bi', None)
    thr_uni  = thr_cfg.get('thr_uni', None)

    auto_cfg = onnx_cfg.get('auto_threshold', {}) or {}
    auto_threshold   = bool(auto_cfg.get('enabled', True))
    auto_sweep       = tuple(auto_cfg.get('sweep', [0.12, 0.30, 19]))
    auto_target_pct  = tuple(auto_cfg.get('target_mask_pct', [15.0, 25.0]))
    auto_trigger_pct = tuple(auto_cfg.get('trigger_pct', [5.0, 35.0]))

    hyst_cfg = onnx_cfg.get('hysteresis', {}) or {}
    hysteresis_enable = bool(hyst_cfg.get('enabled', True))
    hysteresis_low    = float(hyst_cfg.get('low', 0.08))
    hysteresis_high   = float(hyst_cfg.get('high', 0.18))

    morph = onnx_cfg.get('morphology', {}) or {}
    morph_mode = str(morph.get('mode', 'directional'))
    morph_close_open_close = (morph_mode == 'classic')
    directional_close      = (morph_mode == 'directional')
    dir_kv = int((morph.get('directional', {}) or {}).get('kv', 5))
    dir_kh = int((morph.get('directional', {}) or {}).get('kh', 5))
    diag_bridge = bool((morph.get('directional', {}) or {}).get('diag_bridge', True))

    comp = onnx_cfg.get('components', {}) or {}
    comp_min_px   = int(comp.get('min_px', 5))
    comp_min_rows = int(comp.get('min_rows', 5))

    # --- NEW: skeleton guardrails & clamps ---
    skel = onnx_cfg.get('skeleton', {}) or {}
    keep_ratio = float(skel.get('keep_ratio', 0.60))
    keep_min_px = int(skel.get('keep_min_px', 2000))
    skel_prob_floor_min = float(skel.get('prob_floor_min', 0.06))
    skel_prob_floor_max = float(skel.get('prob_floor_max', 0.10))
    prune_iters = int(skel.get('prune_iters', 0))

    # postproc → adapter arg names
    post = onnx_cfg.get('postproc', {}) or {}
    extend_rows = int(post.get('extend_rows', 22))
    dx_win = int(post.get('dx_win', 4))
    refine_prob_min = float(post.get('prob_min', 0.11))        # map YAML prob_min → adapter refine_prob_min
    max_gap_rows = int(post.get('max_gap_rows', 13))
    max_dx = int(post.get('max_dx', 6))
    prob_bridge_min = float(post.get('prob_bridge_min', 0.11))
    dedupe = post.get('dedupe', {}) or {}
    dedupe_enable = bool(dedupe.get('enabled', True))
    dedupe_min_rows = int(dedupe.get('min_rows', 30))
    dedupe_min_score = float(dedupe.get('min_score', 0.11))
    dedupe_overlap_iou = float(dedupe.get('overlap_iou', 0.80))
    dedupe_dx_tol = float(dedupe.get('dx_tol', 2.5))

    # --- NEW: debug toggle for image dumps ---
    dbg = onnx_cfg.get('debug', {}) or {}
    debug_save_images = bool(dbg.get('save_debug_images', True))

    track = onnx_cfg.get('tracking', {}) or {}
    min_length = int(track.get('min_length', 30))

    kwargs = dict(
        # core
        min_length=min_length,
        verbose=bool(verbose),
        export_dir=export_dir,
        seg_size=seg_size,
        thr=thr,

        # per-mode thresholds & mode
        force_mode=force_mode,
        thr_uni=thr_uni,
        thr_bi=thr_bi,

        # morphology & components
        morph_close_open_close=morph_close_open_close,
        directional_close=directional_close,
        comp_min_px=comp_min_px,
        comp_min_rows=comp_min_rows,

        # skeleton guardrails & clamps (NEW)
        skel_keep_ratio=keep_ratio,
        skel_keep_min_px=keep_min_px,
        skel_prob_floor_min=skel_prob_floor_min,
        skel_prob_floor_max=skel_prob_floor_max,
        prune_iters=prune_iters,

        # auto-threshold & hysteresis
        auto_threshold=auto_threshold,
        auto_sweep=auto_sweep,
        auto_target_pct=auto_target_pct,
        auto_trigger_pct=auto_trigger_pct,
        hysteresis_enable=hysteresis_enable,
        hysteresis_low=hysteresis_low,
        hysteresis_high=hysteresis_high,

        # directional kernels & bridging
        dir_kv=dir_kv,
        dir_kh=dir_kh,
        diag_bridge=diag_bridge,

        # fusion
        fuse_uni_into_bi=fuse_uni_into_bi,
        fuse_uni_weight=fuse_uni_weight,

        # refine/merge
        extend_rows=extend_rows,
        dx_win=dx_win,
        refine_prob_min=refine_prob_min,
        max_gap_rows=max_gap_rows,
        max_dx=max_dx,
        prob_bridge_min=prob_bridge_min,

        # dedupe
        dedupe_enable=dedupe_enable,
        dedupe_min_rows=dedupe_min_rows,
        dedupe_min_score=dedupe_min_score,
        dedupe_overlap_iou=dedupe_overlap_iou,
        dedupe_dx_tol=dedupe_dx_tol,

        # debug (NEW)
        debug_save_images=debug_save_images,
    )
    return _runner, kwargs, 'onnx'


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

    # features (optional knobs)
    features_cfg = cfg.get('features', {}) or {}

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

    # determine plotting directory (requires CLI flag and viz.enabled)
    plots_dir = None
    if args.plots_out:
        if viz_enabled:
            plots_dir = ensure_dir(args.plots_out)
            log.info(f'Plot outputs will be written to {plots_dir}')
        else:
            log.info('viz.enabled is False in config; skipping plot generation despite --plots-out')

    # Choose backend runner + kwargs from YAML
    run_kymo, run_kwargs, backend = _build_kymo_runner(cfg, log, args.verbose)
    log.info(f"Using KymoButler backend: {backend}")

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
            base_dir = run_kymo(str(img), **run_kwargs)
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
    peak_results: list[dict] = []
    for arr in arr_paths:
        log.debug(f'Processing track {arr.name}...')
        try:
            metrics, wave_rows, peak_rows = process_track(
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
                features_cfg=features_cfg,
            )
            track_results.append(metrics)
            wave_results.extend(wave_rows)
            peak_results.extend(peak_rows)
        except Exception as e:
            log.exception(f'Failed on {arr}: {e}')

    # write track-level results
    df_tracks = pd.DataFrame(track_results)
    save_dataframe(df_tracks, args.output_csv)
    log.info(f"Track metrics saved to {args.output_csv}")

    # write per-wave results (sorted)
    waves_csv = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_waves.csv")
    df_waves = pd.DataFrame(wave_results)
    if not df_waves.empty:
        df_waves["Track"] = pd.to_numeric(df_waves["Track"], errors="coerce").astype("Int64")
        df_waves = df_waves.sort_values(
            ["Track", "Wave number", "Frame position 1"],
            kind="mergesort",
            ignore_index=True,
        )
    df_waves.to_csv(waves_csv, index=False, na_rep="NA")
    log.info(f"Per-wave metrics saved to {waves_csv}")

    # write per-peak results (sorted)
    peaks_csv = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_peaks.csv")
    df_peaks = pd.DataFrame(peak_results)
    if not df_peaks.empty:
        df_peaks["Track"] = pd.to_numeric(df_peaks["Track"], errors="coerce").astype("Int64")
        # sort by frame (time) then by amplitude descending as a tie-breaker
        sort_cols = ["Track"]
        if "Peak frame" in df_peaks.columns:
            sort_cols.append("Peak frame")
        if "Amplitude (pixels)" in df_peaks.columns:
            sort_cols.append("Amplitude (pixels)")
            df_peaks = df_peaks.sort_values(sort_cols, kind="mergesort", ascending=[True, True, False], ignore_index=True)
        else:
            df_peaks = df_peaks.sort_values(sort_cols, kind="mergesort", ignore_index=True)
    df_peaks.to_csv(peaks_csv, index=False, na_rep="NA")
    log.info(f"Per-peak metrics saved to {peaks_csv}")

    # summary plots (honor viz toggles) — based on track-level stats
    plots_root = plots_dir
    if plots_root is not None and make_summary_hists:
        plot_summary_histograms(df_tracks, plots_root, bins=hist_bins, dpi=dpi)
        log.info(f"Summary plots saved under {plots_root}")


if __name__ == '__main__':
    main()
