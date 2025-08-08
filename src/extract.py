import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# kymo interface (Mathematica/WolframScript wrappers)
from kymo_interface import run_kymobutler

# signal-processing modules (package-relative)
from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks
from .signal.period import estimate_dominant_frequency, frequency_to_period

# utilities
from .utils import setup_logging, get_logger, load_config, list_files, save_dataframe


def process_track(
    arr_path: Path,
    detrend_cfg: dict,
    peaks_cfg: dict,
    period_cfg: dict
) -> dict:
    """
    Load a track (.npy), compute residual, detect peaks, and estimate frequency.
    Returns a dict of track-level metrics.
    """
    data = np.load(arr_path)
    # assume data shape (N, 2): columns [x, y]
    x = data[:, 0]
    y = data[:, 1]

    # detrend
    residual = detrend_residual(x, y, **detrend_cfg)

    # detect peaks on residual
    peaks_idx, props = detect_peaks(residual, **peaks_cfg)

    # global frequency and period
    freq = estimate_dominant_frequency(residual, **period_cfg)
    period = frequency_to_period(freq)

    amps = residual[peaks_idx] if len(peaks_idx) > 0 else np.array([])

    return {
        'track': arr_path.stem,
        'num_peaks': int(len(peaks_idx)),
        'mean_amplitude': float(np.mean(amps)) if amps.size > 0 else float('nan'),
        'median_amplitude': float(np.median(amps)) if amps.size > 0 else float('nan'),
        'std_amplitude': float(np.std(amps)) if amps.size > 0 else float('nan'),
        'dominant_frequency': float(freq),
        'period': float(period)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract wave characteristics from KymoButler outputs"
    )
    parser.add_argument(
        '--input-dir', '-i', required=True,
        help='Directory containing kymograph images (or .npy outputs)'
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

    detrend_cfg = cfg.get('detrend', {})
    peaks_cfg = cfg.get('peaks', {})
    period_cfg = cfg.get('period', {}).copy()
    # ensure sampling_rate present for period estimation
    period_cfg.setdefault('sampling_rate', io_cfg.get('sampling_rate', 1.0))

    input_path = Path(args.input_dir)
    results = []

    # discover existing track arrays
    npy_files = list(input_path.rglob(track_glob))
    if npy_files:
        log.info(f'Found {len(npy_files)} existing track arrays; skipping KymoButler run')
        arr_paths = sorted(npy_files)
    else:
        # gather images and run KymoButler per image
        img_files = list_files(input_path, image_globs)
        if not img_files:
            log.error('No images or track arrays found under input directory')
            raise SystemExit(2)
        log.info(f'Found {len(img_files)} images; running KymoButler...')
        arr_paths = []
        for img in img_files:
            log.debug(f'Running KymoButler on {img}...')
            base_dir = run_kymobutler(img, verbose=args.verbose)
            out_dir = base_dir / 'kymobutler_output'
            found = sorted(out_dir.glob(track_glob))
            log.debug(f"{img.name}: found {len(found)} tracks")
            arr_paths.extend(found)

    if not arr_paths:
        log.error('No track arrays (.npy) found after processing')
        raise SystemExit(2)

    # process each track array
    for arr in arr_paths:
        log.debug(f'Processing track {arr.name}...')
        try:
            metrics = process_track(arr, detrend_cfg, peaks_cfg, period_cfg)
            results.append(metrics)
        except Exception as e:
            log.exception(f'Failed on {arr}: {e}')

    # write results
    df = pd.DataFrame(results)
    save_dataframe(df, args.output_csv)
    log.info(f"Metrics saved to {args.output_csv}")


if __name__ == '__main__':
    main()
