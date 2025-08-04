import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

# import kymo interface
from kymo_interface import run_kymobutler, preprocess_image

# import signal modules
from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks, peaks_to_dataframe
from .signal.period import estimate_dominant_frequency, frequency_to_period


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

    # detect peaks
    peaks_idx, props = detect_peaks(residual, **peaks_cfg)

    # global frequency
    freq = estimate_dominant_frequency(residual, **period_cfg)
    period = frequency_to_period(freq)

    amps = residual[peaks_idx] if len(peaks_idx) > 0 else np.array([])

    return {
        'track': arr_path.stem,
        'num_peaks': len(peaks_idx),
        'mean_amplitude': float(np.mean(amps)) if amps.size > 0 else float('nan'),
        'median_amplitude': float(np.median(amps)) if amps.size > 0 else float('nan'),
        'std_amplitude': float(np.std(amps)) if amps.size > 0 else float('nan'),
        'dominant_frequency': float(freq),
        'period': float(period)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract wave characteristics from kymobutler outputs"
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

    # load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    detrend_cfg = cfg.get('detrend', {})
    peaks_cfg = cfg.get('peaks', {})
    period_cfg = cfg.get('period', {})

    input_path = Path(args.input_dir)
    results = []

    # find all heatmap images or .npy files
    # if .npy files present, skip kymobutler run
    npy_files = list(input_path.rglob('*.npy'))
    if npy_files:
        if args.verbose:
            print(f'Found {len(npy_files)} .npy files; skipping KymoButler run')
        arr_paths = npy_files
    else:
        # run kymobutler on each image
        img_files = list(input_path.rglob('*.png')) + list(input_path.rglob('*.jpg'))
        if args.verbose:
            print(f'Found {len(img_files)} images; running KymoButler...')
        arr_paths = []
        for img in img_files:
            base_dir = run_kymobutler(img, verbose=args.verbose)
            out_dir = base_dir / 'kymobutler_output'
            arr_paths += list(out_dir.glob('*.npy'))

    # process each track array
    for arr in arr_paths:
        if args.verbose:
            print(f'Processing track {arr.name}...')
        metrics = process_track(arr, detrend_cfg, peaks_cfg, period_cfg)
        results.append(metrics)

    # write results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")


if __name__ == '__main__':
    main()
