#!/usr/bin/env python3
# scripts/test_regressions.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --- Make sure we can import package modules ---
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import (
    load_config,
    ensure_dir,
    setup_logging,
    get_logger,
    save_dataframe,
)
from src.extract import process_track


def _gather_npy_files(dirs: List[Path], pattern: str = "*.npy") -> List[Path]:
    found: List[Path] = []
    for d in dirs:
        d = d.resolve()
        if d.is_file() and d.suffix.lower() == ".npy":
            found.append(d)
        elif d.is_dir():
            found.extend(sorted(d.rglob(pattern)))
    # keep stable order before sampling (so RNG sampling is deterministic)
    return sorted(set(found))


def _sample_files(all_files: List[Path], seed: int, max_total: int | None, per_dir: int | None) -> List[Path]:
    rng = np.random.default_rng(seed)
    if per_dir is not None:
        # sample up to N per parent dir of each .npy (the kymobutler_output folders)
        # parent key → files
        by_parent = {}
        for p in all_files:
            by_parent.setdefault(p.parent.resolve(), []).append(p)
        selected = []
        for parent, files in by_parent.items():
            files = sorted(files)
            k = min(per_dir, len(files))
            if k > 0:
                idx = rng.choice(len(files), size=k, replace=False)
                selected.extend([files[i] for i in sorted(idx)])
        return sorted(selected)
    # else: global sampling
    if max_total is None or max_total >= len(all_files):
        return list(all_files)
    idx = rng.choice(len(all_files), size=max_total, replace=False)
    return [all_files[i] for i in sorted(idx)]


def _sort_waves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Track"] = pd.to_numeric(df.get("Track"), errors="coerce").astype("Int64")
    return df.sort_values(
        ["Track", "Wave number", "Frame position 1"],
        kind="mergesort",
        ignore_index=True,
    )


def _sort_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror pipeline's peak sort: Track ↑, Peak frame ↑ (if present), Amplitude ↓ (if present)."""
    if df.empty:
        return df
    df = df.copy()
    df["Track"] = pd.to_numeric(df.get("Track"), errors="coerce").astype("Int64")
    sort_cols = ["Track"]
    ascending = [True]
    if "Peak frame" in df.columns:
        sort_cols.append("Peak frame")
        ascending.append(True)
    if "Amplitude (pixels)" in df.columns:
        sort_cols.append("Amplitude (pixels)")
        ascending.append(False)
    return df.sort_values(sort_cols, kind="mergesort", ascending=ascending, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(
        description="Sample a subset of .npy tracks and run the exact regression/plotting steps used in the pipeline."
    )
    ap.add_argument(
        "--data-dirs", "-d",
        nargs="+",
        required=True,
        help="One or more directories that contain kymobutler_output/*.npy files."
    )
    ap.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config (same one used by pipeline)."
    )
    ap.add_argument(
        "--out", "-o",
        required=True,
        help="Output folder for images and CSVs."
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling."
    )
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Sample at most this many .npy files across ALL data-dirs."
    )
    sel.add_argument(
        "--per-dir",
        type=int,
        default=10,
        help="Sample at most this many .npy files per directory (default: 10)."
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging."
    )
    args = ap.parse_args()

    out_dir = ensure_dir(args.out)
    plots_dir = ensure_dir(Path(out_dir) / "plots")  # process_track will create per-track subfolders here

    # --- Load config & logging (exact same helpers as pipeline) ---
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("test_regressions")
    if args.verbose:
        log.setLevel("DEBUG")

    # Pull the same knobs extract/pipeline use
    io_cfg = cfg.get("io", {}) or {}
    detrend_cfg = cfg.get("detrend", {}) or {}
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    sampling_rate = io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    viz_cfg = cfg.get("viz", {}) or {}
    viz_enabled = bool(viz_cfg.get("enabled", True))
    per_track_cfg = viz_cfg.get("per_track", {}) or {}
    make_plot_detrended = bool(per_track_cfg.get("detrended_with_peaks", True))
    make_plot_spectrum = bool(per_track_cfg.get("spectrum", True))
    dpi = int(viz_cfg.get("dpi", 180))

    features_cfg = cfg.get("features", {}) or {}

    # --- Find and sample .npy tracks deterministically ---
    data_dirs = [Path(p) for p in args.data_dirs]
    all_npy = _gather_npy_files(data_dirs)
    if not all_npy:
        log.error("No .npy tracks found under the provided --data-dirs.")
        sys.exit(2)

    chosen = _sample_files(
        all_files=all_npy,
        seed=args.seed,
        max_total=args.max_total,
        per_dir=args.per_dir if args.per_dir is not None else None,
    )
    if not chosen:
        log.error("Sampling returned 0 files. Try adjusting --max-total/--per-dir.")
        sys.exit(2)

    # Write manifest of what we’re testing
    selection_json = Path(out_dir) / "selection.json"
    selection_json.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "max_total": args.max_total,
                "per_dir": args.per_dir,
                "selected_files": [str(p) for p in chosen],
            },
            indent=2,
        )
    )
    log.info(f"Selected {len(chosen)} of {len(all_npy)} tracks. Manifest: {selection_json}")

    # --- Process each selected track using the SAME function as pipeline/extract ---
    track_rows: list[dict] = []
    wave_rows: list[dict] = []
    peak_rows: list[dict] = []

    for k, arr in enumerate(chosen, start=1):
        arr = Path(arr)
        try:
            log.debug(f"[{k}/{len(chosen)}] {arr.name}")
            result = process_track(
                arr_path=arr,
                detrend_cfg=detrend_cfg,
                peaks_cfg=peaks_cfg,
                period_cfg=period_cfg,
                plots_dir=(plots_dir if viz_enabled else None),
                sampling_rate=sampling_rate,
                make_plot_detrended=make_plot_detrended,
                make_plot_spectrum=make_plot_spectrum,
                dpi=dpi,
                log=log,
                features_cfg=features_cfg,
            )

            # --- NEW: accept both 2-tuple and 3-tuple signatures ---
            if isinstance(result, tuple) and len(result) == 3:
                metrics, waves, peaks = result
            else:
                metrics, waves = result  # backward compat
                peaks = []

            track_rows.append(metrics)
            wave_rows.extend(waves)
            peak_rows.extend(peaks)

        except Exception as e:
            log.exception(f"Failed on {arr}: {e}")

    # --- Write CSVs (same schema as pipeline outputs) ---
    df_tracks = pd.DataFrame(track_rows)
    df_waves = pd.DataFrame(wave_rows)
    df_peaks = pd.DataFrame(peak_rows)

    tracks_csv = Path(out_dir) / "metrics.sampled.csv"
    waves_csv = Path(out_dir) / "metrics_waves.sampled.csv"
    peaks_csv = Path(out_dir) / "metrics_peaks.sampled.csv"

    save_dataframe(df_tracks, tracks_csv)

    if not df_waves.empty:
        df_waves = _sort_waves(df_waves)
    df_waves.to_csv(waves_csv, index=False, na_rep="NA")

    if not df_peaks.empty:
        df_peaks = _sort_peaks(df_peaks)
    df_peaks.to_csv(peaks_csv, index=False, na_rep="NA")

    log.info(f"Done. Tracks: {len(df_tracks)}  Waves: {len(df_waves)}  Peaks: {len(df_peaks)}")
    log.info(f"CSV: {tracks_csv}")
    log.info(f"CSV: {waves_csv}")
    log.info(f"CSV: {peaks_csv}")
    log.info(f"Plots: {plots_dir}  (per-track folders with 'detrended_with_peaks.png' and 'peak_windows/*')")


if __name__ == "__main__":
    main()
