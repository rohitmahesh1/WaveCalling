from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .signal.detrend import fit_baseline_ransac
from .signal.period import spectrum_dataframe
from .utils import ensure_dir


def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    return p


def plot_track(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path | str,
    *,
    title: str | None = None,
    show: bool = False
) -> Path:
    """Simple plot of raw track (y vs x)."""
    save_path = _ensure_parent(save_path)
    plt.figure()
    plt.plot(x, y, linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_detrended_with_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks_idx: Iterable[int],
    save_path: Path | str,
    *,
    degree: int = 1,
    ransac_kwargs: Optional[dict] = None,
    title: Optional[str] = None,
    show: bool = False
) -> Path:
    """
    Plot raw y, robust baseline (RANSAC poly), residual, and mark peaks.

    If you already have a residual, you can pass y as the residual and set
    ransac_kwargs={'skip_baseline': True} to disable baseline overlay.
    """
    save_path = _ensure_parent(save_path)
    ransac_kwargs = ransac_kwargs or {}
    skip_baseline = ransac_kwargs.pop("skip_baseline", False)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if skip_baseline:
        residual = y
        baseline = None
    else:
        model = fit_baseline_ransac(x, y, degree=degree, **ransac_kwargs)
        baseline = model.predict(x.reshape(-1, 1))
        residual = y - baseline

    plt.figure(figsize=(8, 5))
    # raw and baseline
    if not skip_baseline:
        plt.plot(x, y, linewidth=1, label="raw")
        plt.plot(x, baseline, linewidth=1, linestyle="--", label="baseline")
    # residual
    plt.plot(x, residual, linewidth=1, label="residual")

    peaks_idx = np.asarray(list(peaks_idx), dtype=int)
    if peaks_idx.size:
        plt.scatter(x[peaks_idx], residual[peaks_idx], s=20, label="peaks")

    plt.xlabel("x")
    plt.ylabel("value")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_spectrum(
    residual: np.ndarray,
    sampling_rate: float,
    save_path: Path | str,
    *,
    title: Optional[str] = None,
    show: bool = False
) -> Path:
    """Plot single-sided magnitude spectrum of residual."""
    save_path = _ensure_parent(save_path)
    df = spectrum_dataframe(residual, sampling_rate)
    # zero DC for display
    df = df.copy()
    if len(df):
        df.loc[df.index == 0, "magnitude"] = 0

    plt.figure(figsize=(8, 4))
    plt.plot(df["frequency"].values, df["magnitude"].values, linewidth=1)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_summary_histograms(
    metrics_df: pd.DataFrame,
    save_dir: Path | str,
    *,
    bins: int = 20,
    show: bool = False
) -> dict[str, Path]:
    """
    Plot histograms for key metrics in the aggregated output CSV.
    Returns a dict of figure label -> output path.
    """
    save_dir = ensure_dir(save_dir)
    out_paths: dict[str, Path] = {}
    columns = [
        ("dominant_frequency", "Dominant Frequency"),
        ("period", "Period"),
        ("mean_amplitude", "Mean Amplitude"),
        ("num_peaks", "Num Peaks"),
    ]
    for col, label in columns:
        if col not in metrics_df.columns:
            continue
        fig_path = Path(save_dir) / f"hist_{col}.png"
        plt.figure(figsize=(6, 4))
        series = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            plt.text(0.5, 0.5, f"No data for {label}", ha='center', va='center')
        else:
            plt.hist(series.values, bins=bins)
            plt.xlabel(label)
            plt.ylabel("Count")
            plt.title(f"Distribution of {label}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        if show:
            plt.show()
        plt.close()
        out_paths[col] = fig_path
    return out_paths
