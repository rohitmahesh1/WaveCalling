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
    show: bool = False,
    dpi: int | None = None,
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
    plt.savefig(save_path, dpi=(dpi if dpi is not None else 180))
    if show:
        plt.show()
    plt.close()
    return save_path


def _fit_global_sine(residual: np.ndarray,
                     x: np.ndarray,
                     sampling_rate: float,
                     freq: float,
                     amp_override: float | None = None,
                     phase_override: float | None = None):
    """
    Fit residual ~ a*sin(wt) + b*cos(wt) + c at a fixed frequency.
    Returns (y_fit_residual, A, phi, c).
    """
    if sampling_rate is None or freq is None or freq <= 0:
        return None, None, None, None

    t = x / float(sampling_rate)
    w = 2.0 * np.pi * float(freq)

    X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    # Least-squares solve
    beta, *_ = np.linalg.lstsq(X, residual, rcond=None)
    a, b, c = beta
    A = float(np.hypot(a, b))
    phi = float(np.arctan2(b, a))

    # Optional overrides
    if amp_override is not None and A > 0:
        scale = float(amp_override) / A
        a *= scale
        b *= scale
        A = float(amp_override)
        phi = float(np.arctan2(b, a))
    if phase_override is not None:
        # enforce requested phase with current amplitude
        A_now = float(np.hypot(a, b))
        if A_now == 0 and (amp_override is None):
            A_now = 0.0
        A_use = A if amp_override is not None else A_now
        a = A_use * np.cos(float(phase_override))
        b = A_use * np.sin(float(phase_override))
        phi = float(phase_override)

    yfit_res = X @ np.array([a, b, c])
    return yfit_res, A, phi, float(c)


def plot_detrended_with_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks_idx: Iterable[int],
    save_path: Path | str,
    *,
    degree: int = 1,
    ransac_kwargs: Optional[dict] = None,
    title: Optional[str] = None,
    show: bool = False,
    dpi: int | None = None,
    # NEW: overlay controls
    overlay_fit: bool = False,
    sampling_rate: float | None = None,
    freq: float | None = None,
    amp_override: float | None = None,
    phase_override: float | None = None,
) -> Path:
    """
    Plot raw y, robust baseline (RANSAC poly), residual, and mark peaks,
    with axes flipped so X = position, Y = time.

    If overlay_fit is True (and sampling_rate & freq are provided),
    also draw (baseline + global sine fit) over the raw track.
    """
    save_path = _ensure_parent(save_path)
    ransac_kwargs = (ransac_kwargs or {}).copy()
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

    # raw & baseline (axes flipped)
    if not skip_baseline:
        plt.plot(y, x, linewidth=1, label="raw")
        plt.plot(baseline, x, linewidth=1, linestyle="--", label="baseline")

    # residual (axes flipped)
    plt.plot(residual, x, linewidth=1, label="residual")

    # peaks
    peaks_idx = np.asarray(list(peaks_idx), dtype=int)
    if peaks_idx.size:
        plt.scatter(residual[peaks_idx], x[peaks_idx], s=20, label="peaks")

    # optional global-sine overlay: (baseline + fitted residual)
    if overlay_fit and (sampling_rate is not None) and (freq is not None) and freq > 0:
        yfit_res, A, phi, c = _fit_global_sine(
            residual, x, sampling_rate, freq,
            amp_override=amp_override, phase_override=phase_override
        )
        if yfit_res is not None:
            yfit = (baseline if baseline is not None else 0.0) + yfit_res
            # axes flipped
            plt.plot(yfit, x, linewidth=1.2, alpha=0.9, label=f"sine fit f={freq:.3g}Hz")

    plt.xlabel("Position")
    plt.ylabel("Time")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=(dpi if dpi is not None else 180))
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
    show: bool = False,
    dpi: int | None = None,
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
    plt.savefig(save_path, dpi=(dpi if dpi is not None else 180))
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_summary_histograms(
    metrics_df: pd.DataFrame,
    save_dir: Path | str,
    *,
    bins: int = 20,
    show: bool = False,
    dpi: int | None = None,
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
        plt.savefig(fig_path, dpi=(dpi if dpi is not None else 160))
        if show:
            plt.show()
        plt.close()
        out_paths[col] = fig_path
    return out_paths
