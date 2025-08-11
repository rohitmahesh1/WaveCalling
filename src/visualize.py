from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

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


def _fit_global_sine(
    residual: np.ndarray,
    x: np.ndarray,
    sampling_rate: float,
    freq: float,
    amp_override: float | None = None,
    phase_override: float | None = None,
    center_peak_idx: int | None = None,
):
    """
    Fit residual with a fixed frequency sine.

    If center_peak_idx is provided, choose phase so that the sine's MAXIMUM
    occurs exactly at that peak: phi = pi/2 - w * t_peak. Then fit only A and c:
        residual ~ A * sin(w t + phi_fixed) + c
    Otherwise, use the unconstrained (a*sin + b*cos + c) least-squares parameterization.

    Returns: (y_fit_residual, A, phi, c)
    """
    if sampling_rate is None or freq is None or freq <= 0:
        return None, None, None, None

    t = x / float(sampling_rate)
    w = 2.0 * np.pi * float(freq)

    # Phase-anchored fit (peak-centered)
    if center_peak_idx is not None and 0 <= center_peak_idx < len(x):
        t_peak = float(t[center_peak_idx])
        phi = (np.pi / 2.0) - w * t_peak  # maximum at t_peak
        S = np.column_stack([np.sin(w * t + phi), np.ones_like(t)])  # solve for [A, c]
        beta, *_ = np.linalg.lstsq(S, residual, rcond=None)
        A, c = float(beta[0]), float(beta[1])

        # Ensure A >= 0 so that t_peak is indeed a maximum; if A<0, flip amp and shift phase by pi
        if A < 0:
            A = -A
            phi += np.pi

        # Optional overrides
        if amp_override is not None:
            A = float(amp_override)
        if phase_override is not None:
            phi = float(phase_override)

        yfit_res = A * np.sin(w * t + phi) + c
        return yfit_res, float(A), float(phi), float(c)

    # Unconstrained global fit (fallback)
    X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(X, residual, rcond=None)
    a, b, c = map(float, beta)
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
        A_now = float(np.hypot(a, b))
        A_use = A if amp_override is not None else A_now
        a = A_use * np.cos(float(phase_override))
        b = A_use * np.sin(float(phase_override))
        phi = float(phase_override)

    yfit_res = X @ np.array([a, b, c])
    return yfit_res, float(A), float(phi), float(c)


def plot_peak_windows(
    x: np.ndarray,
    y: np.ndarray,
    peaks_idx: Iterable[int],
    save_dir: Path | str,
    *,
    degree: int = 1,
    ransac_kwargs: Optional[dict] = None,
    sampling_rate: float | None = None,
    freq: float | None = None,
    period_frac: float = 0.5,     # fraction of ONE period to show (total width)
    max_plots: int = 12,          # avoid hundreds of files
    dpi: int | None = None,
    title_prefix: Optional[str] = None,
    overlay_fit: bool = True,
) -> List[Path]:
    """
    For each peak, plot a window of width = period_frac * period centered on that peak.
    Overlays baseline, residual, peak marker, and (optionally) baseline+sine fit
    that is PHASE-ANCHORED at that peak.
    Returns the list of saved file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    peaks_idx = np.asarray(list(peaks_idx), dtype=int)
    if peaks_idx.size == 0:
        return []

    # Fit baseline once; compute residual once
    ransac_kwargs = (ransac_kwargs or {}).copy()
    skip_baseline = ransac_kwargs.pop("skip_baseline", False)
    if skip_baseline:
        baseline = np.zeros_like(y)
        residual = y
    else:
        model = fit_baseline_ransac(x, y, degree=degree, **ransac_kwargs)
        baseline = model.predict(x.reshape(-1, 1))
        residual = y - baseline

    # Frames per period (for window size)
    if sampling_rate and freq and freq > 0:
        frames_per_period = sampling_rate / float(freq)
    else:
        frames_per_period = 40.0  # fallback window width
    half_span = max(1, int(round((period_frac * frames_per_period) / 2.0)))

    saved: List[Path] = []
    for j, pk in enumerate(peaks_idx[:max_plots]):
        lo = max(0, int(pk) - half_span)
        hi = min(len(x) - 1, int(pk) + half_span)

        xs = x[lo:hi + 1]
        raw = y[lo:hi + 1]
        base = baseline[lo:hi + 1]
        res = residual[lo:hi + 1]

        # Phase-anchored local overlay (baseline + sine centered at this peak)
        yfit_seg = None
        if overlay_fit and sampling_rate and freq and freq > 0:
            yfit_res_local, _, _, _ = _fit_global_sine(
                residual, x, sampling_rate, freq,
                center_peak_idx=int(pk)
            )
            if yfit_res_local is not None:
                yfit_seg = (baseline + yfit_res_local)[lo:hi + 1]

        # Plot with axes flipped (X=position, Y=time)
        plt.figure(figsize=(6, 4))
        plt.plot(raw, xs, linewidth=1, label="raw")
        plt.plot(base, xs, linewidth=1, linestyle="--", label="baseline")
        plt.plot(res, xs, linewidth=1, label="residual")
        plt.scatter(residual[pk], x[pk], s=25, label="peak")
        if yfit_seg is not None:
            plt.plot(yfit_seg, xs, linewidth=1.2, alpha=0.95, label="sine fit (anchored)")

        plt.xlabel("Position")
        plt.ylabel("Time")
        ttl = f"{title_prefix or 'track'} – peak {j} @ idx {int(pk)}"
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()

        out = save_dir / f"peak_{j:03d}_window.png"
        plt.savefig(out, dpi=(dpi if dpi is not None else 180))
        plt.close()
        saved.append(out)

    return saved


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
    # overlay controls
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
    also draw (baseline + global sine fit) over the raw track, with phase
    **anchored at the strongest detected peak**.
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

    # optional global-sine overlay: (baseline + fitted residual), anchored
    if overlay_fit and (sampling_rate is not None) and (freq is not None) and freq > 0:
        center_idx = None
        if peaks_idx.size:
            # anchor to the strongest residual peak
            center_idx = int(peaks_idx[np.argmax(residual[peaks_idx])])
        yfit_res, A, phi, c = _fit_global_sine(
            residual, x, sampling_rate, freq,
            amp_override=amp_override, phase_override=phase_override,
            center_peak_idx=center_idx
        )
        if yfit_res is not None:
            yfit = (baseline if baseline is not None else 0.0) + yfit_res
            # axes flipped
            plt.plot(yfit, x, linewidth=1.2, alpha=0.9, label=f"sine fit (anchored) f={freq:.3g}Hz")

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
