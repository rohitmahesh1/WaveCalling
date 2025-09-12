# src/features.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------
# ID helpers
# -----------------------

def sample_id_from_name(name: str) -> int:
    """Stable small positive int from a sample name."""
    # stable across runs (python's hash is salted), so use a simple FNV-like hash
    h = 2166136261
    for ch in name:
        h ^= ord(ch)
        h *= 16777619
        h &= 0xFFFFFFFF
    return int(h % 10_000_000)


def coerce_track_id(stem: str) -> Optional[int]:
    """Return int(stem) if possible, else None."""
    try:
        return int(stem)
    except Exception:
        return None


# -----------------------
# Geometry & descriptors
# -----------------------

def segment_bbox(x: np.ndarray, y: np.ndarray, i: int, j: int) -> Tuple[float, float, float, float]:
    """Bounding box for y over x segment [i:j] inclusive."""
    if j < i:
        i, j = j, i
    xs = x[i:j + 1]
    ys = y[i:j + 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def orientation_deg(x_seg: np.ndarray, y_seg: np.ndarray) -> Tuple[float, float]:
    """
    Orientation of the segment in degrees:
      angle = arctan(|slope|) where slope = dy/dx from a linear fit y ~ a*x+b
      0° ~ vertical trace (little lateral motion per frame),
      larger angles ~ more diagonal ("surf").
    Also returns a crude std from local finite-difference slopes.
    """
    x = np.asarray(x_seg).ravel()
    y = np.asarray(y_seg).ravel()
    if x.size < 2:
        return 0.0, 0.0

    # linear fit slope (dy/dx)
    try:
        a, b = np.polyfit(x, y, deg=1)
        slope = float(a)
    except Exception:
        slope = 0.0

    angle = float(np.degrees(np.arctan(np.abs(slope))))

    # local slope std
    dx = np.diff(x)
    dy = np.diff(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_slopes = np.where(np.abs(dx) > 0, dy / dx, 0.0)
    angle_std = float(np.degrees(np.std(np.arctan(np.abs(local_slopes))))) if local_slopes.size else 0.0
    return angle, angle_std


# -----------------------
# Bulge metrics from find_peaks props
# -----------------------

def _peak_prop_at_index(peaks_idx: np.ndarray, props: Dict[str, np.ndarray], peak_i: int) -> Dict[str, float]:
    """Helper: grab aligned props (prominence, width, etc.) for a specific peak index."""
    out = {}
    if peaks_idx is None or props is None:
        return out
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    try:
        pos = int(np.where(peaks_idx == peak_i)[0][0])
    except Exception:
        return out
    for k, v in props.items():
        try:
            out[k] = float(np.asarray(v)[pos])
        except Exception:
            pass
    return out


def bulge_from_props(
    peak_i: int,
    peaks_idx: np.ndarray,
    props: Dict[str, np.ndarray],
    sampling_rate: float,
) -> Dict[str, float]:
    """
    Return dict with bulge prominence and width for a given peak index.
    Width is expressed in frames and seconds if available.
    """
    md = _peak_prop_at_index(peaks_idx, props, peak_i)
    prom = float(md.get("prominences", np.nan))
    width_frames = float(md.get("widths", np.nan))
    width_s = (width_frames / sampling_rate) if (sampling_rate and np.isfinite(width_frames)) else np.nan
    return {
        "bulge_prominence_px": prom,
        "bulge_width_frames": width_frames,
        "bulge_width_s": width_s,
    }


# -----------------------
# Anchored sine fit (around a peak)
# -----------------------

def anchored_sine_params(
    residual: np.ndarray,
    x: np.ndarray,
    sampling_rate: float,
    freq: float,
    center_idx: int,
    period_frac: float = 0.5,
) -> Dict[str, float]:
    """
    Fit residual with a fixed frequency sine, PHASE-ANCHORED at center_idx (peak maximum).
    Compute vNMSE inside a window of width = period_frac * period centered on center_idx.

    Returns dict with A, phi, c, f, vNMSE, and window [lo, hi] (indices).
    """
    out = {
        "fit_amp_A": np.nan,
        "fit_phase_phi": np.nan,
        "fit_offset_c": np.nan,
        "fit_freq_hz": float(freq) if freq is not None else np.nan,
        "fit_error_vnmse": np.nan,
        "fit_window_lo": np.nan,
        "fit_window_hi": np.nan,
    }
    if sampling_rate is None or freq is None or freq <= 0 or center_idx < 0 or center_idx >= len(x):
        return out

    # Lazy import to avoid circular deps
    from .visualize import _fit_global_sine

    yfit_res, A, phi, c = _fit_global_sine(
        residual, x, sampling_rate, freq, center_peak_idx=int(center_idx)
    )
    if yfit_res is None:
        return out

    # Window in frames (indices), centered at center_idx
    frames_per_period = sampling_rate / float(freq)
    half_span = max(1, int(round((period_frac * frames_per_period) / 2.0)))
    lo = max(0, int(center_idx) - half_span)
    hi = min(len(x) - 1, int(center_idx) + half_span)

    y_slice = residual[lo:hi + 1]
    y_fit = yfit_res[lo:hi + 1]
    if y_slice.size >= 2 and np.var(y_slice) > 0:
        vnmse = float(np.mean((y_slice - y_fit) ** 2) / np.var(y_slice))
    else:
        vnmse = np.nan

    out.update({
        "fit_amp_A": float(A) if A is not None else np.nan,
        "fit_phase_phi": float(phi) if phi is not None else np.nan,
        "fit_offset_c": float(c) if c is not None else np.nan,
        "fit_error_vnmse": vnmse,
        "fit_window_lo": float(lo),
        "fit_window_hi": float(hi),
    })
    return out


# -----------------------
# Type heuristic
# -----------------------

def classify_wave_type(
    angle_deg: float,
    prominence_px: float,
    cfg: Optional[dict] = None,
) -> Tuple[str, float]:
    """
    Simple rule-based:
      ripple if angle <= ripple_max_deg and prominence >= prom_min_px
      surf if angle >= surf_min_deg
      else ambiguous
    Returns (label, score in [0,1]).
    """
    cfg = cfg or {}
    ripple_max = float(cfg.get("ripple_max_deg", 10.0))
    surf_min = float(cfg.get("surf_min_deg", 20.0))
    prom_min = float(cfg.get("prominence_min_px", 1.0))

    if np.isfinite(angle_deg) and np.isfinite(prominence_px):
        if angle_deg <= ripple_max and prominence_px >= prom_min:
            # score increases as angle is small and prom large
            score = float(max(0.0, min(1.0, (prominence_px / (prom_min + 1e-6)) * (1.0 - angle_deg / (ripple_max + 1e-6)) )))
            return "ripple", min(1.0, score)
        if angle_deg >= surf_min:
            score = float(max(0.0, min(1.0, (angle_deg - surf_min) / (90.0 - surf_min))))
            return "surf", score

    return "ambiguous", 0.5


# -----------------------
# Row builders
# -----------------------

def _local_period_frames_from_peaks(peaks_idx: np.ndarray, k: int) -> Optional[float]:
    """
    Estimate local period (in frames) for peak index p[k] using nearby gaps.
    Uses the median of {p[k]-p[k-1], p[k+1]-p[k]} when available.
    """
    p = np.asarray(peaks_idx, dtype=int)
    if p.size == 0 or k < 0 or k >= p.size:
        return None
    gaps: List[float] = []
    if k - 1 >= 0:
        gaps.append(float(p[k] - p[k - 1]))
    if k + 1 < p.size:
        gaps.append(float(p[k + 1] - p[k]))
    if not gaps:
        return None
    return float(np.median(gaps))


def build_peak_rows(
    *,
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    peak_props: dict,
    sampling_rate: float,
    sample: str,
    track_stem: str,
    features_cfg: Optional[dict] = None,
    global_freq_hz: float | None = None,
    period_frac_for_fit: float = 0.5,
) -> List[dict]:
    """
    Build one row PER PEAK with:
      - IDs
      - peak frame & position, amplitude (residual at peak)
      - local period (frames,s) from neighbor gaps (fallback to global)
      - local frequency (Hz)
      - bulge metrics from find_peaks props
      - phase-anchored sine fit params around this peak (vNMSE + window)
      - optional local orientation over the fit window
    """
    rows: List[dict] = []
    features_cfg = features_cfg or {}

    p = np.asarray(peaks_idx, dtype=int)
    if p.size == 0:
        return rows

    sample_id = sample_id_from_name(sample)
    maybe_track_id = coerce_track_id(track_stem)

    # Global fallback frames/period if needed
    global_fpp = (sampling_rate / float(global_freq_hz)) if (sampling_rate and global_freq_hz and global_freq_hz > 0) else None

    for idx_in_list, peak_i in enumerate(p):
        frame = float(x[peak_i])
        pos = float(y[peak_i])
        amp = float(residual[peak_i])

        # Local period estimate from neighbor gaps; fallback to global
        local_fpp = _local_period_frames_from_peaks(p, idx_in_list)
        frames_per_period = local_fpp if (local_fpp and local_fpp > 0) else (global_fpp if (global_fpp and global_fpp > 0) else np.nan)
        period_frames = float(frames_per_period) if np.isfinite(frames_per_period) else np.nan
        period_s = (period_frames / sampling_rate) if (sampling_rate and np.isfinite(period_frames)) else np.nan
        freq_hz = (1.0 / period_s) if (np.isfinite(period_s) and period_s > 0) else (float(global_freq_hz) if (global_freq_hz and global_freq_hz > 0) else np.nan)

        # Bulge metrics
        bulge = bulge_from_props(peak_i, p, peak_props or {}, sampling_rate)

        # Anchored sine fit (phase maximum at this peak)
        fit = anchored_sine_params(
            residual=residual,
            x=x,
            sampling_rate=sampling_rate,
            freq=freq_hz if (freq_hz and freq_hz > 0) else (global_freq_hz or np.nan),
            center_idx=int(peak_i),
            period_frac=float(features_cfg.get("fit_window_period_frac", period_frac_for_fit)),
        )

        # Local orientation over the fit window if available
        lo = fit.get("fit_window_lo", np.nan)
        hi = fit.get("fit_window_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            i0, i1 = int(max(0, lo)), int(min(len(x) - 1, hi))
            ang_mean, ang_std = orientation_deg(x[i0:i1 + 1], y[i0:i1 + 1])
        else:
            ang_mean, ang_std = (np.nan, np.nan)

        rows.append({
            # IDs
            "Sample": sample,
            "sample_id": int(sample_id),
            "Track": maybe_track_id if maybe_track_id is not None else track_stem,
            "track_id": int(maybe_track_id) if maybe_track_id is not None else np.nan,
            "Peak index": idx_in_list + 1,            # 1-based within the track
            "Peak frame": frame,
            "Peak position (px)": pos,
            "Amplitude (pixels)": amp,

            # Local period/freq
            "Local period (frames)": period_frames,
            "Local period (s)": period_s,
            "Local frequency (Hz)": freq_hz,

            # Bulge metrics (from scipy props)
            "bulge_prominence_px": bulge.get("bulge_prominence_px", np.nan),
            "bulge_width_frames": bulge.get("bulge_width_frames", np.nan),
            "bulge_width_s": bulge.get("bulge_width_s", np.nan),

            # Anchored fit params & quality
            "fit_amp_A": fit.get("fit_amp_A", np.nan),
            "fit_phase_phi": fit.get("fit_phase_phi", np.nan),
            "fit_offset_c": fit.get("fit_offset_c", np.nan),
            "fit_freq_hz": fit.get("fit_freq_hz", np.nan),
            "fit_error_vnmse": fit.get("fit_error_vnmse", np.nan),
            "fit_window_lo": fit.get("fit_window_lo", np.nan),
            "fit_window_hi": fit.get("fit_window_hi", np.nan),

            # Local orientation in the fit window (diagnostic)
            "orientation_deg": ang_mean,
            "orientation_std_deg": ang_std,
        })

    return rows


def build_wave_rows(
    *,
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    peak_props: dict,
    sampling_rate: float,
    sample: str,
    track_stem: str,
    features_cfg: Optional[dict] = None,
    freq_hz: float | None = None,
    period_frac_for_fit: float = 0.5,
) -> List[dict]:
    """
    Build per-wave rows from consecutive peak pairs, enriched with:
      - IDs, bbox, orientation, bulge metrics
      - anchored sine fit params & vNMSE in a local window
      - type heuristic (ripple/surf/ambiguous)
    """
    features_cfg = features_cfg or {}
    rows: List[dict] = []

    p = np.asarray(peaks_idx, dtype=int)
    if p.size < 2:
        return rows

    sample_id = sample_id_from_name(sample)
    maybe_track_id = coerce_track_id(track_stem)

    for k in range(p.size - 1):
        i, j = int(p[k]), int(p[k + 1])

        frame1 = float(x[i])
        frame2 = float(x[j])
        period_frames = frame2 - frame1
        period_s = period_frames / sampling_rate if sampling_rate else float("nan")
        freq = (1.0 / period_s) if (period_s and period_s > 0) else (float(freq_hz) if freq_hz else np.nan)

        pos1 = float(y[i])         # raw pixel position at first peak
        pos2 = float(y[j])         # raw pixel position at second peak
        amp = float(residual[i])   # amplitude at first peak (residual height)

        dpos = pos2 - pos1
        vel = (dpos / period_s) if (period_s and period_s != 0) else float("nan")
        wavelength = abs(dpos)

        # bbox & orientation over segment
        xmin, xmax, ymin, ymax = segment_bbox(x, y, i, j)
        ang_mean, ang_std = orientation_deg(x[int(min(i,j)):int(max(i,j))+1], y[int(min(i,j)):int(max(i,j))+1])

        # bulge metrics from find_peaks props (at i)
        bulge = bulge_from_props(i, p, peak_props or {}, sampling_rate)

        # anchored sine fit params around i
        fit = anchored_sine_params(
            residual=residual,
            x=x,
            sampling_rate=sampling_rate,
            freq=freq if np.isfinite(freq) else (freq_hz or np.nan),
            center_idx=i,
            period_frac=float(features_cfg.get("fit_window_period_frac", period_frac_for_fit)),
        )

        # wave type heuristic
        wlabel, wscore = classify_wave_type(
            angle_deg=ang_mean,
            prominence_px=float(bulge.get("bulge_prominence_px", np.nan)),
            cfg=features_cfg.get("classify", {}),
        )

        # IDs
        wave_id = f"{maybe_track_id if maybe_track_id is not None else track_stem}-{k+1}"

        rows.append({
            # Original schema (kept for compatibility)
            "Sample": sample,
            "Track": maybe_track_id if maybe_track_id is not None else track_stem,
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

            # New IDs
            "sample_id": int(sample_id),
            "track_id": int(maybe_track_id) if maybe_track_id is not None else np.nan,
            "wave_id": wave_id,

            # Geometry
            "bbox_xmin": xmin, "bbox_xmax": xmax, "bbox_ymin": ymin, "bbox_ymax": ymax,
            "orientation_deg": ang_mean,
            "orientation_std_deg": ang_std,

            # Bulge metrics
            "bulge_prominence_px": bulge.get("bulge_prominence_px", np.nan),
            "bulge_width_frames": bulge.get("bulge_width_frames", np.nan),
            "bulge_width_s": bulge.get("bulge_width_s", np.nan),

            # Anchored sine fit params
            "fit_amp_A": fit.get("fit_amp_A", np.nan),
            "fit_phase_phi": fit.get("fit_phase_phi", np.nan),
            "fit_offset_c": fit.get("fit_offset_c", np.nan),
            "fit_freq_hz": fit.get("fit_freq_hz", np.nan),
            "fit_error_vnmse": fit.get("fit_error_vnmse", np.nan),
            "fit_window_lo": fit.get("fit_window_lo", np.nan),
            "fit_window_hi": fit.get("fit_window_hi", np.nan),

            # Type
            "wave_type": wlabel,
            "type_score": wscore,
        })

    return rows
