import numpy as np
import pandas as pd
from scipy.signal import find_peaks, cwt, ricker
from typing import Tuple, Optional, Dict, Sequence


def _mad(a: np.ndarray) -> float:
    """Median absolute deviation (robust scale)."""
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def detect_peaks(
    signal: np.ndarray,
    prominence: float = 1.0,
    width: float = 1,
    distance: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks in a 1D signal using scipy.find_peaks.

    Args:
        signal: 1D array of residual signal values.
        prominence: required prominence of peaks.
        width: required MINIMUM width of peaks (in samples).
        distance: minimum horizontal distance (in samples) between peaks.

    Returns:
        peaks: indices of peaks in the signal.
        properties: dict of peak properties from scipy.
    """
    kwargs = {"prominence": float(prominence), "width": float(width)}
    if distance is not None:
        kwargs["distance"] = int(distance)

    peaks, properties = find_peaks(np.asarray(signal, dtype=float), **kwargs)
    return peaks, properties


def detect_peaks_adaptive(
    signal: np.ndarray,
    *,
    frames_per_period: Optional[float] = None,
    distance_frac: float = 0.6,          # min distance = distance_frac * period
    width_frac: float = 0.2,             # min width    = width_frac    * period
    rel_mad_k: float = 2.0,              # prom >= max(abs_min_prom_px, rel_mad_k*MAD)
    abs_min_prom_px: float = 1.0,
    nms_enable: bool = True,
    nms_dominance_frac: float = 0.55,    # keep stronger if neighbor < frac*strong
) -> Tuple[np.ndarray, dict]:
    """
    Period-aware peak detector. If frames_per_period is provided, it sets:
      - distance >= distance_frac * frames_per_period
      - width    >= width_frac    * frames_per_period
    Prominence threshold = max(abs_min_prom_px, rel_mad_k * MAD(signal)).

    After detection, a simple NMS will suppress nearby peaks within ~0.5 period,
    dominated by a stronger neighbor (based on 'prominences').

    Returns (peaks_idx, props) where props are aligned to peaks after NMS.
    """
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    if n == 0:
        return np.array([], dtype=int), {}

    # Robust amplitude scale -> adaptive prominence floor
    prom_floor = max(float(abs_min_prom_px), float(rel_mad_k) * _mad(sig))

    # Period-derived constraints
    if frames_per_period and np.isfinite(frames_per_period) and frames_per_period > 0:
        min_dist = max(1, int(round(distance_frac * frames_per_period)))
        min_width = max(1, int(round(width_frac * frames_per_period)))
    else:
        # Mild defaults if period is unknown
        min_dist = max(1, int(round(0.05 * n)))
        min_width = 1

    peaks, props = find_peaks(sig, prominence=prom_floor, width=min_width, distance=min_dist)

    if peaks.size == 0:
        return peaks, props

    if nms_enable:
        # Strength = prominence if available; else use signal height at peaks
        if "prominences" in props and len(props["prominences"]) == len(peaks):
            strength = np.asarray(props["prominences"], dtype=float)
        else:
            strength = sig[peaks].astype(float, copy=False)

        # NMS window ~ half period
        if frames_per_period and np.isfinite(frames_per_period) and frames_per_period > 0:
            win = max(1, int(round(0.5 * frames_per_period)))
        else:
            win = max(1, int(round(0.03 * n)))

        keep_mask = _nms_1d_by_index(peaks, strength, window=win, dominance_frac=float(nms_dominance_frac))
        if not np.all(keep_mask):
            peaks = peaks[keep_mask]
            # Filter properties that are aligned to peaks
            props = {k: (np.asarray(v)[keep_mask] if hasattr(v, "__len__") and len(v) == len(keep_mask) else v)
                     for k, v in props.items()}

    return peaks, props


def _nms_1d_by_index(
    indices: np.ndarray,
    strength: np.ndarray,
    *,
    window: int,
    dominance_frac: float = 0.55
) -> np.ndarray:
    """
    Greedy 1D NMS: iterate peaks by descending strength, keep a peak and
    suppress neighbors within +/- window samples if they are weaker than
    dominance_frac * current_strength. Returns boolean keep mask.
    """
    idx = np.asarray(indices, dtype=int)
    s = np.asarray(strength, dtype=float)
    order = np.argsort(-s)  # descending
    keep = np.ones(idx.shape[0], dtype=bool)

    for o in order:
        if not keep[o]:
            continue
        i0 = idx[o]
        s0 = s[o]
        # Suppress neighbors that are both close and dominated
        close = np.where(np.abs(idx - i0) <= int(window))[0]
        for j in close:
            if j == o:
                continue
            if s[j] <= dominance_frac * s0:
                keep[j] = False
    return keep


def detect_peaks_cwt(
    signal: np.ndarray,
    widths: np.ndarray = np.arange(1, 10),
    wavelet: str = 'ricker'
) -> np.ndarray:
    """
    Detect peaks using Continuous Wavelet Transform (CWT) across multiple scales.

    Args:
        signal: 1D array of residual signal values.
        widths: array of widths to use for CWT.
        wavelet: type of wavelet, currently only 'ricker' supported.

    Returns:
        peaks_cwt: indices of peaks detected by CWT.
    """
    if wavelet != 'ricker':
        raise ValueError("Only 'ricker' wavelet is supported currently.")

    cwt_matrix = cwt(np.asarray(signal, dtype=float), ricker, widths)
    # Sum across scales to find aggregated response
    aggregated = np.mean(cwt_matrix, axis=0)
    # Peaks in aggregated CWT response
    peaks_cwt, _ = find_peaks(aggregated)
    return peaks_cwt


def peaks_to_dataframe(
    x: np.ndarray,
    signal: np.ndarray,
    peaks: np.ndarray,
    properties: Optional[dict] = None
) -> pd.DataFrame:
    """
    Build a DataFrame listing peak positions and their signal values.

    Args:
        x: 1D array of x-axis values (e.g., frame indices or time).
        signal: 1D array of residual signal values.
        peaks: indices of detected peaks.
        properties: optional dict of properties from detect_peaks.

    Returns:
        df_peaks: DataFrame with columns ['index', 'x', 'value', ...properties]
    """
    x = np.asarray(x)
    sig = np.asarray(signal)
    peaks = np.asarray(peaks, dtype=int)

    data: Dict[str, np.ndarray] = {
        'index': peaks,
        'x': x[peaks] if peaks.size else np.array([], dtype=x.dtype),
        'value': sig[peaks] if peaks.size else np.array([], dtype=sig.dtype),
    }

    if properties:
        for key, vals in properties.items():
            vals = np.asarray(vals)
            # If property is already per-peak, lengths match peaks
            if vals.ndim == 1 and vals.shape[0] == peaks.shape[0]:
                data[key] = vals
            # If property is per-sample (len == len(signal)), index it by peaks
            elif vals.ndim == 1 and vals.shape[0] == sig.shape[0]:
                data[key] = vals[peaks]
            # Otherwise, skip (mismatched shape)
    return pd.DataFrame(data)
