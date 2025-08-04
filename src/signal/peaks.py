import numpy as np
import pandas as pd
from scipy.signal import find_peaks, cwt, ricker
from typing import Tuple


def detect_peaks(
    signal: np.ndarray,
    prominence: float = 1.0,
    width: float = 1,
    distance: int = None
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks in a 1D signal using scipy.find_peaks.

    Args:
        signal: 1D array of residual signal values.
        prominence: required prominence of peaks.
        width: required width of peaks (in samples).
        distance: required minimal horizontal distance (in samples) between peaks.

    Returns:
        peaks: indices of peaks in the signal.
        properties: dict of peak properties from scipy.
    """
    kwargs = {"prominence": prominence, "width": width}
    if distance is not None:
        kwargs["distance"] = distance

    peaks, properties = find_peaks(signal, **kwargs)
    return peaks, properties


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

    cwt_matrix = cwt(signal, ricker, widths)
    # Sum across scales to find aggregated response
    aggregated = np.mean(cwt_matrix, axis=0)
    # Peaks in aggregated CWT response
    peaks_cwt, _ = find_peaks(aggregated)
    return peaks_cwt


def peaks_to_dataframe(
    x: np.ndarray,
    signal: np.ndarray,
    peaks: np.ndarray,
    properties: dict = None
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
    data = {
        'index': peaks,
        'x': x[peaks],
        'value': signal[peaks]
    }
    if properties:
        # flatten properties into columns
        for key, vals in properties.items():
            data[key] = np.array(vals)[peaks] if len(vals) == len(signal) else vals
    return pd.DataFrame(data)
