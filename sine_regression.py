import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema, lombscargle
import scipy.optimize

Y_LIM_RANGE = 60


def merge_close_indices(indices, data, min_distance):
    """
    Merge extrema indices that are closer than min_distance,
    keeping the index with the most extreme data value in each group.
    """
    if len(indices) < 2:
        return indices

    merged = []
    group = [indices[0]]
    for idx in indices[1:]:
        if idx - group[-1] < min_distance:
            group.append(idx)
        else:
            best = max(group, key=lambda x: data[x])
            merged.append(best)
            group = [idx]
    merged.append(max(group, key=lambda x: data[x]))
    return np.array(merged)


def detect_extrema(data, window_length=11, polyorder=3, min_distance=10):
    """
    Smooth data with Savitzky-Golay filter and detect local maxima and minima.
    Returns merged extrema indices and the smoothed data.
    """
    if window_length % 2 == 0:
        window_length += 1
    smoothed = savgol_filter(data, window_length, polyorder)

    maxima = argrelextrema(smoothed, np.greater)[0]
    minima = argrelextrema(smoothed, np.less)[0]
    combined = np.sort(np.concatenate((maxima, minima)))
    merged = merge_close_indices(combined, smoothed, min_distance)
    return merged, smoothed


def fit_sin(tt, yy, x_ext, y_ext):
    """
    Fit a sinusoid to (tt, yy) such that it passes through (x_ext, y_ext)
    and has zero derivative at x_ext (extremum constraint).
    Returns fit parameters and a fit function.
    """
    tt = np.array(tt)
    yy = np.array(yy)

    # Frequency estimate via Lomb-Scargle
    freqs = np.linspace(0.01, 1, 10000)
    power = lombscargle(tt, yy, freqs)
    guess_freq = freqs[np.argmax(power)]

    guess = [np.std(yy) * np.sqrt(2), 2 * np.pi * guess_freq, 0, np.mean(yy)]

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    def objective(params):
        return np.sum((sinfunc(tt, *params) - yy) ** 2)

    def deriv_constraint(params):
        A, w, p, _ = params
        return A * w * np.cos(w * x_ext + p)

    def value_constraint(params):
        return sinfunc(x_ext, *params) - y_ext

    cons = [
        {'type': 'eq', 'fun': deriv_constraint},
        {'type': 'eq', 'fun': value_constraint}
    ]

    res = scipy.optimize.minimize(objective, guess, constraints=cons, options={'maxiter':10000})
    if not res.success:
        raise RuntimeError(f"Sin fit failed: {res.message}")

    A, w, p, c = res.x
    freq = w / (2 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {
        'amp': A, 'omega': w, 'phase': p, 'offset': c,
        'freq': freq, 'period': 1/freq, 'fitfunc': fitfunc
    }


def perform_local_sin_regressions(x, y_smoothed, extrema_indices, radius,
                                  window_length=7, polyorder=3):
    """
    Around each extremum, select data within radius and fit a sinusoid.
    Returns a list of dicts with extremum info and fit results.
    """
    results = []
    for idx in extrema_indices:
        low, high = x[idx] - radius, x[idx] + radius
        mask = (x >= low) & (x <= high)
        x_slice = x[mask]
        y_slice = y_smoothed[mask]
        if len(x_slice) == 0:
            continue
        model = fit_sin(x_slice, y_slice, x[idx], y_smoothed[idx])
        results.append({
            'extremum_index': idx,
            'extremum_coordinates': (x[idx], y_smoothed[idx]),
            'x_slice': x_slice,
            'y_slice': y_slice,
            'fitted_model': model
        })
    return results


def get_plot_slices(result, period_frac=0.5):
    """
    Given a regression result, return x and y arrays spanning a fraction of period
    around the extremum for plotting.
    """
    model = result['fitted_model']
    ext_x, _ = result['extremum_coordinates']
    x0, x1 = result['x_slice'][[0, -1]]
    t = np.linspace(x0, x1, int((x1 - x0) / 0.1))
    y_fit = model['fitfunc'](t)

    period = abs(2 * np.pi / model['omega']) if model['omega'] != 0 else 0
    span = period * period_frac
    mask = (t >= ext_x - span) & (t <= ext_x + span)
    return t[mask], y_fit[mask]


def get_wave_characteristics(regression_results, idx_offset=0):
    """
    From local regression results, compute period, frequency, amplitude, velocity, etc.
    Returns a DataFrame summarizing each wave.
    """
    rows = []
    for i, res in enumerate(regression_results):
        model = res['fitted_model']
        x_s, y_s = get_plot_slices(res)
        ext_x, ext_y = res['extremum_coordinates']

        period = x_s[-1] - x_s[0]
        freq = 5 / period if period != 0 else 0
        amp = ext_y - y_s[0]
        vel = amp / ((ext_x - x_s[0]) / 5) if ext_x != x_s[0] else 0
        wl = vel / freq if freq != 0 else 0
        vnmse = ((res['y_slice'] - model['fitfunc'](res['x_slice']))**2).mean() / np.var(res['y_slice'])

        rows.append({
            'id': f'Wave {idx_offset + 1}-{i+1}',
            'period_frames': period,
            'period_s': period/5,
            'frequency_hz': freq,
            'amplitude_px': amp,
            'velocity_px_s': vel,
            'wavelength_px': wl,
            'vnmse': vnmse
        })
    return pd.DataFrame(rows)


def detect_and_plot_extrema(x, y, filename, kymo_idx):
    """
    Detect extrema in y over x, fit local sinusoids, plot results,
    and save plot to filename. Returns indices, smoothed y, raw y, and DataFrame.
    """
    x = np.array(x)
    y = np.array(y)

    extrema, y_sm = detect_extrema(y)
    if len(extrema) == 0:
        return None, None, None, pd.DataFrame()

    regressions = perform_local_sin_regressions(x, y_sm, extrema, radius=30)
    df = get_wave_characteristics(regressions, idx_offset=kymo_idx)

    # Determine plot limits with padding
    y_min, y_max = y.min(), y.max()
    s_min, s_max = y_sm.min(), y_sm.max()
    span = max(y_max, s_max) - min(y_min, s_min)
    pad = (Y_LIM_RANGE - span) / 2
    y0, y1 = min(y_min, s_min) - pad, max(y_max, s_max) + pad

    plt.figure(figsize=(10, 6))
    plt.plot(y, x, label='Noisy')
    plt.plot(y_sm, x, color='red', label='Smoothed')
    plt.scatter(y_sm[extrema], x[extrema], color='green', s=50, label='Extrema')

    for res in regressions:
        t, yf = get_plot_slices(res)
        plt.plot(yf, t, '-', linewidth=1)

    plt.gca().invert_yaxis()
    plt.xlim(y0, y1)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frame')
    plt.title('Local Extrema and Sin Fits')
    plt.legend()
    plt.savefig(filename)
    plt.close()

    return extrema, y_sm, y, df
